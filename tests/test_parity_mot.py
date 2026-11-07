"""The multi-tower core must reproduce EasyWAM-MoT exactly.

Builds tiny random towers, wires the same modules into the old class and into XAMModel,
and compares losses, gradients and sampled actions. Needs PyTorch, but no GPU, weights or
data:

    python tests/test_parity_mot.py
"""

from __future__ import annotations

import inspect
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import peft  # noqa: F401

    HAS_PEFT = True
except ImportError:
    # Checkpoint code imports peft even when no LoRA is in use.
    HAS_PEFT = False
    for name in ("peft", "peft.tuners", "peft.tuners.lora", "peft.tuners.lora.layer"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["peft"].LoraConfig = object
    sys.modules["peft"].inject_adapter_in_model = None
    sys.modules["peft.tuners.lora.layer"].LoraLayer = type("LoraLayer", (), {})

from openxam.core.build import build_model, video_visibility  # noqa: E402
from openxam.core.model import XAMModel  # noqa: E402
from openxam.core.spec import ModelSpec  # noqa: E402
from openxam.model.backbone.contract import BackboneComponents  # noqa: E402
from openxam.model.backbone.wan22.wan_video_dit import WanVideoDiT  # noqa: E402
from openxam.model.component.action_dit import ActionDiT  # noqa: E402
from openxam.model.component.mot import MoT  # noqa: E402
from openxam.model.easywam_mot import EasyWAMMoT  # noqa: E402
from openxam.model.helpers.inference import configure_inference_compile  # noqa: E402
from openxam.towers.action import ActionTower  # noqa: E402
from openxam.towers.wan import WanTower  # noqa: E402

TEXT_DIM, STATE_DIM, ACTION_DIM, LATENT_DIM = 12, 5, 3, 4
HORIZON, FRAMES = 8, 5


class TinyVAE(nn.Module):
    """Stands in for the video VAE: keeps every fourth frame and shrinks space by eight."""

    temporal_downsample_factor = 4
    upsampling_factor = 8

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mix", torch.randn(LATENT_DIM, 3))
        self.model = types.SimpleNamespace(z_dim=LATENT_DIM)

    def encode(self, video: torch.Tensor, device=None) -> torch.Tensor:
        pooled = F.avg_pool3d(video[:, :, ::4], (1, 8, 8))
        return torch.einsum("lc,bcthw->blthw", self.mix, pooled)

    def decode(self, latents: torch.Tensor, device=None) -> torch.Tensor:
        pixels = torch.einsum("lc,blthw->bcthw", self.mix, latents)
        return pixels.repeat_interleave(8, dim=3).repeat_interleave(8, dim=4)


class TinyText(nn.Module):
    """Stands in for the text encoder: the same words always give the same tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.dim = TEXT_DIM

    def forward(self, prompts):
        tokens = [
            torch.randn(4, TEXT_DIM, generator=torch.Generator().manual_seed(sum(map(ord, prompt))))
            for prompt in prompts
        ]
        return torch.stack(tokens), torch.ones(len(prompts), 4, dtype=torch.bool)


def tiny_modules(seed: int = 0):
    torch.manual_seed(seed)
    video = WanVideoDiT(
        hidden_dim=48,
        in_dim=LATENT_DIM,
        ffn_dim=64,
        out_dim=LATENT_DIM,
        text_dim=TEXT_DIM,
        freq_dim=16,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        attn_head_dim=24,
        num_layers=2,
        has_image_input=False,
        seperated_timestep=True,
        fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode="first_frame_causal",
    )
    action = ActionDiT(
        hidden_dim=32,
        action_dim=ACTION_DIM,
        ffn_dim=64,
        text_dim=TEXT_DIM,
        freq_dim=16,
        num_heads=2,
        attn_head_dim=24,
        num_layers=2,
        projector_hidden_dim=8,
    )
    return video, action, TinyVAE()


def old_model(video, action, vae, text_encoder=None) -> EasyWAMMoT:
    return EasyWAMMoT(
        video_expert=video,
        action_expert=action,
        mot=MoT({"video": video, "action": action}),
        vae=vae,
        text_encoder=text_encoder,
        text_dim=TEXT_DIM,
        state_dim=STATE_DIM,
        state_position="sequence",
        device="cpu",
        torch_dtype=torch.float32,
    )


def new_model(old: EasyWAMMoT, config: dict | None = None) -> XAMModel:
    config = config or yaml.safe_load((ROOT / "configs" / "xam" / "fastwam_wan22.yaml").read_text())
    return XAMModel(
        ModelSpec.from_dict(config),
        {
            "video": WanTower(old.video_expert, old.vae, old.train_video_scheduler, old.infer_video_scheduler),
            "action": ActionTower(
                old.action_expert, old.state_encoder, old.train_action_scheduler, old.infer_action_scheduler
            ),
        },
        text_encoder=old.text_encoder,
        model_variant=old.model_variant,
    )


def pair():
    old = old_model(*tiny_modules())
    return old, new_model(old)


def batch(seed: int = 1) -> dict:
    generator = torch.Generator().manual_seed(seed)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[1, 4:] = False
    pad = torch.zeros(2, HORIZON, dtype=torch.bool)
    pad[0, 6:] = True
    return {
        "video": torch.rand(2, 3, FRAMES, 32, 32, generator=generator) * 2 - 1,
        "context": torch.randn(2, 6, TEXT_DIM, generator=generator),
        "context_mask": mask,
        "proprio": torch.randn(2, 1, STATE_DIM, generator=generator),
        "action": torch.randn(2, HORIZON, ACTION_DIM, generator=generator),
        "action_is_pad": pad,
    }


def gradients(model: nn.Module, loss: torch.Tensor) -> dict:
    model.zero_grad(set_to_none=True)
    loss.backward()
    found = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return found


def by_parameter(model: nn.Module, grads: dict) -> dict:
    """Key gradients by parameter identity, because the two models name parameters differently."""
    names = {name: id(parameter) for name, parameter in model.named_parameters()}
    return {names[name]: value for name, value in grads.items()}


def run_training(model, sample, seed: int = 7):
    model.train()
    torch.manual_seed(seed)
    loss, logs = model.training_loss(sample)
    return loss, logs, by_parameter(model, gradients(model, loss))


def assert_same_gradients(first: dict, second: dict) -> None:
    assert first.keys() == second.keys(), (len(first), len(second))
    assert len(first) > 10
    assert any(value.abs().sum() > 0 for value in first.values())
    for key, value in first.items():
        assert torch.equal(value, second[key]), float((value - second[key]).abs().max())


def test_training_loss_and_gradients_match() -> None:
    old, new = pair()
    sample = batch()
    loss_old, logs_old, grads_old = run_training(old, sample)
    loss_new, logs_new, grads_new = run_training(new, sample)
    assert torch.equal(loss_old, loss_new), (float(loss_old), float(loss_new))
    for key in ("loss_video", "loss_action", "weighted_loss_video", "weighted_loss_action"):
        assert torch.equal(logs_old[key], logs_new[key]), key
    assert_same_gradients(grads_old, grads_new)


def test_padded_video_frames_match() -> None:
    old, new = pair()
    sample = batch()
    sample["image_is_pad"] = torch.zeros(2, FRAMES, dtype=torch.bool)
    sample["image_is_pad"][1, 1:] = True
    assert torch.equal(run_training(old, sample)[0], run_training(new, sample)[0])


def test_gradient_checkpointing_matches() -> None:
    old, new = pair()
    old.video_expert.use_gradient_checkpointing = True
    old.action_expert.use_gradient_checkpointing = True
    sample = batch()
    loss_old, _, grads_old = run_training(old, sample)
    loss_new, _, grads_new = run_training(new, sample)
    assert torch.equal(loss_old, loss_new)
    assert_same_gradients(grads_old, grads_new)


def act(model, sample, **kwargs) -> torch.Tensor:
    return model.infer_action_batch(
        prompt=None,
        input_image=sample["video"][:, :, 0],
        action_horizon=HORIZON,
        proprio=sample["proprio"][:, 0],
        context=sample["context"],
        context_mask=sample["context_mask"],
        num_inference_steps=4,
        seed=[11, 12],
        **kwargs,
    )["action"]


def test_sampled_actions_match() -> None:
    old, new = pair()
    sample = batch()
    actions = act(old, sample)
    assert actions.abs().sum() > 0
    assert torch.equal(actions, act(new, sample))


def test_sampled_actions_match_with_cross_attention_cache() -> None:
    old, new = pair()
    old.inference_cross_kv_reuse = True
    new.inference_cross_kv_reuse = True
    sample = batch()
    assert torch.equal(act(old, sample), act(new, sample))


def action_only_config(detach: bool) -> dict:
    config = yaml.safe_load((ROOT / "configs" / "xam" / "fastwam_wan22.yaml").read_text())
    config["losses"] = [{"stream": "action", "type": "flow"}]
    config["train"][0]["noisy"] = ["action", "future"]
    read = {"from": "obs", "grad": False} if detach else "obs"
    config["reads"]["action"] = [read, "action", "state"]
    config["reads"]["state"] = [read, "action", "state"]
    return config


def test_cutting_a_read_stops_gradients_but_not_values() -> None:
    old = old_model(*tiny_modules())
    plain = new_model(old, action_only_config(detach=False))
    cut = new_model(old, action_only_config(detach=True))
    sample = batch()
    loss_plain, _, grads_plain = run_training(plain, sample)
    loss_cut, _, grads_cut = run_training(cut, sample)
    assert torch.allclose(loss_plain, loss_cut, atol=1e-6)
    video_parameters = {id(parameter) for parameter in old.video_expert.parameters()}
    assert any(key in video_parameters and value.abs().sum() > 0 for key, value in grads_plain.items())
    assert not any(key in video_parameters and value.abs().sum() > 0 for key, value in grads_cut.items())


def imagine(model, sample, **kwargs) -> dict:
    return model.infer_joint_batch(
        prompt=None,
        input_image=sample["video"][:, :, 0],
        num_video_frames=FRAMES,
        action_horizon=HORIZON,
        proprio=sample["proprio"][:, 0],
        context=sample["context"],
        context_mask=sample["context_mask"],
        num_inference_steps=4,
        seed=[11, 12],
        **kwargs,
    )


def same_frames(first: list, second: list) -> bool:
    return len(first) == len(second) and all(
        np.array_equal(np.asarray(a), np.asarray(b)) for x, y in zip(first, second) for a, b in zip(x, y)
    )


def test_joint_inference_matches() -> None:
    old, new = pair()
    sample = batch()
    expected, got = imagine(old, sample), imagine(new, sample)
    assert torch.equal(expected["action"], got["action"])
    assert len(expected["video"]) == 2 and same_frames(expected["video"], got["video"])


def test_joint_inference_matches_with_cross_attention_cache() -> None:
    old, new = pair()
    old.inference_cross_kv_reuse = True
    new.inference_cross_kv_reuse = True
    sample = batch()
    expected, got = imagine(old, sample, decode_video=False), imagine(new, sample, decode_video=False)
    assert torch.equal(expected["action"], got["action"])
    assert "video" not in got


def test_prompts_match() -> None:
    old = old_model(*tiny_modules(), TinyText())
    new = new_model(old)
    sample = batch()
    arguments = dict(
        prompt=["pick up the bowl", "open the drawer"],
        input_image=sample["video"][:, :, 0],
        action_horizon=HORIZON,
        proprio=sample["proprio"][:, 0],
        num_inference_steps=4,
        seed=[11, 12],
    )
    assert torch.equal(old.infer_action_batch(**arguments)["action"], new.infer_action_batch(**arguments)["action"])


def test_single_sample_entry_points_match() -> None:
    old, new = pair()
    sample = batch()
    arguments = dict(
        prompt=None,
        input_image=sample["video"][0, :, 0],
        action_horizon=HORIZON,
        proprio=sample["proprio"][0, 0],
        context=sample["context"][0],
        context_mask=sample["context_mask"][0],
        num_inference_steps=3,
        seed=5,
    )
    assert torch.equal(old.infer_action(**arguments)["action"], new.infer_action(**arguments)["action"])
    expected = old.infer_joint(num_video_frames=FRAMES, **arguments)
    got = new.infer_joint(num_video_frames=FRAMES, **arguments)
    assert torch.equal(expected["action"], got["action"])
    assert same_frames([expected["video"]], [got["video"]])
    expected = old.infer(num_frames=FRAMES, **arguments)
    got = new.infer(num_frames=FRAMES, **arguments)
    assert torch.equal(expected["action"], got["action"])


def test_checkpoints_load_in_both_directions() -> None:
    sample = batch()
    trained_old, trained_new = pair()
    with torch.no_grad():
        for parameter in trained_old.mot.parameters():
            parameter.add_(0.01)
    want = act(trained_old, sample)
    with tempfile.TemporaryDirectory() as folder:
        from_old, from_new = f"{folder}/old.pt", f"{folder}/new.pt"
        trained_old.save_checkpoint(from_old, step=3)
        trained_new.save_checkpoint(from_new, step=3)
        assert set(torch.load(from_old)) == set(torch.load(from_new))

        def untrained() -> EasyWAMMoT:
            """Other weights, but the same VAE, which checkpoints do not carry."""
            video, action, _ = tiny_modules(seed=9)
            return old_model(video, action, trained_old.vae)

        new = new_model(untrained())
        assert not torch.equal(act(new, sample), want)
        assert new.load_checkpoint(from_old)["step"] == 3
        assert torch.equal(act(new, sample), want)

        old = untrained()
        old.load_checkpoint(from_new)
        assert torch.equal(act(old, sample), want)


def test_a_checkpoint_of_another_structure_is_refused() -> None:
    old, new = pair()
    with tempfile.TemporaryDirectory() as folder:
        path = f"{folder}/model.pt"
        old.save_checkpoint(path)
        new.model_variant = "mot_joint"
        try:
            new.load_checkpoint(path)
        except ValueError as error:
            assert "model_variant" in str(error)
        else:
            raise AssertionError("a checkpoint of another structure must be refused")


def test_entry_points_accept_what_the_old_ones_accept() -> None:
    for name in ("infer_action_batch", "infer_joint_batch", "infer_action", "infer_joint", "infer"):
        expected = inspect.signature(getattr(EasyWAMMoT, name)).parameters
        got = inspect.signature(getattr(XAMModel, name)).parameters
        assert list(expected) == list(got), name


def test_parameter_names_match_the_old_class() -> None:
    old, new = pair()
    assert [name for name, _ in new.named_parameters()] == [name for name, _ in old.named_parameters()]


def test_the_trainer_trains_the_same_parameters() -> None:
    from openxam.trainer import EasyWAMTrainer

    old, new = pair()
    trainable = []
    for model in (old, new):
        EasyWAMTrainer._apply_dit_only_train_mode(model)
        trainable.append({id(parameter) for parameter in model.parameters() if parameter.requires_grad})
    assert trainable[0] == trainable[1] and trainable[0]
    assert new.dit.training and not new.vae.training
    assert run_training_keeps_gradients(new)


def run_training_keeps_gradients(model) -> bool:
    torch.manual_seed(1)
    loss, _ = model(batch())
    loss.backward()
    found = any(parameter.grad is not None for parameter in model.dit.parameters())
    model.zero_grad(set_to_none=True)
    return found


def test_compiled_step_matches() -> None:
    old, new = pair()
    sample = batch()
    want = act(old, sample)
    configure_inference_compile(new, True, backend="eager")
    assert torch.equal(want, act(new, sample))


def components(seed: int = 0, text_encoder=None) -> BackboneComponents:
    video, _, vae = tiny_modules(seed)
    return BackboneComponents("wan22", video, vae, text_encoder, None, TEXT_DIM)


ACTION_CONFIG = dict(
    hidden_dim=32,
    action_dim=ACTION_DIM,
    ffn_dim=64,
    text_dim=TEXT_DIM,
    freq_dim=16,
    num_heads=2,
    attn_head_dim=24,
    num_layers=2,
    projector_hidden_dim=8,
)


def built(spec="fastwam_wan22", **kwargs) -> XAMModel:
    with mock.patch("openxam.core.build.load_easywam_backbone", return_value=components()):
        return build_model(
            spec,
            backbone={"name": "wan22"},
            action_dit_config=ACTION_CONFIG,
            state_dim=STATE_DIM,
            projector_hidden_dim=8,
            model_variant="mot",
            device="cpu",
            torch_dtype=torch.float32,
            **kwargs,
        )


def test_build_model_gives_a_working_model() -> None:
    model = built()
    assert model.backbone_name == "wan22" and model.model_variant == "mot"
    sample = batch()
    loss, logs = run_training(model, sample)[:2]
    assert torch.isfinite(loss) and {"loss_video", "loss_action"} <= set(logs)
    assert act(model, sample).shape == (2, HORIZON, ACTION_DIM)


def test_build_model_refuses_a_backbone_that_disagrees_with_the_read_table() -> None:
    config = yaml.safe_load((ROOT / "configs" / "xam" / "fastwam_wan22.yaml").read_text())
    config["reads"]["obs"] = ["obs", "future"]
    spec = ModelSpec.from_dict(config)
    assert video_visibility(spec) == "bidirectional"
    try:
        built(spec)
    except ValueError as error:
        assert "visibility" in str(error)
    else:
        raise AssertionError("the backbone setting and the read table must agree")


if __name__ == "__main__":
    for name, check in sorted(globals().items()):
        if name.startswith("test_"):
            check()
            print(f"ok  {name}")
