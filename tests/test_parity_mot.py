"""The multi-tower core must reproduce EasyWAM-MoT exactly.

Builds tiny random towers, wires the same modules into the old class and into XAMModel,
and compares losses, gradients and sampled actions. Needs PyTorch, but no GPU, weights or
data:

    python tests/test_parity_mot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openxam.core.model import XAMModel  # noqa: E402
from openxam.core.spec import ModelSpec  # noqa: E402
from openxam.model.backbone.wan22.wan_video_dit import WanVideoDiT  # noqa: E402
from openxam.model.component.action_dit import ActionDiT  # noqa: E402
from openxam.model.component.mot import MoT  # noqa: E402
from openxam.model.easywam_mot import EasyWAMMoT  # noqa: E402
from openxam.towers.action import ActionTower  # noqa: E402
from openxam.towers.wan import WanTower  # noqa: E402

TEXT_DIM, STATE_DIM, ACTION_DIM, LATENT_DIM = 12, 5, 3, 4
HORIZON, FRAMES = 8, 5


class TinyVAE(nn.Module):
    """Stands in for the video VAE: keeps every fourth frame and shrinks space by eight."""

    temporal_downsample_factor = 4

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mix", torch.randn(LATENT_DIM, 3))

    def encode(self, video: torch.Tensor, device=None) -> torch.Tensor:
        pooled = F.avg_pool3d(video[:, :, ::4], (1, 8, 8))
        return torch.einsum("lc,bcthw->blthw", self.mix, pooled)


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


def old_model(video, action, vae) -> EasyWAMMoT:
    return EasyWAMMoT(
        video_expert=video,
        action_expert=action,
        mot=MoT({"video": video, "action": action}),
        vae=vae,
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


if __name__ == "__main__":
    for name, check in sorted(globals().items()):
        if name.startswith("test_"):
            check()
            print(f"ok  {name}")
