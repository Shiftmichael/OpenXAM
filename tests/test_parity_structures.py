"""MoT-Joint, MoT-IDM, Unified and Hidden must run on the multi-tower core exactly as before.

The same approach as ``test_parity_mot.py``: tiny random towers, the same modules wired into
the old class and into XAMModel, and losses, gradients, sampled actions and generated video
compared bit for bit. Needs PyTorch, but no GPU, weights or data:

    python tests/test_parity_structures.py
"""

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path
from unittest import mock

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_parity_mot as tiny  # noqa: E402  (also puts src on the path and stands in for peft)

from openxam.core.build import build_model, video_clips  # noqa: E402
from openxam.core.model import XAMModel  # noqa: E402
from openxam.core.spec import ModelSpec  # noqa: E402
from openxam.model.backbone.contract import BackboneComponents  # noqa: E402
from openxam.model.component.mot import MoT  # noqa: E402
from openxam.model.easywam_hidden import EasyWAMHidden  # noqa: E402
from openxam.model.easywam_mot_idm import EasyWAMMoTIDM  # noqa: E402
from openxam.model.easywam_mot_joint import EasyWAMMoTJoint  # noqa: E402
from openxam.model.easywam_unified import EasyWAMUnified  # noqa: E402
from openxam.model.helpers.inference import configure_inference_compile  # noqa: E402
from openxam.towers.action import ActionTower  # noqa: E402
from openxam.towers.unified import UnifiedTower  # noqa: E402
from openxam.towers.wan import WanTower  # noqa: E402

STRUCTURES = ("mot_joint", "mot_idm", "unified", "hidden")
MODES = ("first_frame_causal", "bidirectional", "per_frame_causal")
LAYERS = 3
FEATURE_LAYER = 1  # Hidden reads the middle of three layers, so the last one can be skipped.


def video_rows(name: str, mode: str) -> dict:
    """The read-table rows that give the video tower the backbone's ``video_attention_mask_mode``."""
    if mode == "first_frame_causal":
        return {}
    clips = (("obs", "future"), ("obs_c", "future_c")) if name == "mot_idm" else (("obs", "future"),)
    extra = ["action", "state"] if name == "unified" else []
    rows = {}
    for obs, future in clips:
        if mode == "bidirectional":
            rows[obs] = rows[future] = [obs, future, *extra]
        else:
            rows[obs] = [obs, *(["state"] if extra else [])]
            rows[future] = [obs, {"from": future, "pattern": "frame_causal"}, *(["state"] if extra else [])]
    return rows


def config(name: str, mode: str = "first_frame_causal") -> dict:
    config = yaml.safe_load((tiny.ROOT / "configs" / "xam" / f"{name}_wan22.yaml").read_text())
    config["reads"].update(video_rows(name, mode))
    if name == "hidden":
        config["towers"]["action"]["context"][0]["layer"] = FEATURE_LAYER
    return config


def old_model(name: str, seed: int = 0, *, text_encoder=None, dtype=torch.float32, mode=None, vae=None, **extra):
    video, action, tiny_vae = tiny.tiny_modules(seed, layers=LAYERS)
    vae = vae or tiny_vae
    if mode is not None:
        video.video_attention_mask_mode = mode
    common = dict(vae=vae, text_encoder=text_encoder, text_dim=tiny.TEXT_DIM, device="cpu", torch_dtype=dtype)
    if name in ("mot_joint", "mot_idm"):
        cls = EasyWAMMoTJoint if name == "mot_joint" else EasyWAMMoTIDM
        return cls(
            video_expert=video,
            action_expert=action,
            mot=MoT({"video": video, "action": action}),
            state_dim=tiny.STATE_DIM,
            state_position="sequence",
            **common,
            **extra,
        )
    if name == "unified":
        return EasyWAMUnified(
            video_dit=video,
            action_dim=tiny.ACTION_DIM,
            state_dim=tiny.STATE_DIM,
            state_position="sequence",
            projector_hidden_dim=8,
            **common,
        )
    return EasyWAMHidden(
        video_dit=video,
        action_dit=action,
        state_dim=tiny.STATE_DIM,
        state_position="sequence",
        projector_hidden_dim=8,
        video_hidden_layer=FEATURE_LAYER,
        detach_video_hidden=True,
        **common,
    )


def new_model(name: str, old, spec_config: dict | None = None) -> XAMModel:
    spec = ModelSpec.from_dict(spec_config or config(name))
    first = tiny.noise_first_frame(spec)
    if name == "unified":
        towers = {
            "video": UnifiedTower(
                old.video_dit,
                old.vae,
                old.dit["state_encoder"],
                old.dit["action_encoder"],
                old.dit["action_decoder"],
                old.scheduler,
                old.scheduler,
                action_dim=tiny.ACTION_DIM,
                noise_first_frame=first,
            )
        }
    elif name == "hidden":
        towers = {
            "video": WanTower(
                old.video_dit, old.vae, old.train_video_scheduler, old.infer_video_scheduler, noise_first_frame=first
            ),
            "action": ActionTower(
                old.action_dit,
                old.dit["state_encoder"],
                old.train_action_scheduler,
                old.infer_action_scheduler,
                old.dit["video_context_projector"],
            ),
        }
    else:
        towers = {
            "video": WanTower(
                old.video_expert,
                old.vae,
                old.train_video_scheduler,
                old.infer_video_scheduler,
                clips=video_clips(spec),
                noise_first_frame=first,
            ),
            "action": ActionTower(
                old.action_expert, old.state_encoder, old.train_action_scheduler, old.infer_action_scheduler
            ),
        }
    return XAMModel(
        spec,
        towers,
        text_encoder=old.text_encoder,
        model_variant=old.model_variant,
        torch_dtype=old.torch_dtype,
    )


def pair(name: str, **kwargs):
    old = old_model(name, **kwargs)
    return old, new_model(name, old)


def act(model, sample, **kwargs):
    return tiny.act(model, sample, num_video_frames=tiny.FRAMES, **kwargs)


def same_training(old, new, sample, seed: int = 7) -> None:
    loss_old, logs_old, grads_old = tiny.run_training(old, sample, seed)
    loss_new, logs_new, grads_new = tiny.run_training(new, sample, seed)
    assert torch.equal(loss_old, loss_new), (float(loss_old), float(loss_new))
    for key in ("loss_video", "loss_action", "weighted_loss_video", "weighted_loss_action"):
        assert torch.equal(logs_old[key], logs_new[key]), key
    tiny.assert_same_gradients(grads_old, grads_new)


def test_training_loss_and_gradients_match() -> None:
    for name in STRUCTURES:
        old, new = pair(name)
        for seed in range(4):
            same_training(old, new, tiny.batch(), seed)


def test_padded_video_frames_match() -> None:
    for name in STRUCTURES:
        old, new = pair(name)
        sample = tiny.batch()
        sample["image_is_pad"] = torch.zeros(2, tiny.FRAMES, dtype=torch.bool)
        sample["image_is_pad"][1, 1:] = True
        same_training(old, new, sample)


def test_gradient_checkpointing_matches() -> None:
    for name in STRUCTURES:
        old, new = pair(name)
        old.video_dit.use_gradient_checkpointing = True
        if name != "unified":
            new.towers["action"].dit.use_gradient_checkpointing = True
        same_training(old, new, tiny.batch())


def test_idm_condition_noise_both_ways() -> None:
    """The clean copy of the video gets noise never, or always."""
    for probability in (0.0, 1.0):
        old = old_model("mot_idm", video_cond_noise_prob=probability)
        spec = config("mot_idm")
        spec["train"][0]["sometimes_noisy"] = {"future_c": probability}
        new = new_model("mot_idm", old, spec)
        same_training(old, new, tiny.batch())


def test_sampled_actions_match() -> None:
    for name in STRUCTURES:
        for reuse in (False, True):
            old, new = pair(name)
            old.inference_cross_kv_reuse = new.inference_cross_kv_reuse = reuse
            sample = tiny.batch()
            actions = act(old, sample)
            assert actions.abs().sum() > 0
            assert torch.equal(actions, act(new, sample)), (name, reuse)


def test_joint_inference_matches() -> None:
    for name in STRUCTURES:
        old, new = pair(name)
        sample = tiny.batch()
        expected, got = tiny.imagine(old, sample), tiny.imagine(new, sample)
        assert torch.equal(expected["action"], got["action"]), name
        assert len(expected["video"]) == 2 and tiny.same_frames(expected["video"], got["video"]), name


def test_prompts_match() -> None:
    for name in STRUCTURES:
        old = old_model(name, text_encoder=tiny.TinyText())
        new = new_model(name, old)
        sample = tiny.batch()
        arguments = dict(
            prompt=["pick up the bowl", "open the drawer"],
            input_image=sample["video"][:, :, 0],
            action_horizon=tiny.HORIZON,
            proprio=sample["proprio"][:, 0],
            num_inference_steps=4,
            num_video_frames=tiny.FRAMES,
            seed=[11, 12],
        )
        assert torch.equal(old.infer_action_batch(**arguments)["action"], new.infer_action_batch(**arguments)["action"])


def test_single_sample_entry_points_match() -> None:
    for name in STRUCTURES:
        old, new = pair(name)
        sample = tiny.batch()
        arguments = dict(
            prompt=None,
            input_image=sample["video"][0, :, 0],
            action_horizon=tiny.HORIZON,
            proprio=sample["proprio"][0, 0],
            context=sample["context"][0],
            context_mask=sample["context_mask"][0],
            num_inference_steps=3,
            seed=5,
        )
        expected = old.infer_action(num_video_frames=tiny.FRAMES, **arguments)
        assert torch.equal(expected["action"], new.infer_action(num_video_frames=tiny.FRAMES, **arguments)["action"])
        expected = old.infer_joint(num_video_frames=tiny.FRAMES, **arguments)
        got = new.infer_joint(num_video_frames=tiny.FRAMES, **arguments)
        assert torch.equal(expected["action"], got["action"]), name
        assert tiny.same_frames([expected["video"]], [got["video"]]), name
        expected = old.infer(num_frames=tiny.FRAMES, **arguments)
        assert torch.equal(expected["action"], new.infer(num_frames=tiny.FRAMES, **arguments)["action"]), name


def test_other_video_visibility_modes_match() -> None:
    for name in STRUCTURES:
        for mode in MODES[1:]:
            old = old_model(name, mode=mode)
            new = new_model(name, old, config(name, mode))
            sample = tiny.batch()
            same_training(old, new, sample)
            assert torch.equal(act(old, sample), act(new, sample)), (name, mode)


def test_bfloat16_matches() -> None:
    for name in STRUCTURES:
        old, new = pair(name, dtype=torch.bfloat16)
        sample = tiny.batch()
        same_training(old, new, sample)
        assert torch.equal(act(old, sample), act(new, sample)), name
        assert torch.equal(tiny.imagine(old, sample)["action"], tiny.imagine(new, sample)["action"]), name


def test_checkpoints_load_in_both_directions() -> None:
    for name in STRUCTURES:
        sample = tiny.batch()
        trained_old, trained_new = pair(name)
        with torch.no_grad():
            for parameter in trained_old.dit.parameters():
                parameter.add_(0.01)
        want = act(trained_old, sample)
        with tempfile.TemporaryDirectory() as folder:
            from_old, from_new = f"{folder}/old.pt", f"{folder}/new.pt"
            trained_old.save_checkpoint(from_old, step=3)
            trained_new.save_checkpoint(from_new, step=3)
            assert set(torch.load(from_old)) == set(torch.load(from_new)), name

            def untrained():
                """Other weights, but the same VAE, which checkpoints do not carry."""
                return old_model(name, seed=9, vae=trained_old.vae)

            new = new_model(name, untrained())
            assert not torch.equal(act(new, sample), want), name
            assert new.load_checkpoint(from_old)["step"] == 3
            assert torch.equal(act(new, sample), want), name

            old = untrained()
            old.load_checkpoint(from_new)
            assert torch.equal(act(old, sample), want), name


def test_parameter_names_match_the_old_classes() -> None:
    for name in STRUCTURES:
        old, new = pair(name)
        assert [key for key, _ in new.named_parameters()] == [key for key, _ in old.named_parameters()], name
        assert hasattr(new, "mot") == hasattr(old, "mot"), name


def test_the_trainer_trains_the_same_parameters() -> None:
    from openxam.trainer import EasyWAMTrainer

    for name in STRUCTURES:
        old, new = pair(name)
        trainable = []
        for model in (old, new):
            EasyWAMTrainer._apply_dit_only_train_mode(model)
            trainable.append({id(parameter) for parameter in model.parameters() if parameter.requires_grad})
        assert trainable[0] == trainable[1] and trainable[0], name


def test_compiled_step_matches() -> None:
    for name in STRUCTURES:
        old, new = pair(name)
        sample = tiny.batch()
        want = act(old, sample)
        configure_inference_compile(new, True, backend="eager")
        assert torch.equal(want, act(new, sample)), name


def test_entry_points_accept_keywords_the_old_ones_accept() -> None:
    """Evaluators pass keywords, so each old keyword must still be accepted."""
    for cls in (EasyWAMMoTJoint, EasyWAMMoTIDM, EasyWAMUnified, EasyWAMHidden):
        for method in ("infer_action_batch", "infer_joint_batch", "infer_action", "infer_joint", "infer"):
            expected = inspect.signature(getattr(cls, method)).parameters
            got = inspect.signature(getattr(XAMModel, method)).parameters
            missing = [key for key, value in expected.items() if value.kind is not value.VAR_KEYWORD and key not in got]
            assert not missing, (cls.__name__, method, missing)


def test_hidden_stops_at_the_layer_it_reads() -> None:
    """Acting only needs the video tower up to the layer the action tower reads."""
    old, new = pair("hidden")
    sample = tiny.batch()
    for model in (old, new):
        calls = []
        hooks = [block.register_forward_hook(lambda *_: calls.append(1)) for block in model.video_dit.blocks]
        act(model, sample)
        for hook in hooks:
            hook.remove()
        assert len(calls) == FEATURE_LAYER + 1, len(calls)


def test_changing_the_config_changes_the_result() -> None:
    """The comparisons above would pass trivially if the config did not matter."""
    sample = tiny.batch()
    changes = {
        "mot_joint": lambda c: c["reads"].update(action=["obs", "action", "state"], state=["obs", "action", "state"]),
        "mot_idm": lambda c: c["reads"].update(action=["obs_c", "action", "state"], state=["obs_c", "action", "state"]),
        "unified": lambda c: c["reads"].update(obs=["obs"]),
        "hidden": lambda c: c["towers"]["action"]["context"][0].update(layer=0),
    }
    for name, change in changes.items():
        old = old_model(name)
        changed = config(name)
        change(changed)
        assert not torch.equal(act(old, sample), act(new_model(name, old, changed), sample)), name

    old = old_model("hidden")
    changed = config("hidden")
    changed["towers"]["action"]["context"][0]["grad"] = True
    plain, cut = new_model("hidden", old, changed), new_model("hidden", old)
    video = {id(parameter) for parameter in old.video_dit.parameters()}
    sample["action"] = sample["action"] * 3
    reached = [
        sum(float(value.abs().sum()) for key, value in tiny.run_training(model, sample)[2].items() if key in video)
        for model in (plain, cut)
    ]
    assert reached[0] != reached[1]


def test_training_continues_after_inference() -> None:
    """Evaluation runs in inference mode between training steps, and must not break them."""
    for name in (*STRUCTURES, "mot"):
        old = tiny.old_model(*tiny.tiny_modules()) if name == "mot" else old_model(name)
        new = tiny.new_model(old) if name == "mot" else new_model(name, old)
        for model in (old, new):
            sample = tiny.batch()
            tiny.run_training(model, sample)
            act(model, sample)
            tiny.imagine(model, sample)
            tiny.run_training(model, sample)


def components(seed: int = 0) -> BackboneComponents:
    video, _, vae = tiny.tiny_modules(seed, layers=LAYERS)
    return BackboneComponents("wan22", video, vae, None, None, tiny.TEXT_DIM)


def built(name: str) -> XAMModel:
    action_config = {**tiny.ACTION_CONFIG, "num_layers": LAYERS}
    with mock.patch("openxam.core.build.load_easywam_backbone", return_value=components()):
        return build_model(
            ModelSpec.from_dict(config(name)),
            backbone={"name": "wan22"},
            action_dit_config=None if name == "unified" else action_config,
            action_dim=tiny.ACTION_DIM,
            state_dim=tiny.STATE_DIM,
            projector_hidden_dim=8,
            model_variant=name,
            device="cpu",
            torch_dtype=torch.float32,
        )


def test_build_model_gives_working_models() -> None:
    for name in STRUCTURES:
        model = built(name)
        sample = tiny.batch()
        loss, logs = tiny.run_training(model, sample)[:2]
        assert torch.isfinite(loss) and {"loss_video", "loss_action"} <= set(logs), name
        assert act(model, sample).shape == (2, tiny.HORIZON, tiny.ACTION_DIM), name
        assert hasattr(model, "mot") == (name in ("mot_joint", "mot_idm")), name


if __name__ == "__main__":
    for name, check in sorted(globals().items()):
        if name.startswith("test_"):
            check()
            print(f"ok  {name}")
