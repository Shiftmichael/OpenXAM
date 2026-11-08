"""Build an XAMModel from a spec and the backbone configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Union

import torch
from torch import nn

from openxam.model.backbone import load_easywam_backbone, normalize_backbone_config
from openxam.model.backbone.protocol import BLOCK_PROTOCOL_STANDARD
from openxam.model.component.action_dit import ActionDecoder, ActionDiT, ActionEncoder, StateEncoder
from openxam.model.schedulers.factory import build_video_schedulers
from openxam.model.schedulers.scheduler_continuous import ContinuousFlowMatchScheduler
from openxam.towers.action import ACTION, ActionTower
from openxam.towers.unified import UnifiedTower
from openxam.towers.wan import FUTURE_FRAMES, WanTower

from .model import XAMModel
from .spec import FRAME_CAUSAL, ModelSpec

SPEC_DIR = Path(__file__).resolve().parents[3] / "configs" / "xam"


def load_spec(spec: Union[str, Path, ModelSpec]) -> ModelSpec:
    """A spec given as itself, as a path, or by its name in ``configs/xam``."""
    if isinstance(spec, ModelSpec):
        return spec
    path = Path(spec)
    if path.suffix != ".yaml":
        path = SPEC_DIR / f"{path}.yaml"
    return ModelSpec.load(path)


def video_visibility(spec: ModelSpec) -> str:
    """The backbone's ``video_attention_mask_mode`` that the read table corresponds to."""
    if any(read.source == FUTURE_FRAMES for read in spec.reads["obs"]):
        return "bidirectional"
    if any(read.pattern == FRAME_CAUSAL for read in spec.reads[FUTURE_FRAMES]):
        return "per_frame_causal"
    return "first_frame_causal"


def video_clips(spec: ModelSpec) -> tuple[tuple[str, str], ...]:
    """The video tower's clips: each a current-frame stream and the future-frames stream after it."""
    streams = [stream for stream in spec.streams.values() if stream.tower == "video"]
    firsts = [stream.name for stream in streams if stream.data == ("first_frame",)]
    futures = [stream.name for stream in streams if stream.data == ("future_frames",)]
    if len(firsts) != len(futures):
        raise ValueError(f"Every current frame needs future frames after it, got {firsts} and {futures}.")
    return tuple(zip(firsts, futures))


def build_model(
    spec: Union[str, Path, ModelSpec],
    *,
    backbone: Mapping[str, Any],
    action_dit_config: Optional[Mapping[str, Any]] = None,
    action_dim: Optional[int] = None,
    action_dit_pretrained_path: Optional[str] = None,
    state_dim: Optional[int] = None,
    projector_hidden_dim: int = 64,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler: Optional[Mapping[str, Any]] = None,
    action_scheduler: Optional[Mapping[str, Any]] = None,
    model_variant: Optional[str] = None,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> XAMModel:
    spec = load_spec(spec)
    config = normalize_backbone_config(backbone)
    components = load_easywam_backbone(
        config,
        device=device,
        torch_dtype=torch_dtype,
        skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
    )
    if components.block_protocol != BLOCK_PROTOCOL_STANDARD:
        raise NotImplementedError(f"Backbone {config['name']!r} is not migrated yet.")
    video = components.dit
    if video.video_attention_mask_mode != video_visibility(spec):
        raise ValueError(
            f"The read table means video visibility {video_visibility(spec)!r}, but the backbone "
            f"is configured for {video.video_attention_mask_mode!r}."
        )

    video_config = dict(video_scheduler or {})
    train_video, infer_video = build_video_schedulers(
        family=video.scheduler_family,
        config=config.get("video_scheduler"),
        num_train_timesteps=int(video_config.get("num_train_timesteps", 1000)),
        train_shift=float(video_config.get("train_shift", 5.0)),
        infer_shift=float(video_config.get("infer_shift", 5.0)),
    )
    noise_first_frame = bool(spec.towers["video"].options.get("noise_first_frame", False))
    if spec.streams[ACTION].tower == "video":
        if action_dim is None or state_dim is None:
            raise ValueError("A single tower needs `action_dim` and `state_dim`.")
        width, projector = int(video.hidden_dim), int(projector_hidden_dim)
        parts = (
            StateEncoder(state_dim=int(state_dim), hidden_dim=width, projector_hidden_dim=projector),
            ActionEncoder(action_dim=int(action_dim), hidden_dim=width, projector_hidden_dim=projector),
            ActionDecoder(hidden_dim=width, action_dim=int(action_dim), projector_hidden_dim=projector),
        )
        towers = {
            "video": UnifiedTower(
                video,
                components.vae,
                *(part.to(device=device, dtype=torch_dtype) for part in parts),
                train_video,
                infer_video,
                action_dim=int(action_dim),
                noise_first_frame=noise_first_frame,
            )
        }
    else:
        if action_dit_config is None:
            raise ValueError("A separate action tower needs `action_dit_config`.")
        towers = {
            "video": WanTower(
                video,
                components.vae,
                train_video,
                infer_video,
                clips=video_clips(spec),
                noise_first_frame=noise_first_frame,
            ),
            "action": _action_tower(
                spec,
                video,
                action_dit_config,
                action_dit_pretrained_path,
                state_dim,
                projector_hidden_dim,
                skip_dit_load_from_pretrain,
                action_scheduler,
                device,
                torch_dtype,
            ),
        }

    return XAMModel(
        spec,
        towers,
        text_encoder=components.text_encoder,
        tokenizer=components.tokenizer,
        backbone_name=config["name"],
        model_variant=model_variant,
        device=device,
        torch_dtype=torch_dtype,
    )


def _action_tower(
    spec: ModelSpec,
    video: Any,
    action_dit_config: Mapping[str, Any],
    action_dit_pretrained_path: Optional[str],
    state_dim: Optional[int],
    projector_hidden_dim: int,
    skip_dit_load_from_pretrain: bool,
    action_scheduler: Optional[Mapping[str, Any]],
    device: str,
    torch_dtype: torch.dtype,
) -> ActionTower:
    start = spec.towers[ACTION].options.get("init", "from_video")
    if start not in ("from_video", "random"):
        raise ValueError(f"Unknown action tower `init`: {start!r}")
    action = ActionDiT.from_pretrained(
        action_dit_config=dict(action_dit_config),
        action_dit_pretrained_path=action_dit_pretrained_path if start == "from_video" else None,
        skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
        device=device,
        torch_dtype=torch_dtype,
    )
    action_config = dict(action_scheduler or {})
    steps = int(action_config.get("num_train_timesteps", 1000))
    train_action = ContinuousFlowMatchScheduler(steps, float(action_config.get("train_shift", 5.0)))
    infer_action = ContinuousFlowMatchScheduler(steps, float(action_config.get("infer_shift", 5.0)))

    state_encoder = None
    if state_dim is not None:
        state_encoder = StateEncoder(
            state_dim=int(state_dim),
            hidden_dim=int(action.hidden_dim),
            projector_hidden_dim=int(projector_hidden_dim),
        ).to(device=device, dtype=torch_dtype)
    projector = None
    if ACTION in spec.feature_reads:
        projector = nn.Linear(int(video.hidden_dim), int(action.text_dim)).to(device=device, dtype=torch_dtype)
    return ActionTower(action, state_encoder, train_action, infer_action, projector)
