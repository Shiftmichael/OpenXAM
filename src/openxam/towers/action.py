"""The action tower: actions, with the robot state as extra tokens behind them."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from openxam.core.plan import Plan
from openxam.core.spec import NOISY
from openxam.model.helpers.batching import randn_per_sample

from .base import Context, TowerState, flow_weighted_mean, shared_timestep
from .dit import DiTTower

ACTION = "action"
STATE = "state"


def prepare_action(
    sample: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    with_state: bool,
) -> dict[str, Any]:
    """Actions and robot state of a raw batch, ready for a tower."""
    action = sample["action"]
    if action.ndim != 3:
        raise ValueError(f"`action` must be [B, T, D], got {tuple(action.shape)}")
    pad = sample.get("action_is_pad")
    if pad is not None:
        pad = pad.to(device=device, dtype=torch.bool, non_blocking=True)
    state = None
    if with_state:
        proprio = sample.get("proprio")
        if proprio is None:
            raise ValueError("`proprio` is required when the state stream is enabled.")
        if proprio.ndim != 3:
            raise ValueError(f"`proprio` must be [B, T, D], got {tuple(proprio.shape)}")
        state = proprio[:, 0, :].to(device=device, dtype=dtype)
    return {
        "action": action.to(device=device, dtype=dtype, non_blocking=True),
        "action_is_pad": pad,
        "state": state,
    }


def noise_action(
    batch: dict[str, Any],
    plan: Plan,
    timesteps: dict,
    scheduler: Any,
    device: torch.device,
) -> None:
    action = batch["action"]
    if plan.states.get(ACTION) != NOISY:
        batch["noisy"] = action
        batch["t"] = action.new_zeros(action.shape[0])
        return
    noise = torch.randn_like(action)
    timestep = shared_timestep(
        ACTION,
        plan,
        timesteps,
        lambda: scheduler.sample_training_t(action.shape[0], device, action.dtype),
    )
    batch["noisy"] = scheduler.add_noise(action, noise, timestep)
    batch["t"] = timestep
    batch["target"] = scheduler.training_target(action, noise, timestep)


def action_flow_loss(prediction: Tensor, batch: Mapping[str, Any], scheduler: Any) -> Tensor:
    per_token = F.mse_loss(prediction.float(), batch["target"].float(), reduction="none").mean(dim=2)
    pad = batch["action_is_pad"]
    if pad is None:
        per_sample = per_token.mean(dim=1)
    else:
        valid = (~pad).to(device=per_token.device, dtype=per_token.dtype)
        per_sample = (per_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
    return flow_weighted_mean(per_sample, scheduler, batch["t"])


def action_noise(action_dim: int, request: Mapping[str, Any]) -> Tensor:
    return randn_per_sample(
        (request["action_horizon"], action_dim),
        seeds=request["seed"],
        batch_size=request["batch_size"],
        rand_device=request["rand_device"],
    )


class ActionTower(DiTTower):
    def __init__(
        self,
        dit: nn.Module,
        state_encoder: Optional[nn.Module],
        train_scheduler: Any,
        infer_scheduler: Any,
        feature_projector: Optional[nn.Module] = None,
    ):
        super().__init__(dit)
        self.state_encoder = state_encoder
        self.train_scheduler = train_scheduler
        self.infer_scheduler = infer_scheduler
        self.feature_projector = feature_projector

    @property
    def device(self) -> torch.device:
        return next(self.dit.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.dit.parameters()).dtype

    def checkpoint_parts(self) -> dict[str, nn.Module]:
        parts = {"action_dit": self.dit}
        if self.feature_projector is not None:
            parts["video_context_projector"] = self.feature_projector
        if self.state_encoder is not None:
            parts["state_encoder"] = self.state_encoder
        return parts

    def prepare(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        return prepare_action(sample, self.device, self.dtype, self.state_encoder is not None)

    def add_noise(self, batch: dict[str, Any], plan: Plan, timesteps: dict) -> None:
        noise_action(batch, plan, timesteps, self.train_scheduler, self.device)

    def _with_state(self, pre: dict[str, Any], state: Optional[Tensor]) -> TowerState:
        action_len = int(pre["tokens"].shape[1])
        if self.state_encoder is None or state is None:
            return TowerState(pre["tokens"], {ACTION: action_len}, 1, pre)
        tokens = self.state_encoder(state.to(device=self.device, dtype=pre["tokens"].dtype).unsqueeze(1))
        pre = self.dit.append_state_tokens(pre, tokens.to(dtype=pre["tokens"].dtype))
        return TowerState(pre["tokens"], {ACTION: action_len, STATE: int(tokens.shape[1])}, 1, pre)

    def _pre(self, action: Tensor, timestep: Tensor, context: Context) -> dict[str, Any]:
        return self.dit.pre_dit(
            action_tokens=action,
            timestep=timestep,
            context=context.tokens,
            context_mask=context.mask,
            context_is_projected=context.projected,
            cross_kv_cache=context.kv_cache,
        )

    def embed(self, batch: Mapping[str, Any], context: Context) -> TowerState:
        return self._with_state(self._pre(batch["noisy"], batch["t"], context), batch["state"])

    def readout(self, stream: str, x: Tensor, state: TowerState) -> Tensor:
        if stream != ACTION:
            raise ValueError(f"The action tower predicts `{ACTION}`, not {stream!r}.")
        return self.dit.post_dit(x, state.extras)

    def flow_loss(self, stream: str, prediction: Tensor, batch: Mapping[str, Any]) -> Tensor:
        return action_flow_loss(prediction, batch, self.train_scheduler)

    def project_context(self, context: Tensor) -> Optional[Tensor]:
        return self.dit.project_context(context) if self.dit.cross_attention else None

    def project_features(self, features: Tensor) -> Tensor:
        if self.feature_projector is None:
            raise ValueError("This action tower has no projector for another tower's features.")
        return self.dit.project_context(self.feature_projector(features))

    def start(self, stream: str, request: Mapping[str, Any]) -> Tensor:
        return action_noise(self.dit.action_dim, request).to(device=self.device, dtype=self.dtype)

    def embed_step(
        self,
        plan: Plan,
        request: Mapping[str, Any],
        latents: Mapping[str, Tensor],
        timesteps: Mapping[str, Tensor],
    ) -> TowerState:
        pre = self._pre(latents[ACTION], timesteps[ACTION], request["context"]["action"])
        return self._with_state(pre, request.get("state"))
