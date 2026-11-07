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


class ActionTower(DiTTower):
    def __init__(
        self,
        dit: nn.Module,
        state_encoder: Optional[nn.Module],
        train_scheduler: Any,
        infer_scheduler: Any,
    ):
        super().__init__(dit)
        self.state_encoder = state_encoder
        self.train_scheduler = train_scheduler
        self.infer_scheduler = infer_scheduler

    @property
    def device(self) -> torch.device:
        return next(self.dit.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.dit.parameters()).dtype

    def prepare(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`action` must be [B, T, D], got {tuple(action.shape)}")
        pad = sample.get("action_is_pad")
        if pad is not None:
            pad = pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        state = None
        if self.state_encoder is not None:
            proprio = sample.get("proprio")
            if proprio is None:
                raise ValueError("`proprio` is required when the state stream is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`proprio` must be [B, T, D], got {tuple(proprio.shape)}")
            state = proprio[:, 0, :].to(device=self.device, dtype=self.dtype)
        return {
            "action": action.to(device=self.device, dtype=self.dtype, non_blocking=True),
            "action_is_pad": pad,
            "state": state,
        }

    def add_noise(self, batch: dict[str, Any], plan: Plan, timesteps: dict) -> None:
        action = batch["action"]
        scheduler = self.train_scheduler
        if plan.states.get(ACTION) != NOISY:
            batch["noisy"] = action
            batch["t"] = action.new_zeros(action.shape[0])
            return
        noise = torch.randn_like(action)
        timestep = shared_timestep(
            ACTION,
            plan,
            timesteps,
            lambda: scheduler.sample_training_t(action.shape[0], self.device, action.dtype),
        )
        batch["noisy"] = scheduler.add_noise(action, noise, timestep)
        batch["t"] = timestep
        batch["target"] = scheduler.training_target(action, noise, timestep)

    def _with_state(self, pre: dict[str, Any], state: Optional[Tensor]) -> TowerState:
        action_len = int(pre["tokens"].shape[1])
        if self.state_encoder is None or state is None:
            return TowerState(pre["tokens"], {ACTION: action_len}, 1, pre)
        tokens = self.state_encoder(state.to(device=self.device, dtype=pre["tokens"].dtype).unsqueeze(1))
        pre = self.dit.append_state_tokens(pre, tokens.to(dtype=pre["tokens"].dtype))
        return TowerState(pre["tokens"], {ACTION: action_len, STATE: int(tokens.shape[1])}, 1, pre)

    def _pre(self, action: Tensor, timestep: Tensor, context: Context) -> dict[str, Any]:
        return self.dit.pre_dit(
            prepare_rotary_factors=False,
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
        per_token = F.mse_loss(prediction.float(), batch["target"].float(), reduction="none").mean(dim=2)
        pad = batch["action_is_pad"]
        if pad is None:
            per_sample = per_token.mean(dim=1)
        else:
            valid = (~pad).to(device=per_token.device, dtype=per_token.dtype)
            per_sample = (per_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return flow_weighted_mean(per_sample, self.train_scheduler, batch["t"])

    def project_context(self, context: Tensor) -> Optional[Tensor]:
        return self.dit.project_context(context) if self.dit.cross_attention else None

    def start(self, stream: str, request: Mapping[str, Any]) -> Tensor:
        noise = randn_per_sample(
            (request["action_horizon"], self.dit.action_dim),
            seeds=request["seed"],
            batch_size=request["batch_size"],
            rand_device=request["rand_device"],
        )
        return noise.to(device=self.device, dtype=self.dtype)

    def embed_step(
        self,
        request: Mapping[str, Any],
        latents: Mapping[str, Tensor],
        timesteps: Mapping[str, Tensor],
    ) -> TowerState:
        pre = self._pre(latents[ACTION], timesteps[ACTION], request["context"]["action"])
        return self._with_state(pre, request.get("state"))
