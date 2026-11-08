"""One tower for everything: action and state tokens run through the video transformer."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn

from openxam.core.plan import Plan

from .action import ACTION, STATE, action_flow_loss, action_noise, noise_action, prepare_action
from .base import Context, TowerState
from .wan import CURRENT_FRAME, FUTURE_FRAMES, WanTower


class UnifiedTower(WanTower):
    def __init__(
        self,
        dit: nn.Module,
        vae: nn.Module,
        state_encoder: nn.Module,
        action_encoder: nn.Module,
        action_decoder: nn.Module,
        train_scheduler: Any,
        infer_scheduler: Any,
        *,
        action_dim: int,
        noise_first_frame: bool = False,
    ):
        super().__init__(dit, vae, train_scheduler, infer_scheduler, noise_first_frame=noise_first_frame)
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.action_decoder = action_decoder
        self.action_dim = int(action_dim)

    def checkpoint_parts(self) -> dict[str, nn.Module]:
        return {
            "video_dit": self.dit,
            "state_encoder": self.state_encoder,
            "action_encoder": self.action_encoder,
            "action_decoder": self.action_decoder,
        }

    def prepare(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        return {**super().prepare(sample), **prepare_action(sample, self.device, self.dtype, True)}

    def add_noise(self, batch: dict[str, Any], plan: Plan, timesteps: dict) -> None:
        super().add_noise(batch, plan, timesteps)
        noise_action(batch, plan, timesteps, self.train_scheduler, self.device)

    def _everything(
        self,
        video: Tensor,
        video_timestep: Tensor,
        action: Tensor,
        action_timestep: Tensor,
        state: Tensor,
        context: Context,
    ) -> TowerState:
        pre = self.dit.pre_unified_dit(
            x=video,
            timestep_video=video_timestep,
            action_tokens=self.action_encoder(action),
            timestep_action=action_timestep,
            state_tokens=self.state_encoder(state.to(device=video.device, dtype=video.dtype).unsqueeze(1)),
            timestep_state=torch.zeros_like(action_timestep),
            state_position="sequence",
            context=context.tokens,
            context_mask=context.mask,
            context_is_projected=context.projected,
            cross_kv_cache=context.kv_cache,
        )
        meta = pre["meta"]
        frame = int(meta["tokens_per_frame"])
        lengths = {
            CURRENT_FRAME: frame,
            FUTURE_FRAMES: int(meta["video_len"]) - frame,
            ACTION: int(meta["action_len"]),
            STATE: int(meta["state_len"]),
        }
        return TowerState(pre["tokens"], lengths, frame, pre)

    def embed(self, batch: Mapping[str, Any], context: Context) -> TowerState:
        video = batch["clips"][FUTURE_FRAMES]
        return self._everything(video.latents, video.timestep, batch["noisy"], batch["t"], batch["state"], context)

    def readout(self, stream: str, x: Tensor, state: TowerState) -> Tensor:
        outputs = self.dit.post_unified_dit(x, state.extras)
        if stream == ACTION:
            return self.action_decoder(outputs["action_tokens"])
        return outputs["video"]

    def flow_loss(self, stream: str, prediction: Tensor, batch: Mapping[str, Any]) -> Tensor:
        if stream == ACTION:
            return action_flow_loss(prediction, batch, self.train_scheduler)
        return super().flow_loss(stream, prediction, batch)

    def encode_request(self, request: dict[str, Any]) -> None:
        if "state" not in request:
            raise ValueError("A single tower reads the robot state, so inference needs `proprio`.")
        super().encode_request(request)

    def start(self, stream: str, request: Mapping[str, Any]) -> Tensor:
        if stream == ACTION:
            return action_noise(self.action_dim, request).to(device=self.device, dtype=self.dtype)
        return super().start(stream, request)

    def embed_step(
        self,
        plan: Plan,
        request: Mapping[str, Any],
        latents: Mapping[str, Tensor],
        timesteps: Mapping[str, Tensor],
    ) -> TowerState:
        return self._everything(
            latents[FUTURE_FRAMES],
            timesteps[FUTURE_FRAMES],
            latents[ACTION],
            timesteps[ACTION],
            request["state"],
            request["context"]["video"],
        )
