"""The Wan video tower: the current frame and the future frames run as one clip."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from openxam.core.plan import Plan
from openxam.core.spec import NOISY
from openxam.model.helpers.batching import encode_image_batch

from .base import Context, TowerState, flow_weighted_mean, shared_timestep
from .dit import DiTTower

CURRENT_FRAME = "obs"
FUTURE_FRAMES = "future"


def _video_loss_per_sample(
    prediction: Tensor,
    target: Tensor,
    image_is_pad: Optional[Tensor],
    temporal_factor: int,
) -> Tensor:
    per_step = F.mse_loss(prediction.float(), target.float(), reduction="none").mean(dim=(1, 3, 4))
    if image_is_pad is None:
        return per_step.mean(dim=1)
    if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
        raise ValueError(
            f"Cannot align `image_is_pad` with video latent steps: {image_is_pad.shape[1]} frames, "
            f"temporal factor {temporal_factor}."
        )
    pad = image_is_pad[:, 1:].view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
    if pad.shape[1] != per_step.shape[1]:
        raise ValueError(f"Video-loss mask has {pad.shape[1]} steps, the loss has {per_step.shape[1]}.")
    valid = (~pad).to(device=per_step.device, dtype=per_step.dtype)
    return (per_step * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)


class WanTower(DiTTower):
    def __init__(self, dit: nn.Module, vae: nn.Module, train_scheduler: Any, infer_scheduler: Any):
        super().__init__(dit)
        self.vae = vae
        self.train_scheduler = train_scheduler
        self.infer_scheduler = infer_scheduler

    @property
    def device(self) -> torch.device:
        return self.dit.patch_embedding.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.dit.patch_embedding.weight.dtype

    def train(self, mode: bool = True) -> "WanTower":
        super().train(mode)
        self.vae.eval()
        return self

    def prepare(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        video = sample["video"]
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"`video` must be [B, 3, T, H, W], got {tuple(video.shape)}")
        if video.shape[2] % 4 != 1 or video.shape[2] <= 1:
            raise ValueError(f"Video length must satisfy T % 4 == 1 and T > 1, got {video.shape[2]}")
        video = video.to(device=self.device, dtype=self.dtype, non_blocking=True)
        with torch.no_grad():
            latents = self.vae.encode(video, device=self.device)
        pad = sample.get("image_is_pad")
        if pad is not None:
            pad = pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        return {"latents": latents, "image_is_pad": pad}

    def add_noise(self, batch: dict[str, Any], plan: Plan, timesteps: dict) -> None:
        latents = batch["latents"]
        scheduler = self.train_scheduler
        if plan.states.get(FUTURE_FRAMES) != NOISY:
            batch["noisy"] = latents
            batch["t"] = latents.new_zeros(latents.shape[0])
            return
        noise = torch.randn_like(latents)
        timestep = shared_timestep(
            FUTURE_FRAMES,
            plan,
            timesteps,
            lambda: scheduler.sample_training_t(latents.shape[0], self.device, latents.dtype),
        )
        noisy = scheduler.add_noise(latents, noise, timestep)
        noisy[:, :, 0:1] = latents[:, :, 0:1]
        batch["noisy"] = noisy
        batch["t"] = timestep
        batch["target"] = scheduler.training_target(latents, noise, timestep)

    def _pre(
        self,
        latents: Tensor,
        timestep: Tensor,
        context: Context,
    ) -> TowerState:
        pre = self.dit.pre_dit(
            prepare_rotary_factors=False,
            x=latents,
            timestep=timestep,
            context=context.tokens,
            context_mask=context.mask,
            fuse_vae_embedding_in_latents=True,
            context_is_projected=context.projected,
            cross_kv_cache=context.kv_cache,
        )
        frame = int(pre["meta"]["tokens_per_frame"])
        total = int(pre["tokens"].shape[1])
        return TowerState(pre["tokens"], {CURRENT_FRAME: frame, FUTURE_FRAMES: total - frame}, frame, pre)

    def embed(self, batch: Mapping[str, Any], context: Context) -> TowerState:
        state = self._pre(batch["noisy"], batch["t"], context)
        return state

    def readout(self, stream: str, x: Tensor, state: TowerState) -> Tensor:
        if stream != FUTURE_FRAMES:
            raise ValueError(f"The video tower predicts `{FUTURE_FRAMES}`, not {stream!r}.")
        return self.dit.post_dit(x, state.extras)

    def flow_loss(self, stream: str, prediction: Tensor, batch: Mapping[str, Any]) -> Tensor:
        per_sample = _video_loss_per_sample(
            prediction[:, :, 1:],
            batch["target"][:, :, 1:],
            batch["image_is_pad"],
            int(self.vae.temporal_downsample_factor),
        )
        return flow_weighted_mean(per_sample, self.train_scheduler, batch["t"])

    def encode_request(self, request: dict[str, Any]) -> None:
        request["first_frame_latents"] = encode_image_batch(self.vae, request["input_image"], self.device)

    def project_context(self, context: Tensor) -> Tensor:
        return self.dit.project_context(context)

    def embed_step(
        self,
        request: Mapping[str, Any],
        latents: Mapping[str, Tensor],
        timesteps: Mapping[str, Tensor],
    ) -> TowerState:
        first = request["first_frame_latents"]
        if FUTURE_FRAMES in latents:
            raise NotImplementedError("Denoising future frames at inference is not migrated yet.")
        return self._pre(first, first.new_zeros(first.shape[0]), request["context"]["video"])
