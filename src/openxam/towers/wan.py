"""The Wan video tower: the current frame and the future frames run as one clip."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from openxam.core.plan import Plan
from openxam.core.spec import CLEAN, EITHER, NOISY
from openxam.model.helpers.batching import encode_image_batch, randn_per_sample

from .base import Context, TowerState, flow_weighted_mean, shared_timestep
from .dit import DiTTower

CURRENT_FRAME = "obs"
FUTURE_FRAMES = "future"


class Noised(NamedTuple):
    """A clip as a training step feeds it to the tower."""

    latents: Tensor
    timestep: Tensor
    target: Optional[Tensor] = None


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
    """Runs one or more clips, each a current frame and the future frames after it.

    ``noise_first_frame`` draws noise for the current frame too and discards it. It changes
    only which random numbers land where, and matches the EasyWAM classes that do so.
    """

    def __init__(
        self,
        dit: nn.Module,
        vae: nn.Module,
        train_scheduler: Any,
        infer_scheduler: Any,
        *,
        clips: Sequence[tuple[str, str]] = ((CURRENT_FRAME, FUTURE_FRAMES),),
        noise_first_frame: bool = False,
    ):
        super().__init__(dit)
        self.vae = vae
        self.train_scheduler = train_scheduler
        self.infer_scheduler = infer_scheduler
        self.clips = tuple(clips)
        self.noise_first_frame = noise_first_frame

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

    def checkpoint_parts(self) -> dict[str, nn.Module]:
        return {"video_dit": self.dit}

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
        return {"latents": latents, "image_is_pad": pad, "clips": {}}

    def _noise(self, latents: Tensor) -> Tensor:
        if self.noise_first_frame:
            return torch.randn_like(latents)
        noise = torch.zeros_like(latents)
        noise[:, :, 1:] = torch.randn_like(noise[:, :, 1:])
        return noise

    def add_noise(self, batch: dict[str, Any], plan: Plan, timesteps: dict) -> None:
        latents = batch["latents"]
        scheduler = self.train_scheduler
        for _, future in self.clips:
            state = plan.states.get(future)
            if state == CLEAN:
                batch["clips"][future] = Noised(latents, latents.new_zeros(latents.shape[0]))
            elif state == NOISY:
                noise = self._noise(latents)
                timestep = shared_timestep(
                    future,
                    plan,
                    timesteps,
                    lambda: scheduler.sample_training_t(latents.shape[0], self.device, latents.dtype),
                )
                noisy = scheduler.add_noise(latents, noise, timestep)
                noisy[:, :, 0:1] = latents[:, :, 0:1]
                target = scheduler.training_target(latents, noise, timestep)[:, :, 1:]
                batch["clips"][future] = Noised(noisy, timestep, target)

    def add_optional_noise(self, batch: dict[str, Any], plan: Plan) -> None:
        latents = batch["latents"]
        scheduler = self.train_scheduler
        size = latents.shape[0]
        for _, future in self.clips:
            if plan.states.get(future) != EITHER:
                continue
            chosen = torch.rand((size,), device=self.device) < plan.sometimes[future]
            timestep = latents.new_zeros(size)
            clip = latents
            if bool(chosen.any()):
                sampled = scheduler.sample_training_t(size, self.device, latents.dtype)
                timestep = torch.where(chosen, sampled, timestep)
                noisy = scheduler.add_noise(latents, torch.randn_like(latents), sampled)
                clip = torch.where(chosen.view(size, 1, 1, 1, 1), noisy, latents)
            clip = clip.clone()
            clip[:, :, 0:1] = latents[:, :, 0:1]
            batch["clips"][future] = Noised(clip, timestep)

    def _clip(self, first: str, future: str, latents: Tensor, timestep: Tensor, context: Context) -> TowerState:
        pre = self.dit.pre_dit(
            x=latents,
            timestep=timestep,
            context=context.tokens,
            context_mask=context.mask,
            fuse_vae_embedding_in_latents=bool(self.dit.fuse_vae_embedding_in_latents),
            context_is_projected=context.projected,
            cross_kv_cache=context.kv_cache,
        )
        frame = int(pre["meta"]["tokens_per_frame"])
        total = int(pre["tokens"].shape[1])
        return TowerState(pre["tokens"], {first: frame, future: total - frame}, frame, pre)

    def _side_by_side(self, clips: Sequence[tuple[str, TowerState]]) -> TowerState:
        """One state for clips that sit next to each other in the sequence."""
        extras = dict(clips[0][1].extras)
        extras["clips"] = {}
        lengths: dict[str, int] = {}
        start = 0
        for index, (future, clip) in enumerate(clips):
            end = start + int(clip.tokens.shape[1])
            extras["clips"][future] = (start, end, clip.extras)
            lengths.update(clip.lengths)
            start = end
            if index:
                extras.update(self.dit.merge_prepared_conditioning(extras, clip.extras))
                extras.pop("rope", None)
        tokens = clips[0][1].tokens if len(clips) == 1 else torch.cat([clip.tokens for _, clip in clips], dim=1)
        return TowerState(tokens, lengths, clips[0][1].frame_tokens, extras)

    def embed(self, batch: Mapping[str, Any], context: Context) -> TowerState:
        return self._side_by_side(
            [
                (future, self._clip(first, future, *batch["clips"][future][:2], context))
                for first, future in self.clips
                if future in batch["clips"]
            ]
        )

    def readout(self, stream: str, x: Tensor, state: TowerState) -> Tensor:
        if stream not in state.extras["clips"]:
            raise ValueError(f"The video tower has no clip for {stream!r}.")
        start, end, pre = state.extras["clips"][stream]
        return self.dit.post_dit(x[:, start:end], pre)

    def flow_loss(self, stream: str, prediction: Tensor, batch: Mapping[str, Any]) -> Tensor:
        clip = batch["clips"][stream]
        per_sample = _video_loss_per_sample(
            prediction[:, :, 1:],
            clip.target,
            batch["image_is_pad"],
            int(self.vae.temporal_downsample_factor),
        )
        return flow_weighted_mean(per_sample, self.train_scheduler, clip.timestep)

    def feature_mask(self, batch: Mapping[str, Any], state: TowerState) -> Optional[Tensor]:
        """Padded frames are not there to be read."""
        pad = batch["image_is_pad"]
        if pad is None or not bool(pad.any().item()):
            return None
        factor = int(self.vae.temporal_downsample_factor)
        if (pad.shape[1] - 1) % factor != 0:
            raise ValueError(f"Cannot align `image_is_pad` with latent frames: {pad.shape[1]} frames, factor {factor}.")
        tail = pad[:, 1:].view(pad.shape[0], -1, factor).all(dim=2)
        latent_pad = torch.cat([pad[:, :1], tail], dim=1)
        frames = int(state.extras["meta"]["grid_size"][0])
        if latent_pad.shape[1] != frames:
            raise ValueError(f"Video padding covers {latent_pad.shape[1]} latent frames, the clip has {frames}.")
        return (~latent_pad).repeat_interleave(state.frame_tokens, dim=1)

    def encode_request(self, request: dict[str, Any]) -> None:
        request["first_frame_latents"] = encode_image_batch(self.vae, request["input_image"], self.device)

    def project_context(self, context: Tensor) -> Tensor:
        return self.dit.project_context(context)

    def embed_step(
        self,
        plan: Plan,
        request: Mapping[str, Any],
        latents: Mapping[str, Tensor],
        timesteps: Mapping[str, Tensor],
    ) -> TowerState:
        live = [clip for clip in self.clips if clip[0] in plan.states]
        if len(live) != 1:
            raise NotImplementedError(f"Inference runs one clip at a time, this stage has {len(live)}.")
        first, future = live[0]
        if future in plan.states:
            clip, timestep = latents[future], timesteps[future]
        else:
            clip = request["first_frame_latents"]
            timestep = clip.new_zeros(clip.shape[0])
        state = self._clip(first, future, clip, timestep, request["context"]["video"])
        return self._side_by_side([(future, state)])

    def start(self, stream: str, request: Mapping[str, Any]) -> Tensor:
        """Noise for the whole clip, with the current frame kept clean."""
        height, width = request["input_image"].shape[2:]
        factor = self.vae.upsampling_factor
        frames = (request["num_video_frames"] - 1) // self.vae.temporal_downsample_factor + 1
        noise = randn_per_sample(
            (self.vae.model.z_dim, frames, height // factor, width // factor),
            seeds=request["seed"],
            batch_size=request["batch_size"],
            rand_device=request["rand_device"],
        ).to(device=self.device, dtype=self.dtype)
        noise[:, :, 0:1] = request["first_frame_latents"].clone()
        return noise

    def after_step(self, stream: str, latent: Tensor, request: Mapping[str, Any]) -> Tensor:
        """The solver moves the current frame too, so it is put back."""
        if any(stream == future for _, future in self.clips):
            latent[:, :, 0:1] = request["first_frame_latents"].clone()
        return latent
