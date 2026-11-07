"""What every tower provides to the core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from torch import Tensor, nn

from openxam.core.plan import Plan


@dataclass
class TowerState:
    """What a tower carries through the layer loop for one forward pass."""

    tokens: Tensor
    lengths: dict[str, int]
    frame_tokens: int = 1
    extras: dict[str, Any] = field(default_factory=dict)


class Tower(nn.Module):
    """A stack of layers that one or more streams run through.

    Training calls ``prepare``, ``add_noise`` and ``embed``, then the driver alternates
    ``qkv`` and ``finish`` layer by layer, and ``readout`` and ``flow_loss`` close the step.
    Inference calls ``start`` and ``embed_step`` instead of the first three.
    """

    num_layers: int
    num_heads: int
    head_dim: int
    attention_backend: str
    use_gradient_checkpointing: bool = False

    def prepare(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        """Turn a raw batch into clean model inputs."""
        raise NotImplementedError

    def add_noise(self, batch: dict[str, Any], plan: Plan, timesteps: dict) -> None:
        """Add noise to the streams the plan makes noisy, and keep the training targets."""
        raise NotImplementedError

    def embed(self, batch: Mapping[str, Any], context: "Context") -> TowerState:
        """Turn the (noisy) inputs of a training step into tokens."""
        raise NotImplementedError

    def qkv(self, layer: int, x: Tensor, state: TowerState) -> tuple[Tensor, Tensor, Tensor, Any]:
        """The part of a layer before attention: q, k, v of shape [B, S, H*D], and what ``finish`` needs."""
        raise NotImplementedError

    def finish(self, layer: int, attn_out: Tensor, carry: Any, state: TowerState) -> Tensor:
        """The part of a layer after attention."""
        raise NotImplementedError

    def readout(self, stream: str, x: Tensor, state: TowerState) -> Tensor:
        """The prediction for one stream."""
        raise NotImplementedError

    def flow_loss(self, stream: str, prediction: Tensor, batch: Mapping[str, Any]) -> Tensor:
        """The flow-matching loss of one stream."""
        raise NotImplementedError

    def start(self, stream: str, request: Mapping[str, Any]) -> Tensor:
        """The noise a stream starts from at inference."""
        raise NotImplementedError

    def embed_step(
        self,
        request: Mapping[str, Any],
        latents: Mapping[str, Tensor],
        timesteps: Mapping[str, Tensor],
    ) -> TowerState:
        """Turn the inputs of one inference step into tokens."""
        raise NotImplementedError

    def after_step(self, stream: str, latent: Tensor, request: Mapping[str, Any]) -> Tensor:
        """Adjust a stream after a solver step. By default nothing changes."""
        return latent

    def encode_request(self, request: dict[str, Any]) -> None:
        """Add what this tower needs to an inference request. By default nothing."""

    def project_context(self, context: Tensor) -> Optional[Tensor]:
        """The context as this tower reads it. A tower without cross-attention returns None."""
        return None


@dataclass
class Context:
    """A fixed sequence the towers read through cross-attention."""

    tokens: Tensor
    mask: Optional[Tensor]
    projected: bool = False
    kv_cache: Optional[tuple] = None


def shared_timestep(
    stream: str,
    plan: Plan,
    timesteps: dict,
    sample: Callable[[], Tensor],
) -> Tensor:
    """A noise level for ``stream``, shared with the streams it is tied to."""
    for group in plan.tie:
        if stream in group:
            if group not in timesteps:
                timesteps[group] = sample()
            return timesteps[group]
    return sample()


def flow_weighted_mean(per_sample: Tensor, scheduler: Any, timestep: Tensor) -> Tensor:
    weight = scheduler.training_weight(timestep).to(per_sample.device, dtype=per_sample.dtype)
    return (per_sample * weight).mean()
