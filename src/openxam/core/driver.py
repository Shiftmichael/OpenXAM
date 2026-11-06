"""Run towers together, layer by layer."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from openxam.model.component.attention import (
    AttentionSegment,
    StructuredAttentionMask,
    build_structured_attention_mask,
    run_attention,
)
from openxam.towers.base import Tower, TowerState

from .plan import Plan, Segment


def attention_structure(
    plan: Plan,
    states: Mapping[str, TowerState],
    device: torch.device,
) -> tuple[tuple[Segment, ...], StructuredAttentionMask]:
    """The compiled segments for these token lengths, and the mask attention runs on."""
    lengths: dict[str, int] = {}
    frame_tokens = 1
    for state in states.values():
        lengths.update(state.lengths)
        frame_tokens = max(frame_tokens, state.frame_tokens)
    segments = plan.segments(lengths, frame_tokens)
    total = sum(lengths.values())
    mask = build_structured_attention_mask(
        query_len=total,
        key_len=total,
        segments=[AttentionSegment(s.query_start, s.query_end, s.key_ranges) for s in segments],
        device=device,
    )
    return segments, mask


def _attend_cutting_gradients(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    segments: Sequence[Segment],
    num_heads: int,
    backend: str,
) -> Tensor:
    """Attention one segment at a time, so some keys and values can be detached."""
    outputs = []
    for segment in segments:
        keys, values = [], []
        for start, end in segment.key_ranges:
            held = (start, end) in segment.no_grad
            keys.append(k[:, start:end].detach() if held else k[:, start:end])
            values.append(v[:, start:end].detach() if held else v[:, start:end])
        outputs.append(
            run_attention(
                q[:, segment.query_start : segment.query_end],
                torch.cat(keys, dim=1),
                torch.cat(values, dim=1),
                num_heads=num_heads,
                backend=backend,
            )
        )
    return torch.cat(outputs, dim=1)


def run_layers(
    towers: Mapping[str, Tower],
    states: Mapping[str, TowerState],
    mask: StructuredAttentionMask,
    segments: Sequence[Segment],
    *,
    cached_kv: Optional[Sequence[tuple[Tensor, Tensor]]] = None,
    keep_kv: bool = False,
) -> tuple[dict[str, Tensor], list[tuple[Tensor, Tensor]]]:
    """Advance the towers through every layer, sharing one attention per layer.

    ``cached_kv`` holds the keys and values of towers that sit in front of these ones in
    the sequence and already ran. ``mask`` must then cover only the rows of ``towers``.
    With ``keep_kv`` the keys and values of every layer come back for later reuse.
    """
    names = list(towers)
    first = towers[names[0]]
    has_cut = any(segment.no_grad for segment in segments)
    use_checkpoint = any(tower.use_gradient_checkpointing for tower in towers.values())
    kept: list[tuple[Tensor, Tensor]] = []

    def one_layer(layer: int, *tokens: Tensor) -> tuple[Tensor, ...]:
        queries, keys, values, carries = [], [], [], []
        for name, x in zip(names, tokens):
            q, k, v, carry = towers[name].qkv(layer, x, states[name])
            queries.append(q)
            keys.append(k)
            values.append(v)
            carries.append(carry)
        q = torch.cat(queries, dim=1)
        k = torch.cat(keys, dim=1)
        v = torch.cat(values, dim=1)
        if keep_kv:
            kept.append((k, v))
        if cached_kv is not None:
            k = torch.cat([cached_kv[layer][0], k], dim=1)
            v = torch.cat([cached_kv[layer][1], v], dim=1)
        if has_cut:
            mixed = _attend_cutting_gradients(q, k, v, segments, first.num_heads, first.attention_backend)
        else:
            mixed = run_attention(
                q,
                k,
                v,
                num_heads=first.num_heads,
                attention_mask=mask.to(device=q.device),
                backend=first.attention_backend,
            )
        outputs, start = [], 0
        for name, x, carry in zip(names, tokens, carries):
            end = start + x.shape[1]
            outputs.append(towers[name].finish(layer, mixed[:, start:end], carry, states[name]))
            start = end
        return tuple(outputs)

    tokens = tuple(states[name].tokens for name in names)
    for layer in range(first.num_layers):
        if use_checkpoint and torch.is_grad_enabled():
            tokens = checkpoint(one_layer, layer, *tokens, use_reentrant=False)
        else:
            tokens = one_layer(layer, *tokens)
    return dict(zip(names, tokens)), kept
