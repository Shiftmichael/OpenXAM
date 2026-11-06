"""The layer split shared by the diffusion transformer towers."""

from __future__ import annotations

from typing import Any, Optional

from torch import Tensor, nn

from openxam.model.component.attention import normalize_attention_backend
from openxam.model.component.mot import _modulate, _rope_apply

from .base import Tower, TowerState


def _context_for_layer(extras: dict, layer: int) -> Optional[dict]:
    context = extras.get("context")
    kv_cache = extras.get("cross_kv_cache")
    if kv_cache is None:
        return {"context": context, "mask": extras.get("context_mask")}
    return {"context": context, "mask": extras.get("context_mask"), "kv": kv_cache[layer]}


class DiTTower(Tower):
    """Runs one block of a DiT in two halves, so attention can happen in between."""

    def __init__(self, dit: nn.Module):
        super().__init__()
        self.dit = dit
        self.num_layers = len(dit.blocks)
        self.num_heads = int(dit.num_heads)
        self.head_dim = int(dit.attn_head_dim)
        self.attention_backend = normalize_attention_backend(dit.attention_backend)

    @property
    def use_gradient_checkpointing(self) -> bool:
        return bool(self.dit.use_gradient_checkpointing)

    def qkv(self, layer: int, x: Tensor, state: TowerState) -> tuple[Tensor, Tensor, Tensor, Any]:
        block = self.dit.blocks[layer]
        t_mod, freqs = state.extras["t_mod"], state.extras["freqs"]
        if block.uses_mixed_attention_api:
            q, k, v, carry = block.prepare_mixed_attention(
                x, t_mod["embedding"], t_mod["adaln_lora"], freqs, t_mod.get("token_to_timestep")
            )
            return q, k, v, carry

        base = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base + t_mod).unbind(dim=2)
        attn_input = _modulate(block.norm1(x), shift_msa, scale_msa)
        if hasattr(block.self_attn, "project_qkv"):
            q, k, v = block.self_attn.project_qkv(attn_input)
        else:
            q = block.self_attn.norm_q(block.self_attn.q(attn_input))
            k = block.self_attn.norm_k(block.self_attn.k(attn_input))
            v = block.self_attn.v(attn_input)
        q = _rope_apply(q, freqs, block.num_heads)
        k = _rope_apply(k, freqs, block.num_heads)
        return q, k, v, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def finish(self, layer: int, attn_out: Tensor, carry: Any, state: TowerState) -> Tensor:
        block = self.dit.blocks[layer]
        context = _context_for_layer(state.extras, layer)
        if block.uses_mixed_attention_api:
            arguments = (attn_out, carry, context["context"], context["mask"])
            if "kv" in context:
                return block.finish_mixed_attention(*arguments, context_kv=context["kv"])
            return block.finish_mixed_attention(*arguments)

        residual, gate_msa, shift_mlp, scale_mlp, gate_mlp = carry
        x = block.gate(residual, gate_msa, block.self_attn.o(attn_out))
        if context["context"] is not None and block.cross_attn is not None:
            x = x + block.cross_attn(
                block.norm3(x), context["context"], ctx_mask=context["mask"], projected_kv=context.get("kv")
            )
        return block.gate(x, gate_mlp, block.ffn(_modulate(block.norm2(x), shift_mlp, scale_mlp)))
