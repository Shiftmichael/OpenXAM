"""XAMModel: a model described by a spec, behind the interface trainers and evaluators use."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import torch
from torch import Tensor, nn

from openxam.model.component.attention import (
    elide_fully_valid_attention_mask,
    prepare_key_padding_mask,
)
from openxam.model.helpers.batching import ActionBatchResult, validate_batch_inputs
from openxam.towers.base import Context, Tower, TowerState

from .driver import attention_structure, run_layers
from .plan import Plan, plan_infer, plan_train, sample_weights
from .spec import FLOW, NOISY, ModelSpec

# Names the trainer logs under.
LOG_NAMES = {"future": "video", "action": "action"}


class XAMModel(nn.Module):
    def __init__(
        self,
        spec: ModelSpec,
        towers: Mapping[str, Tower],
        *,
        device: Union[str, torch.device] = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        act_program: str = "act",
    ):
        super().__init__()
        spec.validate()
        self.spec = spec
        self.towers = nn.ModuleDict(dict(towers))
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.act_program = act_program
        self.inference_cross_kv_reuse = False
        self.train_plans = {mode.name: plan_train(spec, mode) for mode in spec.train}
        self.infer_plans = {program: plan_infer(spec, program) for program in spec.infer}
        self.mode_names = list(self.train_plans)
        self.mode_weights = torch.tensor([sample_weights(spec)[name] for name in self.mode_names])

    def _next_plan(self) -> Plan:
        """One training mode per minibatch, so the attention structure stays fixed within it."""
        if len(self.mode_names) == 1:
            return self.train_plans[self.mode_names[0]]
        index = int(torch.multinomial(self.mode_weights, 1).item())
        return self.train_plans[self.mode_names[index]]

    def _towers_of(self, names: Sequence[str]) -> dict[str, Tower]:
        return {name: self.towers[name] for name in names}

    def _training_context(self, sample: Mapping[str, Any]) -> Context:
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError("Training requires `sample['context']` and `sample['context_mask']`.")
        mask = elide_fully_valid_attention_mask(sample["context_mask"])
        tokens = sample["context"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if mask is not None:
            mask = mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        return Context(tokens, mask)

    def training_loss(self, sample: Mapping[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
        plan = self._next_plan()
        towers = self._towers_of(plan.active_towers)
        context = self._training_context(sample)

        batches = {name: tower.prepare(sample) for name, tower in towers.items()}
        timesteps: dict = {}
        for name, tower in towers.items():
            tower.add_noise(batches[name], plan, timesteps)

        states = {name: tower.embed(batches[name], context) for name, tower in towers.items()}
        segments, mask = attention_structure(plan, states, self.device)
        tokens, _ = run_layers(towers, states, mask, segments)

        total: Optional[Tensor] = None
        logs: dict[str, Tensor] = {}
        for loss in self.spec.losses:
            if loss.type != FLOW:
                raise NotImplementedError(f"Loss type {loss.type!r} is not implemented yet.")
            if plan.states.get(loss.stream) != NOISY:
                continue
            name = plan.towers[loss.stream]
            prediction = towers[name].readout(loss.stream, tokens[name], states[name])
            value = towers[name].flow_loss(loss.stream, prediction, batches[name])
            weighted = value * loss.weight
            total = weighted if total is None else total + weighted
            label = LOG_NAMES.get(loss.stream, loss.stream)
            logs[f"loss_{label}"] = value.detach()
            logs[f"weighted_loss_{label}"] = weighted.detach()
        return total, logs

    def _inference_context(
        self,
        prompt: Optional[Sequence[str]],
        context: Optional[Tensor],
        context_mask: Optional[Tensor],
    ) -> tuple[Tensor, Optional[Tensor]]:
        if prompt is not None:
            raise NotImplementedError("Encoding prompts is not migrated yet; pass `context`.")
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        return context, prepare_key_padding_mask(context_mask)

    def _tower_contexts(self, context: Tensor, mask: Optional[Tensor]) -> dict[str, Context]:
        contexts = {}
        for name, tower in self.towers.items():
            projected = tower.project_context(context)
            kv_cache = None
            if self.inference_cross_kv_reuse and projected is not None:
                kv_cache = tower.dit.build_cross_attention_kv_cache(projected)
            contexts[name] = Context(projected, mask, True, kv_cache)
        return contexts

    def _run_stage(
        self,
        plan: Plan,
        request: Mapping[str, Any],
        steps: int,
        sigma_shift: Optional[float],
    ) -> dict[str, Tensor]:
        moving = [name for name in plan.active_towers if name not in plan.cached_towers]
        denoised = [stream for stream in plan.streams if plan.states[stream] == NOISY]
        owner = {stream: self.towers[plan.towers[stream]] for stream in denoised}

        latents = {stream: owner[stream].start(stream, request) for stream in denoised}
        schedules = {
            stream: owner[stream].infer_scheduler.build_inference_schedule(
                num_inference_steps=steps,
                device=self.device,
                dtype=latents[stream].dtype,
                shift_override=sigma_shift,
            )
            for stream in denoised
        }

        fixed_states: dict[str, TowerState] = {
            name: self.towers[name].embed_step(request, {}, {}) for name in plan.cached_towers
        }
        structure = None
        cached_kv = None
        batch_size = request["batch_size"]
        for step in range(steps):
            timesteps = {
                stream: schedules[stream][0][step].expand(batch_size).to(dtype=latents[stream].dtype)
                for stream in denoised
            }
            states = {name: self.towers[name].embed_step(request, latents, timesteps) for name in moving}
            if structure is None:
                segments, mask = attention_structure(plan, {**fixed_states, **states}, self.device)
                fixed_length = sum(sum(state.lengths.values()) for state in fixed_states.values())
                query_mask = mask.slice(fixed_length, mask.shape[0])
                query_segments = tuple(s for s in segments if s.query_start >= fixed_length)
                structure = (query_mask, query_segments)
                if fixed_states:
                    fixed_mask = mask.slice(0, fixed_length, 0, fixed_length)
                    fixed_segments = tuple(s for s in segments if s.query_end <= fixed_length)
                    _, cached_kv = run_layers(
                        self._towers_of(plan.cached_towers),
                        fixed_states,
                        fixed_mask,
                        fixed_segments,
                        keep_kv=True,
                    )
            tokens, _ = run_layers(
                self._towers_of(moving),
                states,
                structure[0],
                structure[1],
                cached_kv=cached_kv,
            )
            for stream in denoised:
                name = plan.towers[stream]
                prediction = owner[stream].readout(stream, tokens[name], states[name])
                latents[stream] = owner[stream].infer_scheduler.step(
                    prediction, schedules[stream][1][step], latents[stream]
                )
        return latents

    @torch.inference_mode()
    def infer_action_batch(
        self,
        prompt: Optional[Sequence[str]],
        input_image: Tensor,
        action_horizon: int,
        proprio: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        num_inference_steps: Optional[int] = None,
        sigma_shift: Optional[float] = None,
        seed: Optional[Union[int, Sequence[Optional[int]]]] = None,
        rand_device: str = "cpu",
    ) -> ActionBatchResult:
        batch_size = validate_batch_inputs(input_image, prompt, context, context_mask)
        self.eval()
        context, context_mask = self._inference_context(prompt, context, context_mask)
        request: dict[str, Any] = {
            "batch_size": batch_size,
            "input_image": input_image.to(device=self.device, dtype=self.torch_dtype),
            "action_horizon": action_horizon,
            "seed": seed,
            "rand_device": rand_device,
            "context": self._tower_contexts(context, context_mask),
        }
        if proprio is not None:
            request["state"] = proprio.to(device=self.device, dtype=self.torch_dtype)
        for tower in self.towers.values():
            if hasattr(tower, "encode_request"):
                tower.encode_request(request)

        outputs: dict[str, Tensor] = {}
        for plan in self.infer_plans[self.act_program]:
            outputs.update(self._run_stage(plan, request, num_inference_steps or plan.steps, sigma_shift))
        return {"action": outputs["action"].detach().to(device="cpu", dtype=torch.float32)}
