"""XAMModel: a model described by a spec, behind the interface trainers and evaluators use.

Attribute names and the checkpoint format follow the EasyWAM classes, so the trainer, the
evaluators and existing checkpoints work with this class unchanged.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import torch
from torch import Tensor, nn

from openxam.model.component.action_dit import (
    policy_action_config,
    validate_checkpoint_state_position,
    validate_policy_action_config,
)
from openxam.model.component.attention import (
    StructuredAttentionMask,
    elide_fully_valid_attention_mask,
    prepare_key_padding_mask,
)
from openxam.model.helpers.batching import (
    ActionBatchResult,
    ActionInferenceResult,
    JointBatchResult,
    JointInferenceResult,
    batch_single_state,
    batch_single_tensor,
    decode_video_batch,
    validate_batch_inputs,
)
from openxam.towers.base import Context, Tower, TowerState
from openxam.utils.logging_config import get_logger

from .driver import attention_structure, run_layers
from .plan import Plan, Segment, plan_infer, plan_train, sample_weights
from .spec import FLOW, HOLD, NOISY, ContextRead, ModelSpec

logger = get_logger(__name__)

# Names the trainer logs under.
LOG_NAMES = {"future": "video", "action": "action"}

Seed = Optional[Union[int, Sequence[Optional[int]]]]


class TowerStack(nn.Module):
    """The diffusion transformers, laid out as EasyWAM's ``MoT`` so checkpoints carry over."""

    def __init__(self, towers: Mapping[str, Tower]):
        super().__init__()
        self.mixtures = nn.ModuleDict({name: tower.dit for name, tower in towers.items()})


def _check_towers(towers: Mapping[str, Tower], plans: Sequence[Plan]) -> None:
    """What the spec cannot check: the shapes of the towers actually built."""
    for plan in plans:
        for group in plan.groups:
            shapes = {(towers[name].num_layers, towers[name].num_heads, towers[name].head_dim) for name in group}
            if len(shapes) > 1:
                raise ValueError(
                    f"Towers {list(group)} attend to each other, so they need the same layers, heads and "
                    f"head size, got {sorted(shapes)}."
                )
        for reader, read in plan.features.items():
            if not 0 <= read.layer < towers[read.source].num_layers:
                raise ValueError(
                    f"Tower {reader} reads layer {read.layer} of {read.source}, which has "
                    f"{towers[read.source].num_layers} layers."
                )


class XAMModel(nn.Module):
    """A model described by a spec.

    When towers attend to each other, the trained weights sit under ``mot`` as in EasyWAM's
    MoT classes. Otherwise each tower names its parts under ``dit``, as in EasyWAM's Unified
    and Hidden classes.
    """

    inference_compile_targets = ("_predict_step",)

    def __init__(
        self,
        spec: ModelSpec,
        towers: Mapping[str, Tower],
        *,
        text_encoder: Optional[nn.Module] = None,
        tokenizer: Any = None,
        backbone_name: Optional[str] = None,
        model_variant: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        act_program: str = "act",
        imagine_program: str = "act_and_imagine",
    ):
        super().__init__()
        spec.validate()
        self.spec = spec
        self.train_plans = {mode.name: plan_train(spec, mode) for mode in spec.train}
        self.infer_plans = {program: plan_infer(spec, program) for program in spec.infer}
        plans = [*self.train_plans.values(), *(plan for stages in self.infer_plans.values() for plan in stages)]
        _check_towers(towers, plans)
        mixed = any(len(group) > 1 for plan in plans for group in plan.groups)
        video, action = towers.get("video"), towers.get("action")

        # Registered in this order so parameter names match EasyWAM's, which LoRA checkpoints rely on.
        if mixed:
            self.checkpoint_key = "mot"
            self.video_expert = video.dit
            self.action_expert = action.dit
            self.mot = TowerStack(towers)
            self.dit = self.mot
        else:
            self.checkpoint_key = "dit"
            self.dit = nn.ModuleDict(
                {name: part for tower in towers.values() for name, part in tower.checkpoint_parts().items()}
            )
        self.vae = getattr(video, "vae", None)
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.state_encoder = getattr(action, "state_encoder", None) if mixed else None
        self.towers = nn.ModuleDict(dict(towers))

        self.backbone_name = backbone_name or getattr(self.video_dit, "backbone_name", None)
        self.model_variant = model_variant
        self.state_position = "sequence"
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        # A model that cannot act without imagining runs its joint program to act.
        self.act_program = act_program if act_program in spec.infer else imagine_program
        self.imagine_program = imagine_program
        self.inference_cross_kv_reuse = False
        self.mode_names = list(self.train_plans)
        shares = sample_weights(spec)
        self.mode_weights = torch.tensor([shares[name] for name in self.mode_names])

    @property
    def video_dit(self) -> Optional[nn.Module]:
        return self.towers["video"].dit if "video" in self.towers else None

    @property
    def action_dit(self) -> Optional[nn.Module]:
        return self.towers["action"].dit if "action" in self.towers else None

    def train(self, mode: bool = True) -> "XAMModel":
        super().train(mode)
        if self.vae is not None:
            self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        return self

    def _towers_of(self, names: Sequence[str]) -> dict[str, Tower]:
        return {name: self.towers[name] for name in names}

    # ---- training ----

    def _next_plan(self) -> Plan:
        """One training mode per minibatch, so the attention structure stays fixed within it."""
        if len(self.mode_names) == 1:
            return self.train_plans[self.mode_names[0]]
        index = int(torch.multinomial(self.mode_weights, 1).item())
        return self.train_plans[self.mode_names[index]]

    def _training_context(self, sample: Mapping[str, Any]) -> Context:
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError("Training requires `sample['context']` and `sample['context_mask']`.")
        mask = elide_fully_valid_attention_mask(sample["context_mask"])
        tokens = sample["context"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if mask is not None:
            mask = mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        return Context(tokens, mask)

    def _feature_context(
        self,
        reader: str,
        read: ContextRead,
        features: Tensor,
        mask: Optional[Tensor],
        reuse_kv: bool = False,
    ) -> Context:
        """Another tower's features, as the context ``reader`` attends to."""
        tower = self.towers[reader]
        projected = tower.project_features(features if read.grad else features.detach())
        kv_cache = tower.dit.build_cross_attention_kv_cache(projected) if reuse_kv else None
        return Context(projected, prepare_key_padding_mask(mask), True, kv_cache)

    def training_loss(self, sample: Mapping[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
        plan = self._next_plan()
        towers = self._towers_of(plan.active_towers)
        text = self._training_context(sample)

        batches = {name: tower.prepare(sample) for name, tower in towers.items()}
        timesteps: dict = {}
        for name, tower in towers.items():
            tower.add_noise(batches[name], plan, timesteps)
        for name, tower in towers.items():
            tower.add_optional_noise(batches[name], plan)

        states: dict[str, TowerState] = {}
        tokens: dict[str, Tensor] = {}
        features: dict[str, Tensor] = {}
        for group in plan.groups:
            here = {}
            for name in group:
                read = plan.features.get(name)
                context = text
                if read is not None:
                    source = read.source
                    mask = towers[source].feature_mask(batches[source], states[source])
                    context = self._feature_context(name, read, features[source], mask)
                here[name] = towers[name].embed(batches[name], context)
            segments, mask = attention_structure(plan, here, self.device)
            out, _, found = run_layers(self._towers_of(group), here, mask, segments, capture=plan.captures(group))
            states.update(here)
            tokens.update(out)
            features.update(found)

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

    def forward(self, sample: Mapping[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
        return self.training_loss(sample)

    # ---- inference ----

    def encode_prompt(self, prompt: Union[str, Sequence[str]]) -> tuple[Tensor, Tensor]:
        if self.text_encoder is None:
            raise ValueError(
                "Prompt encoding requires a loaded text encoder. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        if self.tokenizer is None:
            context, mask = self.text_encoder(prompt)
            return context.to(device=self.device), mask.to(device=self.device, dtype=torch.bool)
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        return self.text_encoder(ids, mask).to(device=self.device), mask

    @torch.no_grad()
    def _encode_video_latents(self, video: Tensor) -> Tensor:
        return self.vae.encode(video, device=self.device)

    def _decode_latents(self, latents: Tensor) -> list:
        return decode_video_batch(self.vae, latents, self.device)

    def _tower_contexts(self, context: Tensor, mask: Optional[Tensor]) -> dict[str, Context]:
        contexts = {}
        readers = self.spec.feature_reads
        for name, tower in self.towers.items():
            if name in readers:
                continue  # it reads another tower's features, which exist only once that tower ran
            projected = tower.project_context(context)
            kv_cache = None
            if self.inference_cross_kv_reuse and projected is not None:
                kv_cache = tower.dit.build_cross_attention_kv_cache(projected)
            contexts[name] = Context(projected, mask, True, kv_cache)
        return contexts

    def _request(
        self,
        *,
        prompt: Optional[Sequence[str]],
        input_image: Tensor,
        action_horizon: int,
        proprio: Optional[Tensor],
        context: Optional[Tensor],
        context_mask: Optional[Tensor],
        seed: Seed,
        rand_device: str,
        num_video_frames: int,
    ) -> dict[str, Any]:
        batch_size = validate_batch_inputs(input_image, prompt, context, context_mask)
        height, width = input_image.shape[2:]
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`input_image` must be resized to multiples of 16, got {height}x{width}.")
        if num_video_frames % 4 != 1:
            raise ValueError(f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}.")
        if prompt is not None:
            context, context_mask = self.encode_prompt(prompt)
        else:
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        context_mask = prepare_key_padding_mask(context_mask)

        request: dict[str, Any] = {
            "batch_size": batch_size,
            "input_image": input_image.to(device=self.device, dtype=self.torch_dtype),
            "action_horizon": action_horizon,
            "num_video_frames": num_video_frames,
            "seed": seed,
            "rand_device": rand_device,
            "context": self._tower_contexts(context, context_mask),
        }
        if proprio is not None:
            if proprio.ndim == 3:
                proprio = proprio[:, 0]  # a history of states: the model reads the first one
            if proprio.ndim != 2 or proprio.shape[0] != batch_size:
                raise ValueError(f"`proprio` must be [B, D] or [B, T, D], got {tuple(proprio.shape)}")
            request["state"] = proprio.to(device=self.device, dtype=self.torch_dtype)
        for tower in self.towers.values():
            tower.encode_request(request)
        return request

    def _prefill(
        self,
        plan: Plan,
        group: tuple[str, ...],
        moving: tuple[str, ...],
        request: Mapping[str, Any],
        latents: Mapping[str, Tensor],
        timesteps: Mapping[str, Tensor],
        states: Mapping[str, TowerState],
    ) -> tuple[StructuredAttentionMask, tuple[Segment, ...], Optional[list]]:
        """Compile a group's attention structure and run its towers that stay fixed, once."""
        fixed = {
            name: self.towers[name].embed_step(plan, request, latents, timesteps)
            for name in group
            if name not in moving
        }
        segments, mask = attention_structure(plan, {**fixed, **states}, self.device)
        if not fixed:
            return mask, segments, None
        if tuple(fixed) != group[: len(fixed)]:
            raise NotImplementedError("Towers that stay fixed must come first in the sequence.")
        fixed_length = sum(sum(state.lengths.values()) for state in fixed.values())
        _, cached_kv, _ = run_layers(
            self._towers_of(tuple(fixed)),
            fixed,
            mask.slice(0, fixed_length, 0, fixed_length),
            tuple(segment for segment in segments if segment.query_end <= fixed_length),
            keep_kv=True,
        )
        return (
            mask.slice(fixed_length, mask.shape[0]),
            tuple(segment for segment in segments if segment.query_start >= fixed_length),
            cached_kv,
        )

    def _predict_step(
        self,
        plan: Plan,
        moving: Sequence[str],
        solved: Sequence[str],
        states: Mapping[str, TowerState],
        mask: StructuredAttentionMask,
        segments: Sequence[Segment],
        cached_kv: Optional[list],
        capture: Mapping[str, int],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        tokens, _, features = run_layers(
            self._towers_of(moving),
            states,
            mask,
            segments,
            cached_kv=cached_kv,
            capture=capture,
            stop_after_capture=not solved,
        )
        predictions = {
            stream: self.towers[plan.towers[stream]].readout(
                stream, tokens[plan.towers[stream]], states[plan.towers[stream]]
            )
            for stream in solved
        }
        return predictions, features

    def _run_stage(
        self,
        plan: Plan,
        request: Mapping[str, Any],
        outputs: Mapping[str, Tensor],
        steps: int,
        sigma_shift: Optional[float],
    ) -> dict[str, Tensor]:
        """One stage of an inference program.

        Streams being denoised move every step. Held streams stay at the first noise level,
        and streams fed by ``use`` are clean. A tower reading another tower's features reads
        them once, at the first step.
        """
        request = {**request, "context": dict(request["context"])}
        size = request["batch_size"]
        started = [stream for stream in plan.streams if plan.states[stream] in (NOISY, HOLD)]
        solved = [stream for stream in started if plan.states[stream] == NOISY]
        owner = {stream: self.towers[plan.towers[stream]] for stream in started}

        latents = {stream: owner[stream].start(stream, request) for stream in started}
        schedules = {
            stream: owner[stream].infer_scheduler.build_inference_schedule(
                num_inference_steps=steps,
                device=self.device,
                dtype=latents[stream].dtype,
                shift_override=sigma_shift,
            )
            for stream in started
        }
        for tied in plan.tie:
            for stream in tied[1:]:
                schedules[stream] = schedules[tied[0]]

        def level(stream: str, step: int) -> Tensor:
            return schedules[stream][0][step].expand(size).to(dtype=latents[stream].dtype, device=self.device)

        fixed_levels = {stream: level(stream, 0) for stream in started if stream not in solved}
        for target, source in plan.use.items():
            latents[target] = outputs[source]
            fixed_levels[target] = outputs[source].new_zeros(size)

        prepared: dict[tuple[str, ...], tuple] = {}
        for step in range(steps):
            levels = {**fixed_levels, **{stream: level(stream, step) for stream in solved}}
            predictions: dict[str, Tensor] = {}
            for group in plan.groups:
                moving = tuple(name for name in group if name not in plan.cached_towers)
                capture = plan.captures(group) if step == 0 else {}
                if not moving:
                    if not capture:
                        continue  # nothing here changes, and nobody reads it
                    moving = group  # runs once, for the features another tower reads
                states = {name: self.towers[name].embed_step(plan, request, latents, levels) for name in moving}
                if group not in prepared:
                    prepared[group] = self._prefill(plan, group, moving, request, latents, levels, states)
                here = [stream for stream in solved if plan.towers[stream] in moving]
                found, features = self._predict_step(plan, moving, here, states, *prepared[group], capture)
                predictions.update(found)
                for reader, read in plan.features.items():
                    if read.source in features:
                        request["context"][reader] = self._feature_context(
                            reader, read, features[read.source], None, self.inference_cross_kv_reuse
                        )
            for stream in solved:
                stepped = owner[stream].infer_scheduler.step(predictions[stream], schedules[stream][1][step], latents[stream])
                latents[stream] = owner[stream].after_step(stream, stepped, request)
        return {stream: latents[stream] for stream in solved}

    def _run_program(
        self,
        program: str,
        request: Mapping[str, Any],
        num_inference_steps: Optional[int],
        sigma_shift: Optional[float],
    ) -> dict[str, Tensor]:
        outputs: dict[str, Tensor] = {}
        for plan in self.infer_plans[program]:
            outputs.update(self._run_stage(plan, request, outputs, num_inference_steps or plan.steps, sigma_shift))
        return outputs

    @torch.inference_mode()
    def infer_action_batch(
        self,
        prompt: Optional[Sequence[str]],
        input_image: Tensor,
        action_horizon: int,
        proprio: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: Optional[int] = None,
        sigma_shift: Optional[float] = None,
        seed: Seed = None,
        rand_device: str = "cpu",
        num_video_frames: int = 5,
    ) -> ActionBatchResult:
        self.eval()
        request = self._request(
            prompt=prompt,
            input_image=input_image,
            action_horizon=action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            seed=seed,
            rand_device=rand_device,
            num_video_frames=num_video_frames,
        )
        outputs = self._run_program(self.act_program, request, num_inference_steps, sigma_shift)
        return {"action": outputs["action"].detach().to(device="cpu", dtype=torch.float32)}

    @torch.inference_mode()
    def infer_joint_batch(
        self,
        prompt: Optional[Sequence[str]],
        input_image: Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[Tensor] = None,
        proprio: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: Optional[int] = None,
        sigma_shift: Optional[float] = None,
        seed: Seed = None,
        rand_device: str = "cpu",
        test_action_with_infer_action: bool = False,
        decode_video: bool = True,
    ) -> JointBatchResult:
        if test_action_with_infer_action:
            raise ValueError("Batch inference does not support `test_action_with_infer_action`.")
        self.eval()
        request = self._request(
            prompt=prompt,
            input_image=input_image,
            action_horizon=action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            seed=seed,
            rand_device=rand_device,
            num_video_frames=num_video_frames,
        )
        outputs = self._run_program(self.imagine_program, request, num_inference_steps, sigma_shift)
        result: JointBatchResult = {"action": outputs["action"].detach().to(device="cpu", dtype=torch.float32)}
        if decode_video:
            result["video"] = self._decode_latents(outputs["future"])
        return result

    @torch.inference_mode()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: Tensor,
        action_horizon: int,
        proprio: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: Optional[int] = None,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        num_video_frames: int = 5,
    ) -> ActionInferenceResult:
        result = self.infer_action_batch(
            prompt=[prompt] if prompt is not None else None,
            input_image=batch_single_tensor(input_image, 3, "input_image"),
            action_horizon=action_horizon,
            proprio=batch_single_state(proprio, history=True),
            context=batch_single_tensor(context, 2, "context"),
            context_mask=batch_single_tensor(context_mask, 1, "context_mask"),
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            num_video_frames=num_video_frames,
        )
        return {"action": result["action"][0]}

    @torch.inference_mode()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[Tensor] = None,
        proprio: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: Optional[int] = None,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        test_action_with_infer_action: bool = False,
        decode_video: bool = True,
    ) -> JointInferenceResult:
        result = self.infer_joint_batch(
            prompt=[prompt] if prompt is not None else None,
            input_image=batch_single_tensor(input_image, 3, "input_image"),
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            action=batch_single_tensor(action, 2, "action"),
            proprio=batch_single_state(proprio, history=True),
            context=batch_single_tensor(context, 2, "context"),
            context_mask=batch_single_tensor(context_mask, 1, "context_mask"),
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            test_action_with_infer_action=False,
            decode_video=decode_video,
        )
        result["action"] = result["action"][0]
        if "video" in result:
            result["video"] = result["video"][0]
        return result

    @torch.inference_mode()
    def infer(
        self,
        prompt: Optional[str],
        input_image: Tensor,
        num_frames: int,
        action: Optional[Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: Optional[int] = None,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> JointInferenceResult:
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
        )

    # ---- checkpoints ----

    def save_checkpoint(self, path: str, optimizer: Any = None, step: Optional[int] = None) -> None:
        from openxam.model.component.lora import build_lora_checkpoint_payload, has_lora

        if has_lora(self.video_dit):
            payload = build_lora_checkpoint_payload(self, step=step)
        else:
            payload = {
                self.checkpoint_key: self.dit.state_dict(),
                "step": step,
                "torch_dtype": str(self.torch_dtype),
                "backbone_name": self.backbone_name,
            }
            if self.checkpoint_key == "mot":
                payload["model_variant"] = self.model_variant
                if self.state_encoder is not None:
                    payload["state_encoder"] = self.state_encoder.state_dict()
        payload["state_position"] = self.state_position
        payload["action_config"] = policy_action_config(self)
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    @staticmethod
    def _load_weights(module: nn.Module, state: Mapping[str, Tensor]) -> None:
        from openxam.model.component.lora import load_standard_state_dict

        load_standard_state_dict(module, state, strict=False)

    def load_checkpoint(self, path: str, optimizer: Any = None, merge_lora: bool = False) -> dict:
        payload = torch.load(path, map_location="cpu", mmap=True)
        validate_checkpoint_state_position(payload, self.state_position)
        validate_policy_action_config(self, payload)
        for key, current in (("backbone_name", self.backbone_name), ("model_variant", self.model_variant)):
            saved = payload.get(key)
            if saved is not None and saved != current:
                raise ValueError(f"Checkpoint {key} {saved!r} does not match this model's {current!r}.")

        lora = payload.get("format") == "trainable_lora"
        if lora:
            from openxam.model.component.lora import load_lora_model_checkpoint_state

            load_lora_model_checkpoint_state(self, payload, merge_after_load=bool(merge_lora))
        elif self.checkpoint_key in payload:
            self._load_weights(self.dit, payload[self.checkpoint_key])
        elif "dit" in payload:
            logger.warning("Loading an older `dit` checkpoint into the video tower only.")
            self._load_weights(self.video_dit, payload["dit"])
        else:
            raise ValueError(f"Checkpoint has no `{self.checkpoint_key}` weights: {path}")
        if not lora and self.state_encoder is not None:
            if "state_encoder" in payload:
                self.state_encoder.load_state_dict(payload["state_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `state_encoder` weights; keeping current ones.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload
