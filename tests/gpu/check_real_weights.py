#!/usr/bin/env python
"""Compare the multi-tower core (XAMModel) with the old EasyWAM class on REAL trained weights.

Only structure ``mot`` (old ``EasyWAMMoT`` vs ``XAMModel`` built from ``configs/xam/fastwam_wan22.yaml``)
is wired. Other structures can be added to ``STRUCTURES`` once their builders exist.

What it does (real mode, one GPU):

1. Determinism settings first (CUBLAS_WORKSPACE_CONFIG before CUDA init, deterministic algorithms in
   warn-only mode, cuDNN deterministic, benchmark off, TF32 off, attention backend ``sdpa``), and
   HF offline mode with the datasets cache inside ``--out``.
2. OLD model: Hydra compose of ``task=libero_easywam_mot_wan22`` with the model overrides of
   ``configs/benchmark/sim_libero.yaml`` (no base DiT / ActionDiT init file needed), instantiate,
   ``configure_model_execution``, ``load_checkpoint(ckpt, merge_lora=True)``, as
   ``experiments/libero/eval_libero_single.py:build_eval_runtime`` does. Every loaded tensor is checked
   against the checkpoint, because both classes load with ``strict=False``.
3. Inputs: a real LIBERO sample from the repo's own dataset/processor code (val transforms, the
   checkpoint's dataset stats), text context from the model's own ``encode_prompt`` (T5), padded
   positions zeroed exactly like ``scripts/precompute_text_embeds.py``. Falls back to synthetic tensors
   with the training shapes if the real sample cannot be built (``--input auto``), and says so.
4. Per dtype block: an untimed warm-up, the old reference run, ``--floor-runs`` more old runs (noise
   floor, per tensor), two negative controls (other seeds; a few-ulp nudge of one proprio value),
   NEW (A) "shared modules" (the old model's own modules wrapped in XAMModel, as the parity tests do),
   NEW (B) "real build" (``task=libero_xam_fastwam_wan22`` through ``create_xam``, the SAME checkpoint
   through XAMModel's own ``load_checkpoint``; parameters/buffers compared bitwise, then the same forward
   comparisons), optionally a checkpoint round trip.
5. ``--fp32-check``: the block again in float32 for old vs A, and PRECISION_REF = old bf16 vs old fp32 on
   the same inputs; every bf16 new-vs-old metric is also reported as a ratio to PRECISION_REF.

Verdicts, per comparison and category (loss+logs, grads, actions), never merged:
  BITWISE_IDENTICAL        every tensor has the same dtype, shape and bits
  EQUAL_UP_TO_SIGNED_ZERO  not bitwise, but every difference is exactly 0 (e.g. +0 vs -0)
  WITHIN_NOISE_FLOOR       each differing tensor <= floor_factor x its own old-vs-old floor (max over runs)
  DIFFERENT_FLOOR_ZERO     a tensor differs where old-vs-old was exactly 0 (no floor to compare with);
                           see rel_l2 in bf16 roundoff units and the PRECISION_REF ratio
  DIFFERENT                above the floor, non-finite values or limits, missing names or shapes

``--tiny`` runs the same code path on the tiny random CPU modules of ``tests/test_parity_mot.py`` (no
checkpoint, T5 or data needed). New-vs-old must be BITWISE_IDENTICAL there, and the controls (a model
with one weight nudged, a unit check of the verdict function) must behave; see ``tiny_expectations``.

Usage (from the repo root, where ``checkpoints/`` and ``data/`` live):

    python tests/gpu/check_real_weights.py --tiny --out runs/check_tiny [--fp32-check]
    python tests/gpu/check_real_weights.py --structure mot --out runs/check_mot \
        --ckpt checkpoints/EasyWAM-MoT-Wan22/libero_mot_wan22.pt [--fp32-check] [--autocast] \
        [--floor-runs 3] [--roundtrip-dir DIR]

Writes ``<out>/results.json`` after every phase. Exit code 0 when every phase ran (whatever the
verdicts), 1 when a phase failed (``results.json`` names it), 2 on bad arguments.
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True  # no __pycache__ next to the repo's sources

# Must be set before CUDA is initialised (cuBLAS reads it once). Recorded in results.json.
_ENV_BEFORE_SCRIPT = {
    key: os.environ.get(key)
    for key in ("CUBLAS_WORKSPACE_CONFIG", "HF_DATASETS_CACHE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
}
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse  # noqa: E402
import copy  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from contextlib import contextmanager, nullcontext  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest import mock  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import torch  # noqa: E402  (torch does not import `datasets`; every repo import below is lazy)

# Old task, new task and spec for each structure. Only these are wired.
STRUCTURES = {
    "mot": {
        "old_task": "libero_easywam_mot_wan22",
        "new_task": "libero_xam_fastwam_wan22",
        "spec": "fastwam_wan22",
        "model_variant": "mot",
    },
}
# To add mot_joint / mot_idm / unified / hidden: add an entry above (tasks per
# tests/test_hydra_configs.py:35-41), a shared-module builder like shared_new_model() following
# tests/test_parity_structures.py:105-161, and a tiny builder.

VERDICT_BITWISE = "BITWISE_IDENTICAL"
VERDICT_SIGNED_ZERO = "EQUAL_UP_TO_SIGNED_ZERO"
VERDICT_FLOOR = "WITHIN_NOISE_FLOOR"
VERDICT_FLOOR_ZERO = "DIFFERENT_FLOOR_ZERO"
VERDICT_DIFFERENT = "DIFFERENT"
DIFFERENT_VERDICTS = (VERDICT_DIFFERENT, VERDICT_FLOOR_ZERO)
IDENTICAL_VERDICTS = (VERDICT_BITWISE, VERDICT_SIGNED_ZERO)
CATEGORIES = ("loss_and_logs", "grads", "actions")
BF16_ROUNDOFF = 2.0**-8  # unit roundoff of bfloat16 (half its machine epsilon 2^-7)


class PhaseFailed(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------


def jsonable(value):
    """Plain JSON: tensors become lists/scalars, non-finite floats become strings."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return jsonable(value.item() if value.numel() == 1 else value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, (torch.dtype, torch.device, Path)):
        return str(value)
    return repr(value)


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def gb(n_bytes: int) -> float:
    return round(n_bytes / 1024**3, 3)


_INT_VIEW = {
    torch.float64: torch.int64,
    torch.float32: torch.int32,
    torch.bfloat16: torch.int16,
    torch.float16: torch.int16,
}


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """True bitwise equality: same dtype, shape and bit pattern (so +0/-0 differ, equal NaNs match)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.device != b.device:
        b = b.to(a.device)
    view = _INT_VIEW.get(a.dtype)
    if view is None:
        return bool(torch.equal(a, b))
    return bool(torch.equal(a.contiguous().view(view), b.contiguous().view(view)))


def tensor_metrics(test: torch.Tensor, ref: torch.Tensor, device=None) -> dict:
    """Metrics of ``test`` against ``ref`` (the old model), in float64.

    ``rel_l2 = ||test - ref|| / ||ref||``; when ``||ref|| == 0`` it is the absolute ``||test - ref||``
    instead, and ``rel_l2_is_absolute`` says so."""
    test = test.detach()
    if device is not None:
        test = test.to(device)
    ref = ref.detach().to(test.device)
    out = {
        "bitwise_equal": bits_equal(test, ref),
        "dtype": [str(test.dtype), str(ref.dtype)],
        "shape": list(test.shape),
    }
    if test.shape != ref.shape:
        out.update(shape_mismatch=[list(test.shape), list(ref.shape)], max_abs_diff=float("inf"),
                   rel_l2=float("inf"), rel_l2_is_absolute=False, cosine=float("nan"), finite=False,
                   _diff_sq=float("inf"), _ref_sq=0.0, _dot=0.0, _test_sq=0.0)
        return out
    t = test.to(torch.float64)
    r = ref.to(torch.float64)
    all_finite = bool(torch.isfinite(t).all()) and bool(torch.isfinite(r).all())
    diff = t - r
    diff_norm = float(torch.linalg.vector_norm(diff)) if diff.numel() else 0.0
    ref_norm = float(torch.linalg.vector_norm(r)) if r.numel() else 0.0
    test_norm = float(torch.linalg.vector_norm(t)) if t.numel() else 0.0
    dot = float((t * r).sum()) if t.numel() else 0.0
    if ref_norm > 0:
        rel, absolute = diff_norm / ref_norm, False
    else:
        rel, absolute = diff_norm, True
    if ref_norm > 0 and test_norm > 0:
        cosine = dot / (test_norm * ref_norm)
    else:
        cosine = 1.0 if ref_norm == test_norm == 0 else 0.0
    out.update(
        max_abs_diff=float(diff.abs().max()) if diff.numel() else 0.0,
        rel_l2=rel,
        rel_l2_is_absolute=absolute,
        cosine=cosine,
        ref_norm=ref_norm,
        finite=all_finite,
        _diff_sq=diff_norm**2,
        _ref_sq=ref_norm**2,
        _dot=dot,
        _test_sq=test_norm**2,
    )
    return out


def strip_private(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if not k.startswith("_")}


# ----------------------------------------------------------------------------------------------
# Comparisons and verdicts
# ----------------------------------------------------------------------------------------------


def compare_items(test: dict, ref: dict, device=None) -> dict:
    """Name -> tensor dicts compared name by name. Keeps every item's metrics (floats only)."""
    items = {name: tensor_metrics(test[name], ref[name], device) for name in sorted(set(test) & set(ref))}
    return {
        "items": items,
        "only_test": sorted(set(test) - set(ref)),
        "only_ref": sorted(set(ref) - set(test)),
    }


def summarize(cmp: dict, top_k: int = 20) -> dict:
    """The JSON view of a comparison (top-k items only)."""
    rows = [{"name": name, **strip_private(m)} for name, m in cmp["items"].items()]
    diff_sq = sum(m["_diff_sq"] for m in cmp["items"].values())
    ref_sq = sum(m["_ref_sq"] for m in cmp["items"].values())
    test_sq = sum(m["_test_sq"] for m in cmp["items"].values())
    dot = sum(m["_dot"] for m in cmp["items"].values())
    if ref_sq > 0:
        global_rel = math.sqrt(diff_sq) / math.sqrt(ref_sq)
    else:
        global_rel = math.sqrt(diff_sq)
    global_cos = dot / math.sqrt(ref_sq * test_sq) if ref_sq > 0 and test_sq > 0 else float("nan")

    def worst(key, reverse=True):
        if not rows:
            return None
        row = sorted(rows, key=lambda r: (r[key] if finite(r[key]) else float("inf")), reverse=reverse)[0]
        return {"name": row["name"], key: row[key]}

    groups: dict = {}
    for row in rows:
        g = groups.setdefault(row["name"].split(".")[0], {"n": 0, "n_bitwise": 0, "max_abs_diff": 0.0,
                                                          "max_rel_l2": 0.0})
        g["n"] += 1
        g["n_bitwise"] += int(row["bitwise_equal"])
        g["max_abs_diff"] = max(g["max_abs_diff"], row["max_abs_diff"])
        g["max_rel_l2"] = max(g["max_rel_l2"], row["rel_l2"])
    return {
        "all_bitwise": bool(rows) and all(r["bitwise_equal"] for r in rows)
        and not cmp["only_test"] and not cmp["only_ref"],
        "n_items": len(rows),
        "n_bitwise": sum(r["bitwise_equal"] for r in rows),
        "names_only_in_test": cmp["only_test"][:50],
        "names_only_in_ref": cmp["only_ref"][:50],
        "all_finite": all(r["finite"] for r in rows),
        "max_abs_diff": max((r["max_abs_diff"] for r in rows), default=0.0),
        "worst_rel_l2": max((r["rel_l2"] for r in rows), default=0.0),
        "global_rel_l2": global_rel,
        "global_rel_l2_is_absolute": ref_sq == 0,
        "global_cosine": global_cos,
        "min_cosine": worst("cosine", reverse=False),
        "worst_max_abs_diff_item": worst("max_abs_diff"),
        "worst_rel_l2_item": worst("rel_l2"),
        "per_group": groups,
        f"top{top_k}_by_rel_l2": sorted(rows, key=lambda r: (r["rel_l2"] if finite(r["rel_l2"]) else float("inf")),
                                        reverse=True)[:top_k],
    }


def floor_from(runs: list) -> dict:
    """Per-item noise floor: the max of each metric over the old-vs-old runs. A name missing from a
    run gets NaN (so any comparison against it is DIFFERENT)."""
    names = set()
    for cmp in runs:
        names |= set(cmp["items"])
    floor = {}
    for name in names:
        entry = {}
        for key in ("max_abs_diff", "rel_l2"):
            values = [cmp["items"][name][key] if name in cmp["items"] else float("nan") for cmp in runs]
            entry[key] = float("nan") if any(not finite(v) for v in values) else max(values)
        floor[name] = entry
    return floor


def judge(cmp: dict, floor: dict | None, factor: float) -> dict:
    """The verdict of one category; every differing item is held against its own floor."""
    items = cmp["items"]
    if cmp["only_test"] or cmp["only_ref"]:
        return {"verdict": VERDICT_DIFFERENT, "reason": "names on one side only",
                "only_test": cmp["only_test"][:10], "only_ref": cmp["only_ref"][:10]}
    if not items:
        return {"verdict": VERDICT_DIFFERENT, "reason": "nothing was compared"}
    if all(m["bitwise_equal"] for m in items.values()):
        return {"verdict": VERDICT_BITWISE, "n_items": len(items)}
    bad = [n for n, m in items.items() if not m["bitwise_equal"] and (not m["finite"] or "shape_mismatch" in m)]
    if bad:
        return {"verdict": VERDICT_DIFFERENT, "reason": "non-finite values or shape mismatch", "items": bad[:10]}
    differing = [n for n, m in items.items() if not m["bitwise_equal"] and m["max_abs_diff"] > 0]
    worst_rel = max((items[n]["rel_l2"] for n in differing), default=0.0)
    out = {
        "n_items": len(items),
        "n_not_bitwise": sum(not m["bitwise_equal"] for m in items.values()),
        "n_differing": len(differing),
        "worst_rel_l2": worst_rel,
        "worst_rel_l2_in_bf16_roundoff_units": worst_rel / BF16_ROUNDOFF,
        "max_abs_diff": max((items[n]["max_abs_diff"] for n in differing), default=0.0),
        "any_rel_l2_is_absolute": any(items[n]["rel_l2_is_absolute"] for n in differing),
    }
    if not differing:
        return {"verdict": VERDICT_SIGNED_ZERO, **out}
    if floor is None:
        return {"verdict": VERDICT_DIFFERENT, "reason": "not bitwise and no noise floor was measured", **out}
    over, floor_zero, worst_ratio = [], [], 0.0
    for name in differing:
        m, f = items[name], floor.get(name)
        if f is None:
            over.append({"name": name, "reason": "no floor for this item"})
            continue
        limits = {key: factor * f[key] for key in ("max_abs_diff", "rel_l2")}
        if not all(finite(v) for v in limits.values()):
            over.append({"name": name, "reason": "non-finite limit", "floor": f})
            continue
        if f["max_abs_diff"] == 0:
            floor_zero.append(name)
            continue
        for key, limit in limits.items():
            ratio = m[key] / limit if limit > 0 else float("inf")
            worst_ratio = max(worst_ratio, ratio)
            if m[key] > limit:
                over.append({"name": name, "metric": key, "test": m[key], "floor": f[key], "limit": limit})
    out.update(n_over_floor=len(over), over_floor_sample=over[:10], n_where_floor_is_zero=len(floor_zero),
               floor_zero_sample=floor_zero[:10], worst_ratio_to_limit=worst_ratio, floor_factor=factor)
    if over:
        return {"verdict": VERDICT_DIFFERENT, "reason": "above floor_factor x noise floor (or no usable floor)", **out}
    if floor_zero:
        return {"verdict": VERDICT_FLOOR_ZERO,
                "reason": "differs where old vs old was exactly 0; see bf16 roundoff units / PRECISION_REF", **out}
    return {"verdict": VERDICT_FLOOR, **out}


def verdict_self_test(factor: float) -> dict:
    """Synthetic cases through compare_items + floor_from + judge, so the verdict logic itself is tested."""
    f64 = torch.float64
    ref = {"x": torch.tensor([1.0, 2.0, 3.0], dtype=f64)}

    def shifted(delta, index=0, base=None):
        values = (base if base is not None else ref["x"]).clone()
        values[index] += delta
        return {"x": values}

    floor = floor_from([compare_items(shifted(1e-3), ref), compare_items(shifted(-5e-4), ref)])
    zero_floor = floor_from([compare_items({"x": ref["x"].clone()}, ref)])
    nan_floor = {"x": {"max_abs_diff": float("nan"), "rel_l2": float("nan")}}
    signed_ref = {"x": torch.tensor([0.0, 1.0], dtype=f64)}
    signed_test = {"x": torch.tensor([-0.0, 1.0], dtype=f64)}
    zero_ref = {"x": torch.zeros(3, dtype=f64)}
    zero_floor_abs = floor_from([compare_items({"x": torch.tensor([1e-3, 0.0, 0.0], dtype=f64)}, zero_ref)])
    cases = {
        "identical -> BITWISE": (compare_items({"x": ref["x"].clone()}, ref), floor, VERDICT_BITWISE),
        "within floor -> WITHIN": (compare_items(shifted(5e-4), ref), floor, VERDICT_FLOOR),
        "10x floor -> DIFFERENT": (compare_items(shifted(1e-2), ref), floor, VERDICT_DIFFERENT),
        "NaN -> DIFFERENT": (compare_items(shifted(float("nan")), ref), floor, VERDICT_DIFFERENT),
        "floor 0 + diff -> FLOOR_ZERO": (compare_items(shifted(1e-3), ref), zero_floor, VERDICT_FLOOR_ZERO),
        "non-finite limit -> DIFFERENT": (compare_items(shifted(1e-3), ref), nan_floor, VERDICT_DIFFERENT),
        "+0 vs -0 -> SIGNED_ZERO": (compare_items(signed_test, signed_ref), None, VERDICT_SIGNED_ZERO),
        "zero ref norm, within absolute floor -> WITHIN": (
            compare_items({"x": torch.tensor([5e-4, 0.0, 0.0], dtype=f64)}, zero_ref), zero_floor_abs, VERDICT_FLOOR),
        "missing name -> DIFFERENT": (compare_items({"y": ref["x"].clone()}, ref), floor, VERDICT_DIFFERENT),
        "no floor -> DIFFERENT": (compare_items(shifted(5e-4), ref), None, VERDICT_DIFFERENT),
    }
    out = {}
    for name, (cmp, case_floor, expected) in cases.items():
        got = judge(cmp, case_floor, factor)["verdict"]
        out[name] = {"expected": expected, "got": got, "ok": got == expected}
    return out


def ratio_to_precision_ref(summary: dict, precision: dict) -> dict:
    out = {}
    for key in ("max_abs_diff", "worst_rel_l2", "global_rel_l2"):
        ref_value = precision.get(key)
        value = summary.get(key)
        out[f"{key}_ratio"] = value / ref_value if finite(ref_value) and ref_value > 0 and finite(value) else None
    return out


# ----------------------------------------------------------------------------------------------
# Running a model
# ----------------------------------------------------------------------------------------------


def apply_train_mode(model) -> None:
    """The trainer's own switch (only the DiTs and the state encoder train)."""
    from openxam.trainer import EasyWAMTrainer

    EasyWAMTrainer._apply_dit_only_train_mode(model)


def clone_sample(sample: dict) -> dict:
    return {k: (v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)) for k, v in sample.items()}


def autocast_context(enabled: bool, device: str):
    """trainer.py:1103-1108 wraps the forward in accelerator.autocast(), i.e. bf16 autocast."""
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16)


def run_model(model, sample: dict, inference: dict, *, train_seed: int, infer_seed: int, autocast: bool,
              device: str, refs: dict, keep_grads: bool):
    """One training step and one action sample. Returns this run's outputs and its comparisons
    against each reference in ``refs`` (gradients are compared while they are still on the device)."""
    apply_train_mode(model)
    model.zero_grad(set_to_none=True)
    batch = clone_sample(sample)
    torch.manual_seed(train_seed)
    with autocast_context(autocast, device):
        loss, logs = model.training_loss(batch)
    loss.backward()
    grads, trainable_without_grad = {}, []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            grads[name] = parameter.grad
        elif parameter.requires_grad:
            trainable_without_grad.append(name)
    scalars = {"loss": loss.detach().cpu()}
    scalars.update({f"logs.{k}": v.detach().cpu() for k, v in logs.items()})
    del loss, logs
    outputs = {"scalars": scalars, "trainable_without_grad": trainable_without_grad, "n_grads": len(grads)}
    cmps = {label: {"grads": compare_items(grads, ref["grads"])} for label, ref in refs.items()}
    if keep_grads:
        outputs["grads"] = {name: g.detach().to("cpu", copy=True) for name, g in grads.items()}
    del grads
    model.zero_grad(set_to_none=True)
    free_memory()

    torch.manual_seed(infer_seed)  # the sampler uses per-sample generators; reseed the global RNG anyway
    request = {**clone_sample(inference), "seed": [infer_seed + i for i in range(int(inference["input_image"].shape[0]))]}
    outputs["actions"] = {"actions": model.infer_action_batch(**request)["action"].detach().cpu()}
    for label, ref in refs.items():
        cmps[label]["loss_and_logs"] = compare_items(scalars, ref["scalars"])
        cmps[label]["actions"] = compare_items(outputs["actions"], ref["actions"])
    return outputs, cmps


def nudge_by_bf16_ulps(value: float, ulps: int) -> tuple[float, float]:
    """``value`` rounded to bfloat16, and that plus ``ulps`` bfloat16 ulps (exactly representable)."""
    base = float(torch.tensor(value, dtype=torch.float32).to(torch.bfloat16).float())
    if base == 0:
        return base, float(ulps) * 2.0**-10
    exponent = math.floor(math.log2(abs(base)))
    return base, base + ulps * 2.0 ** (exponent - 7)


def perturbed_inputs(sample: dict, inference: dict, ulps: int) -> tuple[dict, dict, dict]:
    """Same inputs with one proprio value (sample 0, step 0, the largest |value|) moved by a few bf16 ulps.
    Training and inference both read proprio[:, 0]."""
    sample = clone_sample(sample)
    index = int(sample["proprio"][0, 0].abs().argmax())
    original = float(sample["proprio"][0, 0, index])
    base, nudged = nudge_by_bf16_ulps(original, ulps)
    sample["proprio"][0, 0, index] = nudged
    inference = {**inference, "proprio": sample["proprio"][:, 0]}
    info = {"element": [0, 0, index], "original": original, "as_bf16": base, "nudged": nudged, "bf16_ulps": ulps,
            "nudged_is_bf16_exact": float(torch.tensor(nudged).to(torch.bfloat16).float()) == nudged}
    return sample, inference, info


class Harness:
    def __init__(self, args, out_dir: Path, device: str):
        self.args = args
        self.out_dir = out_dir
        self.device = device
        self.results: dict = {
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
            "argv": sys.argv,
            "mode": "tiny" if args.tiny else "real",
            "structure": args.structure,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "phases": {},
            "comparisons": {},
            "verdicts": {},
            "noise_floor": {},
            "controls": {},
            "warnings": [],
        }
        self.failed_phases: list = []
        self._seen_warnings: set = set()
        self.sample: dict | None = None
        self.inference: dict | None = None
        self.summaries: dict = {}  # key -> {category: summary}, for PRECISION_REF ratios

    # -- bookkeeping --

    def save(self) -> None:
        path = self.out_dir / "results.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(jsonable(self.results), indent=2))
        os.replace(tmp, path)

    def record_warning(self, message, category, filename, lineno, file=None, line=None):
        text = f"{category.__name__}: {str(message)[:400]} ({Path(filename).name}:{lineno})"
        if text not in self._seen_warnings:
            self._seen_warnings.add(text)
            self.results["warnings"].append(text)
            print(f"[warning] {text}", file=sys.stderr)

    @contextmanager
    def phase(self, name: str):
        entry = {"status": "running"}
        self.results["phases"][name] = entry
        free_memory()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        print(f"\n=== phase {name} ===", flush=True)
        try:
            yield entry
            entry["status"] = "ok"
        except Exception as error:  # noqa: BLE001 - every failure is recorded, then re-raised
            entry["status"] = "failed"
            entry["error"] = repr(error)
            entry["traceback"] = traceback.format_exc()
            self.failed_phases.append(name)
            print(f"!!! phase {name} FAILED: {error!r}", file=sys.stderr, flush=True)
            raise PhaseFailed(name) from error
        finally:
            entry["seconds"] = round(time.perf_counter() - start, 2)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                entry["peak_gpu_allocated_gb"] = gb(torch.cuda.max_memory_allocated())
                entry["peak_gpu_reserved_gb"] = gb(torch.cuda.max_memory_reserved())
                entry["end_gpu_allocated_gb"] = gb(torch.cuda.memory_allocated())
            else:
                entry["peak_gpu_allocated_gb"] = None
            entry["process_peak_rss_gb_so_far"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3)
            print(f"=== phase {name}: {entry['status']} in {entry['seconds']} s, peak GPU "
                  f"{entry.get('peak_gpu_allocated_gb')} GB ===", flush=True)
            self.save()

    def skip(self, name: str, reason: str) -> None:
        self.results["phases"][name] = {"status": "skipped", "reason": reason}
        print(f"=== phase {name}: skipped ({reason}) ===", flush=True)
        self.save()

    def record(self, key: str, cmps: dict, floor: dict | None, extra: dict | None = None) -> dict:
        """Store the summaries and the verdicts of one comparison (all three categories)."""
        summaries = {category: summarize(cmps[category]) for category in CATEGORIES}
        verdicts = {category: judge(cmps[category], floor[category] if floor else None, self.args.floor_factor)
                    for category in CATEGORIES}
        self.results["comparisons"][key] = {**summaries, **(extra or {})}
        self.results["verdicts"][key] = verdicts
        self.summaries[key] = summaries
        return verdicts


# ----------------------------------------------------------------------------------------------
# Model state checks
# ----------------------------------------------------------------------------------------------


def verify_loaded(model, payload: dict) -> dict:
    """Both classes load with strict=False, so check every tensor actually came from the checkpoint."""
    report = {}
    parts = [(model.checkpoint_key if hasattr(model, "checkpoint_key") else "mot", model.mot)]
    if getattr(model, "state_encoder", None) is not None:
        parts.append(("state_encoder", model.state_encoder))
    ok = True
    for key, module in parts:
        if key not in payload:
            report[key] = {"error": f"checkpoint has no `{key}`"}
            ok = False
            continue
        current = module.state_dict()
        saved = payload[key]
        missing = sorted(set(current) - set(saved))
        unexpected = sorted(set(saved) - set(current))
        mismatched = []
        for name in sorted(set(current) & set(saved)):
            want = saved[name].to(dtype=current[name].dtype)
            if not bits_equal(current[name].detach().cpu(), want):
                mismatched.append(name)
        report[key] = {
            "n_model_tensors": len(current),
            "n_checkpoint_tensors": len(saved),
            "missing_in_checkpoint": missing[:50],
            "n_missing_in_checkpoint": len(missing),
            "unexpected_in_checkpoint": unexpected[:50],
            "n_unexpected_in_checkpoint": len(unexpected),
            "n_values_not_from_checkpoint": len(mismatched),
            "values_not_from_checkpoint": mismatched[:50],
        }
        ok = ok and not missing and not unexpected and not mismatched
    report["every_tensor_from_checkpoint"] = ok
    return report


def compare_named(test_items, ref_items, cast_ref_to_test_dtype: bool = False) -> dict:
    """Bitwise, name by name. ``all_bitwise`` needs the same name set; order is reported separately."""
    test, ref = dict(test_items), dict(ref_items)
    mismatched, dtype_differs = [], []
    for name in sorted(set(test) & set(ref)):
        a, b = test[name].detach(), ref[name].detach()
        if a.dtype != b.dtype:
            dtype_differs.append(name)
            if cast_ref_to_test_dtype:
                b = b.to(a.dtype)
        if not bits_equal(a, b):
            mismatched.append(name)
    names_equal = set(test) == set(ref)
    return {
        "same_name_set": names_equal,
        "same_order": list(test) == list(ref),
        "n_dtype_differs": len(dtype_differs),
        "dtype_differs_sample": dtype_differs[:10],
        "ref_cast_to_test_dtype": cast_ref_to_test_dtype,
        "n_test": len(test),
        "n_ref": len(ref),
        "only_in_test": sorted(set(test) - set(ref))[:50],
        "only_in_ref": sorted(set(ref) - set(test))[:50],
        "n_not_bitwise": len(mismatched),
        "not_bitwise": mismatched[:50],
        "all_bitwise": names_equal and not mismatched,
    }


def compare_states(new, old) -> dict:
    """Parameters, buffers and the checkpointed state dicts of ``new`` against ``old``, bitwise."""
    report = {
        "parameters": compare_named(new.named_parameters(), old.named_parameters()),
        "buffers": compare_named(new.named_buffers(), old.named_buffers()),
        "mot_state_dict": compare_named(new.mot.state_dict().items(), old.mot.state_dict().items()),
    }
    if getattr(old, "state_encoder", None) is not None or getattr(new, "state_encoder", None) is not None:
        report["state_encoder_state_dict"] = compare_named(
            new.state_encoder.state_dict().items(), old.state_encoder.state_dict().items()
        )
    full_new, full_old = set(new.state_dict()), set(old.state_dict())
    report["full_state_dict_info"] = {
        "note": "XAMModel also registers the towers under `towers.*`; extra alias keys are expected.",
        "n_only_in_new": len(full_new - full_old),
        "only_in_new_sample": sorted(full_new - full_old)[:10],
        "n_only_in_old": len(full_old - full_new),
        "only_in_old_sample": sorted(full_old - full_new)[:10],
    }
    report["all_bitwise"] = all(v["all_bitwise"] for k, v in report.items() if k != "full_state_dict_info")
    return report


def attention_settings(model) -> dict:
    out = {}
    for name in ("video_expert", "action_expert", "mot"):
        module = getattr(model, name, None)
        if module is not None:
            out[name] = getattr(module, "attention_backend", None)
    return out


# ----------------------------------------------------------------------------------------------
# NEW model (A): the old model's own modules wired into XAMModel
# ----------------------------------------------------------------------------------------------


def shared_new_model(old, spec_name: str):
    """Mirrors tests/test_parity_mot.py:new_model (and the MoT branch of
    tests/test_parity_structures.py:new_model), plus device/tokenizer/backbone_name, which the CPU
    tests leave at their defaults (XAMModel's device defaults to cpu)."""
    from openxam.core.build import load_spec, video_clips
    from openxam.core.model import XAMModel
    from openxam.towers.action import ActionTower
    from openxam.towers.wan import WanTower

    spec = load_spec(spec_name)
    video = WanTower(
        old.video_expert,
        old.vae,
        old.train_video_scheduler,
        old.infer_video_scheduler,
        clips=video_clips(spec),
        noise_first_frame=bool(spec.towers["video"].options.get("noise_first_frame", False)),
    )
    action = ActionTower(old.action_expert, old.state_encoder, old.train_action_scheduler, old.infer_action_scheduler)
    new = XAMModel(
        spec,
        {"video": video, "action": action},
        text_encoder=old.text_encoder,
        tokenizer=old.tokenizer,
        backbone_name=old.backbone_name,
        model_variant=old.model_variant,
        device=old.device,
        torch_dtype=old.torch_dtype,
    )
    new.inference_cross_kv_reuse = old.inference_cross_kv_reuse
    return new


# ----------------------------------------------------------------------------------------------
# Real mode: Hydra configs, checkpoint, LIBERO sample
# ----------------------------------------------------------------------------------------------

# configs/benchmark/sim_libero.yaml:11-14 -- how the evaluator loads a trained checkpoint.
EVAL_MODEL_OVERRIDES = {
    "backbone": {"load_text_encoder": True},
    "skip_dit_load_from_pretrain": True,
    "action_dit_pretrained_path": None,
}


class RealCase:
    def __init__(self, args, harness: Harness):
        self.args = args
        self.h = harness
        self.structure = STRUCTURES[args.structure]
        from openxam.utils.config_resolvers import register_default_resolvers

        register_default_resolvers()
        import yaml

        bench = yaml.safe_load((ROOT / "configs" / "benchmark" / "sim_libero.yaml").read_text())
        harness.results["eval_model_overrides"] = {
            "used": EVAL_MODEL_OVERRIDES,
            "sim_libero_yaml_model_block": bench.get("model"),
            "match": bench.get("model") == EVAL_MODEL_OVERRIDES,
        }
        if bench.get("model") != EVAL_MODEL_OVERRIDES:
            print("[note] configs/benchmark/sim_libero.yaml model block differs from the overrides used here; "
                  "see results.json eval_model_overrides", file=sys.stderr)

    def compose(self, task: str, *, load_text_encoder: bool):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf, open_dict

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
            cfg = compose("train", overrides=[f"task={task}"])
        wan = Path(self.args.wan_dir)
        with open_dict(cfg):
            cfg.model.backbone.load_text_encoder = bool(load_text_encoder)
            cfg.model.skip_dit_load_from_pretrain = EVAL_MODEL_OVERRIDES["skip_dit_load_from_pretrain"]
            cfg.model.action_dit_pretrained_path = EVAL_MODEL_OVERRIDES["action_dit_pretrained_path"]
            cfg.model.backbone.attention_backend = "sdpa"
            cfg.model.backbone.model_id = str(wan)
            cfg.model.backbone.tokenizer_model_id = str(wan / "google" / "umt5-xxl")
            cfg.data.train.dataset_dirs = [str(Path(self.args.data_root) / self.args.suite)]
            cfg.data.train.pretrained_norm_stats = str(self.args.dataset_stats)
        (self.h.out_dir / f"config_{task}.yaml").write_text(OmegaConf.to_yaml(cfg.model, resolve=True))
        return cfg

    def instantiate_model(self, cfg, dtype):
        from hydra.utils import instantiate
        from openxam.model.helpers.inference import configure_model_execution

        model = instantiate(cfg.model, model_dtype=dtype, device=self.h.device)
        return configure_model_execution(
            model,
            vae_micro_batch_size=cfg.get("vae_micro_batch_size", 1),
            inference_cross_kv_reuse=self.args.cross_kv_reuse == "on",
        )

    def load_payload(self):
        return torch.load(self.args.ckpt, map_location="cpu", mmap=True)

    def build_old(self, dtype, *, load_text_encoder: bool):
        cfg = self.compose(self.structure["old_task"], load_text_encoder=load_text_encoder)
        self.cfg = cfg
        old = self.instantiate_model(cfg, dtype)
        old.load_checkpoint(str(self.args.ckpt), merge_lora=True)  # eval_libero_single.py:133
        return old.to(self.h.device)

    def build_new_real(self, dtype):
        cfg = self.compose(self.structure["new_task"], load_text_encoder=False)
        new = self.instantiate_model(cfg, dtype)
        return new, lambda: new.load_checkpoint(str(self.args.ckpt), merge_lora=True)

    # -- inputs --

    def training_shapes(self) -> dict:
        train = self.cfg.data.train
        num_frames = int(train.num_frames)
        ratio = int(train.action_video_freq_ratio)
        return {
            "video_frames": len(range(0, num_frames, ratio)),
            "action_horizon": num_frames - 1,
            "height": int(train.video_size[0]),
            "width": int(train.video_size[1]),
            "action_dim": int(train.processor.action_output_dim),
            "state_dim": int(train.processor.proprio_output_dim),
        }

    def real_samples(self, info: dict) -> tuple[dict, list]:
        from hydra.utils import instantiate
        from openxam.utils import misc

        misc.register_work_dir(str(self.h.out_dir / "work"))  # the dataset writes a copy of its stats here
        dataset = instantiate(self.cfg.data.train)
        dataset.lerobot_dataset.processor.eval()  # val transforms: no ColorJitter, same resize/normalisation
        # No text cache exists; context comes from encode_prompt below.
        dataset._get_cached_text_context = lambda prompt: (None, None)
        seen = []
        getitem = dataset.lerobot_dataset.__getitem__

        def spy(index):
            item = getitem(index)
            seen.append({"requested": int(index), "returned": int(item["idx"])})
            return item

        dataset.lerobot_dataset.__getitem__ = spy  # _get calls it by attribute, so this sees substitutions
        n = len(dataset)
        if self.args.sample_indices:
            indices = [int(i) for i in self.args.sample_indices.split(",")]
        else:
            indices = [n * (k + 1) // (self.args.batch_size + 1) for k in range(self.args.batch_size)]
        items = [dataset._get(i) for i in indices]  # _get, not __getitem__: no silent random fallback
        info.update(dataset_len=n, indices=indices, lerobot_indices=seen,
                    index_substituted=any(s["requested"] != s["returned"] for s in seen),
                    processor_is_train=dataset.lerobot_dataset.processor.is_train)
        sample = {
            "video": torch.stack([it["video"] for it in items]),
            "action": torch.stack([it["action"] for it in items]),
            "proprio": torch.stack([it["proprio"] for it in items]),
            "action_is_pad": torch.stack([torch.as_tensor(it["action_is_pad"]) for it in items]).bool(),
            "image_is_pad": torch.stack([torch.as_tensor(it["image_is_pad"]) for it in items]).bool(),
        }
        return sample, [it["prompt"] for it in items]

    def synthetic_samples(self) -> tuple[dict, list]:
        from openxam.data.lerobot.prompts import DEFAULT_PROMPT

        s = self.training_shapes()
        g = torch.Generator().manual_seed(1234)
        b = self.args.batch_size
        sample = {
            "video": torch.rand(b, 3, s["video_frames"], s["height"], s["width"], generator=g) * 2 - 1,
            "action": torch.randn(b, s["action_horizon"], s["action_dim"], generator=g),
            "proprio": torch.randn(b, s["action_horizon"], s["state_dim"], generator=g),
            "action_is_pad": torch.zeros(b, s["action_horizon"], dtype=torch.bool),
            "image_is_pad": torch.zeros(b, s["video_frames"], dtype=torch.bool),
        }
        task = "pick up the black bowl between the plate and the ramekin and place it on the plate"
        return sample, [DEFAULT_PROMPT.format(task=task)] * b

    def make_inputs(self, old) -> tuple[dict, dict, dict]:
        info: dict = {}
        source = None
        if self.args.input in ("auto", "real"):
            try:
                sample, prompts = self.real_samples(info)
                source = f"real LIBERO sample ({self.args.suite})"
            except Exception as error:  # noqa: BLE001
                if self.args.input == "real":
                    raise
                info["real_sample_error"] = repr(error)
                info["real_sample_traceback"] = traceback.format_exc()
                print(f"!!! real sample failed ({error!r}); FALLING BACK TO SYNTHETIC INPUTS", file=sys.stderr)
        if source is None:
            sample, prompts = self.synthetic_samples()
            source = "SYNTHETIC tensors with training shapes (no real data)"
        info["source"] = source
        info["prompts"] = prompts
        info["shapes"] = {k: list(v.shape) for k, v in sample.items()}
        info["training_shapes_from_config"] = self.training_shapes()

        # Text context as in training: T5 output, padded positions zeroed (precompute_text_embeds.py:385-386).
        with torch.no_grad():
            context, mask = old.encode_prompt(prompts)
        context = context.clone()
        context[~mask] = 0
        sample["context"] = context.detach().cpu()
        sample["context_mask"] = mask.detach().cpu()
        info["context"] = {"dtype": str(context.dtype), "shape": list(context.shape),
                           "valid_tokens": mask.sum(dim=1).tolist(), "source": "old.encode_prompt (T5 umt5-xxl)"}

        shapes = self.training_shapes()
        steps = self.args.num_inference_steps or int(self.cfg.eval_num_inference_steps)
        inference = {
            "prompt": None,
            "input_image": sample["video"][:, :, 0],
            "action_horizon": shapes["action_horizon"],
            "proprio": sample["proprio"][:, 0],
            "context": sample["context"],
            "context_mask": sample["context_mask"],
            "num_inference_steps": steps,
            "seed": None,  # set per run
            "rand_device": "cpu",
            "num_video_frames": shapes["video_frames"],  # eval_libero_single.py:245-246
        }
        return sample, inference, info


# ----------------------------------------------------------------------------------------------
# Tiny mode: the parity tests' random CPU modules, same harness
# ----------------------------------------------------------------------------------------------


class TinyCase:
    def __init__(self, args, harness: Harness):
        import test_parity_mot as tiny  # also stands in for peft when it is missing

        self.tiny = tiny
        self.args = args
        self.h = harness
        self.structure = STRUCTURES[args.structure]
        # A "trained" checkpoint saved by the old class: seed-0 weights, shifted so loading matters.
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32  # like the real bf16 checkpoint
        source = tiny.old_model(*tiny.tiny_modules(seed=0), dtype=dtype)
        # The state encoder's projector width of the old class (EasyWAMMoT default 64,
        # easywam_mot.py:69,103); B must be built with the same one or its strict load fails.
        self.state_projector_hidden_dim = int(source.projector_hidden_dim)
        with torch.no_grad():
            for parameter in source.mot.parameters():
                parameter.add_(0.01)
            for parameter in source.state_encoder.parameters():
                parameter.add_(0.01)
        self.ckpt = harness.out_dir / "tiny_checkpoint.pt"
        source.save_checkpoint(str(self.ckpt), step=3)
        args.ckpt = self.ckpt
        del source

    def load_payload(self):
        return torch.load(str(self.ckpt), map_location="cpu", mmap=True)

    def build_old(self, dtype, *, load_text_encoder: bool):
        from openxam.model.helpers.inference import configure_model_execution

        old = self.tiny.old_model(*self.tiny.tiny_modules(seed=0), dtype=dtype)
        configure_model_execution(old, vae_micro_batch_size=None,
                                  inference_cross_kv_reuse=self.args.cross_kv_reuse == "on")
        old.load_checkpoint(str(self.ckpt), merge_lora=True)
        return old

    def build_new_real(self, dtype):
        """build_model with the backbone loader replaced by other random tiny towers, as
        tests/test_parity_mot.py:built() does; the VAE has the same values as the old one's, as
        the real VAE file is the same for both."""
        from openxam.core.build import build_model
        from openxam.model.backbone.contract import BackboneComponents
        from openxam.model.helpers.inference import configure_model_execution

        tiny = self.tiny
        video = tiny.tiny_modules(seed=9)[0].to(dtype)
        vae = tiny.tiny_modules(seed=0)[2].to(dtype)
        components = BackboneComponents("wan22", video, vae, None, None, tiny.TEXT_DIM)
        with mock.patch("openxam.core.build.load_easywam_backbone", return_value=components):
            new = build_model(
                self.structure["spec"],
                backbone={"name": "wan22"},
                action_dit_config=tiny.ACTION_CONFIG,
                state_dim=tiny.STATE_DIM,
                projector_hidden_dim=self.state_projector_hidden_dim,
                model_variant=self.structure["model_variant"],
                device="cpu",
                torch_dtype=dtype,
            )
        configure_model_execution(new, vae_micro_batch_size=None,
                                  inference_cross_kv_reuse=self.args.cross_kv_reuse == "on")
        return new, lambda: new.load_checkpoint(str(self.ckpt), merge_lora=True)

    def make_inputs(self, old) -> tuple[dict, dict, dict]:
        tiny = self.tiny
        sample = tiny.batch()
        inference = {
            "prompt": None,
            "input_image": sample["video"][:, :, 0],
            "action_horizon": tiny.HORIZON,
            "proprio": sample["proprio"][:, 0],
            "context": sample["context"],
            "context_mask": sample["context_mask"],
            "num_inference_steps": self.args.num_inference_steps or 4,
            "seed": None,
            "rand_device": "cpu",
            "num_video_frames": tiny.FRAMES,
        }
        info = {"source": "TINY synthetic batch (tests/test_parity_mot.py:batch)",
                "shapes": {k: list(v.shape) for k, v in sample.items()}}
        return sample, inference, info


# ----------------------------------------------------------------------------------------------
# Setup records
# ----------------------------------------------------------------------------------------------


def set_offline_env(out_dir: Path) -> dict:
    """Keep the HF libraries offline and their caches inside --out (runs before any repo import)."""
    os.environ["HF_DATASETS_CACHE"] = str((out_dir / "hf_datasets_cache").resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    return {
        "before_script": _ENV_BEFORE_SCRIPT,
        "HF_DATASETS_CACHE": os.environ["HF_DATASETS_CACHE"],
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "datasets_already_imported": "datasets" in sys.modules,
        "dont_write_bytecode": sys.dont_write_bytecode,
    }


def sdpa_flags() -> dict | None:
    if not torch.cuda.is_available():
        return None
    return {
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
        "cudnn": getattr(torch.backends.cuda, "cudnn_sdp_enabled", lambda: None)(),
    }


def set_determinism(args) -> dict:
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "model_attention_backend_config": "sdpa (model.backbone.attention_backend override)" if not args.tiny
        else "tiny modules' default",
        "sdpa_kernel_restriction": args.sdpa_kernel,
        "inference_cross_kv_reuse": args.cross_kv_reuse,
        "autocast_bf16_in_training_loss": args.autocast,
        "autocast_note": "applied only in bfloat16 blocks; the fp32 block is run without autocast",
        "train_seed": args.train_seed,
        "infer_seed": args.infer_seed,
        "floor_runs": args.floor_runs,
        "floor_factor": args.floor_factor,
        "perturb_ulps": args.perturb_ulps,
        "warmup": "one untimed run of each model before its compared runs; every compared run reseeds",
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
    }


def versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    import importlib.util

    out = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "flash_attn_importable": importlib.util.find_spec("flash_attn") is not None,
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    for package in ("hydra-core", "omegaconf", "numpy", "transformers", "accelerate", "peft", "torchvision",
                    "datasets"):
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            out[package] = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        out["gpu"] = {"name": props.name, "total_memory_gb": gb(props.total_memory),
                      "capability": f"{props.major}.{props.minor}", "count": torch.cuda.device_count()}
    return out


def repo_state() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", "--no-optional-locks", "-C", str(ROOT), *args], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception as error:  # noqa: BLE001
            return f"unavailable: {error!r}"

    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status) and not status.startswith("unavailable"), "status_porcelain": status[:2000]}


def checkpoint_summary(payload: dict) -> dict:
    out = {"top_level_keys": sorted(payload)}
    for key in ("step", "torch_dtype", "backbone_name", "model_variant", "state_position", "format",
                "action_config"):
        out[key] = payload.get(key)
    for key in ("mot", "dit", "state_encoder"):
        if key in payload:
            tensors = payload[key]
            dtypes: dict = {}
            for value in tensors.values():
                dtypes[str(value.dtype)] = dtypes.get(str(value.dtype), 0) + 1
            out[f"{key}_n_tensors"] = len(tensors)
            out[f"{key}_numel"] = sum(int(v.numel()) for v in tensors.values())
            out[f"{key}_dtypes"] = dtypes
    out["has_optimizer_state"] = "optimizer" in payload
    return out


# ----------------------------------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------------------------------


def run_block(h: Harness, case, old, dtype, tag: str, payload, *, include_real_build: bool) -> dict:
    """Warm-up, reference, floor, controls, NEW(A) and (optionally) NEW(B), all on ``old``'s dtype.
    Returns the old reference outputs."""
    args = h.args
    autocast = bool(args.autocast and dtype == torch.bfloat16)
    h.results[f"{tag}block"] = {"dtype": str(dtype), "autocast_applied": autocast}

    def run(model, sample=None, inference=None, *, train_seed=None, infer_seed=None, refs=None, keep_grads=False):
        return run_model(
            model,
            h.sample if sample is None else sample,
            h.inference if inference is None else inference,
            train_seed=args.train_seed if train_seed is None else train_seed,
            infer_seed=args.infer_seed if infer_seed is None else infer_seed,
            autocast=autocast,
            device=h.device,
            refs=refs or {},
            keep_grads=keep_grads,
        )

    with h.phase(f"{tag}old_warmup"):
        run(old, train_seed=args.train_seed + 12345)  # untimed, discarded; compared runs reseed
    with h.phase(f"{tag}old_reference"):
        ref, _ = run(old, keep_grads=True)
        h.results["comparisons"][f"{tag}old_reference"] = {
            "loss": float(ref["scalars"]["loss"].float()), "n_parameters_with_grad": ref["n_grads"],
            "trainable_without_grad": ref["trainable_without_grad"][:20],
            "actions_shape": list(ref["actions"]["actions"].shape)}
    refs = {"ref": ref}

    floor_runs = []
    for k in range(args.floor_runs):
        with h.phase(f"{tag}old_floor_run_{k + 1}"):
            _, cmps = run(old, refs=refs)
            floor_runs.append(cmps["ref"])
            h.results["comparisons"][f"{tag}noise_floor_run_{k + 1}"] = {c: summarize(cmps["ref"][c]) for c in CATEGORIES}
    floor = {c: floor_from([cmps[c] for cmps in floor_runs]) for c in CATEGORIES}
    h.results["noise_floor"][tag or "main"] = {
        c: {
            "runs": len(floor_runs),
            "old_model_deterministic": all(summarize(cmps[c])["all_bitwise"] for cmps in floor_runs),
            "n_items": len(floor[c]),
            "n_items_with_zero_floor": sum(1 for f in floor[c].values() if f["max_abs_diff"] == 0),
            "max_abs_diff": max((f["max_abs_diff"] for f in floor[c].values()), default=0.0),
            "max_rel_l2": max((f["rel_l2"] for f in floor[c].values()), default=0.0),
        }
        for c in CATEGORIES
    }
    del floor_runs

    with h.phase(f"{tag}control_other_seeds"):
        _, cmps = run(old, train_seed=args.train_seed + 1, infer_seed=args.infer_seed + 1000, refs=refs)
        verdicts = h.record(f"{tag}control_other_seeds", cmps["ref"], floor)
        ok = {c: verdicts[c]["verdict"] in DIFFERENT_VERDICTS for c in CATEGORIES}
        h.results["controls"][f"{tag}other_seeds_judged_different"] = ok
        if not all(ok.values()):
            print(f"!!! other-seeds control not judged DIFFERENT: {ok} -- harness problem", file=sys.stderr)
    with h.phase(f"{tag}control_small_perturbation"):
        sample, inference, info = perturbed_inputs(h.sample, h.inference, args.perturb_ulps)
        _, cmps = run(old, sample, inference, refs=refs)
        verdicts = h.record(f"{tag}control_small_perturbation", cmps["ref"], floor, {"perturbation": info})
        ok = {c: verdicts[c]["verdict"] not in IDENTICAL_VERDICTS for c in CATEGORIES}
        h.results["controls"][f"{tag}small_perturbation_not_identical"] = ok
        if not all(ok.values()):
            print(f"!!! small-perturbation control judged identical: {ok}", file=sys.stderr)
        del sample, inference

    with h.phase(f"{tag}new_shared_A"):
        new = shared_new_model(old, case.structure["spec"])
        same = [n for n, _ in new.named_parameters()] == [n for n, _ in old.named_parameters()]
        shares = all(p is q for (_, p), (_, q) in zip(new.named_parameters(), old.named_parameters()))
        h.results[f"{tag}new_shared_A_wiring"] = {"parameter_names_match_old": same, "shares_old_parameters": shares}
        run(new, train_seed=args.train_seed + 12345)  # warm-up
        _, cmps = run(new, refs=refs)
        h.record(f"{tag}new_A_vs_old", cmps["ref"], floor)
        del new, cmps

    if include_real_build:
        run_real_build(h, case, old, tag, dtype, payload, run, refs, floor)
    elif not tag:
        h.skip("new_real_B_build_and_load", "--skip-new-real")
    return ref


def run_real_build(h, case, old, tag, dtype, payload, run, refs, floor) -> None:
    args = h.args
    with h.phase(f"{tag}new_real_B_build_and_load"):
        new, load = case.build_new_real(dtype)
        before = compare_states(new, old)
        h.results[f"{tag}new_real_B_before_load_n_params_differing"] = before["parameters"]["n_not_bitwise"]
        returned = load()
        new = new.to(h.device)
        h.results[f"{tag}new_real_B_load_returned_step"] = returned.get("step")
        h.results[f"{tag}new_real_B_load_check"] = verify_loaded(new, payload)
        h.results[f"{tag}new_real_B_attention_backend"] = attention_settings(new)
        state = compare_states(new, old)
        h.results["comparisons"][f"{tag}new_B_vs_old_state"] = state
        h.results["verdicts"][f"{tag}new_B_vs_old_state"] = {
            "parameters_and_buffers": {"verdict": VERDICT_BITWISE if state["all_bitwise"] else VERDICT_DIFFERENT}
        }
        del returned
    with h.phase(f"{tag}new_real_B_forward"):
        run(new, train_seed=args.train_seed + 12345)  # warm-up
        _, cmps = run(new, refs=refs)
        h.record(f"{tag}new_B_vs_old", cmps["ref"], floor)
        del cmps
    if args.tiny:
        with h.phase(f"{tag}control_injected_bug"):
            injected_bug_control(h, new, tag, run, refs, floor)
    if args.tiny or args.roundtrip_dir:
        with h.phase(f"{tag}checkpoint_roundtrip"):
            checkpoint_roundtrip(h, new, old, tag, payload)
    else:
        h.skip(f"{tag}checkpoint_roundtrip", "not requested (--roundtrip-dir); writes a full checkpoint (~12 GB)")
    del new
    free_memory()


def injected_bug_control(h, new, tag, run, refs, floor) -> None:
    """NEW(B) with one action-tower weight nudged must be judged different; the weight is restored."""
    name, parameter = next((n, p) for n, p in new.named_parameters() if n.startswith("action_expert.") and p.ndim == 2)
    with torch.no_grad():
        saved = parameter.detach().clone()
        parameter.view(-1)[0] += 0.0625
    try:
        _, cmps = run(new, refs=refs)
        verdicts = h.record(f"{tag}control_injected_bug", cmps["ref"], floor,
                            {"nudged_parameter": name, "nudge": "+0.0625 on element 0"})
    finally:
        with torch.no_grad():
            parameter.copy_(saved)
    ok = {c: verdicts[c]["verdict"] in DIFFERENT_VERDICTS for c in CATEGORIES}
    ok["weight_restored"] = bits_equal(parameter.detach(), saved)
    h.results["controls"][f"{tag}injected_bug_judged_different"] = ok


def checkpoint_roundtrip(h, new, old, tag, payload) -> None:
    args = h.args
    folder = Path(args.roundtrip_dir) if args.roundtrip_dir else h.out_dir
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"roundtrip_from_xam_{os.getpid()}.pt"
    try:
        new.save_checkpoint(str(path), step=payload.get("step"))
        saved = torch.load(str(path), map_location="cpu", mmap=True)
        report = {"path": str(path), "top_level_keys_new": sorted(saved), "top_level_keys_original": sorted(payload)}
        for key in ("mot", "state_encoder"):
            if key in payload or key in saved:
                report[key] = compare_named(saved.get(key, {}).items(), payload.get(key, {}).items(),
                                            cast_ref_to_test_dtype=True)
        report["metadata_equal"] = {
            k: saved.get(k) == payload.get(k)
            for k in ("step", "torch_dtype", "backbone_name", "model_variant", "state_position", "action_config")
        }
        old.load_checkpoint(str(path), merge_lora=True)  # the old class must accept what XAMModel saved
        report["old_class_load_check"] = verify_loaded(old, saved)
        report["all_bitwise"] = all(report[k]["all_bitwise"] for k in ("mot", "state_encoder") if k in report)
        h.results["comparisons"][f"{tag}checkpoint_roundtrip"] = report
        h.results["verdicts"][f"{tag}checkpoint_roundtrip"] = {
            "saved_tensors_vs_original_checkpoint": {
                "verdict": VERDICT_BITWISE if report["all_bitwise"] else VERDICT_DIFFERENT},
            "old_class_loads_it": report["old_class_load_check"]["every_tensor_from_checkpoint"],
        }
        del saved
    finally:
        if not args.keep_roundtrip_file:
            path.unlink(missing_ok=True)


def precision_reference(h: Harness, bf16_ref: dict, fp32_ref: dict) -> None:
    """PRECISION_REF = old bf16 vs old fp32 on the same inputs, and every bf16 metric as a ratio to it."""
    device = "cuda" if torch.cuda.is_available() else None
    precision = {
        c: summarize(compare_items(bf16_ref[key], fp32_ref[key], device))
        for c, key in (("loss_and_logs", "scalars"), ("grads", "grads"), ("actions", "actions"))
    }
    h.results["comparisons"]["PRECISION_REF_old_bf16_vs_old_fp32"] = precision
    ratios = {}
    for key, summaries in h.summaries.items():
        if key.startswith("fp32_"):
            continue
        ratios[key] = {c: ratio_to_precision_ref(summaries[c], precision[c]) for c in CATEGORIES}
    h.results["ratios_to_precision_ref"] = ratios


def harness_checks(results: dict) -> dict:
    """Checks that must hold in both modes for the verdicts to mean anything."""
    checks = {}
    for name, case in results.get("verdict_self_test", {}).items():
        checks[f"verdict_self_test: {name}"] = case["ok"]
    for key, value in results["controls"].items():
        for category, ok in value.items():
            checks[f"control {key}.{category}"] = bool(ok)
    for key, value in results.items():
        if key.endswith("load_check") and isinstance(value, dict):
            checks[f"{key}.every_tensor_from_checkpoint"] = bool(value.get("every_tensor_from_checkpoint"))
        if key.endswith("new_real_B_before_load_n_params_differing"):
            checks[f"{key} > 0 (loading matters)"] = value > 0
        if key.endswith("new_shared_A_wiring"):
            checks[key] = all(value.values())
    return checks


def tiny_expectations(results: dict) -> dict:
    """What the tiny dry run must show (the parity tests prove the bitwise parts)."""
    checks = dict(results.get("harness_checks", {}))
    for key, value in results["verdicts"].items():
        if "new_" in key or key.endswith("checkpoint_roundtrip"):
            for category, entry in value.items():
                if isinstance(entry, dict) and "verdict" in entry:
                    checks[f"{key}.{category} is {VERDICT_BITWISE}"] = entry["verdict"] == VERDICT_BITWISE
                elif isinstance(entry, bool):
                    checks[f"{key}.{category}"] = entry
    for tag, floor in results["noise_floor"].items():
        for category, entry in floor.items():
            checks[f"noise floor {tag}.{category}: old deterministic on CPU"] = entry["old_model_deterministic"]
    if "control_injected_bug" not in results["verdicts"]:
        checks["injected-bug control ran"] = False
    return checks


def summary_lines(results: dict) -> list:
    lines = [f"mode={results['mode']} structure={results['structure']} input={results.get('input', {}).get('source')}"]
    for name, entry in results["phases"].items():
        if entry.get("status") != "ok":
            lines.append(f"  phase {name}: {entry.get('status')} {entry.get('error') or entry.get('reason') or ''}")
    for tag, floor in results["noise_floor"].items():
        text = ", ".join(f"{c}: deterministic={v['old_model_deterministic']} max_rel_l2={v['max_rel_l2']:.3g}"
                         for c, v in floor.items())
        lines.append(f"  noise floor [{tag}] {text}")
    for key, value in results["verdicts"].items():
        parts = []
        for category, entry in value.items():
            if isinstance(entry, dict) and "verdict" in entry:
                extra = ""
                if "worst_rel_l2" in entry:
                    extra = f" (worst rel_l2={entry['worst_rel_l2']:.3g} = {entry['worst_rel_l2_in_bf16_roundoff_units']:.3g} u_bf16)"
                parts.append(f"{category}={entry['verdict']}{extra}")
            else:
                parts.append(f"{category}={entry}")
        lines.append(f"  {key}: " + "; ".join(parts))
    for key, value in results.get("ratios_to_precision_ref", {}).items():
        text = ", ".join(f"{c}: worst_rel_l2 x{v['worst_rel_l2_ratio']}" for c, v in value.items())
        lines.append(f"  ratio to PRECISION_REF [{key}] {text}")
    failed = [k for k, v in results.get("harness_checks", {}).items() if not v]
    lines.append(f"  harness checks failed: {failed or 'none'}")
    return lines


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument("--structure", choices=sorted(STRUCTURES), default="mot",
                   help="Only structures wired here are accepted.")
    p.add_argument("--tiny", action="store_true", help="CPU dry run on the parity tests' tiny random modules.")
    p.add_argument("--out", required=True, help="Output directory (results.json, config dumps, caches).")
    p.add_argument("--ckpt", default="checkpoints/EasyWAM-MoT-Wan22/libero_mot_wan22.pt")
    p.add_argument("--dataset-stats", default="checkpoints/EasyWAM-MoT-Wan22/libero_dataset_stats.json")
    p.add_argument("--wan-dir", default="checkpoints/Wan2.2-TI2V-5B")
    p.add_argument("--data-root", default="data/libero-lerobot-v3.0")
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--input", choices=("auto", "real", "synthetic"), default="auto",
                   help="auto: real LIBERO sample, synthetic if that fails (recorded loudly).")
    p.add_argument("--sample-indices", default=None, help="Comma-separated frame indices (default: evenly spaced).")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16",
                   help="Dtype of the main block (old, A, B). fp32 with B needs ~2x memory; prefer --fp32-check.")
    p.add_argument("--fp32-check", action="store_true",
                   help="Extra float32 block (old vs NEW(A)) and PRECISION_REF = old bf16 vs old fp32.")
    p.add_argument("--autocast", action="store_true",
                   help="Wrap training_loss in torch.autocast(bfloat16) as the trainer does (bf16 blocks only).")
    p.add_argument("--skip-new-real", action="store_true", help="Skip NEW(B) (real build) in the main block.")
    p.add_argument("--roundtrip-dir", default=None,
                   help="Real mode: save a checkpoint from NEW(B) here and load it into the old class.")
    p.add_argument("--keep-roundtrip-file", action="store_true")
    p.add_argument("--train-seed", type=int, default=7)
    p.add_argument("--infer-seed", type=int, default=11, help="Per-sample seeds are infer_seed + i.")
    p.add_argument("--num-inference-steps", type=int, default=None,
                   help="Default: cfg.eval_num_inference_steps (real), 4 (tiny).")
    p.add_argument("--cross-kv-reuse", choices=("on", "off"), default="on",
                   help="inference_cross_kv_reuse; on is the train/eval config default.")
    p.add_argument("--sdpa-kernel", choices=("default", "math"), default="default",
                   help="Restrict torch SDPA to the math kernel (slower, avoids fused-kernel nondeterminism).")
    p.add_argument("--floor-runs", type=int, default=3, help="Old-vs-old runs for the noise floor (max over runs).")
    p.add_argument("--floor-factor", type=float, default=1.5,
                   help="WITHIN_NOISE_FLOOR if each differing tensor <= factor x its own floor.")
    p.add_argument("--perturb-ulps", type=int, default=4, help="bf16 ulps for the small-perturbation control.")
    args = p.parse_args(argv)
    if args.floor_runs < 1:
        p.error("--floor-runs must be at least 1")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    env_record = set_offline_env(out_dir)  # before any repo (and so `datasets`) import
    if args.tiny:
        device = "cpu"
    else:
        if not torch.cuda.is_available():
            print("Real mode needs a CUDA GPU (use --tiny for the CPU dry run).", file=sys.stderr)
            return 2
        device = "cuda"
        for label in ("ckpt", "dataset_stats", "wan_dir"):
            if not Path(getattr(args, label)).exists():
                print(f"--{label.replace('_', '-')} does not exist: {getattr(args, label)}", file=sys.stderr)
                return 2

    h = Harness(args, out_dir, device)
    warnings.simplefilter("always")
    warnings.showwarning = h.record_warning
    h.results["environment"] = env_record
    h.results["settings"] = set_determinism(args)
    h.results["versions"] = versions()
    h.results["repo"] = repo_state()
    h.results["verdict_rules"] = {
        VERDICT_BITWISE: "every compared tensor has the same dtype, shape and bit pattern",
        VERDICT_SIGNED_ZERO: "not bitwise, but every element difference is exactly 0 (e.g. +0 vs -0)",
        VERDICT_FLOOR: f"each differing tensor has max_abs_diff and rel_l2 <= {args.floor_factor} x its own "
                       f"old-vs-old floor (max over {args.floor_runs} runs)",
        VERDICT_FLOOR_ZERO: "some tensor differs where every old-vs-old run was exactly 0; read rel_l2 in bf16 "
                            "roundoff units (2^-8) and the ratio to PRECISION_REF",
        VERDICT_DIFFERENT: "above the floor, non-finite values or limits, missing names/shapes, or no floor",
        "rel_l2": "||test - old|| / ||old||, or the absolute ||test - old|| when ||old|| = 0 (flagged)",
    }
    if args.sdpa_kernel == "math" and torch.cuda.is_available():
        from torch.nn.attention import SDPBackend, sdpa_kernel

        kernel_context = sdpa_kernel([SDPBackend.MATH])
    else:
        kernel_context = nullcontext()
    h.save()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    try:
        with kernel_context:
            h.results["settings"]["sdpa_backends_enabled_in_context"] = sdpa_flags()
            with h.phase("setup"):
                h.results["verdict_self_test"] = verdict_self_test(args.floor_factor)
                case = TinyCase(args, h) if args.tiny else RealCase(args, h)
                h.results["paths"] = {k: str(getattr(args, k)) for k in
                                      ("ckpt", "dataset_stats", "wan_dir", "data_root")}
            with h.phase("checkpoint_inspect"):
                payload = case.load_payload()
                h.results["checkpoint"] = checkpoint_summary(payload)
            with h.phase("build_old_with_text_encoder"):
                old = case.build_old(dtype, load_text_encoder=True)
                h.results["old_load_check"] = verify_loaded(old, payload)
                h.results["old_attention_backend"] = attention_settings(old)
            with h.phase("inputs"):
                h.sample, h.inference, h.results["input"] = case.make_inputs(old)
                print(f"INPUT SOURCE: {h.results['input']['source']}", flush=True)
                if not args.tiny:
                    old.text_encoder = None  # context is computed; free T5 (~11 GB) before the comparisons
            main_ref = run_block(h, case, old, dtype, "", payload, include_real_build=not args.skip_new_real)
            del old
            free_memory()

            if not args.fp32_check:
                h.skip("fp32_block", "not requested (--fp32-check)")
            elif dtype == torch.float32:
                h.skip("fp32_block", "main block already ran in float32")
            else:
                try:
                    with h.phase("fp32_build_old"):
                        old = case.build_old(torch.float32, load_text_encoder=False)
                        h.results["fp32_old_load_check"] = verify_loaded(old, payload)
                    fp32_ref = run_block(h, case, old, torch.float32, "fp32_", payload, include_real_build=False)
                    del old
                    free_memory()
                    with h.phase("precision_reference"):
                        precision_reference(h, main_ref, fp32_ref)
                    del fp32_ref
                except PhaseFailed:
                    pass  # recorded; nothing after it depends on it
            del main_ref
    except PhaseFailed as failed:
        print(f"Stopped after failed phase {failed}", file=sys.stderr)
    except BaseException as error:  # a harness bug outside any phase, or an interrupt
        h.failed_phases.append("harness_outside_phase")
        h.results["harness_error"] = {"error": repr(error), "traceback": traceback.format_exc()}
        print(f"!!! harness error outside a phase: {error!r}", file=sys.stderr)
    finally:
        h.results["finished_at"] = datetime.now(timezone.utc).isoformat()
        h.results["failed_phases"] = h.failed_phases
        h.results["completed"] = not h.failed_phases
        h.results["harness_checks"] = harness_checks(h.results)
        if args.tiny:
            checks = tiny_expectations(h.results)
            h.results["tiny_expectations"] = checks
            h.results["tiny_expectation_met"] = bool(checks) and all(checks.values()) and not h.failed_phases
        h.save()
        print("\n==== SUMMARY ====")
        for line in summary_lines(h.results):
            print(line)
        if args.tiny:
            failed = [k for k, v in h.results["tiny_expectations"].items() if not v]
            print(f"TINY HARNESS CHECK: {'PASS' if h.results['tiny_expectation_met'] else 'FAIL'}"
                  + (f" -- failed: {failed}" if failed else ""))
        print(f"results: {out_dir / 'results.json'}")
    return 1 if h.failed_phases else 0


if __name__ == "__main__":
    sys.exit(main())
