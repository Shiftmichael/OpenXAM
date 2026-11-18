#!/usr/bin/env python
"""Find where two LIBERO evaluation runs that should be identical start to differ.

OpenXAM's LIBERO evaluator (``experiments/libero/eval_libero_single.py``) is run in-process with
read-only hooks that record, per episode, every environment step and every policy call (replan).
``compare`` then walks two such traces in time order and names the first thing that differs:
the simulator (an observation after an identical action history), the text context, the model
(identical inputs, different output), pre/post-processing, or nothing.

Subcommands (run from anywhere; the script changes into the repo root, where ``checkpoints/`` is):

    run      --model old|new --out DIR [--deterministic] [--context-from FILE_OR_RUN_DIR]
             One evaluation (default: libero_spatial task 0, 2 trials, seed 42, the released MoT-Wan22
             checkpoint), traced. Writes DIR/index.json, DIR/trace.npz, DIR/context.npz,
             DIR/config.yaml and the evaluator's own results JSON under DIR/eval_output/.
    compare  RUN_A RUN_B --out FILE.json
             RUN_A is the reference. Per episode: first divergence and its kind, first policy call
             whose raw input / model output differs, metrics there, context equality, outcomes,
             counts, and a one-line verdict. Needs numpy only (no torch, no GPU).
    suite    --out DIR [--steps ...] [--stop-after-seconds N] [--run-timeout 900]
             The planned sequence, each run in its own fresh process (as the stock evaluator is run):
             selftest, then stock_old_1, stock_old_2, stock_new_1, stock_new_2, det_old, det_new,
             det_new_ctx_old (new model, --deterministic, context from det_old), then the comparisons
             in SUITE_COMPARISONS, only between runs made by this invocation. Writes DIR/suite.json,
             DIR/compare_<ref>__vs__<test>.json and DIR/<step>/run.log. Use a fresh DIR (or --overwrite).
             If the selftest fails, nothing else runs and the exit code is 3. A run that exceeds
             --run-timeout is killed, recorded as "timeout" and left out of the comparisons.
    selftest
             Drives the tracer with a fake evaluator loop (numpy only) and checks that compare gives
             the expected verdict for seeded differences (model output, simulator, context, padded
             context, done flag, incomplete trace). Run it before trusting a verdict.

How the evaluator is run (``run``): the Hydra config is composed with the same primary config
(``benchmark/sim_libero.yaml``) and the same overrides as the stock command line, then the body of
``eval_single_process`` (eval_libero_single.py:906-915) is replayed with the module's own functions:
``set_global_seed`` -> ``build_eval_runtime`` -> ``evaluate_task_with_runtime`` -> ``write_json_atomic``.
This is what the repo's own worker does (experiments/libero/eval_libero_worker.py:58,91). Calling the
``@hydra.main`` function from an imported module was avoided: Hydra then looks the config up as the
package ``configs`` relative to the module name, and ``configs/`` is not a package. What is lost is
only Hydra's job runtime (logging setup, HydraConfig); the LIBERO path does not read HydraConfig.
The composed config is written to DIR/config.yaml for checking.

Hooks (all are wrappers that call the original with the same arguments and return its result
object unchanged; recording happens on CPU copies, with hashlib/numpy only, so no RNG is drawn,
no model call is added and no tensor the evaluator uses is modified):
  * ``ev.get_libero_env``       -> the env is wrapped in a proxy that hashes the observation returned
                                   by reset / set_init_state / every step, and stores every executed
                                   action (float64) with done/reward.
  * ``ev.run_single_episode``   -> episode index, initial-state hash, success, step/call counts.
  * ``ev._predict_action_chunk``-> raw camera images (as they come from the env) and raw proprio
                                   fields, hashed and saved; the env action chunk it returns.
  * ``model.infer_action``      -> (instance attribute) the processed model inputs (image, normalized
                                   proprio, context, mask, scalar arguments) and the raw action output
                                   (before denormalization).
  * ``ev._denormalize_action``  -> the denormalized action (before the gripper transform).
  * ``runtime.prompt_cache.get``-> the prompt and the model's own T5 context; with ``--context-from``
                                   it returns the saved context instead (the T5 still runs, as in
                                   stock, so everything else is unchanged).
A hook that fails is recorded in index.json ``hook_errors`` and the evaluator carries on unchanged;
``compare`` then labels that pair UNRELIABLE. The only hook that may stop a run is the context
override, on a prompt/shape mismatch. The hooks add device-to-host copies (a CUDA sync) and so change
timing only; the repo never enables cudnn.benchmark, the one setting where timing picks kernels.

The env child matches stock: LiberoEnvProcess starts it with `spawn` (env_process.py:112), and spawn
re-runs the parent's __main__ file in the child as __mp_main__ (multiprocessing.spawn:
get_preparation_data uses ``__main__.__file__`` when ``__main__.__spec__`` is None, as for
``python script.py``). In stock that file is eval_libero_single.py (torch, accelerate, hydra, libero and
openxam imported, TOKENIZERS_PARALLELISM set). ``run`` therefore points ``__main__.__file__`` at
eval_libero_single.py for the duration of the evaluation, and records what spawn will use
(index.json ``spawn_main``). The child needs nothing from this script: its target
(experiments.libero.env_process._environment_main) is pickled by module name, and the tracing proxy
wraps the LiberoEnvProcess handle in the parent.

Differences from the stock run that are not hooks (none should change a number; listed for honesty):
no Hydra job runtime (logging is set up with logging.basicConfig instead); HF_HUB_OFFLINE and
HF_DATASETS_OFFLINE=1; sys.argv in the env child is this script's (the child's __mp_main__ does not
read it). EVALUATION.video_mode=all as in stage 2 (rollout MP4s under DIR/eval_output/).

``--deterministic``: before torch is imported CUBLAS_WORKSPACE_CONFIG=:4096:8; then
``torch.use_deterministic_algorithms(True, warn_only=True)``, cudnn.deterministic=True,
cudnn.benchmark=False, TF32 off (matmul and cuDNN). Without it nothing is changed (the stock evaluator
sets none of these); the flags in effect are recorded at start-up and at the first policy call.
Always set: HF_HUB_OFFLINE=1, HF_DATASETS_OFFLINE=1. Defaulted only if unset (as in the stage 2 job):
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1, MUJOCO_GL=egl, PYOPENGL_PLATFORM=egl.

``--context-from``: a ``context.npz`` written by an earlier ``run`` (or its run directory). The saved
context is float32 (exact: bf16 -> fp32 is lossless) and is cast back to the model dtype; the cast is
checked to reproduce the saved bits (index.json ``context_override.exact``). The prompt must match.

Exit codes: 0 ok, 1 failure (index.json says where), 2 bad arguments, 3 (suite) selftest failed.
``compare`` exits 0 when it could compare, whatever the verdict.
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True  # no __pycache__ next to the repo's sources

# Only the standard library and numpy at module level, so `compare`/`selftest` need no torch. During
# `run`, spawn children re-run eval_libero_single.py, not this file (point_spawn_main_at_evaluator).
# Everything else is imported inside the subcommands.
import argparse  # noqa: E402
import functools  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FORMAT_VERSION = 1
TASKS = {"old": "libero_easywam_mot_wan22", "new": "libero_xam_fastwam_wan22"}
DEFAULT_CKPT = "checkpoints/EasyWAM-MoT-Wan22/libero_mot_wan22.pt"
DEFAULT_STATS = "checkpoints/EasyWAM-MoT-Wan22/libero_dataset_stats.json"
STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")  # eval_libero_single.py:213-225
AGENTVIEW_KEY = "agentview_image"  # libero_utils.py:get_libero_image
WRIST_KEY = "robot0_eye_in_hand_image"
TRACED_TENSORS = ("input_image", "proprio", "context", "context_mask")
CALL_ARRAYS = ("agentview", "wrist", "raw_state", "input_image", "proprio", "raw_action", "denorm_action",
               "env_action_chunk")
ENV_KEYS = ("CUBLAS_WORKSPACE_CONFIG", "HF_HOME", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_DATASETS_CACHE",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "MUJOCO_GL", "PYOPENGL_PLATFORM", "PYTHONHASHSEED",
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM",
            "SLURM_JOB_ID", "PYTHONPATH", "PYTHONNOUSERSITE", "HYDRA_FULL_ERROR")

# name, model, deterministic, context-from (name of an earlier step, or None)
SUITE_STEPS = (
    ("stock_old_1", "old", False, None),
    ("stock_old_2", "old", False, None),
    ("stock_new_1", "new", False, None),
    ("stock_new_2", "new", False, None),
    ("det_old", "old", True, None),
    ("det_new", "new", True, None),
    ("det_new_ctx_old", "new", True, "det_old"),
)
# reference, test, question
SUITE_COMPARISONS = (
    ("stock_old_1", "stock_old_2", "is the stock evaluator reproducible, old model?"),
    ("stock_new_1", "stock_new_2", "is the stock evaluator reproducible, new model?"),
    ("stock_old_1", "stock_new_1", "old vs new as users run it"),
    ("det_old", "det_new", "old vs new, deterministic settings"),
    ("det_old", "det_new_ctx_old", "old vs new given the old model's T5 context (deterministic)"),
    ("det_new", "det_new_ctx_old", "new with its own vs the old T5 context (deterministic)"),
)


# ----------------------------------------------------------------------------------------------
# Small helpers (numpy only)
# ----------------------------------------------------------------------------------------------


def jsonable(value):
    """Plain JSON: numpy scalars/arrays become Python values, non-finite floats become strings."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def write_json(path: Path, payload) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(jsonable(payload), indent=1))
    os.replace(tmp, path)


def write_npz(path: Path, arrays: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:  # a file object, so numpy does not append ".npz" to the name
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def sha_exact(arr: np.ndarray, label: str) -> str:
    """sha256 of (dtype label, shape, raw bytes). ``label`` is the torch dtype for tensors (so bf16
    stored as int16 bits does not collide with real int16)."""
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(f"{label}|{tuple(arr.shape)}|".encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def sha_array(value) -> str:
    arr = np.asarray(value)
    if arr.dtype == object:
        return hashlib.sha256(("object|" + repr(value)).encode()).hexdigest()
    return sha_exact(arr, arr.dtype.str)


def obs_digest(obs) -> dict:
    """Hashes of one LIBERO observation: every key (sorted), the two cameras, the proprio fields."""
    full = hashlib.sha256()
    for key in sorted(obs):
        full.update(f"{key}=".encode())
        full.update(sha_array(obs[key]).encode())
    state = hashlib.sha256()
    for key in STATE_KEYS:
        state.update(f"{key}=".encode())
        state.update(sha_array(obs[key]).encode() if key in obs else b"missing")
    return {
        "obs": full.hexdigest(),
        "agentview": sha_array(obs[AGENTVIEW_KEY]) if AGENTVIEW_KEY in obs else None,
        "wrist": sha_array(obs[WRIST_KEY]) if WRIST_KEY in obs else None,
        "state": state.hexdigest(),
    }


def exact_numpy(value) -> tuple[np.ndarray, str]:
    """An independent CPU copy with the exact bits, and its dtype label. bf16 tensors are returned as
    their int16 bit pattern (numpy has no bfloat16). Accepts numpy arrays too (selftest)."""
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True), value.dtype.name
    import torch

    copy = value.detach().to("cpu", copy=True).contiguous()
    label = str(copy.dtype).replace("torch.", "")
    if copy.dtype == torch.bfloat16:
        return copy.view(torch.int16).numpy().copy(), label
    return copy.numpy().copy(), label


def exact_to_float32(arr: np.ndarray, label: str) -> np.ndarray:
    if label == "bfloat16":
        bits = arr.astype(np.int16).view(np.uint16).astype(np.uint32) << np.uint32(16)
        return bits.view(np.float32)
    return arr.astype(np.float32)


def exact_to_float64(arr: np.ndarray, label: str) -> np.ndarray:
    if label == "bfloat16":
        return exact_to_float32(arr, label).astype(np.float64)
    return np.asarray(arr).astype(np.float64)


def stats(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"numel": 0}
    finite = np.isfinite(v)
    f = v[finite]
    return {
        "numel": int(v.size),
        "n_nonfinite": int(v.size - finite.sum()),
        "mean": float(f.mean()) if f.size else None,
        "std": float(f.std()) if f.size else None,
        "min": float(f.min()) if f.size else None,
        "max": float(f.max()) if f.size else None,
        "l2": float(np.sqrt((f * f).sum())) if f.size else None,
    }


def diff_metrics(test: np.ndarray, ref: np.ndarray) -> dict:
    """Metrics of ``test`` against ``ref`` in float64. rel_l2 = ||test-ref|| / ||ref|| (absolute when
    ||ref|| = 0, flagged)."""
    if test is None or ref is None:
        return {"available": False}
    t = np.asarray(test, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    if t.shape != r.shape:
        return {"available": True, "shape_mismatch": [list(t.shape), list(r.shape)]}
    d = t - r
    ref_norm = float(np.sqrt((r * r).sum()))
    diff_norm = float(np.sqrt((d * d).sum()))
    return {
        "available": True,
        "shape": list(t.shape),
        "n_elements_differing": int(np.count_nonzero(d)),
        "numel": int(d.size),
        "max_abs_diff": float(np.abs(d).max()) if d.size else 0.0,
        "mean_abs_diff": float(np.abs(d).mean()) if d.size else 0.0,
        "rel_l2": diff_norm / ref_norm if ref_norm > 0 else diff_norm,
        "rel_l2_is_absolute": ref_norm == 0,
    }


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------------------------------
# The tracer (used by `run`, and by `selftest` with numpy stand-ins)
# ----------------------------------------------------------------------------------------------


class Tracer:
    """Collects the trace. Every public ``on_*`` method is called through ``safe`` from a hook."""

    def __init__(self, out_dir: Path | None, index: dict, torch_module=None):
        self.out = out_dir
        self.index = index
        self.torch = torch_module
        self.arrays: dict = {}
        self.context_file: dict = {}
        self.ep: dict | None = None
        self.call: dict | None = None
        self._actions: list = []
        self.override = None  # the --context-from payload, set by cmd_run
        self._override_tensors = None
        index.setdefault("episodes", [])
        index.setdefault("hook_errors", [])
        index.setdefault("prompts", [])
        index.setdefault("own_context_shas", [])

    # -- bookkeeping --

    def safe(self, where: str, fn, *args) -> None:
        try:
            fn(*args)
        except Exception as error:  # noqa: BLE001 - recorded; the evaluator itself is not affected
            errors = self.index["hook_errors"]
            if len(errors) < 50:
                errors.append({"where": where, "error": repr(error), "traceback": traceback.format_exc()})
            print(f"[trace] hook error in {where}: {error!r}", file=sys.stderr, flush=True)

    def _flush_actions(self) -> None:
        if self.ep is None:
            return
        key = f"ep{self.ep['episode']}_executed_actions"
        if self._actions:
            self.arrays[key] = np.stack(self._actions)
        else:
            self.arrays[key] = np.zeros((0, 7), dtype=np.float64)

    def save(self) -> None:
        self.index["updated_at"] = now()
        if self.out is None:
            return
        self._flush_actions()
        write_npz(self.out / "trace.npz", self.arrays)
        if self.context_file:
            write_npz(self.out / "context.npz", self.context_file)
        write_json(self.out / "index.json", self.index)

    # -- episodes and env --

    def start_episode(self, episode_idx, initial_state) -> None:
        self._flush_actions()
        ep = {
            "episode": int(episode_idx) if episode_idx is not None else len(self.index["episodes"]),
            "initial_state_sha": None,
            "reset_obs_sha": None,
            # index k = the observation after k env steps (k = 0: right after set_init_state)
            "steps": {"obs_sha": [], "agentview_sha": [], "wrist_sha": [], "state_sha": [], "done": [],
                      "reward": []},
            "calls": [],
            "success": None,
            "n_env_steps": 0,
            "n_policy_calls": 0,
            "seconds": None,
            "finished": False,
        }
        self.index["episodes"].append(ep)
        self.ep = ep
        self.call = None
        self._actions = []
        ep["initial_state_sha"] = sha_array(np.asarray(initial_state))

    def _push_obs(self, obs, done, reward) -> None:
        d = obs_digest(obs)
        steps = self.ep["steps"]
        steps["obs_sha"].append(d["obs"])
        steps["agentview_sha"].append(d["agentview"])
        steps["wrist_sha"].append(d["wrist"])
        steps["state_sha"].append(d["state"])
        steps["done"].append(done)
        steps["reward"].append(reward)

    def on_reset(self, obs) -> None:
        self.ep["reset_obs_sha"] = obs_digest(obs)["obs"]

    def on_init(self, obs) -> None:
        self._push_obs(obs, None, None)

    def on_step(self, action, result) -> None:
        obs, reward, done = result[0], result[1], result[2]
        self._actions.append(np.asarray(action, dtype=np.float64).reshape(-1).copy())
        self._push_obs(obs, bool(done), float(reward))
        self.ep["n_env_steps"] = len(self._actions)

    def end_episode(self, result, seconds: float) -> None:
        ep = self.ep
        ep["success"] = bool(result[0])
        ep["n_env_steps"] = len(self._actions)
        ep["n_policy_calls"] = len(ep["calls"])
        ep["seconds"] = round(seconds, 3)
        ep["finished"] = True
        self.save()

    # -- policy calls --

    def _store(self, name: str, arr: np.ndarray, label: str) -> dict:
        ep, call = self.ep, self.call
        self.arrays[f"ep{ep['episode']}_c{call['call']}_{name}"] = arr
        return {"sha": sha_exact(arr, label), "dtype": label, "shape": list(arr.shape)}

    def begin_call(self, obs) -> None:
        ep = self.ep
        call = {"call": len(ep["calls"]), "env_step": len(ep["steps"]["obs_sha"]) - 1, "_t0": time.perf_counter(),
                "raw": None, "inputs": None, "outputs": {}}
        ep["calls"].append(call)
        ep["n_policy_calls"] = len(ep["calls"])
        self.call = call
        d = obs_digest(obs)
        last = ep["steps"]["obs_sha"][-1] if ep["steps"]["obs_sha"] else None
        raw = {"obs_sha": d["obs"], "agentview_sha": d["agentview"], "wrist_sha": d["wrist"],
               "state_sha": d["state"], "obs_matches_last_step": d["obs"] == last}
        self._store("agentview", np.array(obs[AGENTVIEW_KEY], copy=True), "uint8")
        self._store("wrist", np.array(obs[WRIST_KEY], copy=True), "uint8")
        raw_state = np.concatenate([np.asarray(obs[k], dtype=np.float64).reshape(-1) for k in STATE_KEYS])
        raw["raw_state"] = self._store("raw_state", raw_state, "float64")
        call["raw"] = raw

    def on_model_input(self, args, kwargs) -> None:
        call = self.call
        inputs: dict = {}
        if args:
            inputs["positional_args_not_traced"] = len(args)
        for name in TRACED_TENSORS:
            value = kwargs.get(name)
            if value is None:
                inputs[name] = None
                continue
            arr, label = exact_numpy(value)
            entry = {"sha": sha_exact(arr, label), "dtype": label, "shape": list(arr.shape),
                     "device": str(getattr(value, "device", "cpu"))}
            if name in ("input_image", "proprio"):
                entry["stats"] = stats(exact_to_float64(arr, label))
                self.arrays[f"ep{self.ep['episode']}_c{call['call']}_{name}"] = arr
            if name == "context_mask":
                entry["valid_tokens"] = np.asarray(arr).astype(bool).sum(axis=-1).tolist()
            inputs[name] = entry
        if "context" not in self.context_file and kwargs.get("context") is not None:
            self._save_context("", kwargs["context"], kwargs.get("context_mask"))
        scalars = {k: repr(v) for k, v in sorted(kwargs.items()) if k not in TRACED_TENSORS}
        inputs["scalars"] = scalars
        inputs["scalars_sha"] = hashlib.sha256(json.dumps(scalars, sort_keys=True).encode()).hexdigest()
        call["inputs"] = inputs
        if "flags_at_first_policy_call" not in self.index and self.torch is not None:
            self.index["flags_at_first_policy_call"] = torch_flags(self.torch)

    def on_model_output(self, out) -> None:
        arr, label = exact_numpy(out["action"])
        entry = self._store("raw_action", arr, label)
        entry["stats"] = stats(exact_to_float64(arr, label))
        self.call["outputs"]["raw_action"] = entry

    def on_denorm(self, result) -> None:
        arr, label = exact_numpy(np.asarray(result))
        self.call["outputs"]["denorm_action"] = self._store("denorm_action", arr, label)

    def end_call(self, result) -> None:
        arr, label = exact_numpy(np.asarray(result[0]))
        call = self.call
        call["outputs"]["env_action_chunk"] = self._store("env_action_chunk", arr, label)
        call["seconds"] = round(time.perf_counter() - call.pop("_t0"), 4)

    # -- prompt context --

    def _save_context(self, prefix: str, context, mask) -> None:
        arr, label = exact_numpy(context)
        self.context_file[f"{prefix}context"] = exact_to_float32(arr, label)
        self.context_file[f"{prefix}dtype"] = np.array(label)
        self.context_file[f"{prefix}sha"] = np.array(sha_exact(arr, label))
        if mask is not None:
            mask_arr, mask_label = exact_numpy(mask)
            self.context_file[f"{prefix}context_mask"] = mask_arr.astype(bool)
            self.context_file[f"{prefix}mask_sha"] = np.array(sha_exact(mask_arr, mask_label))
        if self.index["prompts"]:
            self.context_file["prompt"] = np.array(self.index["prompts"][0])
        info = self.index.setdefault("context", {})
        info[f"{prefix or 'used_'}context"] = {"sha": sha_exact(arr, label), "dtype": label, "shape": list(arr.shape)}

    def on_prompt(self, prompt, context, mask) -> None:
        if prompt not in self.index["prompts"]:
            self.index["prompts"].append(prompt)
        if "prompt" not in self.context_file:
            self.context_file["prompt"] = np.array(prompt)
        arr, label = exact_numpy(context)
        sha = sha_exact(arr, label)
        if sha not in self.index["own_context_shas"]:
            self.index["own_context_shas"].append(sha)
        if self.override is not None and "own_context" not in self.context_file:
            self._save_context("own_", context, mask)

    def override_context(self, prompt, own_context, own_mask):
        """The saved context instead of the model's own. Raises on a mismatch: a silent fallback
        would make the run look like a context swap when it is not."""
        ov = self.override
        if prompt != ov["prompt"]:
            raise RuntimeError(f"--context-from prompt {ov['prompt']!r} != this run's prompt {prompt!r}")
        if self._override_tensors is None:
            torch = self.torch
            context = torch.from_numpy(ov["context"].copy()).to(device=own_context.device, dtype=own_context.dtype)
            mask = torch.from_numpy(ov["context_mask"].copy()).to(device=own_mask.device, dtype=own_mask.dtype)
            if tuple(context.shape) != tuple(own_context.shape) or tuple(mask.shape) != tuple(own_mask.shape):
                raise RuntimeError(f"--context-from shapes {tuple(context.shape)}/{tuple(mask.shape)} != own "
                                   f"{tuple(own_context.shape)}/{tuple(own_mask.shape)}")
            arr, label = exact_numpy(context)
            record = self.index["context_override"]
            record["cast_to"] = label
            record["exact"] = label == ov["dtype"] and sha_exact(arr, label) == ov["sha"]
            if not record["exact"]:
                print("[trace] WARNING: the override context does not reproduce the saved bits "
                      f"(saved {ov['dtype']}, model {label})", file=sys.stderr, flush=True)
            self._override_tensors = (context, mask)
        return self._override_tensors


class TracingEnv:
    """Forwards to the LiberoEnvProcess and records what it returns. Returns the same objects."""

    def __init__(self, env, tracer: Tracer):
        self._env = env
        self._tracer = tracer

    def reset(self):
        obs = self._env.reset()
        self._tracer.safe("env.reset", self._tracer.on_reset, obs)
        return obs

    def set_init_state(self, initial_state):
        obs = self._env.set_init_state(initial_state)
        self._tracer.safe("env.set_init_state", self._tracer.on_init, obs)
        return obs

    def step(self, action):
        result = self._env.step(action)
        self._tracer.safe("env.step", self._tracer.on_step, action, result)
        return result

    def close(self):
        return self._env.close()

    def __getattr__(self, name):
        return getattr(self._env, name)


def install_eval_hooks(ev, tracer: Tracer) -> dict:
    """Module-level wrappers. eval_libero_single looks these names up as module globals at call time
    (run_single_task -> get_libero_env / run_single_episode; run_single_episode -> _predict_action_chunk;
    _predict_action_chunk -> _denormalize_action), so replacing the module attributes is enough."""
    originals = {name: getattr(ev, name) for name in
                 ("get_libero_env", "run_single_episode", "_predict_action_chunk", "_denormalize_action")}

    @functools.wraps(originals["get_libero_env"])
    def get_libero_env(*args, **kwargs):
        env, description = originals["get_libero_env"](*args, **kwargs)
        return TracingEnv(env, tracer), description

    @functools.wraps(originals["run_single_episode"])
    def run_single_episode(*args, **kwargs):
        tracer.safe("start_episode", tracer.start_episode, kwargs.get("episode_idx"), kwargs.get("initial_state"))
        started = time.perf_counter()
        result = originals["run_single_episode"](*args, **kwargs)
        tracer.safe("end_episode", tracer.end_episode, result, time.perf_counter() - started)
        return result

    @functools.wraps(originals["_predict_action_chunk"])
    def predict_action_chunk(*args, **kwargs):
        obs = kwargs["obs"] if "obs" in kwargs else args[0]
        tracer.safe("begin_call", tracer.begin_call, obs)  # before the evaluator reads (and may clip) obs
        result = originals["_predict_action_chunk"](*args, **kwargs)
        tracer.safe("end_call", tracer.end_call, result)
        return result

    @functools.wraps(originals["_denormalize_action"])
    def denormalize_action(*args, **kwargs):
        result = originals["_denormalize_action"](*args, **kwargs)
        tracer.safe("denormalize", tracer.on_denorm, result)
        return result

    ev.get_libero_env = get_libero_env
    ev.run_single_episode = run_single_episode
    ev._predict_action_chunk = predict_action_chunk
    ev._denormalize_action = denormalize_action
    return originals


def install_model_hooks(runtime, tracer: Tracer) -> None:
    model = runtime.model
    original_infer = model.infer_action  # bound method

    @functools.wraps(original_infer)
    def infer_action(*args, **kwargs):
        tracer.safe("infer_action.inputs", tracer.on_model_input, args, kwargs)
        out = original_infer(*args, **kwargs)
        tracer.safe("infer_action.output", tracer.on_model_output, out)
        return out

    model.infer_action = infer_action  # instance attribute; nn.Module.__setattr__ stores plain callables as-is

    cache = runtime.prompt_cache
    original_get = cache.get

    @functools.wraps(original_get)
    def get(prompt):
        context, mask = original_get(prompt)  # the model's own T5 runs exactly as in stock
        tracer.safe("prompt_cache.get", tracer.on_prompt, prompt, context, mask)
        if tracer.override is None:
            return context, mask
        return tracer.override_context(prompt, context, mask)

    cache.get = get


def load_context_file(path: Path) -> dict:
    if path.is_dir():
        path = path / "context.npz"
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in ("context", "context_mask", "dtype", "sha", "prompt") if k not in z.files]
        if missing:
            raise ValueError(f"{path} lacks {missing}")
        return {"path": str(path), "context": z["context"].astype(np.float32), "context_mask": z["context_mask"],
                "dtype": str(z["dtype"]), "sha": str(z["sha"]), "prompt": str(z["prompt"])}


# ----------------------------------------------------------------------------------------------
# `run`
# ----------------------------------------------------------------------------------------------


def torch_flags(torch) -> dict:
    out = {
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    try:  # torch >= 2.1; True by default, and only acts when deterministic algorithms are on
        import importlib

        deterministic = importlib.import_module("torch.utils.deterministic")
        out["fill_uninitialized_memory"] = bool(deterministic.fill_uninitialized_memory)
    except Exception as error:  # noqa: BLE001
        out["fill_uninitialized_memory"] = f"unavailable: {error!r}"
    try:
        out["sdpa_enabled"] = {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
            "cudnn": getattr(torch.backends.cuda, "cudnn_sdp_enabled", lambda: None)(),
        }
    except Exception as error:  # noqa: BLE001
        out["sdpa_enabled"] = repr(error)
    return out


def versions(torch) -> dict:
    import importlib.util
    from importlib.metadata import PackageNotFoundError, version

    out = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "flash_attn_importable": importlib.util.find_spec("flash_attn") is not None,
        "flash_attn_interface_importable": importlib.util.find_spec("flash_attn_interface") is not None,
    }
    for package in ("numpy", "hydra-core", "omegaconf", "transformers", "accelerate", "peft", "mujoco",
                    "robosuite", "libero", "flash-attn", "imageio"):
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            out[package] = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        out["gpu"] = {"name": props.name, "total_memory_gb": round(props.total_memory / 1024**3, 2),
                      "capability": f"{props.major}.{props.minor}", "count": torch.cuda.device_count()}
    return out


def repo_state() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", "--no-optional-locks", "-C", str(ROOT), *args], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception as error:  # noqa: BLE001
            return f"unavailable: {error!r}"

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_tracked": bool(status) and not status.startswith("unavailable"), "status": status[:2000]}


def model_info(model) -> dict:
    info = {
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "torch_dtype": str(getattr(model, "torch_dtype", None)),
        "device": str(getattr(model, "device", None)),
        "training": bool(model.training),
        "inference_cross_kv_reuse": getattr(model, "inference_cross_kv_reuse", None),
        "eval_supports_num_video_frames": getattr(model, "_eval_supports_num_video_frames", None),
        "vae_micro_batch_size": getattr(getattr(model, "vae", None), "micro_batch_size", None),
        "n_parameters": sum(int(p.numel()) for p in model.parameters()),
    }
    for name in ("video_expert", "action_expert"):
        module = getattr(model, name, None)
        info[f"{name}_attention_backend"] = getattr(module, "attention_backend", None)
    text_encoder = getattr(model, "text_encoder", None)
    if text_encoder is not None:
        first = next(iter(text_encoder.parameters()), None)
        info["text_encoder"] = {"class": type(text_encoder).__name__,
                                "dtype": str(first.dtype) if first is not None else None,
                                "device": str(first.device) if first is not None else None,
                                "n_parameters": sum(int(p.numel()) for p in text_encoder.parameters())}
    else:
        info["text_encoder"] = None
    return info


def eval_overrides(args, eval_out: Path) -> list:
    """The stage 2 command line (ai-lab slurm/openxam_libero_eval.sbatch), output_dir aside."""
    task = args.task or TASKS[args.model]
    return [
        f"task={task}",
        f"ckpt={args.ckpt}",
        f"seed={args.seed}",
        f"gpu_id={args.gpu_id}",
        f"EVALUATION.dataset_stats_path={args.dataset_stats}",
        f"EVALUATION.task_suite_name={args.suite_name}",
        f"EVALUATION.task_id={args.task_id}",
        f"EVALUATION.num_trials={args.trials}",
        "EVALUATION.video_mode=all",
        "EVALUATION.timing_enabled=true",
        f"EVALUATION.output_dir={eval_out}",
        *args.extra_override,
    ]


STOCK_MAIN_FILE = ROOT / "experiments" / "libero" / "eval_libero_single.py"
_SAVED_MAIN_FILE: list = []


def point_spawn_main_at_evaluator(index: dict) -> None:
    """Make `spawn` children re-run eval_libero_single.py as __mp_main__, as in a stock run.

    multiprocessing.spawn.get_preparation_data: if ``__main__.__spec__`` has a name, the child imports
    that module; otherwise it runs ``__main__.__file__`` (made absolute against ORIGINAL_DIR) with
    runpy.run_path(..., run_name="__mp_main__"). With ``python tests/gpu/trace_libero_eval.py`` the spec
    is None, so setting ``__file__`` is enough. Note ``__main__.__dict__`` is this script's globals, so
    ``__file__`` inside this script changes too until restore_main_file(); nothing in cmd_run reads
    it in between (``index["script"]`` is set before). What spawn will actually send is recorded by
    calling get_preparation_data itself (read-only)."""
    import multiprocessing.spawn as mp_spawn

    main = sys.modules["__main__"]
    record = {"original_file": getattr(main, "__file__", None),
              "spec_name": getattr(getattr(main, "__spec__", None), "name", None),
              "set_to": str(STOCK_MAIN_FILE)}
    _SAVED_MAIN_FILE.append(record["original_file"])
    main.__file__ = str(STOCK_MAIN_FILE)
    data = mp_spawn.get_preparation_data("trace-check")
    record["preparation_init_main_from_path"] = data.get("init_main_from_path")
    record["preparation_init_main_from_name"] = data.get("init_main_from_name")
    record["matches_stock"] = (record["preparation_init_main_from_name"] is None
                               and record["preparation_init_main_from_path"] == os.path.normpath(str(STOCK_MAIN_FILE)))
    index["spawn_main"] = record
    if not record["matches_stock"]:
        print(f"[trace] WARNING: spawn children will not run eval_libero_single.py as __mp_main__: {record}",
              file=sys.stderr, flush=True)


def restore_main_file(index: dict) -> None:
    if _SAVED_MAIN_FILE:
        original = _SAVED_MAIN_FILE.pop()
        if original is not None:
            sys.modules["__main__"].__file__ = original
        (index.get("spawn_main") or {})["restored"] = True


EXPECTED_MAIN_CALLS = ("set_global_seed(int(cfg.seed), get_worker_init_fn=False)", "build_eval_runtime(cfg)",
                       "evaluate_task_with_runtime(cfg, runtime)", "write_json_atomic(output_file, results)")


def cmd_run(args) -> int:
    out = Path(args.out).resolve()
    if (out / "index.json").exists() and not args.overwrite:
        print(f"{out}/index.json exists; pass --overwrite or use a new --out", file=sys.stderr)
        return 2
    override = None
    if args.context_from:
        try:
            override = load_context_file(Path(args.context_from).resolve())
        except Exception as error:  # noqa: BLE001
            print(f"--context-from {args.context_from}: {error!r}", file=sys.stderr)
            return 2
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("index.json", "trace.npz", "context.npz"):
        (out / stale).unlink(missing_ok=True)

    env_before = {k: os.environ.get(k) for k in ENV_KEYS}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    for key, value in (("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1"), ("MUJOCO_GL", "egl"), ("PYOPENGL_PLATFORM", "egl")):
        os.environ.setdefault(key, value)
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # before torch (and so cuBLAS) is loaded
    torch_was_imported = "torch" in sys.modules
    original_cwd = os.getcwd()
    os.chdir(ROOT)  # the stock command ran from the repo root; relative paths in the configs need it
    for path in (str(ROOT / "src"), str(ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import torch

    index: dict = {
        "format_version": FORMAT_VERSION,
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "argv": sys.argv,
        "original_cwd": original_cwd,
        "started_at": now(),
        "status": "running",
        "model": args.model,
        "task": args.task or TASKS[args.model],
        "deterministic": bool(args.deterministic),
        "context_override": None if override is None else {
            "source": override["path"], "prompt": override["prompt"], "saved_dtype": override["dtype"],
            "saved_sha": override["sha"], "shape": list(override["context"].shape)},
        "environment_before": env_before,
        "torch_imported_before_settings": torch_was_imported,
        "flags_before": torch_flags(torch),
        "warnings": [],
    }
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif env_before["CUBLAS_WORKSPACE_CONFIG"]:
        print("[trace] NOTE: CUBLAS_WORKSPACE_CONFIG is set in the environment, so this stock run is not "
              "exactly as users run it", file=sys.stderr)
    index["flags_after_settings"] = torch_flags(torch)
    index["environment"] = {k: os.environ.get(k) for k in ENV_KEYS}
    index["repo"] = repo_state()
    # versions() initialises CUDA (device properties), so it runs after the evaluation, not before.

    seen_warnings: set = set()
    original_showwarning = warnings.showwarning

    def showwarning(message, category, filename, lineno, file=None, line=None):
        text = f"{category.__name__}: {str(message)[:300]} ({Path(filename).name}:{lineno})"
        if text not in seen_warnings and len(seen_warnings) < 200:
            seen_warnings.add(text)
            index["warnings"].append(text)
        original_showwarning(message, category, filename, lineno, file, line)

    warnings.showwarning = showwarning  # display only; the filters are left alone

    tracer = Tracer(out, index, torch)
    tracer.override = override
    tracer.save()
    started = time.perf_counter()
    status = 1
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf

        import experiments.libero.eval_libero_single as ev

        point_spawn_main_at_evaluator(index)
        try:
            source = inspect.getsource(ev.eval_single_process)  # getsourcelines unwraps hydra's wrapper
        except Exception as error:  # noqa: BLE001
            source = f"unavailable: {error!r}"
        index["stock_main_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        missing = [call for call in EXPECTED_MAIN_CALLS if call not in source]
        index["stock_main_matches_replayed_calls"] = not missing
        if missing:
            print(f"[trace] WARNING: eval_single_process no longer contains {missing}; the replay below may "
                  "not match the stock evaluator", file=sys.stderr, flush=True)

        overrides = eval_overrides(args, out / "eval_output")
        index["hydra"] = {"config_dir": str(ROOT / "configs"), "config_name": "benchmark/sim_libero.yaml",
                          "overrides": overrides}
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
            cfg = compose(config_name="benchmark/sim_libero.yaml", overrides=overrides)
        (out / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
        evaluation = cfg.EVALUATION
        index["eval_settings"] = {key: (None if evaluation.get(key) is None else str(evaluation.get(key))) for key in
                                  ("task_suite_name", "task_id", "num_trials", "num_steps_wait", "replan_steps",
                                   "max_steps", "binarize_gripper", "use_action_ensembler", "num_inference_steps",
                                   "sigma_shift", "text_cfg_scale", "rand_device", "torch_compile", "device")}
        index["eval_settings"].update(seed=cfg.get("seed"), mixed_precision=str(cfg.get("mixed_precision")),
                                      inference_cross_kv_reuse=cfg.get("inference_cross_kv_reuse"),
                                      vae_micro_batch_size=cfg.get("vae_micro_batch_size"),
                                      attention_backend_config=str(cfg.model.backbone.get("attention_backend")),
                                      load_text_encoder=cfg.model.backbone.get("load_text_encoder"),
                                      model_target=str(cfg.model.get("_target_")))
        tracer.save()

        install_eval_hooks(ev, tracer)
        # eval_single_process body (eval_libero_single.py:907-915), with the module's own functions.
        if cfg.get("seed") is not None:
            ev.set_global_seed(int(cfg.seed), get_worker_init_fn=False)
        t0 = time.perf_counter()
        runtime = ev.build_eval_runtime(cfg)
        index["model_build_seconds"] = round(time.perf_counter() - t0, 1)
        try:
            index["model_info"] = model_info(runtime.model)
        except Exception as error:  # noqa: BLE001 - informational only
            index["model_info"] = {"error": repr(error)}
        install_model_hooks(runtime, tracer)
        tracer.save()

        t0 = time.perf_counter()
        results = ev.evaluate_task_with_runtime(cfg, runtime)
        index["eval_seconds"] = round(time.perf_counter() - t0, 1)
        output_file = (Path(cfg.EVALUATION.output_dir) / cfg.EVALUATION.task_suite_name
                       / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_results.json")
        ev.write_json_atomic(output_file, results)
        index["eval_results"] = json.loads(output_file.read_text())
        index["eval_results_file"] = str(output_file)
        try:
            from openxam.model.component import attention

            index["attention_backends_selected"] = sorted(list(pair) for pair in attention._LOGGED_SELECTIONS)
        except Exception as error:  # noqa: BLE001
            index["attention_backends_selected"] = repr(error)
        if torch.cuda.is_available():
            index["peak_gpu_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        index["status"] = "ok"
        status = 0
    except BaseException as error:  # noqa: BLE001 - recorded, then reported through the exit code
        index["status"] = "failed"
        index["error"] = repr(error)
        index["traceback"] = traceback.format_exc()
        print(f"!!! run failed: {error!r}", file=sys.stderr, flush=True)
    finally:
        restore_main_file(index)
        try:
            index["versions"] = versions(torch)
        except Exception as error:  # noqa: BLE001
            index["versions"] = {"error": repr(error)}
        index["finished_at"] = now()
        index["wall_seconds"] = round(time.perf_counter() - started, 1)
        index["trace_complete"] = (index["status"] == "ok" and not index["hook_errors"]
                                   and all(ep.get("finished") for ep in index["episodes"]))
        tracer.save()
        print(f"\nstatus={index['status']} trace_complete={index['trace_complete']} "
              f"hook_errors={len(index['hook_errors'])} wall={index['wall_seconds']} s")
        for ep in index["episodes"]:
            print(f"  episode {ep['episode']}: success={ep['success']} env_steps={ep['n_env_steps']} "
                  f"policy_calls={ep['n_policy_calls']}")
        print(f"trace: {out}")
    return status


# ----------------------------------------------------------------------------------------------
# `compare`
# ----------------------------------------------------------------------------------------------


class RunData:
    def __init__(self, name: str, index: dict, arrays: dict, context: dict):
        self.name = name
        self.index = index
        self.arrays = arrays
        self.context = context

    @classmethod
    def load(cls, path: Path) -> "RunData":
        index = json.loads((path / "index.json").read_text())
        arrays, context = {}, {}
        if (path / "trace.npz").exists():
            with np.load(path / "trace.npz", allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
        if (path / "context.npz").exists():
            with np.load(path / "context.npz", allow_pickle=False) as z:
                context = {k: z[k] for k in z.files}
        return cls(str(path), index, arrays, context)

    def reliable(self) -> tuple[bool, list]:
        problems = []
        if self.index.get("status") != "ok":
            problems.append(f"status={self.index.get('status')}")
        if self.index.get("hook_errors"):
            problems.append(f"{len(self.index['hook_errors'])} hook errors")
        if not all(ep.get("finished") for ep in self.index.get("episodes", [])):
            problems.append("an episode did not finish")
        if self.index.get("trace_complete") is False:
            problems.append("trace_complete is False")
        if self.index.get("stock_main_matches_replayed_calls") is False:
            problems.append("eval_single_process no longer matches the replayed calls")
        if (self.index.get("spawn_main") or {}).get("matches_stock") is False:
            problems.append("the env child did not run eval_libero_single.py as __mp_main__")
        override = self.index.get("context_override")
        if override is not None and override.get("exact") is not True:
            problems.append("context override not applied exactly" if "exact" in override
                            else "context override never applied")
        return not problems, problems

    def call_array(self, episode: int, call: dict, name: str):
        """float64 view of a stored per-call array (exact bits decoded via the recorded dtype)."""
        arr = self.arrays.get(f"ep{episode}_c{call['call']}_{name}")
        if arr is None:
            return None
        label = call_field(call, ARRAY_LABEL_PATH.get(name, ())) if name in ARRAY_LABEL_PATH else arr.dtype.name
        return exact_to_float64(arr, label or arr.dtype.name)


ARRAY_LABEL_PATH = {
    "input_image": ("inputs", "input_image", "dtype"),
    "proprio": ("inputs", "proprio", "dtype"),
    "raw_action": ("outputs", "raw_action", "dtype"),
    "denorm_action": ("outputs", "denorm_action", "dtype"),
    "env_action_chunk": ("outputs", "env_action_chunk", "dtype"),
}

# Checked in this order at each policy call; the first mismatch names the divergence.
CALL_CHECKS = (
    ("raw_input", (("raw", "agentview_sha"), ("raw", "wrist_sha"), ("raw", "state_sha"))),
    ("processed_input", (("inputs", "input_image", "sha"), ("inputs", "proprio", "sha"))),
    ("infer_args", (("inputs", "scalars_sha"),)),
    ("model_output", (("outputs", "raw_action", "sha"),)),
    ("denormalized_action", (("outputs", "denorm_action", "sha"),)),
    ("env_action_chunk", (("outputs", "env_action_chunk", "sha"),)),
)
CONTEXT_FIELDS = (("inputs", "context", "sha"), ("inputs", "context_mask", "sha"))
RAW_INPUT_FIELDS = CALL_CHECKS[0][1]
PROCESSED_INPUT_FIELDS = CALL_CHECKS[1][1]
OUTPUT_FIELDS = CALL_CHECKS[3][1]


def call_field(call: dict, path: tuple):
    value = call
    for key in path:
        if not isinstance(value, dict) or value.get(key) is None:
            return None
        value = value[key]
    return value


def fields_differ(ca: dict, cb: dict, fields) -> tuple[bool, str | None, bool]:
    """(differs, which field, data missing)."""
    for path in fields:
        a, b = call_field(ca, path), call_field(cb, path)
        if a is None or b is None:
            return True, ".".join(path), True
        if a != b:
            return True, ".".join(path), False
    return False, None, False


def bytes_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()


def first_divergence(ep_a: dict, ep_b: dict, act_a: np.ndarray, act_b: np.ndarray) -> dict | None:
    """Walk both episodes in time order: observation k, then the policy call made on it (if any),
    then the action executed at k. The text context is not a stop condition (see context_differs)."""
    sa, sb = ep_a["steps"], ep_b["steps"]
    n_a, n_b = len(sa["obs_sha"]) - 1, len(sb["obs_sha"]) - 1  # env steps recorded
    calls_a = {c["env_step"]: c for c in ep_a["calls"]}
    calls_b = {c["env_step"]: c for c in ep_b["calls"]}
    if n_a < 0 or n_b < 0:
        return {"kind": "missing_trace_data", "env_step": 0, "field": "steps (no observation recorded)"}
    if ep_a.get("initial_state_sha") != ep_b.get("initial_state_sha"):
        return {"kind": "initial_state", "env_step": 0}
    for k in range(0, max(n_a, n_b) + 1):
        if k > n_a or k > n_b:
            return {"kind": "length", "env_step": k}
        for key in ("obs_sha", "done", "reward"):
            if sa[key][k] != sb[key][k]:
                parts = [p for p in ("agentview_sha", "wrist_sha", "state_sha") if sa[p][k] != sb[p][k]]
                if key == "obs_sha" and not parts:
                    parts = ["other observation keys only (object/robot state; cameras and eef state equal)"]
                return {"kind": "observation", "env_step": k, "field": key, "obs_parts_differing": parts}
        ca, cb = calls_a.get(k), calls_b.get(k)
        if (ca is None) != (cb is None):
            return {"kind": "call_schedule", "env_step": k}
        if ca is not None:
            for kind, fields in CALL_CHECKS:
                differs, field, missing = fields_differ(ca, cb, fields)
                if differs:
                    return {"kind": "missing_trace_data" if missing else kind, "env_step": k,
                            "call": ca["call"], "call_b": cb["call"], "field": field}
        if k < n_a and k < n_b:
            if k >= len(act_a) or k >= len(act_b):
                return {"kind": "missing_trace_data", "env_step": k, "field": "executed_actions"}
            if not bytes_equal(act_a[k], act_b[k]):
                return {"kind": "executed_action", "env_step": k}
    return None


def first_call_where(ep_a: dict, ep_b: dict, fields, *, missing_counts: bool = True) -> int | None:
    """First aligned call where ``fields`` differ. With missing_counts=False a call whose data is
    missing on either side is skipped (see calls_missing)."""
    for ca, cb in zip(ep_a["calls"], ep_b["calls"]):
        differs, _, missing = fields_differ(ca, cb, fields)
        if differs and (missing_counts or not missing):
            return ca["call"]
    return None


def calls_missing(ep_a: dict, ep_b: dict, fields) -> list:
    return [ca["call"] for ca, cb in zip(ep_a["calls"], ep_b["calls"]) if fields_differ(ca, cb, fields)[2]]


def metrics_at_call(a: RunData, b: RunData, episode: int, c: int | None) -> dict | None:
    """Metrics of b (test) against a (reference) at policy call c (calls aligned by index)."""
    if c is None:
        return None
    ep_a = next(ep for ep in a.index["episodes"] if ep["episode"] == episode)
    ep_b = next(ep for ep in b.index["episodes"] if ep["episode"] == episode)
    if c >= len(ep_a["calls"]) or c >= len(ep_b["calls"]):
        return {"call": c, "available": False}
    ca, cb = ep_a["calls"][c], ep_b["calls"][c]
    out = {"call": c, "env_step": [ca["env_step"], cb["env_step"]]}
    for name in ("raw_action", "env_action_chunk", "input_image", "proprio", "agentview", "wrist", "raw_state"):
        out[name] = diff_metrics(b.call_array(episode, cb, name), a.call_array(episode, ca, name))
        if name in ARRAY_LABEL_PATH:
            sha_path = ARRAY_LABEL_PATH[name][:-1] + ("sha",)
            out[name]["bitwise_equal"] = call_field(ca, sha_path) is not None and call_field(ca, sha_path) == call_field(cb, sha_path)
    for name in ("raw_action", "env_action_chunk"):
        ref, test = a.call_array(episode, ca, name), b.call_array(episode, cb, name)
        if ref is not None and test is not None and ref.shape == test.shape and ref.ndim >= 1:
            out[f"{name}_first_row"] = diff_metrics(test.reshape(-1, ref.shape[-1])[0], ref.reshape(-1, ref.shape[-1])[0])
    return out


def compare_context(a: RunData, b: RunData, prefix_a: str = "", prefix_b: str = "") -> dict:
    """Context ``prefix_b`` of b (test) against context ``prefix_a`` of a (reference). Prefix "" is the
    context the model received; "own_" is the model's own T5 output in a run with --context-from."""
    ca, cb = a.context, b.context
    key_a, key_b = f"{prefix_a}context", f"{prefix_b}context"
    if key_a not in ca or key_b not in cb:
        return {"available": False, "in_reference": key_a in ca, "in_test": key_b in cb}
    x, y = ca[key_a].astype(np.float64), cb[key_b].astype(np.float64)
    out = {
        "available": True,
        "which": [key_a, key_b],
        "dtype": [str(ca.get(f"{prefix_a}dtype")), str(cb.get(f"{prefix_b}dtype"))],
        "shape": [list(x.shape), list(y.shape)],
        "sha_equal": str(ca.get(f"{prefix_a}sha")) == str(cb.get(f"{prefix_b}sha")),
        "prompt_equal": str(ca.get("prompt")) == str(cb.get("prompt")),
    }
    ma, mb = ca.get(f"{prefix_a}context_mask"), cb.get(f"{prefix_b}context_mask")
    out["mask_equal"] = ma is not None and mb is not None and bytes_equal(ma.astype(bool), mb.astype(bool))
    if x.shape != y.shape:
        out["bitwise_equal"] = False
        return out
    out["bitwise_equal"] = out["sha_equal"] and bytes_equal(ca[key_a], cb[key_b])
    out["all_positions"] = diff_metrics(y, x)
    if out["mask_equal"]:
        valid = ma.astype(bool)
        out["valid_tokens"] = valid.sum(axis=-1).tolist()
        out["valid_positions"] = diff_metrics(y[valid], x[valid])
        out["padded_positions"] = diff_metrics(y[~valid], x[~valid])
    return out


def episode_verdict(div: dict | None, ctx_equal: bool | None, info: dict, wait_steps: int | None,
                    *, same_model: bool = False, valid_ctx_diff: float | None = None) -> tuple[str, str]:
    """(code, one-line text). ``same_model``: both runs used the same task and no context override.
    ``valid_ctx_diff``: max |diff| of the context at valid (unpadded) token positions, if known."""
    c_raw = info["first_call_raw_input_differs"]
    same_counts = info["outcome_equal"] and info["n_env_steps"][0] == info["n_env_steps"][1] and \
        info["n_policy_calls"][0] == info["n_policy_calls"][1]
    if div is None:
        if not same_counts:
            return "COUNTS_DIFFER", "every recorded step and call matches, yet outcomes or counts differ (inspect)"
        if ctx_equal is False:
            if valid_ctx_diff == 0:
                return ("IDENTICAL_EXCEPT_CONTEXT",
                        "identical trajectory and actions; the text context differs only at padded positions "
                        "(valid-position max|diff| 0)")
            return ("CONTEXT_DIFFERS_NO_EFFECT",
                    "identical trajectory and actions although the text context differs at valid positions "
                    f"(max|diff| {valid_ctx_diff}; None = unknown); the difference changed no output bit")
        return "IDENTICAL", "identical"
    k, c = div.get("env_step"), div.get("call")
    kind = div["kind"]
    if kind == "model_output":
        if ctx_equal is False:
            return ("CONTEXT_AND_MODEL_OUTPUT_DIFFER",
                    f"image/proprio inputs identical but the text context differs (valid-position max|diff| "
                    f"{valid_ctx_diff}), and outputs differ at call {c} (env step {k}): T5 context and/or model "
                    "difference; a context-swap run separates them")
        cause = "nondeterministic model output (same model)" if same_model else "model difference"
        return ("MODEL_OUTPUT_DIFFERS",
                f"inputs identical but outputs differ at call {c} (env step {k}) ({cause})")
    if kind in ("observation", "initial_state"):
        wait = ""
        if wait_steps is not None and k is not None and k <= wait_steps:
            wait = f", during the {wait_steps}-step no-op wait phase"
        later = f"inputs differ from call {c_raw}" if c_raw is not None else "no later policy call sees it"
        parts = f" [{', '.join(div['obs_parts_differing'])}]" if div.get("obs_parts_differing") else ""
        what = "initial state differs" if kind == "initial_state" else \
            f"{div['field']}{parts} differs at env step {k} after an identical action history{wait}"
        return "ENV_DIVERGENCE", f"{later} (simulator/env difference: {what})"
    texts = {
        "processed_input": ("PREPROCESSING_DIFFERS",
                            f"raw observation identical but the processed model input ({div.get('field')}) differs "
                            f"at call {c} (env step {k})"),
        "raw_input": ("HARNESS_INCONSISTENCY",
                      f"full observation hash equal but a raw-input hash differs at call {c}; trace bug?"),
        "infer_args": ("INFER_ARGS_DIFFER", f"inference arguments differ at call {c} (config difference)"),
        "denormalized_action": ("POSTPROCESS_DIFFERS",
                                f"model output identical but the denormalized action differs at call {c}"),
        "env_action_chunk": ("POSTPROCESS_DIFFERS",
                             f"model output identical but the env action chunk differs at call {c}"),
        "executed_action": ("EXECUTION_DIFFERS",
                            f"policy outputs identical but the executed action differs at env step {k}"),
        "call_schedule": ("CALL_SCHEDULE_DIFFERS", f"one run calls the policy at env step {k}, the other not"),
        "length": ("LENGTH_ONLY",
                   f"everything recorded is identical up to env step {k - 1}, but one episode is longer"),
        "missing_trace_data": ("MISSING_TRACE_DATA",
                               f"{div.get('field')} missing at env step {k}; cannot compare further"),
    }
    return texts.get(kind, (kind.upper(), json.dumps(div)))


def compare_runs(a: RunData, b: RunData) -> dict:
    ok_a, problems_a = a.reliable()
    ok_b, problems_b = b.reliable()
    ctx = compare_context(a, b)
    ctx_equal = ctx.get("bitwise_equal") if ctx.get("available") else None
    result: dict = {
        "reference": a.name,
        "test": b.name,
        "compared_at": now(),
        "reliable": ok_a and ok_b,
        "problems": {"reference": problems_a, "test": problems_b},
        "runs": {
            name: {k: run.index.get(k) for k in ("model", "task", "deterministic", "status", "trace_complete")}
            | {"context_override": (run.index.get("context_override") or {}).get("source"),
               "successes": (run.index.get("eval_results") or {}).get("successes")}
            for name, run in (("reference", a), ("test", b))
        },
        "settings_differences": settings_differences(a.index, b.index),
        "context": ctx,
        "context_used_vs_own": {
            "reference_override": a.index.get("context_override") is not None,
            "test_override": b.index.get("context_override") is not None,
            "own_context_shas": [a.index.get("own_context_shas"), b.index.get("own_context_shas")],
        },
        "episodes": [],
    }
    # With --context-from, the model's own T5 output is kept as own_context: compare it too.
    if "own_context" in b.context:
        result["context_test_own_vs_reference_used"] = compare_context(a, b, "", "own_")
    if "own_context" in a.context:
        result["context_reference_own_vs_test_used"] = compare_context(a, b, "own_", "")
    if "own_context" in a.context and "own_context" in b.context:
        result["context_own_vs_own"] = compare_context(a, b, "own_", "own_")
    wait = None
    try:
        wait = int((a.index.get("eval_settings") or {}).get("num_steps_wait"))
    except (TypeError, ValueError):
        pass
    valid_ctx_diff = (ctx.get("valid_positions") or {}).get("max_abs_diff") if ctx.get("available") else None
    same_model = (a.index.get("task") is not None and a.index.get("task") == b.index.get("task")
                  and a.index.get("context_override") is None and b.index.get("context_override") is None)
    result["same_model_no_override"] = same_model
    pair_problems = []
    eps_b = {ep["episode"]: ep for ep in b.index.get("episodes", [])}
    for ep_a in a.index.get("episodes", []):
        e = ep_a["episode"]
        ep_b = eps_b.get(e)
        if ep_b is None:
            result["episodes"].append({"episode": e, "verdict_code": "MISSING_EPISODE",
                                       "verdict": "episode missing in the test run"})
            continue
        act_a = a.arrays.get(f"ep{e}_executed_actions", np.zeros((0, 7)))
        act_b = b.arrays.get(f"ep{e}_executed_actions", np.zeros((0, 7)))
        div = first_divergence(ep_a, ep_b, act_a, act_b)
        # The saved context is the first call's; the per-call hashes cover the rest of the episode.
        # Missing context data is "unknown" (None), never "differs", and makes the pair unreliable.
        ctx_calls = first_call_where(ep_a, ep_b, CONTEXT_FIELDS, missing_counts=False)
        ctx_missing = calls_missing(ep_a, ep_b, CONTEXT_FIELDS)
        if ctx_calls is not None:
            ep_ctx_equal = False
        elif ctx_missing:
            ep_ctx_equal = None
            pair_problems.append(f"ep{e}: context data missing at calls {ctx_missing[:10]}")
        elif ctx_equal is None:
            ep_ctx_equal = True if ep_a["calls"] and ep_b["calls"] else None
        else:
            ep_ctx_equal = ctx_equal
        info = {
            "episode": e,
            "success": [ep_a.get("success"), ep_b.get("success")],
            "outcome_equal": ep_a.get("success") == ep_b.get("success"),
            "n_env_steps": [ep_a.get("n_env_steps"), ep_b.get("n_env_steps")],
            "n_policy_calls": [ep_a.get("n_policy_calls"), ep_b.get("n_policy_calls")],
            "first_divergence": div,
            "first_call_raw_input_differs": first_call_where(ep_a, ep_b, RAW_INPUT_FIELDS),
            "first_call_processed_input_differs": first_call_where(ep_a, ep_b, PROCESSED_INPUT_FIELDS),
            "first_call_output_differs": first_call_where(ep_a, ep_b, OUTPUT_FIELDS),
            "first_call_context_differs": ctx_calls,
            "calls_with_context_missing": ctx_missing,
            "context_bitwise_equal": ep_ctx_equal,
        }
        info["at_first_output_difference"] = metrics_at_call(a, b, e, info["first_call_output_differs"])
        info["at_first_raw_input_difference"] = metrics_at_call(a, b, e, info["first_call_raw_input_differs"])
        if div is not None and div.get("call") is not None:
            info["at_first_divergence"] = metrics_at_call(a, b, e, div["call"])
        info["verdict_code"], info["verdict"] = episode_verdict(
            div, ep_ctx_equal, info, wait, same_model=same_model, valid_ctx_diff=valid_ctx_diff)
        result["episodes"].append(info)
    result["problems"]["pair"] = pair_problems
    result["reliable"] = result["reliable"] and not pair_problems
    if not result["reliable"]:
        for info in result["episodes"]:
            info["verdict_code"] = f"UNRELIABLE_{info['verdict_code']}"
            info["verdict"] = f"UNRELIABLE (see problems): {info['verdict']}"
    codes = [ep["verdict_code"] for ep in result["episodes"]]
    if codes and all(code == "IDENTICAL" for code in codes):
        result["verdict"] = "identical"
    else:
        result["verdict"] = "; ".join(f"ep{ep['episode']}: {ep['verdict']}" for ep in result["episodes"]) or \
            "no episodes to compare"
    result["verdict_codes"] = codes
    return result


def settings_differences(a: dict, b: dict) -> dict:
    """Settings that should be equal for a fair pair (the model/task and override are expected to differ)."""
    def flat(index):
        out = {"deterministic": index.get("deterministic")}
        for key, value in (index.get("flags_at_first_policy_call") or index.get("flags_after_settings") or {}).items():
            out[f"flags.{key}"] = value
        for key, value in (index.get("eval_settings") or {}).items():
            out[f"eval.{key}"] = value
        hydra = [o for o in (index.get("hydra") or {}).get("overrides", [])
                 if not o.startswith(("task=", "EVALUATION.output_dir="))]
        out["hydra_overrides_except_task_and_output"] = hydra
        versions_ = index.get("versions") or {}
        for key in ("torch", "torch_cuda", "cudnn", "gpu", "flash_attn_importable"):
            out[f"versions.{key}"] = versions_.get(key)
        out["repo.commit"] = (index.get("repo") or {}).get("commit")
        out["attention_backends_selected"] = index.get("attention_backends_selected")
        info = index.get("model_info") or {}
        for key in ("torch_dtype", "inference_cross_kv_reuse", "eval_supports_num_video_frames",
                    "vae_micro_batch_size", "video_expert_attention_backend", "action_expert_attention_backend"):
            out[f"model.{key}"] = info.get(key)
        out["model.text_encoder_dtype"] = (info.get("text_encoder") or {}).get("dtype")
        return out

    fa, fb = flat(a), flat(b)
    return {key: [fa.get(key), fb.get(key)] for key in sorted(set(fa) | set(fb)) if fa.get(key) != fb.get(key)}


def summary_lines(result: dict) -> list:
    lines = [f"{result['reference']}  vs  {result['test']}", f"  verdict: {result['verdict']}"]
    if not result["reliable"]:
        lines.append(f"  UNRELIABLE: {result['problems']}")
    ctx = result["context"]
    if ctx.get("available"):
        extra = ""
        if not ctx.get("bitwise_equal") and "valid_positions" in ctx:
            extra = (f" (max|diff| valid {ctx['valid_positions'].get('max_abs_diff')}, padded "
                     f"{ctx['padded_positions'].get('max_abs_diff')})")
        lines.append(f"  context bitwise equal: {ctx.get('bitwise_equal')}{extra}")
    for key in ("context_test_own_vs_reference_used", "context_reference_own_vs_test_used", "context_own_vs_own"):
        if result.get(key, {}).get("available"):
            other = result[key]
            lines.append(f"  {key}: bitwise equal {other.get('bitwise_equal')}, max|diff| valid positions "
                         f"{(other.get('valid_positions') or {}).get('max_abs_diff')}")
    for ep in result["episodes"]:
        if "success" not in ep:
            lines.append(f"  ep{ep['episode']}: {ep['verdict']}")
            continue
        lines.append(f"  ep{ep['episode']}: success {ep['success']} steps {ep['n_env_steps']} calls "
                     f"{ep['n_policy_calls']} | first raw-input diff call {ep['first_call_raw_input_differs']}, "
                     f"first output diff call {ep['first_call_output_differs']}")
        m = ep.get("at_first_output_difference")
        if m and m.get("raw_action", {}).get("available"):
            ra, im = m["raw_action"], m["input_image"]
            lines.append(f"    at call {m['call']}: raw action max|diff| {ra.get('max_abs_diff')} rel_l2 "
                         f"{ra.get('rel_l2')}; input image bitwise {im.get('bitwise_equal')} max|diff| "
                         f"{im.get('max_abs_diff')}")
    if result["settings_differences"]:
        lines.append(f"  settings that differ: {sorted(result['settings_differences'])}")
    return lines


def cmd_compare(args) -> int:
    a, b = Path(args.run_a), Path(args.run_b)
    for path in (a, b):
        if not (path / "index.json").exists():
            print(f"{path}/index.json not found", file=sys.stderr)
            return 2
    result = compare_runs(RunData.load(a), RunData.load(b))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    for line in summary_lines(result):
        print(line)
    print(f"written: {out}")
    return 0


# ----------------------------------------------------------------------------------------------
# `suite`
# ----------------------------------------------------------------------------------------------


def common_run_args(args) -> list:
    out = ["--ckpt", args.ckpt, "--dataset-stats", args.dataset_stats, "--suite-name", args.suite_name,
           "--task-id", str(args.task_id), "--trials", str(args.trials), "--seed", str(args.seed),
           "--gpu-id", str(args.gpu_id)]
    for override in args.extra_override:
        out += ["--extra-override", override]
    return out


# Stage 2 (job 8658726, stock evaluator, video_mode=all): rollout-video frames per episode were old 73/75
# and new 72/75. One frame per executed step after the 30 no-op wait steps, so env steps = frames + 30.
STAGE2_ENV_STEPS = {"old": [103, 105], "new": [102, 105]}


def run_summary(run_dir: Path, model: str, args) -> dict:
    """Per-episode counts from a run's index.json, and how they compare with stage 2 (only meaningful
    for the stage 2 setup: libero_spatial task 0, 2 trials, seed 42, the default checkpoint)."""
    path = run_dir / "index.json"
    if not path.exists():
        return {"episodes": None}
    try:
        index = json.loads(path.read_text())
    except Exception as error:  # noqa: BLE001 - e.g. a run killed while writing
        return {"episodes": None, "index_error": repr(error)}
    episodes = [{"episode": ep.get("episode"), "success": ep.get("success"), "n_env_steps": ep.get("n_env_steps"),
                 "n_calls": ep.get("n_policy_calls"), "finished": ep.get("finished")}
                for ep in index.get("episodes", [])]
    out = {"episodes": episodes, "run_status": index.get("status"), "trace_complete": index.get("trace_complete")}
    stage2_setup = (args.suite_name == "libero_spatial" and args.task_id == 0 and args.trials == 2 and args.seed == 42
                    and args.ckpt == DEFAULT_CKPT and not args.extra_override)
    if stage2_setup:
        steps = [ep["n_env_steps"] for ep in episodes if ep["finished"]]
        out["stage2_env_steps"] = STAGE2_ENV_STEPS[model]
        out["matches_stage2_env_steps"] = steps == STAGE2_ENV_STEPS[model]
        out["stage2_note"] = "stage 2 was one stock run per model; a mismatch alone says nothing if stock is nondeterministic"
    return out


def cmd_suite(args) -> int:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = [step[0] for step in SUITE_STEPS]
    wanted = args.steps or names
    unknown = [s for s in wanted if s not in names]
    if unknown:
        print(f"unknown steps {unknown}; known: {names}", file=sys.stderr)
        return 2
    started = time.perf_counter()
    report: dict = {"started_at": now(), "argv": sys.argv, "steps": {}, "comparisons": {}}
    print("=== selftest of the tracer and compare (numpy only)", flush=True)
    try:
        report["selftest_passed"] = cmd_selftest(None) == 0
    except Exception as error:  # noqa: BLE001
        report["selftest_passed"] = False
        report["selftest_error"] = repr(error)
    if not report["selftest_passed"]:
        # The selftest drives the same Tracer the runs use, so a failure may mean the traces themselves
        # are wrong, not only the comparisons: no GPU time is spent until it passes.
        report["finished_at"] = now()
        report["status"] = "stopped: selftest failed; no run was started"
        write_json(out / "suite.json", report)
        print("!!! selftest FAILED: no run started (exit 3)", flush=True)
        return 3
    write_json(out / "suite.json", report)
    for name, model, deterministic, context_from in SUITE_STEPS:
        if name not in wanted:
            continue
        entry: dict = {"model": model, "deterministic": deterministic, "context_from": context_from}
        report["steps"][name] = entry
        elapsed = time.perf_counter() - started
        if args.stop_after_seconds is not None and elapsed > args.stop_after_seconds:
            entry["status"] = f"skipped: {elapsed:.0f} s elapsed > --stop-after-seconds {args.stop_after_seconds}"
            write_json(out / "suite.json", report)
            continue
        command = [sys.executable, str(Path(__file__).resolve()), "run", "--model", model, "--out",
                   str(out / name), *common_run_args(args)]
        if args.overwrite:
            command.append("--overwrite")
        if deterministic:
            command.append("--deterministic")
        if context_from:
            source = out / context_from / "context.npz"
            if (report["steps"].get(context_from) or {}).get("status") != "ok" or not source.exists():
                entry["status"] = f"skipped: {context_from} did not run successfully in this suite"
                write_json(out / "suite.json", report)
                continue
            command += ["--context-from", str(source)]
        entry["command"] = command
        print(f"\n=== [{now()}] {name}: {' '.join(command[1:])}", flush=True)
        t0 = time.perf_counter()
        (out / name).mkdir(parents=True, exist_ok=True)
        code, timed_out = None, False
        with open(out / name / "run.log", "w") as log:
            # Own process group, so a timeout also stops the run's spawned LIBERO env child.
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT),
                                    start_new_session=True)
            try:
                code = proc.wait(timeout=args.run_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                for sig, grace in ((signal.SIGTERM, 30), (signal.SIGKILL, 30)):
                    try:
                        os.killpg(proc.pid, sig)
                    except ProcessLookupError:
                        break
                    try:
                        proc.wait(timeout=grace)
                        break
                    except subprocess.TimeoutExpired:
                        continue
        entry["exit_code"] = code
        entry["seconds"] = round(time.perf_counter() - t0, 1)
        entry["status"] = "timeout" if timed_out else ("ok" if code == 0 else "failed")
        entry.update(run_summary(out / name, model, args))
        print(f"=== {name}: {entry['status']} (exit {code}) in {entry['seconds']} s; episodes "
              f"{entry.get('episodes')}", flush=True)
        if code != 0:
            tail = (out / name / "run.log").read_text(errors="replace").splitlines()[-25:]
            print("\n".join(tail), flush=True)
        write_json(out / "suite.json", report)

    print("\n==== comparisons ====")
    # Only runs made by this invocation (a leftover directory from an earlier suite is never compared
    # here; use `compare` by hand for that). Exit code 2 means the run refused to start.
    ran = {n for n, e in report["steps"].items() if e.get("exit_code") not in (None, 2)}
    for ref, test, question in SUITE_COMPARISONS:
        key = f"{ref}__vs__{test}"
        if not ({ref, test} <= ran and (out / ref / "index.json").exists() and (out / test / "index.json").exists()):
            report["comparisons"][key] = {"question": question, "status": "skipped: a run is missing in this suite"}
            continue
        try:
            result = compare_runs(RunData.load(out / ref), RunData.load(out / test))
            write_json(out / f"compare_{key}.json", result)
            report["comparisons"][key] = {"question": question, "verdict": result["verdict"],
                                          "verdict_codes": result["verdict_codes"], "reliable": result["reliable"],
                                          "file": f"compare_{key}.json"}
            print(f"\n[{question}]")
            for line in summary_lines(result):
                print(line)
        except Exception as error:  # noqa: BLE001
            report["comparisons"][key] = {"question": question, "status": f"compare failed: {error!r}",
                                          "traceback": traceback.format_exc()}
            print(f"compare {key} failed: {error!r}", file=sys.stderr)
    report["finished_at"] = now()
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    write_json(out / "suite.json", report)
    print(f"\nsuite: {out / 'suite.json'}")
    failed = [n for n, e in report["steps"].items() if e.get("status") == "failed"]
    return 1 if failed else 0


# ----------------------------------------------------------------------------------------------
# `selftest`: a fake evaluator loop through the real Tracer, then compare (numpy only)
# ----------------------------------------------------------------------------------------------


def _digest(*parts) -> bytes:
    h = hashlib.sha256()
    for part in parts:
        h.update(part if isinstance(part, bytes) else str(part).encode())
    return h.digest()


def _rng(state: bytes, tag: str) -> np.random.Generator:
    return np.random.default_rng(int.from_bytes(_digest(state, tag)[:8], "little"))


class _FakeEnv:
    def __init__(self, jitter_step=None, done_at=40):
        self.state = _digest("init")
        self.k = 0
        self.jitter_step = jitter_step
        self.done_at = done_at

    def obs(self):
        def vec(tag, n):
            return _rng(self.state, tag).standard_normal(n)
        return {AGENTVIEW_KEY: _rng(self.state, "a").integers(0, 256, (8, 8, 3), dtype=np.uint8),
                WRIST_KEY: _rng(self.state, "w").integers(0, 256, (8, 8, 3), dtype=np.uint8),
                "robot0_eef_pos": vec("p", 3), "robot0_eef_quat": vec("q", 4),
                "robot0_gripper_qpos": vec("g", 2), "object_state": vec("o", 5)}

    def reset(self):
        return self.obs()

    def set_init_state(self, _state):
        return self.obs()

    def step(self, action):
        self.k += 1
        self.state = _digest(self.state, np.asarray(action, dtype=np.float64).tobytes())
        if self.jitter_step == self.k:
            self.state = _digest(self.state, "jitter")
        return self.obs(), 0.0, self.k >= self.done_at, {}


def _fake_run(*, perturb_call=None, jitter_step=None, ctx_shift=0.0, ctx_pad_shift=0.0, done_at=40,
              hook_error=False, episodes=2, task="fake_task", model_reads_context=True,
              drop_context_call=None) -> RunData:
    """A run through Tracer with the evaluator's control flow (eval_libero_single.py:450-606), with
    wait=5, replan=10, horizon 12, 7-dim actions; the fake model ignores padded context positions
    (and all of the context with model_reads_context=False). drop_context_call: the tracer is not
    shown the context at that call (missing data; the model still reads it)."""
    index = {"status": "ok", "eval_settings": {"num_steps_wait": "5"}, "model": "fake", "task": task}
    tracer = Tracer(None, index)
    context = _rng(b"ctx", "c").standard_normal((1, 4, 6)).astype(np.float32)
    mask = np.array([[True, True, True, False]])
    context[0, :3] += np.float32(ctx_shift)
    context[0, 3] += np.float32(ctx_pad_shift)
    tracer.on_prompt("fake prompt", context, mask)
    wait, replan, max_steps = 5, 10, 60
    for e in range(episodes):
        env = TracingEnv(_FakeEnv(jitter_step=jitter_step, done_at=done_at), tracer)
        tracer.start_episode(e, np.arange(4.0) + e)
        env.reset()
        obs = env.set_init_state(None)
        t, pending, done = 0, [], False
        while t < max_steps + wait:
            if t < wait:
                obs, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                t += 1
                continue
            if not pending:
                tracer.safe("begin_call", tracer.begin_call, obs)
                image = np.concatenate([obs[AGENTVIEW_KEY], obs[WRIST_KEY]], axis=1).astype(np.float32)
                image = (image * np.float32(2 / 255) - 1).transpose(2, 0, 1)[None]
                proprio = np.concatenate([obs[k] for k in STATE_KEYS])[None, :8].astype(np.float32)
                kwargs = {"prompt": None, "input_image": image, "proprio": proprio, "context": context,
                          "context_mask": mask, "seed": 42, "num_inference_steps": 10}
                if drop_context_call is not None and len(tracer.ep["calls"]) - 1 == drop_context_call:
                    kwargs = {**kwargs, "context": None, "context_mask": None}
                tracer.safe("infer_action.inputs", tracer.on_model_input, (), kwargs)
                key = _digest(image.tobytes(), proprio.tobytes(),
                              context[mask].tobytes() if model_reads_context else b"")
                raw = _rng(key, "model").standard_normal((12, 7)).astype(np.float32)
                if perturb_call is not None and len(tracer.ep["calls"]) - 1 == perturb_call:
                    raw[0, 0] += np.float32(1e-3)
                tracer.safe("infer_action.output", tracer.on_model_output, {"action": raw})
                denorm = raw[None] * np.float32(2)
                tracer.safe("denormalize", tracer.on_denorm, denorm)
                chunk = denorm[0].copy()
                chunk[:, -1] = np.sign(chunk[:, -1])
                tracer.safe("end_call", tracer.end_call, (chunk, None, None))
                pending = chunk[:replan].tolist()
            obs, _, done, _ = env.step(pending.pop(0))
            if done:
                break
            t += 1
        tracer.end_episode((done, [], [], None), 0.0)
    if hook_error:
        index["hook_errors"].append({"where": "selftest", "error": "injected"})
    tracer._flush_actions()
    return RunData("fake", index, tracer.arrays, tracer.context_file)


def cmd_selftest(_args) -> int:
    base = _fake_run()

    def ep0(result):
        return result["episodes"][0]

    def model_case(r):
        m = ep0(r)["at_first_output_difference"]
        return (ep0(r)["first_call_output_differs"] == 1 and ep0(r)["first_call_raw_input_differs"] == 2
                and ep0(r)["first_divergence"]["env_step"] == 15 and m["raw_action"]["max_abs_diff"] > 0
                and m["input_image"]["bitwise_equal"] is True and m["proprio"]["bitwise_equal"] is True
                and "(same model)" in ep0(r)["verdict"])

    cases = {
        "other task, output nudged -> MODEL_OUTPUT_DIFFERS (model difference)": (
            _fake_run(perturb_call=1, task="other_task"), ["MODEL_OUTPUT_DIFFERS"] * 2,
            lambda r: "(model difference)" in ep0(r)["verdict"] and r["same_model_no_override"] is False),
        "valid context shifted, model ignores it -> CONTEXT_DIFFERS_NO_EFFECT": (
            _fake_run(ctx_shift=0.5, model_reads_context=False), ["CONTEXT_DIFFERS_NO_EFFECT"] * 2,
            lambda r: r["context"]["valid_positions"]["max_abs_diff"] > 0,
            _fake_run(model_reads_context=False)),  # its own reference
        "context missing at call 1 -> UNRELIABLE, not a difference": (
            _fake_run(drop_context_call=1), ["UNRELIABLE_IDENTICAL"] * 2,
            lambda r: ep0(r)["first_call_context_differs"] is None and ep0(r)["calls_with_context_missing"] == [1]
            and ep0(r)["context_bitwise_equal"] is None and r["problems"]["pair"]),
        "same run -> IDENTICAL": (
            _fake_run(), ["IDENTICAL"] * 2, lambda r: r["verdict"] == "identical" and r["context"]["bitwise_equal"]),
        "output nudged at call 1 -> MODEL_OUTPUT_DIFFERS": (
            _fake_run(perturb_call=1), ["MODEL_OUTPUT_DIFFERS"] * 2, model_case),
        "obs jitter at step 3 (wait) -> ENV_DIVERGENCE": (
            _fake_run(jitter_step=3), ["ENV_DIVERGENCE"] * 2,
            lambda r: ep0(r)["first_divergence"]["env_step"] == 3 and "wait phase" in ep0(r)["verdict"]
            and ep0(r)["first_call_raw_input_differs"] == 0),
        "obs jitter at step 20 -> ENV_DIVERGENCE": (
            _fake_run(jitter_step=20), ["ENV_DIVERGENCE"] * 2,
            lambda r: ep0(r)["first_divergence"]["env_step"] == 20 and "wait phase" not in ep0(r)["verdict"]
            and ep0(r)["first_call_raw_input_differs"] == 2 and ep0(r)["first_call_output_differs"] == 2),
        "valid context shifted -> CONTEXT_AND_MODEL_OUTPUT_DIFFER": (
            _fake_run(ctx_shift=0.5), ["CONTEXT_AND_MODEL_OUTPUT_DIFFER"] * 2,
            lambda r: ep0(r)["first_call_output_differs"] == 0 and r["context"]["valid_positions"]["max_abs_diff"] > 0),
        "padded context shifted -> IDENTICAL_EXCEPT_CONTEXT": (
            _fake_run(ctx_pad_shift=0.5), ["IDENTICAL_EXCEPT_CONTEXT"] * 2,
            lambda r: r["context"]["valid_positions"]["max_abs_diff"] == 0
            and r["context"]["padded_positions"]["max_abs_diff"] > 0),
        "done one step later -> ENV_DIVERGENCE": (
            _fake_run(done_at=41), ["ENV_DIVERGENCE"] * 2,
            lambda r: ep0(r)["first_divergence"]["field"] == "done"),
        "hook error -> UNRELIABLE_*": (
            _fake_run(hook_error=True), ["UNRELIABLE_IDENTICAL"] * 2, lambda r: r["reliable"] is False),
    }
    failed = []
    for name, (test, expected, extra, *reference) in cases.items():
        result = compare_runs(reference[0] if reference else base, test)
        got = result["verdict_codes"]
        detail = {"expected": expected, "got": got, "verdict": result["verdict"]}
        try:
            ok = got == expected and bool(extra(result))
        except Exception as error:  # noqa: BLE001
            ok, detail["error"] = False, repr(error)
        detail["ok"] = ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got}")
        if not ok:
            failed.append(name)
            print(f"      {json.dumps(jsonable(detail))[:600]}")
    # the hashes must see a one-ulp change and the bf16 bit decoding must be exact
    x = np.array([1.0, -2.5, 3.140625], dtype=np.float32)
    bits = (x.view(np.uint32) >> np.uint32(16)).astype(np.uint16).view(np.int16)
    ulp = np.nextafter(x, np.float32(10))
    unit = {"bf16 decode exact": bool(np.array_equal(exact_to_float32(bits, "bfloat16"), x)),
            "one-ulp change changes the hash": sha_exact(x, "float32") != sha_exact(ulp, "float32"),
            "+0 vs -0 differ": sha_exact(np.array([0.0]), "float64") != sha_exact(np.array([-0.0]), "float64")}
    for name, ok in unit.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            failed.append(name)
    print(f"SELFTEST: {'PASS' if not failed else 'FAIL'}" + (f" -- {failed}" if failed else ""))
    return 0 if not failed else 1


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------


def add_eval_args(p) -> None:
    p.add_argument("--ckpt", default=DEFAULT_CKPT, help="Relative to the repo root (as in the stage 2 command).")
    p.add_argument("--dataset-stats", default=DEFAULT_STATS)
    p.add_argument("--suite-name", default="libero_spatial")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--extra-override", action="append", default=[],
                   help="Extra Hydra override KEY=VALUE (repeatable), appended after the stock ones.")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing trace in --out.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="One traced evaluation.", allow_abbrev=False)
    run.add_argument("--model", choices=sorted(TASKS), required=True)
    run.add_argument("--task", default=None, help=f"Hydra task name; default from --model {TASKS}.")
    run.add_argument("--out", required=True)
    run.add_argument("--deterministic", action="store_true")
    run.add_argument("--context-from", default=None,
                     help="context.npz from an earlier run (or its directory): use that text context.")
    add_eval_args(run)

    compare = sub.add_parser("compare", help="Compare two traces (numpy only).", allow_abbrev=False)
    compare.add_argument("run_a", help="Reference run directory.")
    compare.add_argument("run_b", help="Test run directory.")
    compare.add_argument("--out", required=True, help="compare.json to write.")

    suite = sub.add_parser("suite", help="The planned runs and comparisons, one process per run.",
                           allow_abbrev=False)
    suite.add_argument("--out", required=True)
    suite.add_argument("--steps", nargs="+", default=None, help=f"Subset, in suite order: {[s[0] for s in SUITE_STEPS]}")
    suite.add_argument("--stop-after-seconds", type=float, default=None,
                       help="Start no new run after this many seconds (the comparisons still run).")
    suite.add_argument("--run-timeout", type=float, default=900.0,
                       help="Kill a run after this many seconds (status 'timeout'; left out of the comparisons).")
    add_eval_args(suite)

    sub.add_parser("selftest", help="Check the tracer + compare on a fake evaluator (numpy only).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    handlers = {"run": cmd_run, "compare": cmd_compare, "suite": cmd_suite, "selftest": cmd_selftest}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
