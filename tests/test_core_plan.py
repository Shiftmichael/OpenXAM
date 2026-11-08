"""The compiled attention structure against the structures the current models use.

Needs PyYAML, but not PyTorch, a GPU or a simulator. The reference functions below are
retyped from the mask builders of the EasyWAM classes and of NUS OpenWAM, because those
need PyTorch to run. Tests that call the real functions belong with the parity tests.

    python tests/test_core_plan.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openxam.core import ModelSpec, Plan, Segment, plan_infer, plan_stage, plan_train, sample_weights  # noqa: E402

CONFIGS = ROOT / "configs" / "xam"
MODES = ("first_frame_causal", "bidirectional", "per_frame_causal")


def spec_of(name: str, **rows) -> ModelSpec:
    config = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())
    config["reads"].update(rows)
    return ModelSpec.from_dict(config)


def video_rows(mode: str, obs: str = "obs", future: str = "future", extra: tuple = ()) -> dict:
    if mode == "first_frame_causal":
        return {}
    if mode == "bidirectional":
        return {obs: [obs, future, *extra], future: [obs, future, *extra]}
    return {obs: [obs, *extra], future: [obs, {"from": future, "pattern": "frame_causal"}, *extra]}


def normalized(segments: list) -> tuple:
    """Merge touching key ranges and neighbouring rows, as ``Plan.segments`` does."""
    out: list[Segment] = []
    for start, end, ranges in segments:
        merged: list[tuple[int, int]] = []
        for key_start, key_end in ranges:
            if merged and merged[-1][1] == key_start:
                merged[-1] = (merged[-1][0], key_end)
            else:
                merged.append((key_start, key_end))
        segment = Segment(start, end, tuple(merged))
        if out and out[-1].key_ranges == segment.key_ranges:
            out[-1] = Segment(out[-1].query_start, end, segment.key_ranges)
        else:
            out.append(segment)
    return tuple(out)


def video_segments(offset: int, length: int, frame: int, mode: str) -> list:
    first = min(frame, length)
    if mode == "bidirectional":
        return [(offset, offset + length, [(offset, offset + length)])]
    if mode == "first_frame_causal":
        out = [(offset, offset + first, [(offset, offset + first)])]
        if first < length:
            out.append((offset + first, offset + length, [(offset, offset + length)]))
        return out
    return [
        (offset + start, offset + start + frame, [(offset, offset + start + frame)])
        for start in range(0, length, frame)
    ]


def mot(video: int, action: int, frame: int, mode: str) -> tuple:
    total, first = video + action, min(frame, video)
    keys = [(0, total)] if first == video else [(0, first), (video, total)]
    return normalized(video_segments(0, video, frame, mode) + [(video, total, keys)])


def mot_joint(video: int, action: int, frame: int, mode: str) -> tuple:
    total = video + action
    return normalized(video_segments(0, video, frame, mode) + [(video, total, [(0, video), (video, total)])])


def mot_idm(video: int, action: int, frame: int, mode: str) -> tuple:
    condition_end = 2 * video
    total = condition_end + action
    segments = video_segments(0, video, frame, mode) + video_segments(video, video, frame, mode)
    return normalized(segments + [(condition_end, total, [(video, condition_end), (condition_end, total)])])


def unified(clean: int, future: int, action: int, state: int, mode: str) -> tuple:
    video = clean + future
    state_start = video + action
    total = state_start + state
    state_keys = [(state_start, total)] if state else []
    if mode == "bidirectional":
        segments = [(0, total, [(0, total)])]
    elif mode == "first_frame_causal":
        segments = [(0, clean, [(0, clean)] + state_keys), (clean, state_start, [(0, total)])]
    else:
        segments = [
            (start, start + clean, [(0, start + clean)] + state_keys) for start in range(0, video, clean)
        ]
        segments.append((video, state_start, [(0, total)]))
    if mode != "bidirectional" and state:
        segments.append((state_start, total, [(0, total)]))
    return normalized(segments)


def dense(segments: tuple, total: int) -> list:
    matrix = [[False] * total for _ in range(total)]
    for segment in segments:
        for start, end in segment.key_ranges:
            for query in range(segment.query_start, segment.query_end):
                for key in range(start, end):
                    matrix[query][key] = True
    return matrix


def nus_openwam(video: int, action: int, tail: int, first: int, mode: str) -> list:
    """``build_cross_modal_attention_mask`` with first-frame-causal video."""
    total = video + action + tail
    matrix = [[False] * total for _ in range(total)]
    for query, key in itertools.product(range(first), range(first)):
        matrix[query][key] = True
    for query, key in itertools.product(range(first, video), range(video)):
        matrix[query][key] = True
    for query, key in itertools.product(range(video, video + action), range(video, video + action)):
        matrix[query][key] = True
    for query, key in itertools.product(range(video, video + action), range(video)):
        matrix[query][key] = mode in ("action_sees_video", "mutual") or key < first
    for query, key in itertools.product(range(video), range(video, video + action)):
        matrix[query][key] = mode in ("mutual", "video_sees_action") and query >= first
    for query, key in itertools.product(range(video + action), range(video + action, total)):
        matrix[query][key] = True
    for query, key in itertools.product(range(video + action, total), range(video + action, total)):
        matrix[query][key] = True
    return matrix


def sizes():
    return itertools.product((1, 3, 4), (2, 3, 5), (1, 4, 8), (0, 1, 2))


def lengths(frame: int, frames: int, action: int, state: int) -> dict:
    return {"obs": frame, "future": frame * (frames - 1), "action": action, "state": state}


def test_mot_and_joint_match_the_current_structure() -> None:
    for mode, (frame, frames, action, state) in itertools.product(MODES, sizes()):
        size = lengths(frame, frames, action, state)
        video = frame * frames
        for name, reference in (("fastwam_wan22", mot), ("mot_joint_wan22", mot_joint)):
            plan = plan_train(spec_of(name, **video_rows(mode)), "joint")
            assert plan.segments(size, frame) == reference(video, action + state, frame, mode), (name, mode, size)


def test_action_only_prefill_matches_the_current_structure() -> None:
    plan = plan_infer(spec_of("fastwam_wan22"), "act")[0]
    for frame, _, action, state in sizes():
        size = {"obs": frame, "action": action, "state": state}
        assert plan.segments(size, frame) == mot(frame, action + state, frame, "first_frame_causal")


def test_idm_matches_the_current_structure() -> None:
    for mode, (frame, frames, action, state) in itertools.product(MODES, sizes()):
        rows = {**video_rows(mode), **video_rows(mode, "obs_c", "future_c")}
        plan = plan_train(spec_of("mot_idm_wan22", **rows), "teacher_forcing")
        size = lengths(frame, frames, action, state)
        size.update(obs_c=size["obs"], future_c=size["future"])
        assert plan.segments(size, frame) == mot_idm(frame * frames, action + state, frame, mode), (mode, size)


def test_unified_matches_the_current_structure() -> None:
    every = ["obs", "future", "action", "state"]
    rows = {
        "first_frame_causal": {},
        "bidirectional": {"obs": every},
        "per_frame_causal": {"future": ["obs", {"from": "future", "pattern": "frame_causal"}, "state"]},
    }
    for mode, (frame, frames, action, state) in itertools.product(MODES, sizes()):
        plan = plan_train(spec_of("unified_wan22", **rows[mode]), "joint")
        size = lengths(frame, frames, action, state)
        assert plan.segments(size, frame) == unified(frame, size["future"], action, state, mode), (mode, size)


def test_nus_openwam_visibility_modes() -> None:
    video_action = ["obs", "future", "action", "state"]
    tables = {
        "mutual": {"future": video_action, "action": video_action},
        "action_sees_video": {"future": ["obs", "future"], "action": video_action},
        "video_sees_action": {"future": video_action, "action": ["obs", "action", "state"]},
        "isolated": {"future": ["obs", "future"], "action": ["obs", "action", "state"]},
    }
    config = yaml.safe_load((CONFIGS / "fastwam_wan22.yaml").read_text())
    config["towers"]["reader"] = {"kind": "frozen_features"}
    config["streams"]["tail"] = {"tower": "reader", "data": "instruction"}
    config["reads"]["tail"] = ["tail"]
    for mode, table in tables.items():
        for frame, frames, action, state, tail in itertools.product((1, 3), (2, 4), (2, 5), (0, 1), (0, 3)):
            rows = {"obs": ["obs"], **table, "state": table["action"], "tail": ["tail"]}
            rows = {key: [*value, "tail"] if key != "tail" else value for key, value in rows.items()}
            config["reads"].update(rows)
            plan = plan_train(ModelSpec.from_dict(config), "joint")
            size = {**lengths(frame, frames, action, state), "tail": tail}
            video, total = frame * frames, frame * frames + action + state + tail
            got = dense(plan.segments(size, frame), total)
            assert got == nus_openwam(video, action + state, tail, frame, mode), (mode, size)


def test_fixed_streams_match_what_is_cached_today() -> None:
    expected = {
        ("fastwam_wan22", "act", 0): {"obs"},
        ("vla_qwen3vl", "act", 0): {"prompt"},
        ("mot_joint_wan22", "act_and_imagine", 0): {"obs"},
        ("mot_idm_wan22", "act_and_imagine", 1): {"obs_c", "future_c"},
        ("hidden_wan22", "act", 0): {"obs", "future"},
        ("unified_wan22", "act_and_imagine", 0): set(),
    }
    for (name, program, index), want in expected.items():
        assert plan_stage(spec_of(name), program, index).static == want, name


def test_idm_stages_pass_results_along() -> None:
    first, second = plan_infer(spec_of("mot_idm_wan22"), "act_and_imagine")
    assert first.streams == ("obs", "future")
    assert second.streams == ("obs_c", "future_c", "action", "state")
    assert second.use == {"future_c": "future"}


def test_frame_causal_and_causal_reads() -> None:
    spec = spec_of("vla_qwen3vl")
    plan = plan_train(spec, "act")
    segments = plan.segments({"prompt": 3, "action": 2, "state": 1})
    assert segments == (
        Segment(0, 1, ((0, 1),)),
        Segment(1, 2, ((0, 2),)),
        Segment(2, 3, ((0, 3),)),
        Segment(3, 6, ((0, 6),)),
    )


def test_detached_reads_keep_the_structure() -> None:
    plain = plan_train(spec_of("vla_qwen3vl"), "act")
    cut = plan_train(
        spec_of("vla_qwen3vl", action=[{"from": "prompt", "grad": False}, "action", "state"]), "act"
    )
    size = {"prompt": 4, "action": 2, "state": 1}
    assert plain.detached == ()
    assert cut.detached == (("action", "prompt"),)
    assert dense(plain.segments(size), 7) == dense(cut.segments(size), 7)
    assert [segment.no_grad for segment in cut.segments(size)] == [(), (), (), (), ((0, 4),), ()]


def test_cached_towers() -> None:
    assert plan_infer(spec_of("fastwam_wan22"), "act")[0].cached_towers == ("video",)
    assert plan_infer(spec_of("vla_qwen3vl"), "act")[0].cached_towers == ("vlm",)
    assert plan_infer(spec_of("mot_joint_wan22"), "act_and_imagine")[0].cached_towers == ()
    assert plan_infer(spec_of("mot_idm_wan22"), "act_and_imagine")[1].cached_towers == ("video",)


def test_cut_removes_a_read_at_test_time() -> None:
    config = yaml.safe_load((CONFIGS / "fastwam_wan22.yaml").read_text())
    config["infer"]["act"][0]["cut"] = ["action<-obs"]
    plan = plan_stage(ModelSpec.from_dict(config), "act", 0)
    assert [read.source for read in plan.rows["action"]] == ["action", "state"]


def test_empty_streams_take_no_tokens() -> None:
    plan = plan_train(spec_of("fastwam_wan22"), "joint")
    assert "state" not in plan.spans({"obs": 2, "future": 2, "action": 2})


def test_tower_order_decides_token_order() -> None:
    plan: Plan = plan_train(spec_of("mot_idm_wan22"), "teacher_forcing")
    assert plan.streams == ("obs", "future", "obs_c", "future_c", "action", "state")
    assert plan.active_towers == ("video", "action")


def test_mode_shares_add_up() -> None:
    assert sample_weights(spec_of("predict_mix_wan22")) == {"joint": 0.5, "predict": 0.5}


def test_towers_run_in_groups() -> None:
    assert plan_train(spec_of("fastwam_wan22"), "joint").groups == (("video", "action"),)
    assert plan_train(spec_of("unified_wan22"), "joint").groups == (("video",),)
    first, second = plan_infer(spec_of("mot_idm_wan22"), "act_and_imagine")
    assert (first.groups, second.groups) == ((("video",),), (("video", "action"),))
    hidden = plan_train(spec_of("hidden_wan22"), "joint")
    assert hidden.groups == (("video",), ("action",))
    assert hidden.captures(("video",)) == {"video": 17} and hidden.captures(("action",)) == {}


def test_a_tower_reading_features_runs_after_its_source() -> None:
    config = yaml.safe_load((CONFIGS / "hidden_wan22.yaml").read_text())
    config["towers"] = {"action": config["towers"]["action"], "video": config["towers"]["video"]}
    plan = plan_train(ModelSpec.from_dict(config), "joint")
    assert plan.active_towers == ("action", "video")
    assert plan.groups == (("video",), ("action",))


def test_idm_noise_shares_reach_the_plan() -> None:
    assert plan_train(spec_of("mot_idm_wan22"), "teacher_forcing").sometimes == {"future_c": 0.5}


if __name__ == "__main__":
    for name, check in sorted(globals().items()):
        if name.startswith("test_"):
            check()
            print(f"ok  {name}")
