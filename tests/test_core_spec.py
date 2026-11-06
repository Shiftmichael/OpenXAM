"""Configuration checks. Needs PyYAML, but not PyTorch, a GPU or a simulator.

    python tests/test_core_spec.py
"""

from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openxam.core import ModelSpec, SpecError  # noqa: E402

CONFIGS = ROOT / "configs" / "xam"
DRAFT = ROOT / "docs" / "design" / "multi_tower_core_zh.md"


def raw(name: str) -> dict:
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())


def codes(config: dict) -> set[str]:
    return {problem.code for problem in ModelSpec.from_dict(config).check()}


def test_every_config_is_valid() -> None:
    files = sorted(CONFIGS.glob("*.yaml"))
    assert len(files) >= 8
    for path in files:
        ModelSpec.load(path)


def test_every_example_in_the_design_note_is_valid() -> None:
    blocks = re.findall(r"```yaml\n(.*?)```", DRAFT.read_text(), re.S)
    assert len(blocks) >= 9
    for block in blocks:
        ModelSpec.from_dict(yaml.safe_load(block)).validate()


def test_unknown_keys_are_refused() -> None:
    config = raw("fastwam_wan22")
    config["noise"] = {}
    try:
        ModelSpec.from_dict(config)
    except SpecError as error:
        assert error.problems[0].code == "format"
    else:
        raise AssertionError("a misspelled section must be refused")


def test_misspelled_stream() -> None:
    config = raw("fastwam_wan22")
    config["reads"]["action"][0] = "obss"
    assert codes(config) == {"name"}


def test_stream_without_a_row() -> None:
    config = raw("fastwam_wan22")
    del config["reads"]["state"]
    assert "row" in codes(config)


def test_joint_read_across_different_shapes() -> None:
    config = raw("fastwam_wan22")
    config["towers"]["vlm"] = {"kind": "qwen3_vl"}
    config["streams"]["prompt"] = {"tower": "vlm", "data": "instruction"}
    config["reads"]["prompt"] = ["prompt", "obs"]
    assert "shape" in codes(config)


def test_flow_loss_on_a_stream_that_never_gets_noise() -> None:
    config = raw("fastwam_wan22")
    config["train"][0]["noisy"] = ["action"]
    assert "loss" in codes(config)


def test_denoising_a_stream_never_trained_with_noise() -> None:
    config = raw("fastwam_wan22")
    config["infer"]["act"][0]["denoise"] = ["action", "state"]
    assert "denoise" in codes(config)


def test_reading_a_stream_that_is_absent_at_inference() -> None:
    config = raw("mot_joint_wan22")
    config["infer"]["bad"] = [{"denoise": ["action"], "absent": ["future"], "steps": 10}]
    assert "unseen" in codes(config)


def test_state_never_seen_in_training() -> None:
    config = raw("mot_joint_wan22")
    config["infer"]["bad"] = [{"denoise": ["future"], "steps": 10}]
    assert "unseen" in codes(config)


def test_read_table_never_used_in_training() -> None:
    config = raw("mot_joint_wan22")
    row = ["obs", "future", "action", "state"]
    config["infer"]["bad"] = [{"denoise": ["future"], "steps": 10, "reads": {"future": row}}]
    assert "table" in codes(config)


def test_use_needs_an_earlier_stage() -> None:
    config = raw("mot_idm_wan22")
    config["infer"]["act_and_imagine"][1]["use"] = {"future_c": "action"}
    assert "stage" in codes(config)


def test_cut_must_name_a_read() -> None:
    config = raw("fastwam_wan22")
    good = copy.deepcopy(config)
    good["infer"]["act"][0]["cut"] = ["action<-obs"]
    assert codes(good) == set()
    config["infer"]["act"][0]["cut"] = ["action<-future"]
    assert "stage" in codes(config)


def test_context_layer_must_exist() -> None:
    config = raw("hidden_wan22")
    config["towers"]["action"]["context"][0]["layer"] = 30
    assert "name" in codes(config)


def test_tied_streams_must_be_noisy() -> None:
    config = raw("unified_wan22")
    config["train"][0]["tie"] = [["future", "state"]]
    assert "mode" in codes(config)


def test_a_stream_must_read_something() -> None:
    config = raw("fastwam_wan22")
    config["reads"]["state"] = ["future"]
    config["train"][0]["absent"] = ["future"]
    config["train"][0]["noisy"] = ["action"]
    assert "mode" in codes(config)


if __name__ == "__main__":
    for name, check in sorted(globals().items()):
        if name.startswith("test_"):
            check()
            print(f"ok  {name}")
