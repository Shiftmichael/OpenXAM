"""The Hydra config of the new model must compose, and match the old one where they overlap.

Needs Hydra, but not PyTorch weights, a GPU or data:

    python tests/test_hydra_configs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openxam.utils.config_resolvers import register_default_resolvers  # noqa: E402

register_default_resolvers()


def composed(task: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        config = compose("train", overrides=[f"task={task}"])
    return OmegaConf.to_container(config, resolve=True)


def test_the_new_task_differs_from_the_old_one_only_in_the_model_entry() -> None:
    old = composed("libero_easywam_mot_wan22")
    new = composed("libero_xam_fastwam_wan22")
    for section in ("data", "batch_size", "learning_rate", "max_steps", "weight_decay"):
        assert old[section] == new[section], section
    old_model, new_model = old["model"], new["model"]
    assert new_model["_target_"] == "openxam.runtime.create_xam"
    assert new_model["spec"] == "fastwam_wan22"
    assert new_model["model_variant"] == "mot"
    for key in (
        "backbone",
        "state_dim",
        "projector_hidden_dim",
        "action_dit_pretrained_path",
        "skip_dit_load_from_pretrain",
        "action_dit_config",
        "video_scheduler",
        "action_scheduler",
    ):
        assert old_model[key] == new_model[key], key
    assert old_model["state_position"] == "sequence"
    assert old_model["loss"] == {"lambda_video": 1.0, "lambda_action": 1.0}


if __name__ == "__main__":
    for name, check in sorted(globals().items()):
        if name.startswith("test_"):
            check()
            print(f"ok  {name}")
