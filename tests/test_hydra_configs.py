"""The Hydra config of the new model must compose, and match the old one where they overlap.

Needs Hydra, but not PyTorch weights, a GPU or data:

    python tests/test_hydra_configs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
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


# old model name, new spec, model_variant
TASKS = {
    "fastwam": ("mot", "fastwam_wan22", "mot"),
    "mot_joint": ("mot_joint", "mot_joint_wan22", "mot_joint"),
    "mot_idm": ("mot_idm", "mot_idm_wan22", "mot_idm"),
    "unified": ("unified", "unified_wan22", "unified"),
    "hidden": ("hidden", "hidden_wan22", "hidden"),
}
SHARED = (
    "backbone",
    "action_dim",
    "state_dim",
    "projector_hidden_dim",
    "action_dit_pretrained_path",
    "skip_dit_load_from_pretrain",
    "action_dit_config",
    "video_scheduler",
    "action_scheduler",
)


def spec(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / "xam" / f"{name}.yaml").read_text())


def test_each_new_task_differs_from_the_old_one_only_in_the_model_entry() -> None:
    for new_name, (old_name, spec_name, variant) in TASKS.items():
        old = composed(f"libero_easywam_{old_name}_wan22")
        new = composed(f"libero_xam_{new_name}_wan22")
        for section in ("data", "batch_size", "learning_rate", "max_steps", "weight_decay"):
            assert old[section] == new[section], (new_name, section)
        old_model, new_model = old["model"], new["model"]
        assert new_model["_target_"] == "openxam.runtime.create_xam"
        assert (new_model["spec"], new_model["model_variant"]) == (spec_name, variant), new_name
        for key in SHARED:
            if key == "action_dim" and key not in new_model and "action_dit_config" in new_model:
                assert old_model.get(key, new_model["action_dit_config"][key]) == new_model["action_dit_config"][key]
            elif key in old_model or key in new_model:
                assert old_model.get(key) == new_model.get(key), (new_name, key)
        assert old_model["state_position"] == "sequence", new_name
        assert old_model["loss"] == {"lambda_video": 1.0, "lambda_action": 1.0}, new_name


def test_settings_that_moved_into_the_spec_agree() -> None:
    hidden = composed("libero_easywam_hidden_wan22")["model"]
    read = spec("hidden_wan22")["towers"]["action"]["context"][0]
    assert read["layer"] == hidden["video_hidden_layer"] and read["grad"] is not hidden["detach_video_hidden"]
    idm = composed("libero_easywam_mot_idm_wan22")["model"]
    assert spec("mot_idm_wan22")["train"][0]["sometimes_noisy"]["future_c"] == idm["video_cond_noise_prob"]


if __name__ == "__main__":
    for name, check in sorted(globals().items()):
        if name.startswith("test_"):
            check()
            print(f"ok  {name}")
