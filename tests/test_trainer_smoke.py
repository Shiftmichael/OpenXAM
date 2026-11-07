"""The real trainer must be able to train XAMModel: a few steps on tiny towers, on CPU.

Needs PyTorch and Accelerate, but no GPU, weights or data:

    python tests/test_trainer_smoke.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_parity_mot as tiny  # noqa: E402

from openxam.trainer import EasyWAMTrainer  # noqa: E402


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, length: int = 8):
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        generator = torch.Generator().manual_seed(index)
        return {
            "video": torch.rand(3, tiny.FRAMES, 32, 32, generator=generator) * 2 - 1,
            "context": torch.randn(6, tiny.TEXT_DIM, generator=generator),
            "context_mask": torch.ones(6, dtype=torch.bool),
            "proprio": torch.randn(1, tiny.STATE_DIM, generator=generator),
            "action": torch.randn(tiny.HORIZON, tiny.ACTION_DIM, generator=generator),
            "action_is_pad": torch.zeros(tiny.HORIZON, dtype=torch.bool),
        }


def settings(folder: str, steps: int):
    return OmegaConf.create(
        {
            "output_dir": folder,
            "learning_rate": 1e-3,
            "warmup_ratio": 0.0,
            "weight_decay": 0.0,
            "batch_size": 2,
            "num_workers": 0,
            "max_steps": steps,
            "log_every": 1,
            "save_every": steps,
            "checkpoint_save_limit": 2,
            "eval_every": 10_000,
            "eval_num_inference_steps": 2,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "seed": 3,
            "resume": None,
            "mixed_precision": "no",
            "lr_scheduler_type": "cosine",
            "wandb": {"enabled": False},
        }
    )


def test_the_trainer_trains_and_saves_xam_model() -> None:
    model = tiny.built()
    before = {name: value.clone() for name, value in model.dit.state_dict().items()}
    with tempfile.TemporaryDirectory() as folder:
        accelerator = Accelerator(cpu=True, mixed_precision="no", step_scheduler_with_optimizer=False)
        trainer = EasyWAMTrainer(
            model,
            TinyDataset(),
            None,
            cfg=settings(folder, steps=3),
            accelerator=accelerator,
        )
        trainer.train()
        accelerator.end_training()

        trained = accelerator.unwrap_model(trainer.model)
        moved = [name for name, value in trained.dit.state_dict().items() if not torch.equal(value, before[name])]
        assert len(moved) > 10, len(moved)

        saved = sorted(Path(folder).rglob("*.pt"))
        assert saved, list(Path(folder).rglob("*"))
        fresh = tiny.built()
        sample = tiny.batch()
        assert not torch.equal(tiny.act(fresh, sample), tiny.act(trained, sample))
        fresh.towers["video"].vae.load_state_dict(trained.towers["video"].vae.state_dict())
        fresh.load_checkpoint(str(saved[-1]))
        assert torch.equal(tiny.act(fresh, sample), tiny.act(trained, sample))


def trainer_for(folder: str, steps: int, save_every: int, resume=None) -> tuple[EasyWAMTrainer, Accelerator]:
    AcceleratorState._reset_state(reset_partial_state=True)
    accelerator = Accelerator(cpu=True, mixed_precision="no", step_scheduler_with_optimizer=False)
    config = settings(folder, steps)
    config.save_every = save_every
    config.resume = resume
    return EasyWAMTrainer(tiny.built(), TinyDataset(), None, cfg=config, accelerator=accelerator), accelerator


def test_training_resumes_from_saved_state() -> None:
    """Weights, step and optimizer come back. Later steps are not bit-identical to an unbroken
    run, for the old classes either, so that is not asserted."""
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        trainer, accelerator = trainer_for(first, steps=2, save_every=2)
        trainer.train()
        accelerator.end_training()
        state = next(Path(first).rglob("trainer_state.json")).parent
        saved = torch.load(next(Path(first).rglob("step_000002.pt")), map_location="cpu")["mot"]

        trainer, accelerator = trainer_for(second, steps=4, save_every=4, resume=str(state))
        trainer._resume_or_load_checkpoint()
        restored = accelerator.unwrap_model(trainer.model).mot.state_dict()
        assert all(torch.equal(value, restored[key]) for key, value in saved.items())
        assert trainer.global_step == 2
        assert len(trainer.optimizer.state_dict()["state"]) > 0

        trainer, accelerator = trainer_for(second, steps=4, save_every=4, resume=str(state))
        trainer.train()
        accelerator.end_training()
        assert trainer.global_step == 4


if __name__ == "__main__":
    for name, check in sorted(globals().items()):
        if name.startswith("test_"):
            check()
            print(f"ok  {name}")
