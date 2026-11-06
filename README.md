# OpenXAM

**An open platform for X-Action Models.**

[中文](README_zh.md)

Vision-language-action models (VLAs) and world-action models (WAMs) differ mainly in what information reaches the action generator: a VLM's knowledge, a video model's sense of dynamics, a written plan, optical flow, touch. OpenXAM treats that X as a setting. The goal is one codebase where a lab can fine-tune, run and evaluate these models on a modest GPU budget, starting from public pretrained weights.

> **Status: early.** Today's code is [EasyWAM](https://github.com/OpenMOSS/EasyWAM) at commit `d7c7c5a`, moved into the `openxam` package. The models listed below are EasyWAM's and have not been rerun here yet. The multi-tower core that makes X configurable does not exist yet; it is the next step.

## Design

A model will be described by four settings instead of a class:

| Setting | What it decides |
| --- | --- |
| Towers | The parameter groups and their pretrained weights, such as VLM, video and action |
| Read table | Which token streams each stream may read, and whether training signal flows back through that read |
| Noise plan | Whether each stream is clean, noisy or absent, in training and at inference |
| Losses | What is supervised: actions, future video, future features and so on |

Known models then become entries in a config:

| Model | Towers | Read table | At inference | Extra loss |
| --- | --- | --- | --- | --- |
| π0-style VLA | VLM, action | Action reads VLM | Denoise action only | None |
| JAM | Backbone, action | Action reads backbone | Denoise action only | Future features, head dropped after training |
| FastWAM | Video, action | Action reads the current frame only | Denoise action only | Future video |
| OpenWAM-α | Video, action | Action and future video read each other | Denoise both in step | Future video |
| AW-0 | Code, video, action | Video reads code; action reads code and video | Denoise all three | Future video, code |
| Forward dynamics | Video, action | Future video reads action | Action is clean, denoise video | Future video |

## What is here today

Everything below comes from EasyWAM and has not been rerun in this repository. EasyWAM's own numbers are in [docs/results/result.md](docs/results/result.md).

- **Architectures:** Unified, MoT, MoT-Joint, MoT-IDM and Hidden, each with full fine-tuning and LoRA.
- **Backbones:** Wan2.2-TI2V-5B, Cosmos-Predict2.5-2B and FLUX.2 Klein-4B.
- **Benchmarks:** LIBERO, LIBERO-Plus, LIBERO-Pro, RoboTwin, RoboDojo, RoboCasa365 and VLABench.

## Roadmap

- [ ] Multi-tower core: towers, read table, noise plan and losses as configuration
- [ ] VLM tower and a VLA path
- [ ] Given-action video prediction
- [ ] More sources: future features, subgoal images, optical flow, text or code plans
- [ ] Diagnostics: action following, cutting a read at test time

## Quick start

```bash
conda create -n openxam python=3.10 -y
conda activate openxam
pip install -U pip
pip install torch==2.7.1 torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

Prepare a [backbone](docs/instructions/backbone/README.md) and [training data](docs/instructions/data/README.md), then:

```bash
python scripts/precompute_text_embeds.py task=libero_easywam_mot_wan22
NPROC_PER_NODE=8 bash scripts/train_zero1.sh task=libero_easywam_mot_wan22
python experiments/libero/run_libero_manager.py task=libero_easywam_mot_wan22 ckpt=<path/to/checkpoint.pt>
```

Task and class names keep the `easywam` prefix until the new core replaces them.

To check the package layout without PyTorch, a GPU or a simulator:

```bash
python tests/test_package_imports.py
```

## Documentation

| Section | English | 中文 |
| --- | --- | --- |
| Backbones | [Index](docs/instructions/backbone/README.md) | [索引](docs/instructions/backbone/README_zh.md) |
| Training data | [Index](docs/instructions/data/README.md) | [索引](docs/instructions/data/README_zh.md) |
| Benchmarks | [Index](docs/instructions/benchmark/README.md) | [索引](docs/instructions/benchmark/README_zh.md) |
| Configuration | [Index](docs/instructions/config/README.md) | [索引](docs/instructions/config/README_zh.md) |

## Repository layout

```text
OpenXAM/
├── configs/        # Data, model, task and training configs
├── docs/           # Guides, plus EasyWAM's results and blog
├── experiments/    # Benchmark evaluators
├── scripts/        # Training and preprocessing entry points
├── src/openxam/    # Models, data pipeline, runtime and trainer
└── tests/          # Tests that run on CPU
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License and credits

OpenXAM is released under the [MIT License](LICENSE). It starts from [EasyWAM](https://github.com/OpenMOSS/EasyWAM) by OpenMOSS, imported once at commit `d7c7c5a`; OpenXAM does not follow EasyWAM as an upstream. Code origins and their licenses are listed in [NOTICE.md](NOTICE.md). Model weights are not part of this repository and carry their own licenses.
