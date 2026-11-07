# OpenXAM

**X-Action Models 的开放平台。**

[English](README.md)

视觉语言动作模型（VLA）和世界动作模型（WAM）的主要区别，是什么信息进入了动作生成器：VLM 的知识、视频模型对动态的理解、写下来的计划、光流、触觉。OpenXAM 把这个 X 当成一项设置。目标是一套代码，让实验室用不多的 GPU，从公开的预训练权重出发，完成这些模型的微调、推理和评测。

> **状态：早期。** 代码从 [EasyWAM](https://github.com/OpenMOSS/EasyWAM) 的 `d7c7c5a` 提交出发。下面列的模型都来自 EasyWAM，在本仓库里还没有重跑过。让 X 可以配置的多塔核心正在 `src/openxam/core` 里搭建：配置已经能编译，MoT 在小的随机模型上和 EasyWAM-MoT 完全一致，权重和训练器也接上了，用 `task=libero_xam_fastwam_wan22` 运行。真实权重和 GPU 还没试过，其余结构还在用 EasyWAM 的类。

## 设计

一个模型将由四项设置决定，而不是一个类：

| 设置 | 决定什么 |
| --- | --- |
| 塔 | 有哪几组参数，各用什么预训练权重，比如 VLM、视频、动作 |
| 读取表 | 每一路 token 能读哪几路，读的时候训练信号回不回传 |
| 噪声安排 | 每一路在训练和推理时是干净、带噪，还是不存在 |
| 损失 | 监督什么：动作、未来视频、未来特征等 |

已知的模型就成了配置里的条目：

| 模型 | 塔 | 读取表 | 推理时 | 额外损失 |
| --- | --- | --- | --- | --- |
| π0 这类 VLA | VLM、动作 | 动作读 VLM | 只去噪动作 | 无 |
| JAM | 骨干、动作 | 动作读骨干 | 只去噪动作 | 未来特征，训完丢掉预测头 |
| FastWAM | 视频、动作 | 动作只读当前画面 | 只去噪动作 | 未来视频 |
| OpenWAM-α | 视频、动作 | 动作和未来视频互读 | 两路同步去噪 | 未来视频 |
| AW-0 | 代码、视频、动作 | 视频读代码，动作读代码和视频 | 三路一起去噪 | 未来视频、代码 |
| 给定动作预测未来 | 视频、动作 | 未来视频读动作 | 动作干净，只去噪视频 | 未来视频 |

## 现在有什么

下面这些都来自 EasyWAM，在本仓库里还没有重跑过。EasyWAM 自己报告的数字见 [docs/results/result_zh.md](docs/results/result_zh.md)。

- **结构：** Unified、MoT、MoT-Joint、MoT-IDM、Hidden，都支持全量微调和 LoRA。
- **骨干：** Wan2.2-TI2V-5B、Cosmos-Predict2.5-2B、FLUX.2 Klein-4B。
- **基准：** LIBERO、LIBERO-Plus、LIBERO-Pro、RoboTwin、RoboDojo、RoboCasa365、VLABench。

## 路线

- [x] 塔、读取表、噪声安排、损失都写成配置，在用 GPU 之前就能检查
- [ ] 多塔核心跑通所有结构（MoT 已完成，见[设计说明](docs/design/multi_tower_core_zh.md)）
- [ ] VLM 塔和 VLA 路径
- [ ] 给定动作预测未来
- [ ] 更多信息来源：未来特征、子目标图、光流、文字或代码计划
- [ ] 诊断工具：动作跟随、测试时切断某一路读取

## 快速开始

```bash
conda create -n openxam python=3.10 -y
conda activate openxam
pip install -U pip
pip install torch==2.7.1 torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

准备好[骨干](docs/instructions/backbone/README_zh.md)和[训练数据](docs/instructions/data/README_zh.md)后：

```bash
python scripts/precompute_text_embeds.py task=libero_easywam_mot_wan22
NPROC_PER_NODE=8 bash scripts/train_zero1.sh task=libero_easywam_mot_wan22
python experiments/libero/run_libero_manager.py task=libero_easywam_mot_wan22 ckpt=<path/to/checkpoint.pt>
```

task 和类名暂时保留 `easywam` 前缀，等新的核心替换它们时再改。

不需要 GPU 和仿真器的测试：

```bash
python tests/test_package_imports.py
python tests/test_core_spec.py
python tests/test_core_plan.py
python tests/test_parity_mot.py     # 需要 PyTorch
python tests/test_trainer_smoke.py  # 需要 PyTorch 和 Accelerate
python tests/test_hydra_configs.py  # 需要 Hydra
```

## 文档

| 内容 | 中文 | English |
| --- | --- | --- |
| 骨干 | [索引](docs/instructions/backbone/README_zh.md) | [Index](docs/instructions/backbone/README.md) |
| 训练数据 | [索引](docs/instructions/data/README_zh.md) | [Index](docs/instructions/data/README.md) |
| 基准 | [索引](docs/instructions/benchmark/README_zh.md) | [Index](docs/instructions/benchmark/README.md) |
| 配置 | [索引](docs/instructions/config/README_zh.md) | [Index](docs/instructions/config/README.md) |

## 目录结构

```text
OpenXAM/
├── configs/        # 数据、模型、task 和训练配置
├── docs/           # 使用指南，以及 EasyWAM 的结果和博客
├── experiments/    # 基准评测
├── scripts/        # 训练和预处理入口
├── src/openxam/    # 核心、塔、模型、数据流程、运行时和训练器
└── tests/          # 能在 CPU 上跑的测试
```

## 参与开发

见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证和致谢

OpenXAM 使用 [MIT 许可证](LICENSE)。它从 OpenMOSS 的 [EasyWAM](https://github.com/OpenMOSS/EasyWAM) 出发，在 `d7c7c5a` 提交一次性导入，之后不再跟随它的更新。代码来源和对应许可证见 [NOTICE.md](NOTICE.md)。模型权重不在本仓库里，各有自己的许可证。
