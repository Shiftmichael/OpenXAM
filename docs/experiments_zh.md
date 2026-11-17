# 实验计划

[English](experiments.md) | [文档索引](README_zh.md) | [返回项目 README](../README_zh.md)

现在有几位同学同时用 OpenXAM 跑基准实验。这一页写清楚第一轮做什么、每次评测跑多少局，以及权重和结果放在哪里。谁在跑什么，记在项目 README 的[状态一节](../README.md#status)。

## 第一轮

第一轮的目标是：每个基准都能在 OpenXAM 里用 Wan2.2 的结构从训练跑到评测，并按官方协议得到我们自己的第一个数字。EasyWAM 发布过权重的，也把发布的权重评测一遍，让我们的数字有个参照。

每个基准一位负责人。LIBERO-Plus 和 LIBERO-Pro 评测的是在 LIBERO 上训练的权重，所以这三个归同一个人。

| 基准 | 起点 | 要跑的 | 对照 |
| --- | --- | --- | --- |
| LIBERO、LIBERO-Plus、LIBERO-Pro | EasyWAM 发布的 LIBERO 权重；重新训练时用 Wan2.2-TI2V-5B | 在 LIBERO 上评测五个发布的 Wan2.2 权重，在 LIBERO-Plus 和 LIBERO-Pro 上评测发布的 MoT；用两颗种子在 LIBERO 上重新训练 MoT | EasyWAM [公布的数字](results/result_zh.md) |
| RoboTwin 2.0 | Wan2.2-TI2V-5B | 用两颗种子训练 MoT，两个都评测 | 基准自己公布的基线 |
| RoboDojo | Wan2.2-TI2V-5B | 同上 | 同上 |
| RoboCasa365 | Wan2.2-TI2V-5B | 同上 | 同上 |
| VLABench | Wan2.2-TI2V-5B | 同上 | 同上 |

合计 10 次训练、18 次评测。

LIBERO 的第一次评测，把发布的 MoT 权重跑两遍：一遍用 EasyWAM 的任务 `libero_easywam_mot_wan22`，一遍用新核心的 `libero_xam_fastwam_wan22`。两边结果应该一致，这也是新核心第一次用真实权重运行。其他基准在新核心上还没有任务配置，暂时用各自的 `<基准>_easywam_mot_wan22`。

先做 MoT，是因为它是 EasyWAM 的主要结构，而且有发布的权重。之后几轮再把每个基准扩展到其他结构、LoRA 和 Cosmos-Predict2.5，顺序看第一轮的结果再定。

## 每次评测跑多少

按各评测器实现的官方协议来。环境怎么装，见[基准指南](instructions/benchmark/README_zh.md)。

| 基准 | 协议 | 局数 | 仿真器 |
| --- | --- | ---: | --- |
| LIBERO | 4 组 × 10 个任务 × 50 次 | 2,000 | MuJoCo 3.3.2、robosuite 1.4.0 |
| LIBERO-Plus | 10,030 个扰动任务，每个 1 局 | 10,030 | LIBERO-Plus 的 fork，单独一个环境 |
| LIBERO-Pro | 20 个"组 × 扰动"单元里的 200 个任务 × 50 次 | 10,000 | LIBERO-Pro 的源码 |
| RoboTwin 2.0 | 50 个任务 × 干净和随机化两种场景 × 100 局 | 10,000 | RoboTwin `96c1fea` 提交 |
| RoboDojo | 42 个任务 × 种子 0、1、2 × 50 局 | 6,300 | RoboDojo 环境里的 Isaac Sim |
| RoboCasa365 | Human300：50 个任务 × 50 局 | 2,500 | RoboCasa v1.0.1 或更新；记下 robosuite 的提交号 |
| VLABench | 官方第 1、2、3、4、6 组，每个任务取前 10 个配置 | 500 | MuJoCo 3.2.2 |

一个数字要进结果表，需要满足：

- 来自最后一个检查点。不要按基准分数挑检查点。想用别的步数，要在评测前定好，并且同一组比较里每个 run 都用这个步数。
- 用两颗种子训练（`seed=0` 和 `seed=1`），报告两个值和平均值。只是检查流程能不能跑通，一颗种子就够。
- 跑的是完整协议。调试时只跑一部分没问题。
- 写明仿真器版本、提交号、任务和命令行改动。同一组比较里仿真器版本保持不变。

## 命名和存放

训练输出在 `runs/<任务>/<run-id>/`。run 的名字用 `<年月日>-<GitHub 用户名>-s<种子>`，日志记到实验室共用的 Weights & Biases 团队里，大家都能看到曲线：

```bash
RUN_ID=20261118-yourname-s0 NPROC_PER_NODE=8 bash scripts/train_zero1.sh \
  task=robotwin_easywam_mot_wan22 seed=0 \
  wandb.enabled=true wandb.workspace=<lab-team> wandb.project=openxam \
  wandb.group=robotwin_easywam_mot_wan22 wandb.name=20261118-yourname-s0
```

| 什么 | 放在哪 | 留什么 |
| --- | --- | --- |
| 训练中的检查点 | 训练所在集群上的 `runs/` | 训练器只保留最近三个（`checkpoint_save_limit: 3`） |
| 有数字进结果表的检查点 | 实验室 Hugging Face 组织下的私有模型仓库，名字用 `<任务>-<run-id>` | 最终权重、`dataset_stats.json`、完整配置和评测汇总 |
| 评测输出 | 集群上的 `evaluate_results/<基准>/...` | `summary.json` 和 `summary.csv`，同时贴到这个 run 的 issue 里 |
| EasyWAM 发布的权重 | 每个集群从 Hub 上的 `OpenMOSS-Team/EasyWAM-<结构>-Wan22` 下载一次 | 不另存副本 |

Wan2.2 的完整检查点每个 10–12 GB，和 EasyWAM 发布的文件一样大；续训用的训练状态更大，开跑前先确认存储配额。评测完后，删掉训练状态和中间检查点。要不要公开某个检查点，另外决定。检查点、数据集、评测输出和本机路径都不要提交到这个仓库。

## 跟踪一次 run

1. 开跑前，用 **Experiment** 表单开一个 issue，并在 README 状态表里把自己的格子标成运行中（🏃），链接到这个 issue。
2. 运行中，进展和问题都写在 issue 里。
3. 跑完后，把评测汇总贴到 issue 里，再用一个 Pull Request 把数字加进结果表，并把 README 里的格子改成完成（✅ 加分数）。

## 结果

结果表只在[英文版](experiments.md#results)里维护一份，避免两份对不上。

## 哪些不放进这个仓库

这个仓库是公开的，它的所有分支和 fork 也都是公开的。还没发表的想法、配置和结果，放在自己的私有仓库里，把 OpenXAM 当依赖装进去；这类工作的 run 也私下跟踪。发表以后，再到这里开 Pull Request。
