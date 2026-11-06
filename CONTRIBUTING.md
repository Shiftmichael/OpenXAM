# Contributing to OpenXAM

[中文](#中文)

## Before you start

Open an issue for anything larger than a small fix, so we can agree on the approach first. The multi-tower core is still being built, and new information sources should wait for it unless we have talked about them.

## Pull requests

- Keep each change focused, and work on a branch rather than on `main`.
- Update the English and Chinese documents together when user-facing behavior changes.
- Do not commit datasets, checkpoints, generated outputs, credentials or machine-specific paths.
- Say in the pull request what you changed, how you checked it, and what you did not check.

Before submitting, run:

```bash
python tests/test_package_imports.py
python tests/test_core_spec.py
python tests/test_core_plan.py
python tests/test_parity_mot.py
git diff --check
```

## Reporting results

Differences between training runs of the same recipe can be as large as differences between methods, so:

- Use at least two training seeds, and compare methods on the same evaluation episodes.
- State the commit, config, simulator version and number of episodes.
- Keep simulator versions fixed within a comparison. LIBERO success rates depend on the MuJoCo version.

## 中文

## 开始之前

小修改以外的工作，请先开 issue，把做法说好再动手。多塔核心还在写，新的信息来源除非已经讨论过，否则先等它完成。

## 提交 Pull Request

- 每次改动聚焦一件事，在分支上做，不要直接改 `main`。
- 用户能感知的行为有变化时，中英文文档一起更新。
- 不要提交数据集、权重、生成结果、密钥和只在自己机器上有效的路径。
- 在 Pull Request 里写清楚改了什么、怎么验证的、哪些没验证。

提交前运行：

```bash
python tests/test_package_imports.py
python tests/test_core_spec.py
python tests/test_core_plan.py
python tests/test_parity_mot.py
git diff --check
```

## 报告结果

同一配方不同训练种子的差距，可能和不同方法之间的差距一样大，所以：

- 至少用两颗训练种子，并在相同的评测局上比较不同方法。
- 写明提交号、配置、仿真器版本和评测局数。
- 同一组比较里仿真器版本要固定。LIBERO 的成功率和 MuJoCo 版本有关。
