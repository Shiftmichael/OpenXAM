# Experiment Plan

[中文](experiments_zh.md) | [Documentation index](README.md) | [Back to project README](../README.md)

Several people now run benchmark experiments with OpenXAM at the same time. This page sets out what the first round covers, how much each evaluation runs, and where checkpoints and results go. Who runs what is tracked in the [status section](../README.md#status) of the project README.

## Round 1

Round 1 aims to run every benchmark end to end in OpenXAM with the Wan2.2 structures, and to give each benchmark a first number of our own under its official protocol. Where EasyWAM released weights, we evaluate those too, so our numbers have a reference.

Each benchmark has one owner. LIBERO-Plus and LIBERO-Pro evaluate checkpoints trained on LIBERO, so one person owns all three.

| Benchmark | Starts from | Runs | Compare with |
| --- | --- | --- | --- |
| LIBERO, LIBERO-Plus, LIBERO-Pro | EasyWAM's released LIBERO weights; Wan2.2-TI2V-5B for retraining | Evaluate the five released Wan2.2 checkpoints on LIBERO, and the released MoT on LIBERO-Plus and LIBERO-Pro; train MoT on LIBERO with two seeds | EasyWAM's [published numbers](results/result.md) |
| RoboTwin 2.0 | Wan2.2-TI2V-5B | Train MoT with two seeds and evaluate both | Baselines published with the benchmark |
| RoboDojo | Wan2.2-TI2V-5B | Same | Same |
| RoboCasa365 | Wan2.2-TI2V-5B | Same | Same |
| VLABench | Wan2.2-TI2V-5B | Same | Same |

That is 10 training runs and 18 evaluations in total.

The first LIBERO evaluation runs the released MoT checkpoint twice: through the EasyWAM task `libero_easywam_mot_wan22` and through the new core with `libero_xam_fastwam_wan22`. The two should agree, and this is also the new core's first run with real weights. The other benchmarks have no tasks on the new core yet, so they use their `<benchmark>_easywam_mot_wan22` tasks for now.

MoT comes first because it is EasyWAM's main structure and has released weights. Later rounds extend each benchmark to the other structures, LoRA and Cosmos-Predict2.5, in an order chosen after round 1.

## How much each evaluation runs

Use each benchmark's official protocol as its evaluator implements it. The [benchmark guides](instructions/benchmark/README.md) explain the setup.

| Benchmark | Protocol | Episodes | Simulator |
| --- | --- | ---: | --- |
| LIBERO | 4 suites × 10 tasks × 50 trials | 2,000 | MuJoCo 3.3.2, robosuite 1.4.0 |
| LIBERO-Plus | 10,030 perturbed tasks, one episode each | 10,030 | LIBERO-Plus fork in its own environment |
| LIBERO-Pro | 200 tasks in 20 suite/perturbation cells × 50 trials | 10,000 | LIBERO-Pro checkout |
| RoboTwin 2.0 | 50 tasks × clean and randomized × 100 episodes | 10,000 | RoboTwin at `96c1fea` |
| RoboDojo | 42 tasks × seeds 0, 1, 2 × 50 episodes | 6,300 | Isaac Sim in the RoboDojo environment |
| RoboCasa365 | Human300: 50 tasks × 50 episodes | 2,500 | RoboCasa v1.0.1 or newer; record the robosuite commit |
| VLABench | Tracks 1, 2, 3, 4 and 6, first 10 configurations per task | 500 | MuJoCo 3.2.2 |

A number goes into the results table only if:

- It comes from the final checkpoint. Do not choose a checkpoint by its benchmark score. If another step is wanted, fix it before evaluating and use it for every run in the comparison.
- It is reported for two training seeds, `seed=0` and `seed=1`, with both values and their mean. One seed is enough to check that the pipeline works.
- It covers the full protocol. Subsets are fine for debugging.
- It lists the simulator version, commit, task and command-line overrides. Keep the simulator version fixed within a comparison.

## Names and storage

Training writes to `runs/<task>/<run-id>/`. Name runs `<yyyymmdd>-<github-handle>-s<seed>`, and log them to the lab's shared Weights & Biases team so everyone sees the curves:

```bash
RUN_ID=20261118-yourname-s0 NPROC_PER_NODE=8 bash scripts/train_zero1.sh \
  task=robotwin_easywam_mot_wan22 seed=0 \
  wandb.enabled=true wandb.workspace=<lab-team> wandb.project=openxam \
  wandb.group=robotwin_easywam_mot_wan22 wandb.name=20261118-yourname-s0
```

| What | Where | What to keep |
| --- | --- | --- |
| Checkpoints during training | `runs/` on the cluster you train on | The trainer keeps the last three (`checkpoint_save_limit: 3`) |
| Checkpoints behind a reported number | A private model repository in the lab's Hugging Face organization, named `<task>-<run-id>` | Final weights, `dataset_stats.json`, the resolved config and the evaluation summary |
| Evaluation outputs | `evaluate_results/<benchmark>/...` on the cluster | `summary.json` and `summary.csv`, also posted in the run's issue |
| EasyWAM's released weights | Downloaded once per cluster from `OpenMOSS-Team/EasyWAM-<structure>-Wan22` on the Hub | No second copy |

A full Wan2.2 checkpoint takes 10–12 GB, the size of EasyWAM's released files, and the training states kept for resuming take more, so check your storage quota before starting. Once a run is evaluated, delete its training states and intermediate checkpoints. Making a checkpoint public is a separate decision. Never commit checkpoints, datasets, evaluation outputs or machine paths to this repository.

## Tracking a run

1. Before starting, open an issue with the **Experiment** form, and mark your cell in the README status table as running (🏃), linked to the issue.
2. While it runs, post progress and problems in the issue.
3. When it is done, post the evaluation summary in the issue. Then, in one pull request, add the numbers to the results table below and mark the README cell done (✅ with the score).

## Results

Numbers produced in OpenXAM. EasyWAM's own numbers are in [results/result.md](results/result.md). No results yet.

| Benchmark | Structure | Starts from | Seeds | Score | Commit | Issue |
| --- | --- | --- | --- | --- | --- | --- |

## What stays out of this repository

This repository is public, and so is every branch and fork of it. Ideas, configs and results that are not published yet belong in a private repository that installs OpenXAM, and runs for such work are tracked privately as well. Open a pull request here once they are published.
