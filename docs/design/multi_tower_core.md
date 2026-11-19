# OpenXAM Multi-Tower Core: Interface Draft

[中文](multi_tower_core_zh.md) | [Documentation index](../README.md) | [Back to project README](../../README.md)

Status: config checking and compilation are implemented. All five EasyWAM structures (MoT, MoT-Joint, MoT-IDM, Unified and Hidden) run on this core with the Wan2.2 backbone and go through the trainer. MoT has been checked on a GPU with EasyWAM's released weights; the other four structures still need checking with real weights on a GPU.

## 1. What this solves

Today each structure is a class. EasyWAM's five structures plus the joint attention come to about 5,200 lines, and the attention rules, the noising and the inference loop are hard-coded in each class's own methods. Adding a kind of information, or changing who reads whom, means writing another class.

The core is meant to do three things:

- **Change the structure by changing only the config**, so VLAs and WAMs share one codebase.
- **Add an information source by adding only a tower or a token stream**, without touching the trainer or the evaluators.
- **Report a config error before any GPU time is used.**

The trainer and the benchmark evaluators stay as they are. The new model offers the same methods as today, so they need no changes.

## 2. Five terms

| Term | Name in the config | Meaning |
| --- | --- | --- |
| Stream | `streams` | One kind of token, such as the current frame, future video, actions or robot state |
| Tower | `towers` | A group of parameters with its pretrained weights. One tower can carry several streams |
| Read table | `reads` | Which streams each stream may read, and whether training signal flows back through that read |
| Noise plan | `train`, `infer` | Whether each stream is clean, noisy or absent, in training and at inference |
| Losses | `losses` | Which streams are supervised, and with what objective |

A model's information enters at three moments, which match these settings: pretrained weights are in the **towers**, the training signal is in the **losses**, and the inputs at inference are in the **read table and the noise plan**.

## 3. A complete example: FastWAM

```yaml
# example: fastwam
towers:
  video:  {kind: wan22, train: full, context: [text]}
  action: {kind: action_dit, hidden_dim: 1024, init: from_video, train: full, context: [text]}

streams:
  text:   {encoder: umt5}
  obs:    {tower: video,  data: first_frame}
  future: {tower: video,  data: future_frames}
  action: {tower: action, data: action}
  state:  {tower: action, data: proprio}

reads:
  obs:    [obs]
  future: [obs, future]
  action: [obs, action, state]
  state:  [obs, action, state]

train:
  - {name: joint, noisy: [future, action]}

infer:
  act:
    - {denoise: [action], absent: [future], steps: 10}
  act_and_imagine:
    - {denoise: [future, action], steps: 10}

losses:
  - {stream: action, type: flow, weight: 1.0}
  - {stream: future, type: flow, weight: 1.0}
```

Section by section:

- **`towers`**: `train` is `full`, `lora` or `frozen`. `context` is the fixed sequence that each layer of the tower reads through cross-attention. `init` decides how the action tower is initialized: `from_video` is the current approach, derived from the video tower's weights; `random` is random initialization. The repository's configs that reproduce the EasyWAM MoT family also set `noise_first_frame: true`: the current frame also draws a sample of noise, which is then discarded, as in the old classes. It only changes where the random numbers fall, so new configs don't need it.
- **`streams`**: `tower` says which tower a stream goes through. A stream without a `tower` (here `text`) is only a fixed sequence.
- **`reads`**: one line per reader. Here actions read the current frame and not the future video, which is exactly what defines FastWAM. The future video does not read actions.
- **`train`**: one entry per training mode. Streams in `noisy` get noise; all others are clean. Each stream draws its noise level on its own; to share one, write `tie`.
- **`infer`**: named inference programs, each made of one or more stages. In `act` the future video is absent and only actions are denoised; `act_and_imagine` also generates the future video, to show what the model imagined.
- **`losses`**: `flow` is the current flow-matching loss.

Each stream's noise distribution still comes from the existing schedulers (`video_scheduler`, `action_scheduler`), which are not shown here.

A read can also be written in full form:

```text
{from: future, grad: false, pattern: frame_causal}
```

- `grad: false`: no training signal flows back through the read, which means this stream's K and V are detached from the graph.
- `pattern`: `full` reads everything, `frame_causal` lets each frame read only itself and earlier frames, and `causal` is token-by-token causal attention as in a language model.

## 4. How other structures are written

Below, only the parts that differ from the previous example are described.

### VLA, the π0 family

```yaml
# example: vla
towers:
  vlm:    {kind: qwen3_vl, train: frozen}
  action: {kind: action_dit, hidden_dim: 1024, init: random, train: full}

streams:
  prompt: {tower: vlm,    data: [first_frame, instruction]}
  action: {tower: action, data: action}
  state:  {tower: action, data: proprio}

reads:
  prompt: [{from: prompt, pattern: causal}]
  action: [prompt, action, state]
  state:  [prompt, action, state]

train:
  - {name: act, noisy: [action]}

infer:
  act:
    - {denoise: [action], steps: 10}

losses:
  - {stream: action, type: flow}
```

To train the VLM without the action signal flowing back into it ("knowledge insulation"), make the VLM trainable and change the read to:

```text
action: [{from: prompt, grad: false}, action, state]
```

### JAM: predict future features in training, drop them at inference

The only differences from the VLA are a trainable backbone and one more loss:

```yaml
# example: jam
towers:
  vlm:    {kind: qwen3_vl, train: full}
  action: {kind: action_dit, hidden_dim: 1024, init: random, train: full}

streams:
  prompt: {tower: vlm,    data: [first_frame, instruction]}
  action: {tower: action, data: action}
  state:  {tower: action, data: proprio}

reads:
  prompt: [{from: prompt, pattern: causal}]
  action: [prompt, action, state]
  state:  [prompt, action, state]

train:
  - {name: act, noisy: [action]}

infer:
  act:
    - {denoise: [action], steps: 10}

losses:
  - {stream: action, type: flow}
  - {stream: prompt, type: feature, encoder: dinov2, target: future_frames, training_only: true}
```

### Denoising video and actions together

EasyWAM-MoT-Joint: actions read the future video; the future video does not read actions.

```yaml
# example: mot_joint
towers:
  video:  {kind: wan22, train: full, context: [text]}
  action: {kind: action_dit, hidden_dim: 1024, init: from_video, train: full, context: [text]}

streams:
  text:   {encoder: umt5}
  obs:    {tower: video,  data: first_frame}
  future: {tower: video,  data: future_frames}
  action: {tower: action, data: action}
  state:  {tower: action, data: proprio}

reads:
  obs:    [obs]
  future: [obs, future]
  action: [obs, future, action, state]
  state:  [obs, future, action, state]

train:
  - {name: joint, noisy: [future, action]}

infer:
  act_and_imagine:
    - {denoise: [future, action], steps: 10}

losses:
  - {stream: action, type: flow}
  - {stream: future, type: flow}
```

OpenWAM-α's mutual reading only adds actions to the future video's line:

```text
future: [obs, future, action, state]
```

### Predicting the future from given actions

Add a training mode and an inference program to the model above. When `reads` is written inside an entry, it overrides only the lines it lists, and only for that entry.

```yaml
# example: predict_mix
towers:
  video:  {kind: wan22, train: full, context: [text]}
  action: {kind: action_dit, hidden_dim: 1024, init: from_video, train: full, context: [text]}

streams:
  text:   {encoder: umt5}
  obs:    {tower: video,  data: first_frame}
  future: {tower: video,  data: future_frames}
  action: {tower: action, data: action}
  state:  {tower: action, data: proprio}

reads:
  obs:    [obs]
  future: [obs, future]
  action: [obs, future, action, state]
  state:  [obs, future, action, state]

train:
  - {name: joint, weight: 0.5, noisy: [future, action]}
  - name: predict
    weight: 0.5
    noisy: [future]
    reads:
      future: [obs, future, action, state]
      action: [obs, action, state]
      state:  [obs, action, state]

infer:
  act_and_imagine:
    - {denoise: [future, action], steps: 10}
  predict:
    - denoise: [future]
      steps: 10
      reads:
        future: [obs, future, action, state]
        action: [obs, action, state]
        state:  [obs, action, state]

losses:
  - {stream: action, type: flow}
  - {stream: future, type: flow}
```

### Generate video first, then actions

EasyWAM-MoT-IDM carries two copies of the video: one to learn generation, and one for actions to read. The second copy is clean half the time and noisy the other half. Inference has two stages, and the first stage's result is handed to the second through `use`.

```yaml
# example: mot_idm
towers:
  video:  {kind: wan22, train: full, context: [text]}
  action: {kind: action_dit, hidden_dim: 1024, init: from_video, train: full, context: [text]}

streams:
  text:     {encoder: umt5}
  obs:      {tower: video,  data: first_frame}
  future:   {tower: video,  data: future_frames}
  obs_c:    {tower: video,  data: first_frame}
  future_c: {tower: video,  data: future_frames}
  action:   {tower: action, data: action}
  state:    {tower: action, data: proprio}

reads:
  obs:      [obs]
  future:   [obs, future]
  obs_c:    [obs_c]
  future_c: [obs_c, future_c]
  action:   [obs_c, future_c, action, state]
  state:    [obs_c, future_c, action, state]

train:
  - {name: teacher_forcing, noisy: [future, action], sometimes_noisy: {future_c: 0.5}}

infer:
  act_and_imagine:
    - {denoise: [future], absent: [obs_c, future_c, action, state], steps: 10}
    - {denoise: [action], absent: [obs, future], use: {future_c: future}, steps: 10}

losses:
  - {stream: action, type: flow}
  - {stream: future, type: flow}
```

### Reading the video tower's intermediate features

EasyWAM-Hidden: the action tower reads the video tower's layer-17 features through cross-attention, with no training signal flowing back. At inference the future video is not denoised; it stays at pure noise and runs once, which `hold` expresses. The video tower only needs to run up to layer 17. `act_and_imagine` also generates video, and actions still read the features from the first step: within a stage, features from another tower are read only once, at the first step.

```yaml
# example: hidden
towers:
  video:  {kind: wan22, train: full, context: [text]}
  action:
    kind: action_dit
    hidden_dim: 1024
    init: from_video
    train: full
    context: [{from: video, layer: 17, grad: false}]

streams:
  text:   {encoder: umt5}
  obs:    {tower: video,  data: first_frame}
  future: {tower: video,  data: future_frames}
  action: {tower: action, data: action}
  state:  {tower: action, data: proprio}

reads:
  obs:    [obs]
  future: [obs, future]
  action: [action, state]
  state:  [action, state]

train:
  - {name: joint, noisy: [future, action]}

infer:
  act:
    - {denoise: [action], hold: [future], steps: 10}
  act_and_imagine:
    - {denoise: [future, action], steps: 10}

losses:
  - {stream: action, type: flow}
  - {stream: future, type: flow}
```

### A single tower

EasyWAM-Unified: actions and state also go through the video tower, and the future video and actions share one noise level.

```yaml
# example: unified
towers:
  video: {kind: wan22, train: full, context: [text]}

streams:
  text:   {encoder: umt5}
  obs:    {tower: video, data: first_frame}
  future: {tower: video, data: future_frames}
  action: {tower: video, data: action}
  state:  {tower: video, data: proprio}

reads:
  obs:    [obs, state]
  future: [obs, future, action, state]
  action: [obs, future, action, state]
  state:  [obs, future, action, state]

train:
  - {name: joint, noisy: [future, action], tie: [[future, action]]}

infer:
  act_and_imagine:
    - {denoise: [future, action], tie: [[future, action]], steps: 10}

losses:
  - {stream: action, type: flow}
  - {stream: future, type: flow}
```

### Three streams: code, video, actions

AW-0's information flows code → video → actions.

```yaml
# example: aw0
towers:
  code:   {kind: masked_lm, train: full}
  video:  {kind: cosmos25, train: full, context: [text]}
  action: {kind: action_dit, hidden_dim: 1024, init: from_video, train: full, context: [text]}

streams:
  text:   {encoder: cosmos_reason}
  code:   {tower: code,   data: [first_frame, instruction, program]}
  obs:    {tower: video,  data: first_frame}
  future: {tower: video,  data: future_frames}
  action: {tower: action, data: action}
  state:  {tower: action, data: proprio}

reads:
  code:   [code]
  obs:    [code, obs]
  future: [code, obs, future]
  action: [code, obs, future, action, state]
  state:  [code, obs, future, action, state]

train:
  - {name: joint, noisy: [code, future, action]}

infer:
  act_and_imagine:
    - {denoise: [code, future, action], steps: 4}

losses:
  - {stream: code, type: masked}
  - {stream: action, type: flow}
  - {stream: future, type: flow}
```

## 5. Python interface

```text
src/openxam/core/spec.py     Config data structures and checks; no PyTorch dependency
src/openxam/core/plan.py     Compiles a config into an execution plan; no PyTorch dependency
src/openxam/core/driver.py   Runs layer by layer
src/openxam/core/model.py    XAMModel: the entry point the trainer and evaluators call
src/openxam/towers/          One file per tower kind: base, dit, wan, action, unified
```

### Towers

A tower provides these methods. The code in `MoT._build_expert_attention_io` and `_apply_post_block` that branched on the tower type has moved into `towers/dit.py`.

```python
class Tower(nn.Module):
    num_layers: int
    num_heads: int
    head_dim: int

    # Training
    def prepare(self, sample) -> dict: ...                 # raw batch to clean inputs
    def add_noise(self, batch, plan, timesteps) -> None: ...
    def add_optional_noise(self, batch, plan) -> None: ... # streams that are only sometimes noisy, after the rest of the noise
    def embed(self, batch, context) -> TowerState: ...     # inputs to tokens

    # Each layer
    def qkv(self, layer, x, state): ...                    # before attention: q, k, v
    def finish(self, layer, attn_out, carry, state): ...   # after attention: output projection, cross-attention, feed-forward
    def run_block(self, layer, x, mask, state): ...        # when reading only itself, run the whole layer at once

    # Finishing
    def readout(self, stream, x, state) -> Tensor: ...
    def flow_loss(self, stream, prediction, batch) -> Tensor: ...

    # Other towers reading this tower's features
    def project_features(self, features) -> Tensor: ...    # reader side: turn features into cross-attention inputs
    def feature_mask(self, batch, state): ...              # side being read: which tokens may be read

    # Inference
    def start(self, stream, request) -> Tensor: ...        # starting point for denoising
    def embed_step(self, plan, request, latents, timesteps) -> TowerState: ...

    # Saving weights
    def checkpoint_parts(self) -> dict: ...                # the modules this tower saves, and their names
```

A video tower can run several video segments at once, each a current-frame stream plus a future-video stream, such as IDM's two copies of the video.

A tower that reads only itself (such as a VLM) can be simpler. It doesn't need to alternate layer by layer with other towers: it runs once and hands over each layer's K and V. For a frozen VLM, that is the `past_key_values` a Hugging Face model returns, with no change to its internals.

### Execution plan

```python
spec = ModelSpec.load("configs/xam/fastwam_wan22.yaml")   # all checks happen here
plans = plan_infer(spec, "act")                           # one plan per stage
```

A plan contains:

- **Token layout**: where each stream sits in the sequence.
- **Attention structure**: `plan.segments(...)` maps directly to the current `AttentionSegment`. When a read has `grad: false`, the segment records which ranges to detach, and attention runs segment by segment at run time.
- **Cacheable towers**: a clean stream is computed only once in a stage if every stream it reads is also unchanged. When all of a tower's streams are unchanged, the whole tower runs only once. Today's hand-written `prefill_video_cache` is derived from this rule.
- **Tower groups and order**: towers that read each other go in one group and alternate layer by layer within it. A tower that reads another tower's features forms its own group, placed after the tower it reads. Hidden runs the video tower first, then the action tower.

### Running layer by layer

`run_layers` in `core/driver.py` does three things per layer: each tower computes q, k and v, one joint attention runs over them, and each tower finishes the layer.

When a tower reads only itself, as with Unified's single tower, Hidden's two towers and IDM's video-generation stage, the whole layer runs at once instead of being split in two. The two ways differ in the rotary position encoding: split, it is computed in float32; as a whole layer, in the model's precision, so results differ under bfloat16. This choice matches what each EasyWAM class does.

### Model entry point

`XAMModel` provides the same entry points as the EasyWAM classes: `training_loss`, `infer_action_batch`, `infer_joint_batch`, `infer_action`, `infer_joint`, `infer`, `encode_prompt`, `save_checkpoint` and `load_checkpoint`, with the same attribute names (`dit`, `video_dit`, `state_encoder`, `vae` and so on). So the trainer and the evaluators need no changes.

The weight format is unchanged too: weights saved by either side can be read by the other.

Writing `_target_: openxam.runtime.create_xam` and `spec: fastwam_wan22` in a config uses the new core. All five structures have ready LIBERO tasks: `task=libero_xam_<structure>_wan22`, where the structure is `fastwam`, `mot_joint`, `mot_idm`, `unified` or `hidden`.

When towers read each other, weights are saved under `mot`, as in the EasyWAM MoT family. When they don't, weights are saved under `dit`, as in Unified and Hidden. A model that can only produce actions while imagining, such as MoT-Joint, has no separate `act` program and runs `act_and_imagine` to produce actions.

To cut a read at test time, write `cut: ["action<-future"]` in a stage. The `predict_future` entry point for predicting the future from given actions has not been added yet.

## 6. Config checks

These errors are reported when the config loads, without waiting for a GPU:

| Check | Example |
| --- | --- |
| All names exist | A misspelled stream name in the read table |
| Every stream has a read line | A stream with no line saying what it reads |
| Two towers that read each other jointly have the same number of layers and heads | A 30-layer Wan video tower and a 28-layer VLM reading each other |
| Losses match the noise | A stream with a `flow` loss is never noised in any training mode, so the loss is always 0 |
| Denoised streams were trained | Inference denoises a stream that no training mode ever noised |
| Inference never reads a stream in a condition (clean, noisy or absent) it was not in during training | In training, actions read a noisy future video; at inference the future video is absent |
| The read table at inference matches some training mode | At inference the future video reads actions, but training never read that way |
| Towers that read features can be ordered | The action tower reads the video tower's features, and the video tower reads the action tower's features |
| A tower that reads features does not also read that tower jointly | The action tower reads video layer 17 and also reads the video in joint attention |
| A tower whose features are read is present | An inference stage marks all of the video tower's streams absent, but the action tower reads its features |

The last two catch a kind of error that is very hard to find: the code runs, but the scores are inexplicably low.

## 7. Migrating from the current code

| Now | After |
| --- | --- |
| `EasyWAMMoT` and the other four classes | Config files + `XAMModel` |
| `MoT.forward` | `core/driver.py` |
| `_build_expert_attention_io`, `_apply_post_block` | Each tower's `qkv` and `finish` |
| Each class's `_build_*_mask` | Compiled from the read table |
| `prefill_video_cache` | Cacheable streams in the plan |
| `video_attention_mask_mode` | Written into the read table |
| `runtime.create_easywam_*` | `create_model(spec)` |
| Trainer, evaluators | Unchanged |

Order: MoT → MoT-Joint → Unified → IDM → Hidden → VLM tower. The first five steps are done.

Each structure, once migrated, gets a parity test with very small random towers: the old and new paths use the same weights and inputs, and `training_loss` and the action outputs must match. This needs PyTorch but not a GPU. All five structures pass, but the old classes stay for now: GPU comparisons with real weights still need them, and FLUX still uses them.

## 8. Decisions made

1. **The read table is written per stream and is the same in every layer.** NUS OpenWAM has an option to connect only some layers; the first version does not support it.
2. **Towers that read each other jointly must have the same number of layers and heads.** Towers of different depths read features through cross-attention.
3. **The first VLM is Qwen3-VL-2B.** Real π0.5 weights need a separate tower kind and come later.
4. **The action tower's initialization is the config setting `init`.** Reproducing existing models keeps the current approach; fair comparisons all start from random.
5. **FLUX is not migrated for now.** Its block structure and training objective differ from the other two backbones. The first version covers only Wan and Cosmos, and FLUX stays on the old classes for now.
6. **Each minibatch uses one training mode.** The mode is drawn at random in proportion to its `weight`, so the attention structure is fixed within a minibatch.

## 9. How far this has been checked

**Checked:**

- **Structure**: the attention structures compiled from the read table match EasyWAM's MoT, MoT-Joint, IDM and Unified segment by segment, and match NUS OpenWAM's four visibility modes bit for bit (`tests/test_core_plan.py`). For this comparison, the existing logic was re-implemented from its source code, so the tests don't need PyTorch.
- **Config checks**: every check in section 6 has a test, and each one is confirmed to raise an error on a wrong config (`tests/test_core_spec.py`).
- **Numbers for the five structures**: with very small random towers, the same set of modules is wired into the old class and into `XAMModel`; the loss, all gradients, the sampled actions and the generated video are bitwise identical (`tests/test_parity_mot.py`, `tests/test_parity_structures.py`). They are also identical with gradient checkpointing or the cross-attention cache turned on, with the other two video visibility modes, in bfloat16, and with compiled inference steps.
- **These comparisons are effective**: changing the read table or the layer Hidden reads changes the results. Deliberately splitting every tower makes IDM, Unified and Hidden mismatch; deliberately not drawing noise for the current frame in the MoT family makes their training mismatch.
- **Weights**: after old and new each read the other's saved weights, the actions are bitwise identical; reading another structure's weights raises an error. Parameter names match the old classes one for one.
- **Trainer**: the real `EasyWAMTrainer` trains all five structures on a CPU, saves weights and reads them back (`tests/test_trainer_smoke.py`). MoT was also tested resuming after an interruption: weights, step count and optimizer are restored, and the following steps are not bitwise identical to an uninterrupted run, the same as with the old class.
- **Training after evaluation**: evaluation runs in inference mode. The previous cache passed position encodings computed there to later training steps, which then raised an error; the old classes have the same bug. The error can be reproduced by first producing actions from the current frame only, then generating video, then training. Now each mode keeps its own cache, tested on all five structures.
- **Hydra configs**: all five `task=libero_xam_*_wan22` compose, and are identical to the old tasks except for the model entry point. The settings moved into config files (which layer Hidden reads, IDM's noising probability) match the old configs (`tests/test_hydra_configs.py`).
- **`grad: false`**: detaching leaves the outputs unchanged, and the video tower receives no gradient from the action loss.
- **Real MoT weights**: with EasyWAM's released `EasyWAM-MoT-Wan22` weights, the old class and `XAMModel` ran on the same batch of real LIBERO samples on one A100. Both ways were tested: wiring the old class's modules into `XAMModel`, and building from the config and loading the weights itself. In bfloat16 the loss, all gradients and the sampled actions are bitwise identical, also with autocast on as the trainer does; the parameters `XAMModel` loads are bitwise identical to the old class's. In float32 the loss and actions are bitwise identical, and the gradients differ by as much as the old class differs from itself across two runs (`tests/gpu/check_real_weights.py`).
- **LIBERO evaluation**: `libero_easywam_mot_wan22` and `libero_xam_fastwam_wan22` both run `eval_libero_single.py` to the end with the same released weights. The new core's own T5 text encoding is bitwise identical to the old class's; when the two get bitwise identical observations, their output actions are also bitwise identical (`tests/gpu/trace_libero_eval.py`). The LIBERO rollouts themselves are not exactly repeatable: the wrist camera's rendering differs by one intensity level between runs, also for the old class run twice. So old-vs-new comparisons must look at success rates, not trajectories.

**Not verified yet:**

- Results for the other four structures (MoT-Joint, MoT-IDM, Unified and Hidden) with real weights on a GPU.
- Benchmark scores from the full protocol. The LIBERO evaluation above ran only one task of LIBERO-Spatial.
- Saving and loading LoRA. The test environment does not have `peft` installed.
- The Cosmos tower. `towers/dit.py` keeps its branch, but it is untested. The old Unified and Hidden classes zero the current-frame prediction before the solver step; this has no effect with Wan's solver but does with Cosmos's multistep solver. The Cosmos version of Unified also needs a shared scheduler.
- FLUX is still on the old classes.
- Implementation details of the VLM tower: aligning K and V head counts with the action tower, aligning position encodings, and how Qwen3-VL injects visual features into its first few layers.
