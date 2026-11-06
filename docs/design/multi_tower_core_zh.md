# OpenXAM 多塔核心：接口草案

状态：配置检查和编译已实现，MoT 的训练和动作推理已经跑在这套核心上。其余结构还在迁移。

## 1. 要解决什么

现在每种结构是一个类。EasyWAM 的五种结构加联合注意力，大约 5,200 行，里面的注意力规则、加噪方式、推理循环都写死在各自的方法里。想加一种信息，或者换一下谁读谁，就要再写一个类。

这个核心要做到三件事：

- **换结构只改配置**：VLA 和 WAM 用同一套代码。
- **加信息来源只加一座塔或一路 token**：不碰训练器和评测。
- **配置错了，在用 GPU 之前就报错**。

训练器和各基准的评测不动。新模型对外提供和现在一样的方法，所以它们不用改。

## 2. 五个词

| 词 | 配置里的名字 | 意思 |
| --- | --- | --- |
| 路 | `streams` | 一类 token，比如当前画面、未来视频、动作、机器人状态 |
| 塔 | `towers` | 一组参数和它的预训练权重。一座塔可以走几路 token |
| 读取表 | `reads` | 每一路能读哪几路，读的时候训练信号回不回传 |
| 噪声安排 | `train`、`infer` | 每一路在训练和推理时是干净、带噪，还是不存在 |
| 损失 | `losses` | 监督哪一路，用什么目标 |

模型的信息来源分三个时刻进入，正好对应这几项：预训练权重在**塔**里，训练信号在**损失**里，推理时的输入在**读取表和噪声安排**里。

## 3. 一个完整的例子：FastWAM

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

losses:
  - {stream: action, type: flow, weight: 1.0}
  - {stream: future, type: flow, weight: 1.0}
```

逐段说明：

- **`towers`**：`train` 取 `full`、`lora` 或 `frozen`。`context` 是这座塔每层交叉注意力读的固定序列。`init` 决定动作塔怎么初始化：`from_video` 是现在的做法，从视频塔权重推出来；`random` 是随机初始化。
- **`streams`**：`tower` 说明这一路走哪座塔。没有 `tower` 的路（这里的 `text`）只是一段固定序列。
- **`reads`**：一行一个读者。这里动作读当前画面，不读未来视频，这正是 FastWAM 的定义。未来视频不读动作。
- **`train`**：一种训练方式一项。`noisy` 里的路加噪，其余都是干净的。每路的噪声水平各自抽取，要共用就写 `tie`。
- **`infer`**：一个命名的推理程序，由一个或几个阶段组成。`act` 这个程序里，未来视频不存在，只去噪动作。
- **`losses`**：`flow` 是现在的流匹配损失。

每路的噪声分布继续用现有的调度器（`video_scheduler`、`action_scheduler`），这里没写出来。

读取还可以写成完整形式：

```text
{from: future, grad: false, pattern: frame_causal}
```

- `grad: false`：读的时候不回传训练信号，也就是把这一路的 K、V 从计算图上断开。
- `pattern`：`full` 是全读，`frame_causal` 是每帧只读自己和之前的帧，`causal` 是语言模型的逐 token 因果。

## 4. 其他结构怎么写

下面只写和上一个例子不同的部分。

### VLA，π0 这一类

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

VLM 训练时动作的信号不回传（"知识隔离"），只需把 VLM 改成可训练，再把读取改成：

```text
action: [{from: prompt, grad: false}, action, state]
```

### JAM：训练时预测未来特征，推理时丢掉

和 VLA 的区别只在骨干可训练，以及多一项损失：

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

### 视频和动作一起去噪

EasyWAM-MoT-Joint：动作读未来视频，未来视频不读动作。

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

OpenWAM-α 的互读，只是未来视频那一行多读动作：

```text
future: [obs, future, action, state]
```

### 给定动作预测未来

在上面的模型上加一种训练方式和一个推理程序。`reads` 写在某一项里时，只替换这一项用到的行。

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

### 先生成视频，再出动作

EasyWAM-MoT-IDM 同时放两份视频：一份用来学生成，一份给动作读。第二份一半时间干净、一半时间带噪。推理分两个阶段，第一阶段的结果通过 `use` 交给第二阶段。

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

### 读视频塔的中间特征

EasyWAM-Hidden：动作塔通过交叉注意力读视频塔第 17 层的特征，不回传训练信号。推理时未来视频不去噪，停在纯噪声上跑一次，用 `hold` 表示。

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

losses:
  - {stream: action, type: flow}
  - {stream: future, type: flow}
```

### 单塔

EasyWAM-Unified：动作和状态也走视频塔，两路共用一个噪声水平。

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

### 三路：代码、视频、动作

AW-0 的信息顺序是代码→视频→动作。

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

## 5. Python 接口

```text
src/openxam/core/spec.py     配置的数据结构和检查，不依赖 PyTorch
src/openxam/core/plan.py     把配置编译成执行计划，不依赖 PyTorch
src/openxam/core/driver.py   逐层运行
src/openxam/core/model.py    XAMModel：训练器和评测调用的入口
src/openxam/towers/          每种塔一个文件：base、dit、wan、action
```

### 塔

一座塔要提供这些方法。现在 `MoT._build_expert_attention_io` 和 `_apply_post_block` 里按塔类型分支的代码，都搬进了 `towers/dit.py`。

```python
class Tower(nn.Module):
    num_layers: int
    num_heads: int
    head_dim: int

    # 训练
    def prepare(self, sample) -> dict: ...                 # 原始批次变成干净输入
    def add_noise(self, batch, plan, timesteps) -> None: ...
    def embed(self, batch, context) -> TowerState: ...     # 输入变成 token

    # 每一层
    def qkv(self, layer, x, state): ...                    # 注意力之前：q、k、v
    def finish(self, layer, attn_out, carry, state): ...   # 注意力之后：输出投影、交叉注意力、前馈

    # 收尾
    def readout(self, stream, x, state) -> Tensor: ...
    def flow_loss(self, stream, prediction, batch) -> Tensor: ...

    # 推理
    def start(self, stream, request) -> Tensor: ...        # 去噪的起点
    def embed_step(self, request, latents, timesteps) -> TowerState: ...
```

只读自己的塔（比如 VLM）可以更简单。它不用和别的塔逐层交替，一次跑完，交出每层的 K、V 就行。对冻结的 VLM，这就是 Hugging Face 模型返回的 `past_key_values`，不用改它的内部。

### 执行计划

```python
spec = ModelSpec.load("configs/xam/fastwam_wan22.yaml")   # 这一步做所有检查
plans = plan_infer(spec, "act")                           # 每个阶段一份计划
```

计划里有：

- **token 的排列**：每一路在序列里的位置。
- **注意力结构**：`plan.segments(...)` 直接对应现在的 `AttentionSegment`。有 `grad: false` 的读取时，段里记下哪些范围要断开，运行时逐段做注意力。
- **可以缓存的塔**：干净的路，如果它读的路也都不变，这一阶段只算一次。一座塔的路全部不变时，整座塔只跑一次。现在手写的 `prefill_video_cache` 由这条规则推出来。

### 逐层运行

`core/driver.py` 里的 `run_layers` 每层做三件事：各塔算 q、k、v，合起来做一次注意力，各塔收尾。

### 模型入口

`XAMModel` 现在提供 `training_loss` 和 `infer_action_batch`，和旧类接口一致。`infer_joint_batch`、`save_checkpoint`、`load_checkpoint`、`.dit` 和新方法 `predict_future` 还要补，补齐后训练器和评测才能直接用它。

测试时切断某一路读取，只要在阶段里写 `cut: ["action<-future"]`。

## 6. 配置检查

下面这些错误在加载配置时就报，不用等到 GPU 上：

| 检查 | 例子 |
| --- | --- |
| 名字都存在 | 读取表里写错了路名 |
| 每一路都有读取行 | 某一路没写谁能读 |
| 联合读取的两座塔层数和头数相同 | Wan 30 层的视频塔和 28 层的 VLM 互读 |
| 损失对得上噪声 | `flow` 损失的路在所有训练方式里都没加噪，这项损失永远是 0 |
| 去噪的路训练过 | 推理要去噪一路，但没有任何训练方式给它加过噪 |
| 推理不读训练没见过的状态 | 训练时动作读带噪的未来视频，推理时未来视频不存在 |
| 读取表在推理时和某种训练方式一致 | 推理时未来视频读动作，但训练时从没这样读过 |

最后两条能防住一类很难查的错误：代码能跑，分数却莫名其妙地低。

## 7. 从现有代码迁移

| 现在 | 之后 |
| --- | --- |
| `EasyWAMMoT` 等五个类 | 配置文件 + `XAMModel` |
| `MoT.forward` | `core/driver.py` |
| `_build_expert_attention_io`、`_apply_post_block` | 各塔的 `qkv`、`finish` |
| 各类的 `_build_*_mask` | 读取表编译 |
| `prefill_video_cache` | 计划里的可缓存路 |
| `video_attention_mask_mode` | 写进读取表 |
| `runtime.create_easywam_*` | `create_model(spec)` |
| 训练器、评测 | 不变 |

顺序：MoT → MoT-Joint → Unified → IDM → Hidden → VLM 塔。

每迁移一种结构，用很小的随机塔做一次一致性测试：新旧两条路径用相同的权重和输入，`training_loss` 和动作输出要一致。全部通过后才删旧类。这一步需要 PyTorch，但不需要 GPU。

## 8. 已定的决定

1. **读取表按路写，所有层相同。** NUS OpenWAM 有只在部分层连接的选项，第一版不支持。
2. **联合读取的塔层数和头数必须相同。** 不同深度的走交叉注意力读特征。
3. **第一个 VLM 用 Qwen3-VL-2B。** 真正的 π0.5 权重要单独实现一种塔，放到后面。
4. **动作塔的初始化是配置项 `init`。** 复现已有模型时沿用现在的做法，做公平比较时都从随机开始。
5. **FLUX 暂不迁移。** 它的块结构和训练目标都和另外两种骨干不同，第一版只做 Wan 和 Cosmos，FLUX 暂时留在旧类里。
6. **每个小批只用一种训练方式。** 按权重抽取，这样注意力结构在一个小批内固定。

## 9. 核对到什么程度

**已核对：**

- **结构**：读取表编译出的注意力结构，和 EasyWAM 的 MoT、MoT-Joint、IDM、Unified 逐段一致，也和 NUS OpenWAM 的四种可见方式逐位一致（`tests/test_core_plan.py`）。这里的现有逻辑是照源码重新写的，这样测试不需要 PyTorch。
- **配置检查**：第 6 节的检查都有测试，每一项都用写错的配置确认能报错（`tests/test_core_spec.py`）。
- **MoT 数值**：用很小的随机塔，把同一批模块接到旧类和 `XAMModel` 上，loss、全部梯度和采样出的动作都逐位相同，打开梯度检查点时也相同（`tests/test_parity_mot.py`）。把读取表改掉时，结果确实会变，所以这个比较有效。
- **`grad: false`**：断开后输出不变，视频塔收不到动作损失的梯度。

**还没有验证：**

- 真实权重和 GPU 上的结果。上面都是 CPU 上的小模型。
- Cosmos 塔。`towers/dit.py` 保留了它的分支，但还没有测试。
- MoT 以外的结构，`infer_joint_batch`，以及文字编码和存取权重。
- VLM 塔的实现细节：K、V 头数和动作塔的对齐，位置编码怎么对齐，Qwen3-VL 把视觉特征注入前几层的做法。
