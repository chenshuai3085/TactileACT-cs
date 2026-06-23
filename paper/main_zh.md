# Feel Before Act: 基于触觉预见能量的扩散策略梯度引导

## 作者

陈帅  
小米机器人实验室  
chenshuai@xiaomi.com

---

## 摘要

接触丰富操作要求机器人不仅生成几何上合理的动作，还要在动作执行前预判其未来接触后果。插插座需要避免插入前的侧向碰撞和 bounce；擦黑板、擦花瓶、刷卡和粉笔写字等连续接触任务则要求接触力大小合适、变化柔顺且持续稳定。现有视觉或视触觉扩散策略通常把触觉作为当前条件输入，缺少一种机制来回答：当前正在去噪的候选动作未来会产生怎样的触觉后果，以及这个后果是否好。

本文提出 **Proactive Tactile Guidance (PTG)**，一种面向扩散策略的触觉后果评分与梯度引导框架。PTG 由三部分组成：首先，基础 Diffusion Policy 根据视觉、本体和当前触觉观测生成未来动作 chunk；其次，多步 Tactile Foresight 根据当前触觉历史和候选未来动作预测未来多帧触觉 latent 序列；最后，可微 TacQuality Energy Scorer 对预测的未来触觉后果打分，并将质量分数对动作的梯度注入扩散去噪过程。与后生成 reranking 不同，PTG 直接在去噪后期修正 action sample，使动作朝向“未来触觉更好”的方向移动。

我们的任务覆盖离散接触和连续接触两类场景：插插座、擦黑板、擦花瓶、刷卡和粉笔写字。本文以擦黑板作为当前主要例子：我们不手写力学公式来评分，而是用四类有语义的采集数据监督评分器，包括正常擦拭、压力太小、压力太大和压力不稳定。当前黑板实验表明，基于未来 16 步 action 和未来 16 步 tactile latent 的 latent energy scorer 能在 episode-level split 上区分这些触觉后果类别，并提供有限、稳定的 action 梯度。真实机器人上的最终改进仍需通过 baseline-vs-guided rollout 的力曲线和任务指标验证。

**关键词：** 触觉操作、扩散策略、触觉预见、classifier guidance、能量评分器、接触丰富任务

---

## 1. 引言

接触丰富操作的难点不只在于“到达目标位置”，更在于“以合适的接触方式到达”。插插座中，肉眼看似接近正确的动作可能在插入前产生侧向挤压，导致 bounce、卡住或反复 retry。擦黑板、擦花瓶、刷卡和粉笔写字则属于连续接触任务：力太小会擦不干净、刷卡失败或写不出字；力太大会损坏物体、折断粉笔或产生危险接触；力变化不平稳则会造成轨迹抖动和接触质量下降。

Diffusion Policy 等动作生成模型擅长从示范中学习多模态动作分布，但标准推理过程只根据当前观测生成动作，没有显式评估“这个动作会导致什么未来触觉”。触觉传感器虽然能提供接触后的反馈，但被动反应往往已经太晚。对于高精度接触任务，更理想的机制是：在动作仍处于扩散去噪过程中时，就预测该动作的未来触觉后果，并用质量评分器把梯度传回动作。

本文的核心观点是：**未来触觉后果可以作为动作质量的可微判别信号**。如果一个 candidate action 经过 Foresight 预测会导致 bounce、压力过小、压力过大或力振荡，则应在去噪中被压低；如果它预测会导致稳定、适中、平滑的接触，则应被提升。由此我们提出 Proactive Tactile Guidance (PTG)：用 action-conditioned tactile foresight 将“动作”和“未来触觉后果”连接起来，再用 TacQuality Energy Scorer 将“未来触觉后果”和“任务质量”连接起来，最终形成可对 action 反传的 guidance objective。

我们的贡献如下：

1. **触觉后果驱动的扩散策略梯度引导。** 我们提出一种不同于 reranking 的在线引导方式：在 DDPM/DDIM 去噪后期估计干净动作，预测其未来触觉，并最大化可微质量能量对 action sample 进行小步更新。
2. **多步 Tactile Foresight。** 我们将 Foresight 从单步终点预测扩展为未来 16 步 tactile latent 序列预测，使连续接触任务中的力变化过程、稳定性和振荡模式能够被评分器使用。
3. **任务质量能量评分器。** 我们使用任务后果标签训练可微评分器，而不是手写固定物理规则。以擦黑板为例，评分器学习区分正常擦拭、压力太小、压力太大和压力不稳定四类未来触觉后果，并把四类 logits 转成连续的 expert margin 作为引导分数。
4. **多任务接触操作故事线。** 我们统一讨论插插座、擦黑板、擦花瓶、刷卡、粉笔写字五类任务，将其归纳为离散接触风险规避和连续接触质量保持两类问题。

---

## 2. 相关工作

**扩散策略用于机器人操作。** Diffusion Policy 将动作 chunk 建模为条件去噪过程，在多模态操作任务中表现出较强的生成能力。ACT 通过 action chunking 缓解遥操作数据中的时序不确定性。本文沿用扩散策略作为动作先验，但不满足于只从示范分布采样，而是在推理时用触觉后果质量对去噪轨迹进行引导。

**触觉条件策略。** 许多视触觉策略将触觉作为当前观测输入，用于提升接触状态估计。这类方法通常是反应式的：触觉发生后再影响下一步动作。PTG 的不同点在于使用 action-conditioned Foresight 预测未来触觉，让模型在动作执行前就能评估接触后果。

**触觉世界模型与预见。** DreamTacVLA、ViTacFormer、Visual Foresight 等工作表明，预测未来观测可以为操作提供有用的中间表征。本文关注的不是把未来触觉作为额外条件输入，而是把未来触觉作为动作质量评分的核心证据。

**Classifier guidance 与能量引导。** 扩散模型中的 classifier guidance 通过分类器梯度改变生成过程。DynaGuide 使用视觉动力学模型引导扩散策略，TouchGuide 探索触觉引导去噪。PTG 与这些方法同属 inference-time guidance，但强调任务后果标签和多步触觉后果预测：评分器不是只判断当前观测是否合理，而是判断“当前 action 会导致的未来触觉是否好”。

---

## 3. 问题定义

在决策时刻 \(t\)，机器人观测为：

\[
o_t = \{I_t^{global}, I_t^{wrist}, q_t, m_{t-W+1:t}\},
\]

其中 \(I\) 是图像，\(q_t \in \mathbb{R}^7\) 是关节状态，\(m\) 是 GelSight marker offset 序列。基础扩散策略生成未来动作：

\[
a_{t:t+H-1} \in \mathbb{R}^{H \times 7}.
\]

目标不是单纯最大化示范似然，而是寻找同时满足任务完成和接触质量的动作。我们定义一个任务相关质量函数：

\[
S_\phi(a_{t:t+H-1}, \hat{z}_{t+1:t+H}, \tau),
\]

其中 \(\hat{z}_{t+1:t+H}\) 是由 Foresight 预测的未来触觉 latent 序列，\(\tau\) 是任务 id 或任务语义。PTG 在扩散推理中近似最大化：

\[
\max_a \log p_\theta(a|o_t) + \lambda S_\phi(a, f_\psi(o_t, a), \tau).
\]

其中 \(p_\theta\) 是 DP 动作先验，\(f_\psi\) 是 Tactile Foresight，\(S_\phi\) 是 TacQuality Energy Scorer。

---

## 4. 方法

### 4.1 总体框架

PTG 的在线流程为：

```text
当前观测
  -> Diffusion Policy 去噪生成未来 action
  -> 后期去噪 step 中估计 clean action x0
  -> Multi-step Foresight 预测 future tactile latent
  -> TacQuality Energy Scorer 计算质量分数
  -> 反传 d(score)/d(action sample)
  -> 小步更新 action sample
  -> 继续去噪并输出最终动作
```

这一流程保留 DP 的动作分布先验，同时用触觉后果能量对动作进行局部修正。它不是生成 \(K\) 个候选后选择最优，也不是执行后再反应；它是在生成过程中直接改变动作。

### 4.2 触觉表征：TactileVAE

GelSight marker offset 形状为：

\[
m_t \in \mathbb{R}^{9 \times 9 \times 2}.
\]

我们使用 TactileVAE 将一个短触觉窗口编码到 latent 空间：

\[
z_t = Enc_{vae}(m_{t-W+1:t}) \in \mathbb{R}^{16 \times 3 \times 3},
\]

展平后为 144 维。TactileVAE 在对应任务数据上预训练并冻结。对黑板任务，当前使用的是在 260609/260610 黑板数据上训练的 TactileVAE，以避免旧插座触觉分布和黑板触觉分布不一致。

TactileVAE 的角色是把高维 marker field 压缩成稳定、可预测、可评分的触觉后果表征。后续 Foresight 和 Scorer 都工作在这个 latent 空间，而不是直接处理 raw marker 图像。

### 4.3 多步 Tactile Foresight

当前 Foresight 采用多步 latent 预测，而不是旧的单步终点预测。输入为当前触觉历史、当前 qpos 和未来动作/状态序列：

\[
\hat{z}_{t+1:t+H_f} = f_\psi(m_{t-W+1:t}, q_t, a_{t+1:t+H_f}).
\]

当前黑板部署配置为：

```text
predict_horizon = 16
foresight_horizon = 16
tactile_mode = marker
camera_names = ['gelsight']
use_state_trajectory = true
hidden_dim = 512
foresight_layers = 3
foresight_nheads = 8
```

Foresight 输出完整未来序列：

\[
\hat{z}_{t+1:t+16} \in \mathbb{R}^{16 \times 144}.
\]

相比单步预测 \(z_{t+16}\)，多步预测监督了未来过程中的每一步，因此更适合连续接触任务。擦黑板、擦花瓶、刷卡和写字的质量往往体现在力变化是否平稳，而不是单个终点触觉是否合理。

**训练损失。** 多步 Foresight 使用真实未来 marker 通过 TactileVAE 得到 \(z^{gt}_{t+1:t+H}\)，训练目标为：

\[
L_{fs} = L_{latent} + \lambda_m L_{marker} + \lambda_\Delta L_\Delta.
\]

其中：

\[
L_{latent} =
\text{SmoothL1}(\hat{z}_{1:H}, z^{gt}_{1:H})
+ \alpha \text{SmoothL1}(\hat{z}_{H}, z^{gt}_{H}),
\]

\[
L_{marker} =
\text{SmoothL1}(Dec_{vae}(\hat{z}_{1:H}), m^{gt}_{1:H}),
\]

\[
L_\Delta =
\text{SmoothL1}(\hat{z}_{2:H}-\hat{z}_{1:H-1},
z^{gt}_{2:H}-z^{gt}_{1:H-1}).
\]

当前黑板配置为：

```text
lambda_marker = 0.3
lambda_delta = 0.5
final_weight = 1.0
num_epochs = 100
batch_size = 16
lr = 4e-5
```

### 4.4 TacQuality Energy Scorer

评分器学习一个可微质量函数：

\[
S_\phi(a_{1:H}, z_{1:H}, \tau).
\]

它的输入不是当前触觉，而是“候选动作导致的未来触觉后果”。训练时使用真实未来触觉 latent；推理时使用 Foresight 预测的未来触觉 latent。

#### 4.4.1 任务质量标签

我们将任务分成两类：

**离散接触任务：插插座。**

正样本：顺利插入或成功插入阶段。  
负样本：pre-bounce、bounce、卡住、明显碰到外边缘的阶段。

**连续接触任务：擦黑板、擦花瓶、刷卡、粉笔写字。**

正样本：接触力大小合适，变化平稳，任务效果有效。  
负样本包括：

```text
too_light: 力太小，擦不干净 / 刷卡失败 / 写不出字
too_heavy: 力太大，可能损坏物体 / 粉笔断裂
unstable: 力忽大忽小或接触丢失
rough_motion: 轨迹抖动、接触不柔顺
```

这种标签定义比“只分好坏”更适合梯度引导，因为坏原因会影响梯度方向和安全约束。

需要强调的是，当前部署到 `guide_forshow.sh` 的擦黑板 latent-energy scorer **不是显式 force-band 物理规则评分器**。它没有直接把 “Fz 在某个区间内” 写成损失或公式，而是通过四类数据标签学习：正常擦拭、压力太小、压力太大、压力不稳定。这些标签来自数据采集条件和任务语义，因此具有物理解释，但评分函数本身是从 tactile latent 与 action chunk 中学习出来的。

#### 4.4.2 当前黑板 latent energy scorer

当前 `guide_forshow.sh` 使用的是 2026-06-15 版 blackboard latent energy scorer。输入为未来 16 步 action 和未来 16 步 tactile latent：

```text
action_chunk: (16, 7)
latent_chunk: (16, 144)
```

模型结构：

```text
action_chunk -> Temporal MLP -> action feature
latent_chunk -> Temporal MLP -> tactile feature
[action feature, tactile feature] -> Fusion MLP -> embedding
embedding 与 4 个 learnable prototypes 做相似度 -> 4 类 logits
```

四类为：

```text
0 expert
1 pressure_too_small
2 pressure_too_large
3 pressure_unstable
```

训练 loss：

\[
L_{score} = CE(logits, y) + 0.5 L_{margin}.
\]

其中 \(CE\) 负责四分类，\(L_{margin}\) 拉开 expert 与 bad 的 expert score：

\[
L_{margin}
= \text{mean}\left[\max(0, \delta + s_{bad} - \bar{s}_{expert})\right].
\]

推理时用于引导的分数为：

\[
expert\_margin
= logit_{expert} - \log \sum_{c \in bad} \exp(logit_c).
\]

这个分数越高，表示未来触觉越接近正常擦拭，越远离力太小、力太大和不稳定三种坏后果。

#### 4.4.3 多头 TacQuality Energy 扩展

为了支持插插座和连续接触任务的统一建模，我们进一步采用多头能量评分器作为通用方向：

```text
共享编码器:
  h = Encoder(predicted tactile, action, task_id)

输出头:
  binary head: good / bad 边界
  reason head: 坏原因类别
  quality head: 连续质量分数
  energy head: 用于 guidance 的标量能量
```

部署时主 objective 不建议只用 \(p(good)\)，因为概率容易饱和且局部梯度不稳定。更合理的是使用综合能量：

\[
E = w_q q + w_b margin_{good} + w_r margin_{reason} + w_t teacher,
\]

并可使用 clipped energy 防止过大梯度。当前这一路仍需要最终真实 rollout 证明效果。

### 4.5 去噪中的梯度引导

在扩散推理第 \(i\) 个去噪 step，当前 action sample 为 \(x_i\)。首先由噪声预测网络得到 \(\epsilon_\theta(x_i, i, o_t)\)，再估计 clean action：

\[
\hat{x}_0 =
\frac{x_i - \sqrt{1-\bar{\alpha}_i}\epsilon_\theta(x_i, i, o_t)}
{\sqrt{\bar{\alpha}_i}}.
\]

将 \(\hat{x}_0\) 反归一化到 raw joint action，并根据 Foresight 契约做 shift 对齐：

```text
alignment = shift1
action_aligned = action_raw[:, 1:H+1]
```

然后：

\[
\hat{z}_{1:H} = f_\psi(o_t, action_{aligned}),
\]

\[
s = S_\phi(action_{aligned}, \hat{z}_{1:H}, \tau).
\]

计算梯度：

\[
g = \nabla_{x_i} s.
\]

实际更新采用保守归一化：

\[
x_i \leftarrow x_i + \eta \frac{g}{\|g\| + \epsilon}.
\]

当前黑板 latent-energy guided 默认参数：

```text
num_inference_steps = 100
guidance_steps = 5
guidance_scale = 0.003
score_mode = expert_margin
guidance_path = latent_only
alignment = shift1
normalize_guidance_grad = true
max_grad_norm = 5.0
```

其中 `latent_only` 表示评分器 action 分支 detach，主要梯度路径为：

```text
action sample
  -> clean action estimate
  -> Foresight predicted tactile latent
  -> TacQuality score
  -> d score / d action sample
```

这样可以减少评分器直接利用 action 分布捷径，而更关注“这个 action 会造成什么未来触觉后果”。

### 4.6 训练阶段划分

PTG 采用分阶段训练，避免端到端训练不稳定：

1. **TactileVAE 预训练。** 在任务触觉数据上训练 marker reconstruction，得到稳定 latent 表征。
2. **基础 DP 训练。** 使用图像、qpos、当前触觉 latent 条件训练 Diffusion Policy。
3. **多步 Foresight 训练。** 使用真实未来 action/state 和真实未来 tactile latent 训练未来 16 步预测。
4. **TacQuality Energy Scorer 训练。** 使用人工或事件标签构造 good/bad/reason/quality 监督，episode-level split 评估泛化。
5. **在线 guidance 集成。** 冻结 DP、Foresight 和 Scorer，在推理时通过 classifier guidance 修改去噪过程。

---

## 5. 任务与质量标准

### 5.1 插插座

插插座是离散接触成功任务。好的动作应进入孔内并减少 retry；坏动作通常表现为插入前碰到孔外、pre-bounce、bounce、卡住或接触力突然升高。标签可以来自人工标注或自动事件检测。

指标：

```text
success rate
retry count
time/step to insertion
peak force / side force
bounce count
```

### 5.2 擦黑板

擦黑板是当前实验最完整的连续接触任务。质量标准为：

```text
力大小适中
力变化平稳
接触持续
擦拭轨迹覆盖有效区域
```

当前黑板四类数据：

```text
expert: 正常擦拭
pressure_too_small: z too high，压力太小
pressure_too_large: z too low，压力太大
pressure_unstable: z oscillate，压力不稳定
```

指标：

```text
force band ratio
force std / jerk
contact dropout ratio
coverage
server-side force trace
```

### 5.3 擦花瓶

擦花瓶属于易碎曲面连续接触任务。相比黑板，它更强调安全上界和曲面法向变化：

```text
正样本: 轻柔稳定接触，覆盖目标区域，不推动/碰倒物体
负样本: 力过大、局部冲击、接触丢失、轨迹偏离曲面
```

### 5.4 刷卡

刷卡任务要求卡片与卡槽保持合适接触和姿态。质量标准为：

```text
正样本: 插入/滑动过程接触连续，阻力适中，轨迹通过卡槽
负样本: 力太小导致未进入槽，力太大导致卡住，姿态偏差导致边缘碰撞
```

### 5.5 粉笔写字

粉笔写字任务要求力足够留下痕迹，但不能过大导致粉笔折断或轨迹抖动：

```text
正样本: 线条连续、压力适中、笔画平滑
负样本: 力太小无痕，力太大断裂，力变化不稳定导致线条断续
```

这些任务共享相同 PTG 框架，区别主要体现在标签定义、reason taxonomy 和任务代价权重。

---

## 6. 实验设计

### 6.1 离线评估

离线评估分三层：

**Foresight 预测质量。**

```text
latent SmoothL1 / MSE
decoded marker error
delta error
完整 episode 可视化视频
不同 horizon: t+2, t+4, ..., t+16 对比
```

**Scorer 泛化能力。**

必须使用 episode-level split 或 GroupKFold，不能 frame-level 随机划分，避免同一 episode 的相邻窗口同时出现在训练和测试中导致泄漏。

指标：

```text
binary AUC / balanced accuracy
reason macro-F1
quality correlation / Spearman
score gap between good and bad
```

**Guidance 梯度可用性。**

检查：

```text
finite gradient rate
positive score-improvement rate
gradient norm
action update norm
score delta before/after local update
```

### 6.2 在线真机评估

真机评估采用 paired baseline-vs-guided rollout：

```text
baseline: same DP server with --disable_guidance
guided: DP + Foresight + TacQuality Energy guidance
```

每次真机 rollout 需要保存：

```text
force_trace.csv / npz
metadata.json
guidance_report
action trace
marker trace
force curve png
```

擦黑板当前 server 已支持按 arm 保存 force trace，baseline 和 guided 分开统计。最终必须用真实 rollout 力曲线和任务结果证明 guided 确实优于 baseline，不能只依赖离线分类准确率。

### 6.3 当前已完成的代表性离线证据

黑板 latent energy scorer 当前验证集结果：

```text
val acc = 100%
macro_f1 = 100%
expert_margin_auroc = 100%
finite_grad_rate = 1.0
positive_grad_rate = 1.0
```

解释边界：这说明四类黑板接触窗口在当前数据分布下可分，且 scorer 有可用梯度；但这不等价于真实机器人 guided 一定提升。最终结论必须来自 paired real rollout。

---

## 7. 消融实验计划

| 消融项 | 目的 |
|---|---|
| 无 guidance | 基础 DP 性能 |
| 单步 Foresight | 验证只看终点是否不足 |
| 多步 Foresight | 验证未来过程预测对连续接触的价值 |
| 只用 p_good | 检查概率饱和和梯度不稳定 |
| expert_margin / energy_clipped | 检查 margin/energy 是否更适合 guidance |
| action_only guidance | 检查评分器是否依赖 action shortcut |
| latent_only guidance | 检查通过 Foresight 后果引导是否更合理 |
| 不同 guidance_steps | 检查去噪后期引导步数 |
| 不同 guidance_scale | 检查安全性和改进幅度 |
| 不同任务 scorer | 检查任务质量标准是否可迁移 |

---

## 8. 讨论

**为什么不是 reranking？**  
Reranking 需要生成多个完整候选再选择，不能改变 DP 单个采样轨迹内部的去噪过程。PTG 的目标是 classifier guidance：让质量梯度直接作用在 action sample 上，从而在生成过程中修正动作。

**为什么需要 Foresight？**  
评分器只能判断一段未来触觉是否好，但在推理时未来触觉尚未发生。Foresight 提供 action 到 future tactile 的可微桥梁：

```text
action -> predicted tactile consequence -> quality score
```

没有 Foresight，评分器无法知道当前 action 会造成什么接触后果。

**为什么需要多步预测？**  
连续接触任务的质量通常体现在过程：力是否平稳、是否中途丢失接触、是否振荡。单步 \(t+16\) 预测容易忽略中间过程，因此多步 \(t+1:t+16\) 更适合擦拭、刷卡和写字。

**为什么不能只看分类准确率？**  
一个分类器即使离线准确率高，也可能不适合 guidance：概率可能饱和，梯度可能很小或方向不稳定。因此必须同时评估分类泛化、score 排序、action-gradient sanity 和真实 rollout 改进。

---

## 9. 局限性

1. 当前真实机器人 paired rollout 证据仍需补齐，不能只凭离线 scorer 指标宣称任务成功率提升。
2. 多任务 scorer 的标签质量决定上限。连续接触任务需要覆盖 too_light、too_heavy、unstable、rough_motion 等完整负样本。
3. Foresight 的预测误差会直接影响 guidance。多步预测降低了只看终点的风险，但仍可能在分布外 action 上不可靠。
4. Guidance scale 需要保守设置。过强梯度可能把动作推离 DP 训练分布，造成安全风险。
5. 当前黑板 latent energy scorer 对四类数据分得很开，可能部分依赖采集分布差异；需要跨日期、跨轨迹、跨物体验证泛化。

---

## 10. 结论

本文将方法主线从“触觉重排序”更新为“触觉后果驱动的扩散去噪梯度引导”。PTG 用多步 Tactile Foresight 预测候选动作的未来触觉后果，用 TacQuality Energy Scorer 将未来触觉映射为连续质量能量，再把该能量的梯度注入 DP 去噪过程。该框架统一覆盖插插座、擦黑板、擦花瓶、刷卡和粉笔写字等接触丰富任务。当前黑板实验已经验证了 latent energy scorer 的离线可分性和可微性，下一步关键是通过真实 baseline-vs-guided rollout 的力曲线和任务指标证明 PTG 在真实执行中改善接触质量和任务成功率。

---

## 附录 A：当前黑板部署版本

当前 `for_show_xiaomi/guide_forshow.sh` 对齐版本：

```text
DP:
/home/chenshuai/Project/output/dp_tac_concat_board_260609_260610_left_boardvae_rawimg200x266_ph16_oh2_e1000/dp_best.pth

Foresight:
/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload/foresight_best.ckpt

Scorer:
/home/chenshuai/Project/output/board_latent_energy/ce_margin_e10/board_latent_energy_best.pt
```

默认引导参数：

```text
guidance_path = latent_only
score_mode = expert_margin
alignment = shift1
guidance_scale = 0.003
guidance_steps = 5
```

---

## 参考文献

[1] Chi et al., Diffusion Policy: Visuomotor Policy Learning via Action Diffusion, RSS 2023.  
[2] Zhao et al., Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware, RSS 2023.  
[3] Calandra et al., More Than a Feeling: Learning to Grasp and Regrasp using Vision and Touch, RA-L 2018.  
[4] Lee et al., Making Sense of Vision and Touch, ICRA 2019.  
[5] Prasad et al., Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation, RSS 2024.  
[6] Ebert et al., Visual Foresight: Model-Based Deep RL for Vision-Based Robotic Control, 2018.  
[7] Florence et al., Implicit Behavioral Cloning, CoRL 2021.  
[8] TouchGuide, inference-time touch guidance for diffusion policies.  
[9] DreamTacVLA, learning to feel the future for contact-rich manipulation.  
