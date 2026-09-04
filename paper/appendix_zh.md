# ForeTac 补充材料：用预测接触后果引导机器人动作

作者：匿名作者

## A. 附录概述

本附录在主文之外进一步补充具体的网络架构、数据采集、预处理、训练、评估、分析和局限性细节。第 B 节介绍真实机器人数据格式和训练数据统计。第 C 节介绍网络组件。第 D 节给出模型超参数和训练契约。第 E 节给出 ForeTac 在推理时使用的算法流程。第 F 节到第 I 节分别提供触觉表征、触觉前瞻预测、接触质量评分器和力行为的额外定量与定性证据。后续章节总结主实验对比、消融实验、插孔基准、backbone 适配、失败案例、部署开销和局限性。

## B. 数据采集与格式

### B.1 机器人观测数据结构

所有策略和前瞻模块都从时间戳对齐的 HDF5 episode 中训练。每条轨迹包含同步的 RGB 图像、机器人本体状态、动作标签、触觉 marker 场，以及可选的力/力矩测量。主要的 action-conditioned foresight 训练契约为：

$$
\hat{\mathbf{z}}_{t:t+H-1}
= f_\psi(I_t^g, I_t^w, \mathbf{m}_{t-W+1:t}, q_t,
\mathbf{a}_{t:t+H-1})
$$

其中条件输入是未来动作 chunk，也就是 `actions/joint_abs[t:t+H]`，而不是未来状态轨迹。这个区分很重要：未来状态是执行后的结果，而 rollout 时在执行前可获得的是候选动作 chunk。

**表 1：实现中使用的 HDF5 episode 字段。**

| 字段 | 形状 / 类型 | 用途 |
|---|---|---|
| `observations/images/global` | $T \times H \times W \times 3$ uint8 | 场景上下文 |
| `observations/images/wrist` | $T \times H \times W \times 3$ uint8 | 局部接触几何 |
| `observations/proprio_joint` | $T \times 7$ float | 当前机器人状态 |
| `actions/joint_abs` | $T \times 7$ float | 动作 chunk 目标 / foresight 条件 |
| `observations/tac/left/marker_offset` | $T \times 9 \times 9 \times 2$ float | GelSight marker 位移 |
| `observations/tac/left/force6d` | $T \times 6$ float | 接触质量标签与分析 |

RGB 图像在 ImageNet normalization 之前会进行 resize 和 crop。触觉输入是 GelSight marker 位移场，并按每个 displacement channel 进行归一化。TacVAE 使用 8 帧触觉窗口，并把当前接触编码为紧凑的空间 latent。主实验中的 diffusion policy 使用 `obs_horizon=2`，并预测 16 步动作 chunk。

### B.2 采集流程

每条 demonstration episode 都被记录为一段连续的机器人执行轨迹。根据任务脚本不同，控制器会保存关节绝对命令 `actions/joint_abs`，或末端执行器命令 `actions/eef_abs` / `eef_rel`。Action-conditioned ForeTac 实验中，在动作标签与实测关节状态一致时，黑板、刷卡和插孔类策略使用 joint-absolute chunk。使用较老末端执行器脚本采集的花瓶数据需要单独处理，因为部分保存的 `joint_abs` 标签包含逆运动学分支跳变，而这些跳变并不存在于真实机器人运动中。对于这些序列，策略训练应使用末端执行器标签，或使用重新生成的关节标签。

触觉流与 RGB 和机器人状态流使用相同的 episode step index 保存。本文最重要的触觉字段是 `marker_offset`，即 $9\times9$ 网格上的二维 marker 位移。相比原始触觉 RGB，我们在 foresee-and-score 流程中更偏好这个 marker 场，因为它紧凑、直接对应局部接触形变，并且可以可视化为变形网格。力/力矩信号在可用时用于打标签和分析，但主 foresee-and-score 链路在部署时不要求必须有力传感。

### B.3 训练数据统计

表 2 报告用于触觉表征预训练的数据规模。该语料包含 5,969 条 HDF5 episode，约 2.1M 触觉 marker 帧。帧数统计遵循训练流程中统一使用的 marker-stream 长度定义。

**表 2：触觉预训练数据规模。**

| 训练语料 | Episodes | 近似触觉帧数 |
|---|---:|---:|
| TacVAE 预训练语料 | 5969 | 约 2.1M |

### B.4 TacVAE 预训练语料

TacVAE 不是只在五个任务的 rollout split 上预训练，而是在完整触觉语料上预训练。预训练语料包含 5,969 条 HDF5 episode，约 2.1M 触觉 marker 帧。语料包含成功接触、弹起、压力过小、压力过大、振荡接触、插孔错位和 replay 轨迹等变体。这些负样本或不完美片段对表征学习有价值，因为 encoder 必须保留理想和非理想接触状态；后续接触质量评分器会学习触觉空间中哪些区域应该被偏好。

Task-local TacVAE checkpoint 对黑板擦拭、花瓶擦拭和刷卡使用相同的架构与归一化契约。当部署策略是任务特定策略时使用 task-local encoder；full-corpus TacVAE 则作为更宽泛的触觉预训练参考。

### B.5 任务标签与安全定义

对于每次在线试验，当任务目标在 rollout horizon 内完成时记为 success。safe success 更严格，并从保存的 rollout log 中计算。令 $y_{\rm task}\in\{0,1\}$ 表示任务是否完成，$F_t$ 表示接触力大小，$\mathcal{C}=\{t:F_t>\tau_{\rm contact}\}$ 表示接触帧集合，$r$ 表示自动检测到的 retry 或 bounce 次数，$d_{\rm obj}$ 表示测量或人工审计得到的物体位移。定义：

$$
F_{\max} = \max_t F_t
$$

$$
\sigma_F^2 = {\rm Var}_{t\in\mathcal{C}}(F_t)
$$

$$
\rho_{\rm drop} = 1-\frac{|\mathcal{C}|}{T_{\rm expected}}
$$

$$
B_{\rm band} =
\frac{1}{|\mathcal{C}|}
\sum_{t\in\mathcal{C}}
\mathbb{I}\left[\tau_{\rm low}^{(k)} \le F_t \le \tau_{\rm high}^{(k)}\right]
$$

对于任务 $k$，safe-success 标签为：

$$
\begin{aligned}
y_{\rm safe}
&= y_{\rm task}\,
\mathbb{I}[F_{\max}\le\tau_{\rm peak}^{(k)}]\,
\mathbb{I}[\sigma_F^2\le\tau_{\rm var}^{(k)}]\,
\mathbb{I}[\rho_{\rm drop}\le\tau_{\rm drop}^{(k)}]\\
&\quad\cdot
\mathbb{I}[B_{\rm band}\ge\tau_{\rm band}^{(k)}]\,
\mathbb{I}[r\le\tau_{\rm retry}^{(k)}]\,
\mathbb{I}[d_{\rm obj}\le\tau_{\rm obj}^{(k)}]\,
\mathbb{I}[\neg{\rm damage}]
\end{aligned}
$$

对于插孔任务，$B_{\rm band}$ 可以省略，或替换为 jamming duration 阈值。对于擦拭和刷卡任务，力带占比和力方差是稳定接触的主要指标。任何可见的物体位移、倾倒、被夹持物体损坏、急停或硬件安全干预都会将该 trial 标记为 unsafe；严重物体移动或损坏也会被计为任务失败，即使几何终点已经到达。

### B.6 任务命名

Rollout 表格统一使用五个任务名。`Board wiping` 指平面黑板擦拭；`Vase wiping` 指沿花瓶曲面运动并保持稳定接触；`Card swiping` 指推动或滑动卡片通过受约束的接触区域；`Chip grasping` 指在不压碎和不打滑的情况下夹起易碎物体；`Socket insertion` 指在接触力、卡滞、重试和恢复约束下完成插入。主实验、消融实验和跨架构实验均采用这五项统一任务定义。

## C. 网络架构

### C.1 模块概述

ForeTac 是一个用于 chunk-based action generator 的模块化附加方法。完整系统包含四个可训练或预训练组件：

1. 触觉 encoder-decoder，即 TacVAE，将 GelSight marker 历史映射为紧凑 latent map，并将其解码回 marker 场。
2. 基础动作策略，通常是 Diffusion Policy，提供 demonstration action chunk 上的行为先验。
3. action-conditioned tactile foresight transformer，预测候选动作 chunk 将导致的未来触觉 latent 序列。
4. 接触质量能量模型，将预测的未来触觉映射为标量质量分数，并提供用于动作细化的梯度。

这种拆分是有意设计的。基础策略在推理时不需要学习新的全局目标；它只负责给出候选动作。Foresight 模型针对每个候选 chunk 回答一个反事实接触问题。评分器判断预测接触是否符合任务要求。Guidance 模块随后对动作样本施加有界的局部修正。

**表 3：主要架构参数。**

| 模块 | 输入 | 内部表示 | 输出 |
|---|---|---|---|
| TacVAE encoder | 8 帧 marker，每帧 $9\times9\times2$ | $16\times3\times3$ 空间 latent | 144-D latent vector / 9 个 tactile token |
| TacVAE decoder | $16\times3\times3$ latent | 反卷积 marker decoder | $9\times9\times2$ marker displacement |
| Vision encoder | global 与 wrist RGB | 冻结 ResNet18 token，投影到 512-D | visual context token |
| Foresight transformer | visual token、tactile token、$q_t$、动作 chunk | 3 层、8 heads、hidden 512、FFN 2048 | 未来 latent 序列，$H=16$ |
| Contact-quality scorer | 仅预测 marker latent 序列 | temporal latent MLP + learned prototypes | expert-good margin score |
| Guidance module | noisy diffusion sample 与 score gradient | 后期步骤中的归一化 trust-region 更新 | 细化后的动作 chunk |

### C.2 TacVAE Tokens

TacVAE 输出的是空间 latent，而不是单个无结构向量。在实现中，latent 有 16 个 channel 和 $3\times3$ 空间布局。经过线性投影到 512-D transformer hidden dimension 后，$3\times3$ 的格子被当作 tactile token。这样 foresight 模块仍然知道接触发生在 GelSight 表面的哪个位置。训练和诊断中保留 decoder，使 latent prediction 可以在 marker 空间中检查。

### C.3 Action-Conditioned Foresight Transformer

Foresight transformer 接收当前观测 token 和候选动作 chunk。预测 horizon 被设置为与策略 chunk 相同的 16 步窗口。动作序列作为未来接触动力学的条件信号被嵌入。当前本体状态 $q_t$ 被作为上下文输入，但主模型不使用未来本体状态轨迹，因为它会泄漏执行后的结果，而不是以可执行候选命令作为条件。

### C.4 接触质量能量模型

评分器被训练为区分 expert 或可接受接触与任务特定失败模式。对于黑板擦拭，失败模式包括压力过小、压力过大和压力不稳定。对于插孔，重要负样本包括侧向接触、弹起、retry 过多的运动、卡死和过大峰值力。部署时使用的标量分数是 good-contact class 的 logit 与 bad-contact classes 的 log-sum-exp 之间的 margin。相比校准概率，这个分数更不容易饱和，因此能为后期 denoising 提供更有用的梯度。

### C.5 Guidance Module

Guidance module 只在选定的后期 denoising 或 flow-matching 步骤中运行。它使用当前 clean-action estimate，预测未来 marker latent，计算质量 margin，并对动作样本求该 margin 的梯度。更新会被归一化、裁剪，并在 raw action unit 中检查。正式评分器没有 action 分支，梯度只能沿预期路径传播：

$$
\mathbf{a} \rightarrow f_\psi(\mathbf{o},\mathbf{a}) \rightarrow S
$$

这样可以降低评分器学习 action-only shortcut 的风险，避免分数不再对应预测触觉后果。

## D. 训练细节

### D.1 模型组件

表 4 总结实验中使用的实现设置。这些值与项目页和附录证据所用的本地脚本和配置一致。

**表 4：主要训练超参数。**

| 组件 | 架构 / 输入 | 优化 | 备注 |
|---|---|---|---|
| TacVAE | 8-frame $9 \times 9 \times 2$ marker windows；latent $16 \times 3 \times 3$ | AdamW，lr $10^{-4}$，batch 512，150 epochs，stride 2 或 4 | KL weight $10^{-6}$；direction loss weight 0.2 |
| Visual/tactile DP | global+wrist RGB、TacVAE latent history、$q_t$；action horizon 16 | AdamW，lr $10^{-4}$，batch 64，黑板训练 1000 epochs | DDPM training steps 100；inference steps 100；action space 为 `joint_abs` |
| Action-conditioned foresight | ResNet18 visual tokens、TacVAE tokens、$q_t$、action chunk $H=16$ | AdamW，lr $4{\times}10^{-5}$，weight decay $10^{-4}$ | 3 transformer layers，8 heads，hidden 512，FFN 2048 |
| Contact-quality scorer | 仅 predicted marker latent sequence | cross-entropy / margin objective | 部署分数是 expert-good 相对 negative modes 的 margin |
| Guided denoising | 仅后期 DDPM steps | normalized gradient step；trust-region clipping | action 只通过 Foresight-predicted latent 影响分数 |

### D.2 TacVAE 预训练设置

触觉表征在策略和 foresight 模型之前训练。每个训练样本是来自一个 GelSight 侧的一段 8 帧 marker window。目标是当前窗口中的 marker displacement field，重建损失在将 latent 解码回 marker 空间后计算。Marker fields 按 displacement channel 使用训练集统计量归一化。同一套归一化统计会随 checkpoint 保存，并在策略训练、foresight 训练、可视化和 runtime serving 中复用。

**表 5：TacVAE 预训练数据与优化设置。Episode 和 frame 数量仅报告完整预训练语料。**

| TacVAE 设置 | Episodes | 近似帧数 | Window / stride | 优化 |
|---|---:|---:|---|---|
| Full tactile pretraining corpus | 5969 | 约 2.1M | 8 / mixed | AdamW，lr $10^{-4}$，batch 512 |
| Board task-local TacVAE |  |  | 8 / 4 | 150 epochs，KL $10^{-6}$，dir. 0.2 |
| Vase task-local TacVAE |  |  | 8 / 2 | 150 epochs，KL $10^{-6}$，dir. 0.2 |
| Card task-local TacVAE |  |  | 8 / 2 | 150 epochs，KL $10^{-6}$，dir. 0.2 |

Full-corpus 数量用于描述可用触觉预训练规模。Task-local encoder 使用黑板擦拭、花瓶擦拭和刷卡各自的 marker statistics 与接触模式。

### D.3 策略训练设置

基础 DP 策略使用标准 denoising-action 框架训练。输入条件是视觉特征、触觉 latent history 和短 observation horizon 内归一化机器人状态的拼接。动作目标是归一化 `joint_abs` action space 中的 16 步 chunk。对于黑板擦拭，策略使用 global 和 wrist RGB 视角、`obs_horizon=2`、`pred_horizon=16`、`n_action_steps=8`、batch size 64、100 diffusion training steps，以及 100 inference denoising steps。模型最多训练 1000 epochs，并使用 validation-window monitoring 和周期性的 best-checkpoint saving。

对于已知保存的 joint action labels 含有逆运动学分支跳变的任务，策略应使用末端执行器动作标签或重新生成的关节标签。这一点对较老的花瓶数据尤其重要。本附录保留这个说明，因为如果把正确的触觉模型与错误的 action label space 混用，会让 foresight conditioning 看起来比实际更差。

### D.4 评分器训练设置

接触质量评分器不是只从任务 success 训练，而是从已标注的接触窗口训练。这样可以为接触质量提供密集监督：expert contact、压力不足、压力过大、振荡接触、弹起、边缘接触、卡死，以及其他任务特定负样本模式。对于黑板设置，评分器在 guidance diagnostic 所用的 labeled validation windows 上 held-out validation accuracy 和 macro-F1 均为 1.0。该结果评估的是离线接触标签，而不是在线任务成功率。

部署时，评分器输出被转换为 expert-margin energy。相比校准概率，margin 更适合使用，因为即使分类器已经很自信，它仍然能提供有用信号。同一个分数有两种用法：采样后的 candidate reranking，或 final denoising steps 中的 differentiable guidance。主 ForeTac 配置使用后者，因为它可以在不丢弃基础策略先验的情况下局部改善已采样 chunk。

### D.5 Foresight Loss

Foresight 模型预测与候选动作 chunk 对应的 latent sequence。训练使用冻结 TacVAE target，并组合三个损失项：

$$
\mathcal{L}_{\rm fs}
= \mathcal{L}_{\rm latent}
+ \lambda_m \mathcal{L}_{\rm marker}
+ \lambda_\Delta \mathcal{L}_\Delta
$$

$$
\mathcal{L}_{\rm latent}
= {\rm SmoothL1}(\hat{\mathbf{z}}_{1:H}, \mathbf{z}_{1:H})
+ {\rm SmoothL1}(\hat{\mathbf{z}}_H, \mathbf{z}_H)
$$

$$
\mathcal{L}_{\rm marker}
= {\rm SmoothL1}(D_{\rm tac}(\hat{\mathbf{z}}_{1:H}), \mathbf{m}_{1:H})
$$

$$
\mathcal{L}_\Delta
= {\rm SmoothL1}(\Delta\hat{\mathbf{z}}_{1:H-1},
\Delta\mathbf{z}_{1:H-1})
$$

主配置使用 $\lambda_m=0.3$ 和 $\lambda_\Delta=0.5$。Decoded marker loss 使 latent forecast 在物理上绑定到 marker displacement，而 delta 项惩罚时间上不一致的预测。

### D.6 正确的动作条件化

修正后的训练路径设置 `use_state_trajectory=false`。对于从时间 $t$ 开始的样本，foresight 输入是 `actions/joint_abs[t:t+H]`，目标是未来触觉 `marker_offset[t+1:t+H]` 或 stride-aligned equivalent。之前的 state-trajectory mode 仅作为 legacy option 保留给 ablation，不用于主 ForeTac claim。

## E. 算法流程

**Algorithm S1：推理时的预测接触 guidance。**

1. 读取当前观测 $\mathbf{o}_t=(I_t^g,I_t^w,q_t,\mathbf{m}_{t-W+1:t})$。
2. 使用 TacVAE 编码触觉历史，得到当前 latent $\mathbf{z}_t$ 和 tactile tokens。
3. 运行基础动作生成器，对动作 chunk $\mathbf{a}_{t:t+H-1}$ 采样一条 denoising trajectory。
4. 在选定的后期 denoising steps，将 noisy action sample 转换为 clean action estimate $\hat{x}_0$。
5. 将 $\hat{x}_0$ 输入 action-conditioned foresight model。
6. 预测未来触觉 latent sequence $\hat{\mathbf{z}}_{t:t+H_f-1}$。
7. 使用接触质量能量模型对预测的未来触觉打分：

$$
s_{\rm margin}
= \ell_{\rm good}
- \log \sum_{c \in \mathcal{C}_{\rm bad}} \exp(\ell_c)
$$

8. 通过 $\mathbf{a} \rightarrow \hat{\mathbf{z}} \rightarrow S$ 反传 $s_{\rm margin}$，并在 raw-action trust region 内应用一个归一化的小更新。
9. 只执行 refined chunk 的前几个动作 step，并在下一个控制周期重复该流程。

这个设计将行为生成与接触质量修正分离。基础策略仍然是 demonstration robot motions 上的先验；foresight 询问候选 chunk 会造成什么触觉后果；评分器判断该预测未来接触是否理想。

## F. 触觉 Encoder 与重建

### F.1 Encoder 构建

TacVAE 使用的是 marker displacement fields 上的 encoder-decoder 结构，而不是原始触觉图像。输入是一段 8 帧 marker window；encoder 将其映射为 $16 \times 3 \times 3$ 空间 latent，decoder 重建 $9 \times 9 \times 2$ marker field。空间 latent 在将回归目标从原始 marker grid 压缩为紧凑接触表征的同时，仍保留了局部接触几何信息，供 foresight 和 scorer 使用。

**表 6：不同触觉 encoder 的重建对比。每个任务报告 MAE / cosine similarity。每个任务-指标列中最佳值加粗。**

| Encoder | Board MAE | Board Cos. | Vase MAE | Vase Cos. | Card MAE | Card Cos. | Chip MAE | Chip Cos. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PCA | 0.040 | 0.995 | 0.035 | **0.995** | 0.036 | 0.980 | 0.026 | 0.965 |
| PointNet-AE | 0.048 | 0.956 | **0.027** | 0.955 | 0.044 | 0.992 | 0.032 | **0.990** |
| Conv-AE | 0.055 | 0.997 | 0.042 | 0.994 | 0.104 | **0.999** | 0.036 | 0.959 |
| TacVAE (ours) | **0.039** | **0.998** | 0.037 | 0.975 | **0.031** | **0.999** | **0.015** | 0.899 |

### F.2 解释

重建表说明 marker field 的维度足够低，简单 baseline 也能重建许多静态帧。因此，仅靠重建结果并不能完整证明 TacVAE。对于 ForeTac，真正相关的性质是该 latent 是否能在完整 differentiable chain 中发挥作用：当前触觉历史 $\rightarrow$ 未来触觉预测 $\rightarrow$ 接触质量分数 $\rightarrow$ 动作更新。因此，TacVAE 应与 foresight prediction 和 guidance diagnostics 一起评估，而不是只作为 standalone autoencoder 评估。

## G. Foresight 预测质量

Horizon study 评估的是：虽然更长的预测未来会带来预期中的预测误差上升，它是否仍能改善动作选择。表 7 报告了黑板擦拭在 held-out sliding-window validation set 上的结果。16-step horizon 是主设置，并与控制器使用的主在线成功率对应。短 horizon 的重建误差更低，但无法暴露足够长的未来接触后果，因此对稳定 guidance 不够。

**表 7：不同预测 horizon 下的 foresight 预测质量与 rollout success。**

| Horizon | Eval windows | Latent MAE | Latent cosine | Marker error | Success ↑ |
|---:|---:|---:|---:|---:|---:|
| 1 | 9,631 | 0.167 | 0.9993 | 0.213 px | 60% |
| 4 | 9,631 | 0.188 | 0.9989 | 0.297 px | 65% |
| 8 | 9,631 | 0.230 | 0.9965 | 0.317 px | 75% |
| 12 | 9,631 | 0.261 | 0.9952 | 0.352 px | 80% |
| 16 | 9,631 | 0.274 | 0.9953 | 0.377 px | **85%** |

图 1：不同 horizon 下的 foresight 预测质量。更长 horizon 会有更大的 decoded marker error，但能为 action guidance 提供更有任务相关性的未来接触上下文。评估单位是 held-out board episodes 中的 sliding tactile window，而不是完整 rollout episode。

图 2：跨任务阶段的长 horizon 黑板预测。上方 panel 展示从 approach 到 late sweeping 的四个阶段中 GT 和预测的未来 marker field。下方曲线报告 episode-level marker prediction error，并明确标出所选择的低误差可视化点。

## H. 接触质量评分器验证

### H.1 Preference-Order Accuracy

评分器使用从离线机器人视频及其同步触觉日志中构造的正/负 rollout pair 进行验证。对于每条成功 rollout，我们选择决定性交互附近的 key contact frames；对于每条失败 rollout，我们从匹配的失败阶段选择负样本帧，例如 dropout、压力过大、弹起、侧向接触或卡死。随后构造任务内的正/负样本 pair，并评估：

$$
A_{\rm pref}
= \frac{1}{|\mathcal{P}|}
\sum_{(i,j)\in\mathcal{P}}
\mathbb{I}\left[
S(\mathbf{o}_i^+,\mathbf{m}_i^+) >
S(\mathbf{o}_j^-,\mathbf{m}_j^-)
\right]
$$

其中 $S$ 是接触质量 margin score。这正是能量模型需要的排序性质：绝对分数不需要完美校准，但理想接触应该比匹配失败接触得分更高。

**表 8：由正/负 key-contact pair 得到的 preference-order validation。Accuracy 表示正样本得分高于负样本的 pair 占比。**

| Task family | Correct / pairs | Preference accuracy | Mean score gap |
|---|---:|---:|---:|
| Board wiping | 1856 / 1980 | 93.7% | 0.42 |
| Vase wiping | 936 / 1012 | 92.5% | 0.36 |
| Card swiping | 1292 / 1408 | 91.8% | 0.31 |
| Socket insertion | 1908 / 2020 | 94.5% | 0.47 |
| Pooled | 5992 / 6420 | 93.3% | 0.39 |

相比对所有 windows 求平均，key-contact pairing 更有意义，因为非接触 approach frames 往往是模糊的。该指标直接评估评分器是否能够将理想接触排在匹配失败接触之前。

### H.2 Gradient 与 Sampler Diagnostics

评分器还必须能作为 differentiable guidance signal 使用。黑板 diagnostic 验证了组合链路 action--foresight--score 可以产生有限梯度，并且动作更新是有界的。

**表 9：黑板 guidance diagnostic。**

| Metric | Value |
|---|---:|
| Scorer validation windows | 31,440 sliding windows |
| Gradient-audit windows | 1,280 sampled windows |
| Scorer accuracy / macro-F1 | 1.000 / 1.000 |
| Expert-margin AUROC | 1.000 |
| Rows with finite gradients | 1,280 / 1,280 |
| Mean raw gradient norm | 2.4475 |
| Mean scaled gradient norm | 0.00734 |
| Mean guided-step score delta | 0.0198 |
| Mean applied update norm | 0.0030 |

这里的 `window` 指从 held-out rollout logs 中采样的 8-frame tactile sliding window，而不是完整 episode。Scorer validation window count 用于评估接触质量分类器。Gradient-audit windows 是 phase-balanced subset，在这些窗口上还额外运行开销较大的 backward pass，即 action $\rightarrow$ foresight $\rightarrow$ score。`Rows with finite gradients` 统计的是该 differentiable chain 产生有限 action gradient 的窗口数。Scaled gradient norm 在应用 guidance scale 后、最终 raw-action trust-region clamp 前测量。

## I. 力曲线分析

力轨迹为接触质量目标提供了可解释视角。对于擦拭任务，目标行为不是最大力，而是稳定的力带：既保持接触，又不扰动物体。对于插孔任务，目标行为是有界峰值力和较少 retry；一个策略如果在接触后持续推压，几何上可能看起来合理，但并不安全。插孔力曲线 panel 直接展示了 no-lift jamming、retry-heavy recovery 和低力 foresight-guided recovery 之间的差异。

图 3：Socket insertion force traces。Raw DP 在接触后保持高力并且未能 lift；DP + Contact Observation 通过多次 bounce-lift-reinsert 循环成功；ForeTac 降低了峰值力和恢复时间。

## J. 主实验对比

表 10 报告主 rollout 对比。每个条目报告 20 次尝试中的成功率和成功次数。Safe success 是更严格的子集，还必须满足第 B 节定义的安全准则。

**表 10：五项接触丰富操作任务上的主对比。SR/SSR 分别表示任务成功率和安全成功率（%）。**

| 触觉使用方式 | 方法 | Board SR | Board SSR | Vase SR | Vase SSR | Card SR | Card SSR | Chip SR | Chip SSR | Socket SR | Socket SSR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No touch | DP | 50 | 25 | 55 | 35 | 10 | 5 | 30 | 20 | 35 | -- |
| No touch | RDT | 65 | 55 | 65 | 55 | 40 | 35 | 50 | 45 | -- | -- |
| Reactive touch | DP + Contact Obs. | 60 | 50 | 60 | 50 | 35 | 30 | 40 | 35 | 60 | -- |
| Reactive touch | RDP | 70 | 60 | 65 | 55 | 45 | 35 | 55 | 45 | -- | -- |
| Predictive touch | **ForeTac** | **85** | **80** | **70** | **65** | **55** | **50** | **75** | **70** | **85** | -- |

主实验对比分离了两类提升。第一，直接融合当前接触观测给基础策略增加了接触证据，因此有帮助，但它仍然是反应式的。第二，ForeTac 在动作执行前使用预测触觉后果，使评分器能够更早地惩罚 contact dropout、excessive force 和 unstable contact。提升最大的是视觉几何不足以推断接触质量的任务：黑板擦拭、夹薯片和 socket-style insertion。

## K. 消融实验

表 11 分别检验触觉表征和推理时动作选择机制。ResNet18 变体替换 TacVAE；去掉 Guide 的变体保留 foresight 但不执行动作细化；Predictive Reranking 在多个候选动作中按预测接触质量选择，但不进行梯度更新。

**表 11：消融实验。每个条目报告 20 次尝试中的 success 和同 20 次尝试中的 safe success。**

| Variant | Board SR | Board SSR | Vase SR | Vase SSR | Card SR | Card SSR | Chip SR | Chip SSR | Socket SR | Socket SSR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ForeTac w/o TacVAE (ResNet18) | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| ForeTac w/o Guide | 60 | 55 | 65 | 55 | 40 | 35 | 55 | 50 | -- | -- |
| ForeTac w/ Predictive Reranking | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| **ForeTac (Full)** | **85** | **80** | **70** | **65** | **55** | **50** | **75** | **70** | **85** | -- |

四个变体分别检验触觉表征、是否执行推理时细化，以及梯度引导相对于候选重排序的作用。Foresight 条件化通过 action-conditioned、no-action 和 mismatched-action 预测对照单独验证，因为移除预测器也会同时移除 Guide 和 Predictive Reranking 使用的评分路径。

## L. 插孔基准

Socket insertion benchmark 每种方法使用 20 次试验。除 success 外，该任务还记录 retry 次数、观测到的最大接触力、平均峰值力，以及从峰值力恢复到 2 N 以下所需的时间。

**表 12：Peg-in-hole / socket insertion benchmark。**

| Method | Succ. ↑ | Retry | Max F (N) ↓ | Peak F (N) ↓ | Recover (s) |
|---|---:|---:|---:|---:|---:|
| Raw DP | 35% (7/20) | 0 | 28.1 | 24.3 | - |
| DP + Contact Obs. | 60% (12/20) | 4.2 | 26.4 | 21.5 | 1.85 |
| DP+ForeTac | 85% (17/20) | 1.1 | 15.2 | 12.8 | 0.42 |
| RDT+ForeTac | **90% (18/20)** | **0.9** | **14.1** | **11.5** | **0.35** |

Raw DP 没有显式恢复行为，因此在失败的 no-lift trials 中，`retry` 被报告为 0，而不是稳定插入的迹象。DP + Contact Observation 能够恢复，但恢复表现为重复的高力接触循环。Foresight-guided 变体同时减少了 retry 次数和力峰值大小。

## M. Backbone 适配

ForeTac 被设计为可以包裹任意 chunk-based action generator。对于 DP，guidance gradient 直接施加到 denoising action sample 上。对于 RDT，同一个 predicted-contact score 被接到 reactive diffusion transformer 的 action chunk 上。对于 $\pi_{0.5}$，适配方案保留 VLA backbone 作为 flow-matching action-generation 模块，并将 tactile latent tokens 拼接到 policy prefix；foresight branch 仍然是 action-conditioned contact consequence model。在所有情况下，接触质量评分器都位于基础策略之外，并且可以独立评估。

$\pi_{0.5}$ 实现使用 10 个 flow-matching integration steps，并只在最后 2 步启用 ForeTac。在 flow state $x_s$ 上，clean action estimate 为 $\hat{x}_0=x_s-sv_\theta(x_s,s,\mathbf{o}_t)$。模型动作空间是 32 维，而机器人命令、触觉 foresight 条件和接触质量评分器只使用前 7 个关节维度。因此，score gradient 会被 mask 到这 7 个可执行维度；其余 padding dimensions 保持基础 flow trajectory 不变。引导后的机器人命令还会使用与 DP 实现相同的 raw-action trust region 约束。

**表 13：Backbone 适配总结。**

| Backbone | Adaptation | Additional policy training |
|---|---|---|
| DP | 后期 denoising action update | 任务 DP checkpoint |
| RDT | 在 transformer action chunk 上使用同一个 score | 任务 RDT checkpoint |
| $\pi_{0.5}$ | tactile latent prefix + flow-matching action chunk | $\pi_{0.5}$ fine-tuning / adapter |
| ForeAR | autoregressive foresight baseline |  |

## N. 失败案例

最常见的失败是接触质量失败，而不是纯视觉位姿失败。

**黑板擦拭。** Vision-only DP 可以覆盖目标路径，但可能施加过小压力，导致擦不干净；也可能施加过大压力，导致不安全擦拭。典型轨迹会在 contact dropout 和 high-force burst 之间交替。这一失败动机说明，我们需要对预测的未来触觉打分，而不是只检查视觉上的擦拭轨迹。

**花瓶擦拭。** 曲面且脆弱的物体带来了很窄的安全力带。即使视觉路径正确，如果法向力振荡或出现尖峰，花瓶仍可能被推倒。较旧花瓶数据中还存在一些保存的 `joint_abs` 动作标签质量问题：机器人实际运动是平滑的，但存储的关节标签可能在逆运动学分支之间跳变。这些 episode 应使用末端执行器标签或修正后的关节标签。

**刷卡。** 刷卡任务常见失败是短暂 contact dropout 或槽口错位。关键触觉事件是持续滑动接触，而不只是到达槽口入口。因此，短 horizon foresight 不如更长触觉序列有信息量。

**夹薯片。** 主要预期失败模式是压碎和打滑。成功行为需要在保持足够夹持稳定性的同时，将接触力控制在物体损伤阈值以下。

**插孔。** 插孔任务的失败包括侧向接触、bounce-lift-reinsert 循环和持续卡死。Raw DP 可以生成看似合理的 approach，但在第一次接触后继续推压。Tactile concatenation 提高了反应能力，但仍可能需要多次 retry。Foresight-guided 策略通过预测候选 chunk 的接触后果降低峰值力。

## O. Runtime 与部署细节

ForeTac 在基础动作生成器之外增加了两个推理时计算：触觉 foresight 和接触质量 scoring。两者都运行在候选 action chunk 上，并在 chunk horizon 上 batch 计算。在 DP 路径中，guidance 只在后期 denoising steps 激活，因此开销小于对许多完整 trajectory 运行 planner。Runtime update 会在 raw action units 中归一化并限幅。如果 score 没有改善，或 proposed action update 违反 trust region，系统可以保留基础策略 chunk。

部署 server 在启用时会把 guidance diagnostics 与 rollout traces 一起记录。这些 diagnostics 包括 guidance 前后的 score、gradient norm、accepted update norm、contact gate，以及可选的 per-action dimension gradient statistics。这类日志有助于区分三种情况：评分器较弱；评分器较好但没有 action gradient；guidance 路径有效但仍需要真实机器人验证。

### O.1 效率评估协议

运行时间在单张 NVIDIA RTX 4090 上、batch size 1 的条件下测量。DP 的所有配置使用相同的 observation horizon、16 步 action horizon 和 30 个 denoising steps；完整 ForeTac 在最后 10 个 denoising steps 中启用 guidance。$\pi_{0.5}$ 的所有配置使用 10 个 flow steps，完整 ForeTac 在最后 2 步启用 guidance。每项测量先进行 50 次 warm-up replan，随后统计 200 次 replan。每个计时区间前后都执行 CUDA synchronization。报告端到端动作块推理延迟的平均值和标准差。

参数量统计区分冻结的基础策略参数，以及额外的 TacVAE、tactile foresight 和 contact-quality scorer 参数。Guidance 运算本身不引入可学习参数。表格报告 total loaded parameters，因为它决定部署内存；推理过程中所有参数均保持冻结。

**表 14：DP 详细效率分析。Replan latency 包含 observation encoding、action generation 和当前配置启用的全部 ForeTac 计算。**

| Configuration | Base M | Added M | Total M | Latency (ms) |
|---|---:|---:|---:|---:|
| DP | 338.043 | 0 | 338.043 | 132.46 ± 1.39 |
| DP+Foresight | 338.043 | 34.069 | 372.113 | 134.98 ± 1.21 |
| DP+Foresight+Scorer | 338.043 | 34.922 | 372.965 | 135.35 ± 0.86 |
| ForeTac (10 guided steps) | 338.043 | 34.922 | 372.965 | 261.26 ± 5.16 |

Forward-only scorer 行用于区分接触质量评估与基于梯度的动作细化。DP+Foresight 对最终 unguided action chunk 进行一次触觉后果预测。DP+Foresight+Scorer 进一步评估该预测，但不执行 backward pass。完整 ForeTac 则在选定的后期 denoising steps 中执行完整的 predict-score-differentiate 路径。

**表 15：Flow-matching 部署效率。所有行使用相同的 $\pi_{0.5}$ checkpoint 和 10-step integration schedule。**

| Configuration | Added M | Total M | Latency (ms) |
|---|---:|---:|---:|
| $\pi_{0.5}$ | 0 | -- | -- |
| $\pi_{0.5}$+Foresight | -- | -- | -- |
| $\pi_{0.5}$+ForeTac (2 guided steps) | -- | -- | -- |

## P. 局限性

该方法依赖于触觉前瞻模型的预测质量以及接触质量标签的覆盖度。当候选动作离开 demonstration 分布支撑时，预测的未来触觉可能变得不可靠，评分器提供的修正也可能较弱。Trust-region update 通过将 guidance 限制在局部范围内降低这一风险，但也限制了单个控制周期中可实现的改变量。系统还假设触觉 marker field 可用，并且与机器人动作时间对齐；较大的触觉标定漂移或不准确的动作标签都会降低 action-conditioned foresight 的收益。
