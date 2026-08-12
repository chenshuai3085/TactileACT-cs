# Guidance Diagnostics 数据与表达审查

## 1. 审查范围

本记录审查项目网页中的 `Guidance Diagnostics` 四面板图，以及生成该图所使用的离线日志。审查对象是黑板擦拭任务，目标是区分：

- 评分梯度在去噪过程中的局部作用；
- 梯度更新是否有界；
- 梯度在动作块中的空间分布；
- 策略去噪方向与引导方向的关系。

这组图是离线机制诊断，不等同于真实机器人成功率、安全成功率或力曲线实验。

## 2. 四个面板的数据来源

| 面板 | 数据文件 | 样本量 | 采样设置 | 面板横轴实际含义 |
|---|---|---:|---|---|
| A | `outputs/board_stride3_guidance_gradient_vis/ddpm_gradient_steps.csv` | 61 个重规划窗口；每个窗口 5 个 guided rows | 黑板 `episode_0.hdf5`；窗口间隔 16；20-step DDPM 的最后 5 个步骤；`guidance_scale=0.003` | 同一去噪阶段在 61 个窗口上的平均 score delta |
| B | 同上 | 同上 | 与 A 完全相同 | 同一去噪阶段在 61 个窗口上的平均梯度/更新范数 |
| C | `.../ce_margin_temporalstride3_e40/manifest.json` | 128 个验证窗口，四类各 32 个 | `expert`、`pressure_too_small`、`pressure_too_large`、`pressure_unstable`；动作预测长度 16；temporal stride 3 | 未来 action chunk 的时间索引 $h=0\ldots15$ |
| D | `outputs/board_stride3_ddpm_stage_guidance_audit_20260706/board_ddpm_step_guidance_sweep.json` | 3 个观测点，每点 3 个 guided steps | episode 0@304、episode 1@313、episode 2@363；DDIM 8 steps；seed 1；最后 3 步 | 该次 DDIM audit 内的局部 guided-step 序号 |

A/B 使用的 dense log 元数据为：黑板 `episode_0.hdf5`、stride-3 DP checkpoint、latent-only scorer、`expert_margin` score、`obs_horizon=4`、`pred_horizon=16`。这里有一个必须保留的契约问题：该 foresight checkpoint 的 `args.json` 设置为 `use_state_trajectory=true`，训练数据集实际从 `observations/proprio_joint` 读取未来 qpos/state trajectory；它不是严格的 action-conditioned checkpoint。C 的诊断脚本虽然从 `actions/joint_abs` 读取候选 chunk，但在该开关打开时使用 qpos normalization，因此实际属于“action chunk 作为 qpos-like trajectory 输入”的兼容性诊断。D 也复用了这个旧 checkpoint 和独立的小规模 audit。C 不是 A/B 的 61 个窗口。

## 3. A/B 的统计方式

每个重规划窗口内部记录 5 个 late DDPM steps，因此数据结构是：

$$
\text{61 windows} \times \text{5 denoising steps} \times
\{\text{guided},\text{no-guidance}\} = 610\text{ rows}.
$$

当前绘图代码先按 `ddpm_step_idx` 分组，再对 61 个窗口求均值和 SEM。因而横轴是去噪阶段，而不是窗口编号。对应关系如下：

| 图中 local step | `ddpm_step_idx` | scheduler timestep |
|---:|---:|---:|
| 1 | 15 | 20 |
| 2 | 16 | 15 |
| 3 | 17 | 10 |
| 4 | 18 | 5 |
| 5 | 19 | 0 |

A 中 guided 的逐阶段平均 `post_expert_margin - pre_expert_margin` 为：
`-0.0990, 0.0561, 0.0620, 0.0373, 0.0427`。第一阶段平均值为负，不能写成“每一步都改善”。对应的正向比例分别约为 `85.2%, 96.7%, 98.4%, 100%, 100%`，但第一阶段存在较大的负向离群值。

A 的 no-guidance 曲线恒为 0 是记录定义的结果：该分支没有施加更新，所以 `post_expert_margin == pre_expert_margin`。它不是一个独立重新采样的性能基线。

B 的 guided 平均 raw gradient norm 为：
`15.54, 21.07, 19.07, 11.95, 15.05`；乘以 `lambda=0.003` 后为：
`0.0466, 0.0632, 0.0572, 0.0358, 0.0451`。applied update norm 在五个阶段都约为 `0.003`，因为梯度先归一化，再按 trust-region scale 裁剪。

## 4. C/D 的含义与限制

### C：动作块中的梯度分布

C 对 128 个验证窗口计算：

$$
\text{candidate chunk}
\rightarrow \text{qpos-conditioned foresight checkpoint}
\rightarrow \hat{\mathbf z}_{t:t+15}
\rightarrow S.
$$

`latent_only` 设置下，评分器直接使用 action 的分支被 detach，热图主要反映预测触觉路径对候选 chunk 的梯度。当前热图是 `0.003 × mean(signed gradient)`，形状为 `7 joints × 16 future chunk steps`。由于正负梯度直接求平均可能相互抵消，热图的浅色区域不一定表示“没有梯度”。正式分析应同时报告 `mean(|gradient|)` 或使用逐窗口标准化后的幅值热图。由于 checkpoint 的 `use_state_trajectory=true`，这里的梯度不能直接作为严格 action-conditioned foresight 的证据。

当前最大绝对梯度维度只有 `j4`（77/128）和 `j2`（51/128）两类；`j0` 到 `j6` 尚未映射为实际关节名称，因此论文中不能把它们解释成具体物理关节。

### D：策略方向与引导方向

D 的三个观测点来自独立的 DDIM audit，底层 scheduler timestep 实际为 `24, 12, 0`。当前代码将它们重新编号为 local step `1, 2, 3`，所以图上看起来像与 A/B 相同的去噪横轴，实际上不是同一 scheduler、同一批窗口或同一数量级的统计。三步平均 cosine alignment 为：
`-0.000013, -0.0612, -0.0124`。

这只能说明当前小样本中引导更新大多是侧向或轻微反向修正，不能证明引导方向与 DP 方向一致，也不能单独证明引导有效。`n=3` 时误差带和趋势都不适合作为强结论。

## 5. 网页和论文中的歧义

网页正文写成“denoising logs and 128 validation windows”，图注又只在 C 中写 128，容易让读者误以为 A、B、D 也使用 128 个窗口。四个面板还混用了 61、128、3 三种样本集合。另一个问题是，当前网页图文件与 n=128 重新生成的 artifact 不是字节级相同文件，后续应固定唯一生成脚本和数据清单。

图中 score 是学习到的 `expert_margin`，不是任务成功率、真实力改善或安全指标。评分器在当前验证 split 上的高分类指标也不能替代 paired real-rollout evaluation。

## 6. 建议的正式版本

建议将图题改为 `Offline Guidance Diagnostics for Blackboard Wiping`，并在图内或图注中明确：

1. A/B：`n=61 re-planning windows; DDPM t=20,15,10,5,0`；
2. C：`n=128 balanced scorer-validation windows; h=0...15`；
3. D：`n=3 DDIM observation points; t=24,12,0`；
4. 所有 score 均为 offline learned expert-margin score。

若保留四面板，A 的 no-guidance 零线应改成注释而不是性能对照；B 应标出固定的 `trust-region update norm=0.003`；C 应增加 mean-absolute-gradient 版本；D 最好使用与 A/B 相同的 61 个窗口和 scheduler 重新计算，否则应降为补充材料的小样本方向审计。

更严格的主文版本应先使用 `use_state_trajectory=false`、明确以 `actions/joint_abs[t:t+H]` 为条件重新训练或选择 checkpoint，并在同一组 observation、候选 action 和 noise seed 上统一重算 A-D，再用 paired score change、positive-improvement rate 和 action-update magnitude 作为核心统计量。当前这张图只能作为旧 qpos-conditioned pipeline 的离线机制审计，不能支撑“action-conditioned ForeTac 已验证”的表述。

## 7. 数据选择是否合理

当前选择对于“检查梯度链路能否运行、更新是否有限、梯度是否集中”是合理的：A/B 覆盖一个完整黑板 episode 的多个时间位置，C 按四个接触质量类别等量抽样，便于观察类别平衡下的梯度分布。

但它不适合作为最终的泛化统计：A/B 的 61 个窗口全部来自同一个 episode，窗口之间还存在时间相关性，因此有效独立样本数小于 61；D 只有 3 个观测点，只能算 smoke audit。正式实验应按 episode 划分数据，在多个 held-out episodes 上分层抽取 approach、initial contact、stable contact 和 recovery 阶段，并让 A-D 使用相同的 observation/action/noise 配对。C 的 32 samples/class 适合画热图，最终数值应使用完整验证集并按 episode bootstrap 或报告 episode-level confidence interval。

## 8. 2026-08-12 正式绘图修订

主文 Fig. 6 不再把不同窗口的聚合统计画成连续去噪曲线。上半部分直接按一条记录执行的原始 `Timestep`（0--864）展示零偏置后的 $F_x/F_y/F_z$ 与 $[0,1]$ contact-quality score，并用背景区分 Approach、Contact、Wiping 和 Release。这里不再使用 `Task progress (%)`，避免归一化进度掩盖原始采样长度。

下半部分将机制证据拆成三个互不混淆的统计对象：305 次 same-state paired updates 的 expert-margin 变化分布；finite gradient、positive score gain 和 bounded update 的比例；以及重新计算的 1,024 个 balanced held-out windows（四类各 256）的 mean-absolute scaled-gradient heatmap。后者保存于 `outputs/board_stride3_val_guidance_debug_20260812_n1024/`，1,024 次梯度计算全部成功，没有 skipped window。图内和图注分别明确 305 updates 与 1,024 windows 的统计单位。
