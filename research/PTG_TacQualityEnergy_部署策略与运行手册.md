# PTG TacQualityEnergy 部署策略与运行手册

日期：2026-06-10

## 目标

最终目标不是重排候选动作，而是在 DP 生成动作时提供可微的触觉质量势函数：

```text
action -> Foresight 预测未来触觉 -> TacQualityEnergy 评分 -> d score / d action -> 修正 action
```

因此评分器必须同时满足两个条件：

1. 能准确区分/评估触觉后果好坏；
2. score 对 action 可微，并且局部梯度上升能提高预测触觉质量。

## 好坏标准

### 插座任务

插座任务的坏样本只从 bounce episode 中定义：

- `good_insert`：success episode 的 insert 阶段；
- `pre_bounce_risk`：即将碰外壁、会导致 bounce 的阶段；
- `impact_or_recovery`：已经 bounce 或 bounce 后恢复；
- `weak/approach`：接触不足或靠近阶段，不作为二分类 good/bad 的主监督，只作为 reason/phase 辅助。

主目标是避免 `pre_bounce_risk` 和 `impact_or_recovery`，同时保留正常插入触觉。

### 擦黑板任务

擦黑板目前没有人工好坏标注，因此使用物理弱标签：

- `good_smooth`：擦拭力大小处于合适区间，并且力/动作/marker 变化平滑；
- `too_light`：力过小，接触不足；
- `too_heavy`：力过大或峰值过大；
- `rough_force`：力变化不柔顺；
- `rough_motion`：动作或 marker 变化不平滑。

当前最好弱标签方案来自 `left_force + t5_scoreband + both_marker_actions`。RF/GBM teacher 可以很好复现该标准；可微 MLP 作为部署 scorer。

## 为什么不是只做二分类

只做 `good/bad` 分类不够适合作 DP 梯度引导：

1. `p_good` 和 `log_p_good` 容易饱和，分类准确但梯度弱；
2. 二分类没有说明坏的原因，不能区分力过小、力过大、即将碰壁、动作粗糙；
3. DP guidance 需要连续坡度，最好有可排序的质量分数；
4. 不同任务的“好”标准共享一部分物理结构，但不能用统一阈值硬判定。

当前采用 multi-head scorer：

```text
binary head: good / bad
reason head: weak / good / risk-heavy / impact-rough-force / rough-motion
quality head: continuous quality
energy score: quality_logit + binary_margin + reason_margin 的加权组合
```

分类解释使用 `p_good + reason_prob + quality_score`，梯度引导使用未压缩或 clipped 的 logit energy。

## 当前最佳评分器

推荐名称：`TacQualityEnergy`

### 插座

默认使用：

```text
InsertionRiskScorerRuntime
energy = 0.50 * quality_logit + 0.10 * binary_margin
energy_clipped = 4 * tanh(energy / 4)
```

关键证据：

- GroupKFold binary AUC 约 `0.9877`；
- full-chain gradient 通过，`action -> Foresight -> decoder -> marker -> scorer -> score` 没有断图；
- constrained clean-action refinement 在 N=40 测试中通过；
- controller real-sample audit improved rate 约 `0.9961`；
- 每步使用当前梯度是稳定的，复用旧梯度不稳定。

### 擦黑板

默认使用：

```text
PTGProxyScorerV2Runtime
energy = 0.75 * quality_logit + 0.10 * binary_margin
```

关键证据：

- 黑板 scorer binary AUC 接近 `0.98-0.999`，quality correlation 约 `0.91-0.96`；
- blackboard weak label 由力大小和柔顺度定义，符合任务目标；
- board feature-cache full-chain heldout smoke 通过；
- 当前仍需要真机或人工小样本标注校准弱标签阈值。

## 当前推荐部署方式

推荐模式：

```text
DP 正常完成 denoising
  -> 得到 clean/final action
  -> Foresight 预测未来触觉
  -> TacQualityEnergy 打分
  -> trust-region action refinement
  -> 只接受 score 提高且不越界的更新
```

这是梯度引导，不是简单 reranking。它直接对最终动作做 `d score / d action` 更新，但把更新限制在小 trust region 内。

当前默认约束：

- 插座：`refine_steps=4`，`action_step=0.02`，`max_total_delta=0.08`；
- 擦黑板：`refine_steps=4`，`action_step=0.0002`，`max_total_delta=0.02`；
- 每一步重新计算 `action -> Foresight -> TacQualityEnergy` 的梯度；
- 只接受 constrained objective 提高的更新；
- 使用 smoothness penalty 和 joint/action limit barrier；
- 不允许复用 stale gradient。

## 暂不推荐的方式

以下方式目前只作为研究诊断，不建议直接上线：

1. 每个 DDPM step 都无条件加 classifier guidance；
2. 大 guidance scale；
3. 只用 `p_good` 或 `log_p_good` 作为势函数；
4. 复用上一时刻/上一候选的梯度；
5. 使用统一 `P(good)>0.5` 阈值跨任务硬判定；
6. 把 L1-to-expert 当作最终触觉质量目标。

原因：controller-in-denoising smoke 显示，局部 score 提升不能稳定转化为最终 denoised sample score 提升。final/clean-action trust-region refinement 当前更稳。

## 与 classifier guidance / CFG 的关系

classifier guidance 的核心是：

```text
guided update = diffusion update + scale * grad_x log p(class | x)
```

在本项目中，`x` 是 action chunk，不是图像。对应为：

```text
grad_action TacQualityEnergy(Foresight(obs, action), action)
```

CFG 的思想是用 conditional 与 unconditional score 的差值控制生成方向。对当前阶段，更实际的是先使用外部可微 scorer 的 classifier guidance；未来可以把 TacQualityEnergy 蒸馏进 DP 训练，形成 classifier-free / value-conditioned policy。

## 证据文件

主要代码：

- `TFAC_V5/tac_quality_guidance_runtime.py`
- `TFAC_V5/tac_quality_dp_guidance_controller.py`
- `TFAC_V5/tac_quality_trust_region_guidance.py`
- `TFAC_V5/tac_quality_guidance_config.py`
- `TFAC_V5/train_insertion_risk_scorer.py`
- `TFAC_V5/train_ptg_proxy_scorer_v2.py`

主要输出：

- `/home/chenshuai/Project/output/ptg_guidance_evidence/ptg_guidance_evidence_summary.json`
- `/home/chenshuai/Project/output/tac_quality_guidance_manifest/tac_quality_guidance_manifest.json`
- `/home/chenshuai/Project/output/full_chain_guidance_gradient/insertion_full_chain_energy_clipped_K8_N16.json`
- `/home/chenshuai/Project/output/clean_action_energy_refinement/insertion_clean_refine_constrained_K4_N40.json`
- `/home/chenshuai/Project/output/tac_quality_dp_guidance_controller/controller_real_sample_audit.json`
- `/home/chenshuai/Project/output/tac_quality_controller_denoising_smoke/`

## 当前结论

`TacQualityEnergy` 已经是当前最合理的评分/分类器方案：

- 有明确好坏标准；
- 有多类失败原因；
- 有连续质量分数；
- 有非饱和 energy 势函数；
- 能对 action 求梯度；
- 插座和擦黑板都有离线证据；
- 当前安全部署方式是 final/clean-action trust-region refinement。

但还不能声称最终完成真实机器人部署。剩余关键验证是：

```text
真实机器人或最终生产策略验证：
  baseline DP action
  vs TacQualityEnergy-guided action
  比较真实触觉后果、力大小、力平滑、bounce rate、任务成功率和动作安全性。
```

## 2026-06-10 离线 Gate 增强

本轮将 offline production gate 从“只检查正向通过项”改成同时检查部署策略约束：

1. `final_clean_action_trust_region_refinement` 必须通过；
2. `controller_in_every_ddpm_step` 必须保持 research-only；
3. gate 输出中显式记录 denoising controller diagnostic 的负结果。

原因：之前的 controller-in-denoising smoke 显示，每一步局部 score 可以提高，但最终 denoised action 的逐样本提升不稳定。因此不能因为 scorer 和 controller API 都可用，就默认 every-step guidance 已经能上线。

当前 gate 语义：

```text
offline_production_gate_pass = true
含义：可以进入机器人 dry-run / 生产策略验证
不含义：every-DDPM-step guidance 已经生产可用
推荐：final clean-action trust-region gradient refinement
```

## 2026-06-10 真实 Rollout 质量 Gate

新增真实 rollout 结果评估入口：

- `TFAC_V5/eval_real_rollout_quality_gate.py`

用途：

```text
baseline DP 真机/生产策略 HDF5 rollouts
vs
TacQualityEnergy-guided DP 真机/生产策略 HDF5 rollouts
```

该脚本从 HDF5 中读取：

- `ft`
- `observations/tac/{left,right}/force6d`
- `observations/tac/{left,right}/marker_offset`
- `actions/{eef_abs,joint_abs}`

插座任务指标：

- `risk_proxy`
- `impact_proxy`
- `risk_flag`
- `quality_score = exp(-risk)`

擦黑板任务指标：

- `force_band_score`
- `smoothness_score`
- `too_light_flag`
- `too_heavy_flag`
- `rough_flag`
- `quality_score = force_band_score * smoothness_score`

示例命令：

```bash
python TFAC_V5/eval_real_rollout_quality_gate.py \
  --task board \
  --baseline_dir /path/to/baseline_hdf5_dir \
  --guided_dir /path/to/guided_hdf5_dir \
  --output_dir /home/chenshuai/Project/output/real_rollout_quality_gate \
  --tag board_baseline_vs_guided
```

为了避免误报，默认要求 baseline 和 guided 各至少 10 条 rollout。当前 smoke 只用已有数据做输入管线测试，因此输出应为：

```text
production_validation_pass = false
reason = Insufficient rollout count
```

这说明脚本入口可运行，但真实 production validation 仍需要采集正式 baseline/guided 两组 rollout 后再判断。
