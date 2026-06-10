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

### 生产通过条件

真实 rollout gate 的通过条件被设计得比较保守，避免“小样本均值略高”误判：

```text
baseline_n >= 10
guided_n >= 10
guided_quality_mean - baseline_quality_mean >= 0.03
bootstrap 95% CI lower bound of quality delta > 0
bad/risk/rough flag rate increase <= 0.05
paired episodes, if available, must have positive mean quality delta
```

其中 bootstrap 默认 `2000` 次，可用参数调整：

```bash
--min_episodes 10
--min_quality_delta 0.03
--max_bad_rate_increase 0.05
--bootstrap_samples 2000
```

调试 smoke 使用 `--min_episodes 2 --bootstrap_samples 200` 时，虽然均值看起来提升，但 CI 下界为负，因此仍不通过。这是合理结果，说明 gate 对统计不确定性敏感。

### 显式配对实验

如果 baseline 和 guided 是同一组 trial 的配对实验，但 HDF5 文件名不同，应使用 `--pairing_csv`，不要依赖自动 stem 匹配。

CSV 格式：

```csv
pair_id,baseline,guided
trial_001,baseline_episode_001.hdf5,guided_episode_001.hdf5
trial_002,baseline_episode_002.hdf5,guided_episode_002.hdf5
```

配对文件中的 `baseline` 和 `guided` 可以是绝对路径、相对各自目录的路径，或可唯一匹配的文件名/stem。示例：

```bash
python TFAC_V5/eval_real_rollout_quality_gate.py \
  --task board \
  --baseline_dir /path/to/baseline_hdf5_dir \
  --guided_dir /path/to/guided_hdf5_dir \
  --pairing_csv /path/to/pairs.csv \
  --tag board_paired_validation
```

有显式配对时，gate 会额外计算 paired bootstrap CI。默认配对实验可以用 paired CI 证明提升；如果想同时要求 aggregate CI 也为正，可以加：

```bash
--require_aggregate_ci_for_paired
```

如果用户为了调试把 `--min_episodes` 调低到 10 以下，报告会写入：

```text
debug_or_underpowered = true
```

这种结果只能说明脚本和统计流程可运行，不能作为正式生产验证结论。

### 任务成功元数据

真实生产验证不能只看触觉质量，还必须检查任务是否成功、是否提前停止。若 HDF5 attrs 中没有 `success` 或 `stopped_early`，可用 `--metadata_csv` 提供人工或采集日志标注。

CSV 至少包含 `file`、`path` 或 `stem` 之一，可选字段：

```csv
stem,success,stopped_early
episode_001,1,0
episode_002,0,1
```

也支持同义字段：

```text
task_success -> success
early_stop -> stopped_early
```

示例：

```bash
python TFAC_V5/eval_real_rollout_quality_gate.py \
  --task insertion \
  --baseline_dir /path/to/baseline \
  --guided_dir /path/to/guided \
  --pairing_csv /path/to/pairs.csv \
  --metadata_csv /path/to/rollout_metadata.csv
```

生产通过时额外要求：

```text
guided_success_rate >= baseline_success_rate - max_success_rate_drop
guided_stopped_early_rate <= baseline_stopped_early_rate + max_bad_rate_increase
```

默认 `--max_success_rate_drop 0.0`，即 guided 不能降低任务成功率。

## 2026-06-10 离线 gate 增加强制梯度可用性检查

TacQualityEnergy 的目标是做 DP classifier guidance，因此 offline production gate 不能只检查分类/评分效果，还必须检查 score 是否真的能作为局部可微势能。已将下面两项加入 `TFAC_V5/eval_ptg_offline_production_gate.py` 的 required deployment checks。

### 1. Local guidance scale sweep

输入：

```text
/home/chenshuai/Project/output/tac_quality_guidance_scale_sweep/tac_quality_guidance_scale_sweep.json
```

检查内容：

```text
overall_pass = true
insertion.passes_guidance_scale_sweep = true
board.passes_guidance_scale_sweep = true
insertion.recommended_improved_rate >= 0.95
board.recommended_improved_rate >= 0.95
```

当前结果：

```text
insertion_recommended_scale = 0.08
insertion_recommended_improved_rate = 0.984375
board_recommended_scale = 0.0016
board_recommended_improved_rate = 1.0
```

含义：在真实插座和黑板样本上，沿 `d score / d action` 做小步更新时，TacQualityEnergy 分数能稳定提升，并且更新被 trust region 约束。

### 2. Current-gradient robustness

输入：

```text
/home/chenshuai/Project/output/tac_quality_guidance_robustness/tac_quality_guidance_robustness.json
```

检查内容：

```text
overall_pass = true
insertion.passes_current_gradient_robustness = true
board.passes_current_gradient_robustness = true
insertion.worst_perturbed_gradient_improved_rate >= 0.95
board.worst_perturbed_gradient_improved_rate >= 0.95
```

当前结果：

```text
insertion_worst_perturbed_gradient_improved_rate = 0.99609375
board_worst_perturbed_gradient_improved_rate = 1.0
```

含义：在小的 tactile/action 扰动下，重新计算当前 `d score / d action` 仍能提升分数。这一点比 stale-gradient 稳定性更重要，因为部署时每次 guidance 都应该在当前 action/predicted tactile 上重算梯度。

### 更新后的 offline gate 状态

运行：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_ptg_offline_production_gate.py
```

当前输出：

```text
offline_production_gate_pass = true
remaining_required_step = Real robot / final production policy validation.
```

因此当前离线结论更严格：

```text
TacQualityEnergy 不仅能分类/评分，而且满足作为 DP final-action trust-region gradient guidance 的局部可微性和扰动鲁棒性要求。
```

仍然不能声称：

```text
real robot validation completed
```

## 2026-06-10 目标完成度审计

新增脚本：

```text
TFAC_V5/audit_tac_quality_goal_completion.py
```

该脚本不同于 offline production gate：

| 工具 | 回答的问题 |
|---|---|
| `eval_ptg_offline_production_gate.py` | 当前 scorer/guidance stack 是否足够进入 robot dry-run |
| `audit_tac_quality_goal_completion.py` | 用户目标是否已经真正完成 |

运行：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/audit_tac_quality_goal_completion.py
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_goal_audit/tac_quality_goal_completion_audit.json
/home/chenshuai/Project/output/tac_quality_goal_audit/tac_quality_goal_completion_audit.md
```

当前审计结果：

```text
objective_complete = false
status = incomplete
n_requirements = 10
n_blockers = 2
```

已经满足的要求包括：

```text
1. 插座任务使用 episode-level generalization 评估；
2. 黑板任务使用力大小和柔顺性定义弱监督质量标准；
3. 统一 task-conditioned differentiable scorer 已训练/评估；
4. scorer 通过 local guidance scale sweep；
5. scorer 通过 current-gradient robustness；
6. offline production-readiness gate 通过；
7. deployment manifest 存在；
8. 工作记录和研究文档已记录。
```

仍然缺失的两个 blocker：

```text
1. Formal socket insertion baseline-vs-guided production/robot rollout validation passes.
2. Formal board wiping baseline-vs-guided production/robot rollout validation passes.
```

因此当前最准确的状态是：

```text
offline-ready scorer/guidance package complete;
final user objective incomplete until formal real/production rollouts pass.
```

## 2026-06-10 真实 Rollout 最终验收协议

新增机器可读协议：

```text
TFAC_V5/build_tac_quality_real_rollout_acceptance_protocol.py

/home/chenshuai/Project/output/tac_quality_real_rollout_acceptance_protocol/
  tac_quality_real_rollout_acceptance_protocol.json
  tac_quality_real_rollout_acceptance_protocol.md
```

该协议把“最终怎样证明 TacQualityEnergy 可以用于 DP classifier guidance”固定为 formal paired12 真实 rollout 验收，而不是继续看 frame-level 分类准确率或 synthetic smoke。

### 采集要求

```text
insertion paired_n_pairs = 12
board paired_n_pairs = 12

formal arms:
  baseline
  default_guided
  distilled_guided

optional arm:
  action_aware_guided
```

### 必须关闭的 4 个 blocker

```text
1. insertion baseline-vs-default_guided two-arm real rollout gate
2. board baseline-vs-default_guided two-arm real rollout gate
3. insertion baseline-vs-default_guided-vs-distilled_guided three-arm scorer ablation
4. board baseline-vs-default_guided-vs-distilled_guided three-arm scorer ablation
```

对应 artifact：

```text
/home/chenshuai/Project/output/real_rollout_quality_gate/insertion_baseline_vs_guided/real_rollout_quality_gate.json
/home/chenshuai/Project/output/real_rollout_quality_gate/board_baseline_vs_guided/real_rollout_quality_gate.json
/home/chenshuai/Project/output/real_rollout_scorer_ablation_gate/insertion_baseline_vs_default_vs_distilled/real_rollout_scorer_ablation_gate.json
/home/chenshuai/Project/output/real_rollout_scorer_ablation_gate/board_baseline_vs_default_vs_distilled/real_rollout_scorer_ablation_gate.json
```

### 通过条件

two-arm gate 至少要求：

```text
min_episodes_each_arm >= 10
quality_delta_mean >= 0.03
bootstrap_ci95_low > 0
bad/risk/rough rate increase <= 0.05
success rate must not drop
debug_or_underpowered = false
```

three-arm ablation 至少要求：

```text
production_ablation_pass = true
debug_or_underpowered = false
recommended_real_scorer is not null
at least one guided arm passes vs baseline
guided-vs-guided CI selects a winner or reports tie
```

明确不能算作完成：

```text
synthetic HDF5 smoke outputs
frame-level random cross validation
score-only improvement without non-degradation checks
server launch smoke without recorded rollout HDF5s
optional ActionAware pass without formal baseline/default/distilled gates
```

当前状态：

```text
protocol_pass = true
scientific_evidence = false
```

解释：验收标准已经定义并接入 manifest/audit/evidence summary；但这不是最终科学证据，必须采集真实 rollout 后运行 gates。

### 真实 rollout 验证准备工具

新增脚本：

```text
TFAC_V5/prepare_real_rollout_validation.py
```

用途：真实采集 baseline DP 和 TacQuality-guided DP 两组 HDF5 rollout 后，先用它检查目录是否能进入正式 gate，并自动生成：

```text
pairing_template.csv
metadata_template.csv
real_rollout_validation_readiness.json
real_rollout_validation_readiness.md
```

示例：

```bash
python TFAC_V5/prepare_real_rollout_validation.py \
  --task insertion \
  --baseline_dir /path/to/baseline_insert_rollouts \
  --guided_dir /path/to/guided_insert_rollouts \
  --output_dir /home/chenshuai/Project/output/real_rollout_validation_ready \
  --tag insertion_formal_ready
```

输出中会包含可直接运行的正式 gate 命令：

```bash
python TFAC_V5/eval_real_rollout_quality_gate.py \
  --task insertion \
  --baseline_dir <baseline_hdf5_dir> \
  --guided_dir <guided_hdf5_dir> \
  --pairing_csv <generated_pairing_template.csv> \
  --metadata_csv <generated_metadata_template.csv> \
  --output_dir /home/chenshuai/Project/output/real_rollout_quality_gate \
  --tag insertion_baseline_vs_guided
```

当前用 smoke HDF5 运行 sanity：

```text
/home/chenshuai/Project/output/real_rollout_validation_ready/board_smoke_ready/real_rollout_validation_readiness.json
```

结果：

```text
baseline_n = 2
guided_n = 2
ready_for_quality_gate = false
blocking_issues = success/stopped_early metadata missing
```

这是预期结果：smoke 文件不是正式验证数据，但脚本能正确发现 HDF5 并要求补任务成功/提前停止 metadata。

脚本也支持校验用户已经填写的 CSV：

```bash
python TFAC_V5/prepare_real_rollout_validation.py \
  --task board \
  --baseline_dir /home/chenshuai/Project/output/real_rollout_quality_gate_smoke_input/board_baseline \
  --guided_dir /home/chenshuai/Project/output/real_rollout_quality_gate_smoke_input/board_guided \
  --pairing_csv /tmp/ptg_pairing_smoke.csv \
  --metadata_csv /tmp/ptg_metadata_smoke.csv \
  --output_dir /home/chenshuai/Project/output/real_rollout_validation_ready \
  --tag board_smoke_ready_with_csv \
  --min_episodes 2
```

当前结果：

```text
ready_for_quality_gate = true
pairing_csv_check.ready = true
metadata_csv_check.ready = true
```

输出：

```text
/home/chenshuai/Project/output/real_rollout_validation_ready/board_smoke_ready_with_csv/real_rollout_validation_readiness.json
```

这说明正式数据采集后，可以先用该脚本发现配对路径错误、metadata 覆盖不完整、success/stopped_early 值非法等问题，再运行正式 quality gate。

### 真实 rollout 样本量规划

新增脚本：

```text
TFAC_V5/plan_real_rollout_sample_size.py
```

用途：采集真实 baseline/guided rollout 前，先根据当前 gate 的统计条件估计需要多少条实验。它不替代正式 gate，只用于规划采集规模。

默认运行：

```bash
python TFAC_V5/plan_real_rollout_sample_size.py --task insertion --tag insertion_default_plan
python TFAC_V5/plan_real_rollout_sample_size.py --task board --tag board_default_plan
```

当前默认假设：

```text
expected_quality_delta = 0.08
expected_paired_delta_std = 0.10
expected_group_quality_std = 0.18
min_quality_delta = 0.03
min_episodes = 10
min_recommended = 12
```

输出：

```text
/home/chenshuai/Project/output/real_rollout_sample_size_plan/insertion_default_plan/real_rollout_sample_size_plan.json
/home/chenshuai/Project/output/real_rollout_sample_size_plan/board_default_plan/real_rollout_sample_size_plan.json
```

建议：

```text
paired design: at least 12 baseline/guided pairs per task
unpaired design: about 28 baseline + 28 guided rollouts per task
```

推荐优先 paired design，因为它减少初始条件、场景、轨迹难度差异带来的方差，也和 `eval_real_rollout_quality_gate.py --pairing_csv` 的 paired bootstrap CI 更一致。

### 正式 rollout 实验包

新增脚本：

```text
TFAC_V5/build_real_rollout_experiment_packet.py
```

运行：

```bash
python TFAC_V5/build_real_rollout_experiment_packet.py --tag formal_paired12
```

输出目录：

```text
/home/chenshuai/Project/output/real_rollout_experiment_packet/formal_paired12
```

内容：

```text
README.md
real_rollout_experiment_packet.json
insertion/
  README.md
  pairing_template.csv
  metadata_template.csv
  baseline_metadata_template.csv
  guided_metadata_template.csv
board/
  README.md
  pairing_template.csv
  metadata_template.csv
  baseline_metadata_template.csv
  guided_metadata_template.csv
```

每个任务的 template 都按 12 对 paired trials 生成。`README.md` 中包含采集 checklist、`prepare_real_rollout_validation.py` 命令和 `eval_real_rollout_quality_gate.py` 命令。

该 packet 是两个剩余 blocker 的正式执行包，但仍然不代表真实验证已经完成。只有采集 HDF5 后运行 gate 并得到：

```text
production_validation_pass = true
debug_or_underpowered = false
```

才能关闭对应任务 blocker。

## 2026-06-10 Formal paired12 rollout runbook

新增执行手册生成脚本：

```text
TFAC_V5/build_tac_quality_formal_rollout_runbook.py
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_formal_rollout_runbook/formal_paired12/
  tac_quality_formal_rollout_runbook.json
  tac_quality_formal_rollout_runbook.md
```

用途：把最终真实验证需要的所有操作集中到一份文件，避免在 protocol、launch sheet、readiness、post-collection pipeline 之间切换。

runbook 中包含：

```text
rollout_root:
  /home/chenshuai/Project/output/tac_quality_formal_rollouts

tasks:
  insertion
  board

formal arms:
  baseline
  default_guided
  distilled_guided

optional:
  action_aware_guided
```

每个任务和 arm 都列出：

```text
rollout_dir
launch_command
needed_hdf5
current_hdf5
missing_hdf5
ready
```

采集后按 runbook 中的顺序执行：

```bash
python TFAC_V5/run_tac_quality_post_collection_pipeline.py --tag formal_paired12
python TFAC_V5/run_tac_quality_post_collection_pipeline.py --tag formal_paired12 --run_gates
python TFAC_V5/audit_tac_quality_goal_completion.py
```

当前状态：

```text
runbook_pass = true
scientific_evidence = false
ready_for_gate_runner = false
```

解释：runbook 已经完整；`ready_for_gate_runner=false` 表示真实 rollout 还没采集够，不是脚本失败。采集完成后应先跑 preflight，确认 pairing/metadata/HDF5 schema 都 ready，再跑 `--run_gates`。

### Runbook smoke

新增结构检查脚本：

```text
TFAC_V5/smoke_tac_quality_formal_rollout_runbook.py
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_formal_rollout_runbook_smoke/formal_paired12/
  tac_quality_formal_rollout_runbook_smoke.json
  tac_quality_formal_rollout_runbook_smoke.md
```

该 smoke 不重复启动 server，而是检查 runbook 是否完整且和 launch-sheet smoke 对齐：

```text
overall_pass = true
scientific_evidence = false
```

关键检查：

```text
baseline command contains --disable_guidance
guided commands keep guidance enabled
all formal arms use for_show_xiaomi.serve_dp_tac_quality_guided
post_collection preflight and --run_gates commands exist
goal audit command exists
completion blockers remain exactly 4
cannot_count_as_completion guardrails are present
formal_launch_sheet_smoke already passes
```

这一步的意义是：在真实采集前，操作手册、启动命令、采后 gate、不能算完成的 guardrail 已经一致。它不代替真实 rollout gate。

## 2026-06-10 Formal paired12 collection schedule

新增采集顺序排程脚本：

```text
TFAC_V5/build_tac_quality_collection_schedule.py
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_collection_schedule/formal_paired12/
  tac_quality_collection_schedule.json
  tac_quality_collection_schedule.md
  tac_quality_collection_schedule.csv
```

用途：减少真实 rollout 采集中的顺序偏差。不要总是先采 baseline 或总是先采 guided；应按 CSV 中的 `pair_id` 和 `within_pair_order` 执行。

每个任务 12 个 paired triplets，每个 triplet 包含：

```text
baseline
default_guided
distilled_guided
```

排程使用 6 个 arm 排列，每个重复 2 次，使每个 arm 在第 1/2/3 个执行位置各出现 4 次。

当前验证：

```text
schedule_pass = true
n_rows = 72
scientific_evidence = false
```

执行要求：

```text
1. 同一个 pair_id 内尽量保持初始设置匹配；
2. 不要根据中途结果改变后续 arm 顺序；
3. 每条 HDF5 保存到 schedule 指定的 rollout_dir；
4. 采集完成后运行 post-collection pipeline 生成实际 pairing/metadata；
5. schedule 本身不算完成证据，只有真实 HDF5 gate 通过才算。
```
