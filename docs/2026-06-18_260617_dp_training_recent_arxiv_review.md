# 2026-06-18 260617-only DP 训练监督与近两个月相关工作调研

## 当前训练

训练目标：只使用 `/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617` 训练擦黑板 DP，2000 epoch。

训练脚本：

- `scripts/train/train_dp_tac_concat_board_260617_only.sh`
- `scripts/train/watch_dp_tac_concat_board_260617_only.sh`

输出目录：

- `/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000`

数据与参数：

- 原始图像：`global,wrist`, `200x266`
- proprio：`observations/proprio_joint`
- action：`actions/joint_abs`
- tactile：左手 `observations/tac/left/marker_offset`
- TactileVAE：`/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt`
- `pred_horizon=16`, `obs_horizon=2`, `n_action_steps=8`, `tac_history=8`
- `batch_size=64`, `epochs=2000`, `lr=1e-4`, `val_ratio=0.1`
- `max_train_windows=8192`, `max_val_windows=1024`
- `save_freq=500`, `topk_k=3`

当前观察到的启动结果：

- 可用 episode：79 / 80；`episode_1.hdf5` 缺 `observations/proprio_joint`，训练自动跳过。
- 第 7 epoch：`train=0.036236`, `val=0.036695`。
- `dp_best.pth`, `dp_latest.pth`, top-k checkpoint 正常写入。
- 由于 home 盘空间紧张，将周期 checkpoint 从每 100 epoch 改成每 500 epoch。

2026-06-18 11:46 监督更新：

- 训练进程仍在运行，PID `1544542`。
- GPU：RTX 4090，显存约 14.8GB，训练进程约 13.9GB。
- 磁盘：`/home/chenshuai/Project/output` 剩余约 50GB，checkpoint 较大，仍需持续监督。
- 最新完整日志到第 19 epoch，正在跑第 20 epoch：
  - 第 15 epoch：`train=0.021321`, `val=0.022302`, best 更新；
  - 第 16 epoch：`train=0.020619`, `val=0.020317`, best 更新；
  - 第 17 epoch：`train=0.021183`, `val=0.023113`；
  - 第 18 epoch：`train=0.020093`, `val=0.023806`；
  - 第 19 epoch：`train=0.019390`, `val=0.022509`。
- 当前判断：早期 train loss 继续下降，val 在第 16 epoch 后有回弹，但还属于早期波动，尚不能判断过拟合。
- 当前 checkpoint：
  - `dp_best.pth`
  - `dp_latest.pth`
  - top-k checkpoint 保持 3 个左右。

2026-06-18 11:57 监督更新：

- 第 27 epoch：`train=0.017194`, `val=0.017955`，best 更新；
- 第 28 epoch：`train=0.017041`, `val=0.016746`，best 更新；
- 第 29 epoch：`train=0.016361`, `val=0.016544`，best 更新；
- GPU 利用率约 94%，磁盘剩余约 47GB；
- 当前判断：第 17-26 epoch 的 val 回弹属于早期波动，训练仍在有效下降，不需要暂停或改参数。

2026-06-18 12:01 监督更新：

- 第 33 epoch：`train=0.015655`, `val=0.016474`，best 更新；
- 第 34 epoch：`train=0.015371`, `val=0.016327`，best 更新；
- 第 35 epoch：`train=0.015362`, `val=0.017221`；
- 当前 best：第 34 epoch，`val=0.016327`；
- 磁盘剩余约 49GB；
- 当前判断：训练仍健康，val 有小幅波动但整体明显优于启动初期。

2026-06-18 12:04 监督更新：

- 第 38 epoch：`train=0.014987`, `val=0.015768`，best 更新；
- 第 39 epoch：`train=0.014409`, `val=0.016439`；
- 第 40 epoch：`train=0.014239`, `val=0.015603`，best 更新；
- 当前 best：第 40 epoch，`val=0.015603`；
- 相比第 7 epoch `val=0.036695`，验证 loss 已下降约 57.5%；
- GPU 利用率约 92%，磁盘剩余约 47GB。

2026-06-18 12:07 监督更新：

- 第 43 epoch：`train=0.013797`, `val=0.012587`，best 明显更新；
- 第 46 epoch：`train=0.013545`, `val=0.014939`；
- 第 47 epoch：`train=0.014049`, `val=0.015377`；
- 当前 best：第 43 epoch，`val=0.012587`；
- 说明：第 43 epoch 是明显低谷，后续几个 epoch 回到 `0.015` 左右，因此后续仍需观察这个 best 是否稳定复现，不能只凭单点低谷判断最终泛化。

2026-06-18 12:31 监督更新：

- 训练进程仍在运行，PID `1544542`；watcher PID `1563456`。
- 最新到第 89 epoch：
  - 第 82 epoch：`train=0.011370`, `val=0.011510`，best 更新；
  - 第 83 epoch：`train=0.010874`, `val=0.013345`；
  - 第 88 epoch：`train=0.010858`, `val=0.014583`；
  - 第 89 epoch：`train=0.010593`, `val=0.012491`。
- 当前 best：第 82 epoch，`val=0.011510`。
- 判断：训练仍在有效下降，best 从第 43 epoch 的 `0.012587` 刷新到 `0.011510`；第 83-89 epoch 的 val 有波动，但第 89 epoch 回到 `0.012491`，还不能判断过拟合。
- 磁盘：`/home/chenshuai/Project/output` 可用约 47GB；checkpoint 较大，仍需持续监督。
- 已更新：
  - `training_metrics_latest.csv`
  - `training_curve_latest.png`

## 2026-06-18 引导服务链路修复与验证

目的：在训练 260617-only DP 的同时，检查当前 board PTG/guidance 服务是否能真正接上 `DP clean action -> multistep Foresight -> TacQuality scorer -> trust-region 梯度引导`。

发现的问题：

- `for_show_xiaomi/serve_dp_tac_quality_guided.py` 原来引用的 `TFAC_V5.tac_quality_foresight_bridge` 和 `TFAC_V5.tac_quality_serving_guidance` 已不在正式 root 包中，只在临时脚本目录里有副本，真实启动会 import 失败。
- 旧服务只按 `LatentForesightPretrainModel` 加载 Foresight；当前 board Foresight checkpoint 是 `predict_horizon=16` 的 multistep 模型，直接加载会出现 `future_queries` shape mismatch。
- Foresight 训练输入使用归一化 marker，decoder 输出也是归一化 marker；服务 bridge 需要用 Foresight `args.json/norm_stats.marker_offset_mean/std` 做一致的输入归一化和输出反归一化，否则 scorer 输入尺度会错。
- trust-region refiner 原有每步日志有 finite gradient，但没有顶层 `finite_grad_rate / positive_grad_rate / accept_rate`，dry-run gate 不方便直接判定。

修复内容：

- 新增正式包模块：
  - `TFAC_V5/tac_quality_energy/foresight_bridge.py`
  - `TFAC_V5/tac_quality_energy/serving_guidance.py`
  - `TFAC_V5/tac_quality_energy/ptg_proxy_runtime.py`
  - `TFAC_V5/tac_quality_energy/insertion_runtime.py`
- `serve_dp_tac_quality_guided.py` 改为从正式包导入 bridge/guidance，并根据 `predict_horizon` 自动选择：
  - `LatentForesightPretrainModel` for single-step；
  - `MultiStepLatentForesightModel` for multistep。
- 服务端 marker 处理改为：

```text
raw marker window
  -> normalize by Foresight marker stats
  -> Foresight predicts normalized future latent/marker
  -> decode marker
  -> denormalize by same marker stats
  -> TacQuality scorer computes energy
```

- `TacQualityTrustRegionRefiner` 增加顶层梯度/接受率汇总。
- 正式包补入 `InsertionRiskScorerRuntime`，避免 board 修复后插孔 `default_guided` 隐藏回归。

验证结果：

- `py_compile` 通过：
  - `TFAC_V5/tac_quality_energy/*.py`
  - `for_show_xiaomi/serve_dp_tac_quality_guided.py`
- import smoke 通过：
  - `import for_show_xiaomi.serve_dp_tac_quality_guided`
  - `from TFAC_V5.tac_quality_energy import InsertionRiskScorerRuntime, PTGProxyScorerV2Runtime`
- board guided server synthetic-Foresight dry-run 通过：
  - 输出：`/home/chenshuai/Project/output/tac_quality_guided_server_packet/260617_repair/guided_server_synthetic_foresight_dry_run_smoke.json`
  - `dry_run_guidance_smoke_pass=true`
- board guided server real multistep-Foresight dry-run 通过：
  - 输出：`/home/chenshuai/Project/output/tac_quality_guided_server_packet/260617_repair/guided_server_real_foresight_dry_run_smoke.json`
  - `dry_run_guidance_smoke_pass=true`
  - Foresight load：`kind=multistep`, `predict_horizon=16`, `missing=0`, `unexpected=0`
  - `finite_grad_rate=1.0`, `positive_grad_rate=1.0`
  - `max_delta_within_trust_region=true`
  - `not_reranking=true`
- insertion runtime gradient smoke 通过：
  - marker/action 梯度 finite；
  - marker grad norm 约 `0.0406`；
  - action grad norm 约 `0.0226`；
  - `build_serving_guidance_from_arm('insertion', 'default_guided', ...)` 可构建。

边界说明：

- 以上是离线 dry-run / smoke，不是真机 rollout 结果。
- 当前只证明服务链路可执行、可产生有限非零梯度、action 更新被 trust region 限制。
- 真正是否提升擦黑板接触质量，需要等 DP checkpoint 训练充分后，用 server-side force curves 和 marker/force 指标做真实对比。

## 最近两个月最相关工作

时间窗口按 2026-06-18 往前约两个月筛选，优先选择 tactile / diffusion policy / contact-rich manipulation / guidance 相关工作。

### 1. ViTaL: Inference-time Policy Steering via Vision and Touch

链接：<https://arxiv.org/abs/2606.14981>

提交时间：2026-06-12。

核心思想：把 inference-time steering 分成两层：视觉负责长程模式选择，触觉负责短程接触 refinement。它使用 visuo-tactile latent world model 预测未来 outcome，再用视觉/触觉 verifier 给候选 action 打分或做 diffusion editing。

和本项目关系：

- 非常接近我们的目标：`DP action -> Foresight 预测未来触觉 -> 质量评分器 -> 梯度引导 action`。
- 它明确区分 global visual success 和 local contact success，这正好对应插孔/擦黑板任务中的两类失败：视觉上路径对，但接触力/插入接触不好。
- 对我们的启发：评分器不要只做“好/坏”，应拆成至少两个分量：任务进度/几何模式分数 + 局部接触质量分数。当前我们已有 tactile quality scorer，但视觉/任务进度 verifier 还弱。

### 2. Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation

链接：<https://arxiv.org/abs/2606.08737>

提交时间：2026-06-07。

核心思想：把 action、未来视觉、未来触觉动态一起建模；提出 contact-gated visuotactile fusion 和 contact-aware attention bias，并做缓存加速。

和本项目关系：

- 我们现在的 Foresight 已经是 “action -> future tactile latent/marker” 的方向，但主要预测触觉，没有联合未来视觉。
- Dream-Tac 的 contact-gated fusion 很适合解释为什么擦黑板 approach 阶段不该强用触觉，而 wiping 接触阶段应该强用触觉。
- 对我们的启发：Foresight/评分器可增加 contact gate：只有预测/观测 marker 或 force 达到接触条件时，触觉质量能量才强生效。

### 3. FTP-1: A Generalist Foundation Tactile Policy Across Tactile Sensors

链接：<https://arxiv.org/abs/2606.13102>

提交时间：2026-06-11。

核心思想：多种 tactile sensor 输入通过异构 encoder 映射到统一 morphology-aware latent token，再用共享 tactile Transformer expert 预训练，支持跨 sensor / embodiment transfer。

和本项目关系：

- 我们现在的 TactileVAE 是单传感器/单任务相对局部的 encoder；FTP-1 说明 tactile representation 本身可以作为更通用的“触觉基础模型”。
- 对我们的启发：论文故事里可以把当前 TactileVAE 表述为 task-local tactile consequence latent，未来升级为 multi-task/multi-sensor tactile token encoder。
- 实验上可以增加：插孔 + 擦黑板共用 tactile latent/proxy schema，看评分器是否跨任务复用。

### 4. Multi-Resolution Tactile Imitation Learning

链接：<https://arxiv.org/abs/2606.06281>

提交时间：2026-06-04。

核心思想：融合 RGB、低频 dense tactile（如 GelSight）和高频 event tactile，用 modality-specific stem + transformer fusion，策略用 flow matching。

和本项目关系：

- 我们只有 marker_offset 和 force6d，但同样存在不同时间尺度：marker field 是空间形变，force 曲线是更直接的高频接触强度。
- 对我们的启发：擦黑板评分器最好不要只看 marker 形变大小，还要看力曲线/marker 变化率；短窗口内平滑性是核心质量信号。

### 5. Tube Diffusion Policy

链接：<https://arxiv.org/abs/2604.23609>

提交时间：2026-04-26。

核心思想：传统 action chunking 反应慢；Tube DP 在 diffusion action chunk 外学习 observation-conditioned feedback flow，形成 action tube，可在执行过程中根据 tactile/vision 快速局部修正。

和本项目关系：

- 我们现在是 DP 一次生成 `pred_horizon=16`，执行 `action_horizon=8`，再通过 PTG 做 final clean action refinement。
- Tube DP 支持我们的判断：只做 reranking 不够，接触任务需要局部可微修正或快速反馈。
- 对我们的启发：PTG 可进一步升级成“score-gradient action tube”：不是只改一次完整 action chunk，而是在执行过程中每步根据实时触觉重算局部评分和修正。

### 6. Latent Diffusion Policy: Shaping Latent Spaces for Diffusion-Based Robotic Manipulation

链接：<https://arxiv.org/abs/2606.08657>

提交时间：2026-06-07。

核心思想：把原始 action 空间中的 diffusion / flow matching 拆成两阶段：先用 observation-conditioned CVAE 把动作压到更集中的 latent 分布，再在 latent 空间做生成；同时用 per-token diffusion forcing 和 staircase inference 缓解训练/推理不一致。

和本项目关系：

- 我们当前 DP 仍是在 joint action chunk 上直接 denoise，视觉理解、触觉历史和轨迹生成都压在同一个 denoising noise predictor 里。
- 如果后续 260617-only 或 plus_peg 数据上出现“训练 loss 降但真机动作不稳定”，可以考虑把 action chunk 改成 latent action token，再用 TacQualityEnergy/Foresight 在 latent 或 decoded action 上做小范围引导。
- 这也呼应之前讨论过的 Foresight 是否要加 CVAE：对策略本体来说，latent action space 可能比直接给 Foresight 加随机潜变量更值得优先尝试。

### 7. HapTile: A Haptic-Informed Vision-Tactile-Language-Action Dataset

链接：<https://arxiv.org/html/2606.04825v1>

提交时间：2026-06。

核心思想：构建包含视觉、指尖触觉、proprioception、动作轨迹和 haptic feedback 的 contact-rich imitation learning 数据集，并提供 Diffusion Policy 等 baseline。

和本项目关系：

- 它说明最近社区正在把触觉和 haptic feedback 当成数据集核心字段，而不只是 policy 的附加输入。
- 对我们当前 260617 数据很直接：应把每次真机测试的 force curve、接触阶段、是否擦干净、是否提前停止都作为 rollout metadata 保存，后续 scorer 训练和真实评估才能闭环。
- 这支持 server-side force logging 的必要性：不能只保存动作和图像，否则无法判断 PTG 是否真的改善接触质量。

### 8. DreamTacVLA: Learning to Feel the Future

链接：<https://arxiv.org/html/2512.23864v3>

说明：该工作初版早于两个月，但最近版本在 2026 年更新，且和“预测未来触觉后果再修正动作”的思路高度相关。

核心思想：Think-Dream-Act：先提出草稿动作，再预测该动作导致的未来触觉，最后把真实观测和预测触觉结合起来 refine action。

和本项目关系：

- 它和我们的 `DP action -> Foresight future tactile -> TacQuality score -> action guidance` 几乎是同一个因果结构，只是我们更强调可微质量能量和 bounded gradient refinement。
- 对论文故事有帮助：可以把 Foresight 解释成 task-local tactile consequence model，而不是普通辅助预测头。

## 相关但略超出两个月/非 arXiv 的重要工作

### 9. DPTG: Diffusion Policy with Tactile Feasibility Guidance

链接：<https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1851102/full>

发表时间：2026-06-10，Frontiers。

核心思想：视觉 diffusion policy 负责生成 action，触觉 feasibility classifier 只作为物理可行性约束，不作为并列 action generator；用 feasibility score 自适应调节 guidance 强度，只在接触有信息时启用。

和本项目关系：

- 这是最贴近我们 classifier guidance 的公开工作。
- 它支持一个重要设计选择：DP policy 最好用成功/高质量 demonstrations 训练；坏数据主要用于训练 feasibility / quality scorer，而不是全部混进 policy。
- 对我们当前数据特别重要：260617-only DP 如果是好数据，可作为视觉/触觉 policy；负样本应该主要进入质量分类器/评分器。

### 10. PPGuide: Steering Diffusion Policies with Performance Predictive Guidance

链接：<https://arxiv.org/abs/2603.10980>

提交时间：2026-03-11，略早于两个月窗口，但 classifier guidance 相关性高。

核心思想：用 rollout 中自监督挖掘出来的 success/failure-relevant observation-action chunks 训练 performance predictor，推理时用 predictor gradient 引导 DP 远离失败模式。

和本项目关系：

- 我们现在的评分器标签来自人工定义/规则定义（插孔 bounce，擦黑板力过大/过小/不稳）。PPGuide 提供另一路：从 rollout success/failure 中自动挖关键 chunk。
- 对我们的启发：后续真机 rollout 后，可以把每条擦黑板的 force curve + 完成质量作为 episode label，再自动定位导致失败的 chunk，训练更贴近部署分布的 scorer。

### 11. AdaVTF: Learning When to See and When to Feel

链接：<https://arxiv.org/abs/2604.01414>

提交时间：2026-04-01，略早于严格两个月窗口，但用户已指定参考。

核心思想：非接触阶段忽略 F/T，接触阶段自适应融合 vision 和 torque；在 diffusion policy 中比较多种 F/T-vision 融合方式。

和本项目关系：

- 支持我们在擦黑板中只关注 wiping/contact 阶段的做法。
- 对我们的启发：质量评分器应包含 contact gate 或 phase-aware weight，避免 approach 阶段的触觉噪声影响 action。

## 对当前项目架构的建议

### 建议 A：保持“policy 和 scorer 分工不对称”

更合理的故事：

```text
DP policy: 从成功/高质量动作学习任务动作分布
Foresight: 预测候选 action 的未来触觉后果
TacQuality scorer: 判断未来触觉是否物理可行/质量好
Guidance: 用 scorer 梯度在小 trust region 内修正 action
```

这比“把好坏数据全部混进 DP”更清晰，也更接近 DPTG / ViTaL 的方向。

### 建议 B：擦黑板 scorer 不只做二分类，要做质量分解

当前标签可定义为：

- `expert`: 力大小合适且变化平滑；
- `too_small`: 接触/压力不足，擦不干净；
- `too_large`: 压力过大，风险高；
- `unstable`: 力忽大忽小或 marker 变化不平滑。

部署时不一定用 4-class argmax，而是转成连续能量：

```text
score = expert_logit
      - alpha * too_small_logit
      - beta  * too_large_logit
      - gamma * unstable_logit
      - lambda * action_smooth_penalty
```

这样既能分类解释，又能提供可微梯度。

### 建议 C：评分器必须 phase/contact-aware

擦黑板中，approach 阶段 force/marker 小是正常的，不能被 `too_small` 惩罚；wiping 阶段 force/marker 小才是坏。插孔中，未接触阶段也不该强触觉引导。

因此 guidance scale 应该类似：

```text
guidance_weight = contact_gate * uncertainty_or_risk_gate
```

contact_gate 可由 marker magnitude / contact area / force magnitude 估计。

### 建议 D：从“final action refinement”升级到“局部 action tube”是后续创新点

当前实现：DP 先生成完整 action chunk，然后 PTG 进行 bounded refinement。

下一步创新：借鉴 Tube DP，把 PTG 做成执行时每步/短窗口局部反馈修正：

```text
DP nominal chunk + TacQuality gradient feedback flow -> locally corrected action tube
```

这样更适合擦黑板这种连续接触任务，因为力变化需要实时小修，而不是只在 chunk 开头修一次。

### 建议 E：真机评估指标必须绑定 scorer 目标

擦黑板 rollout 评估不只看是否擦干净，还应保存并统计：

- `Fz_mean` 是否在目标范围；
- `Fz_p95 / max` 是否过大；
- `|dFz/dt|` 或 `force_delta_abs_mean` 是否小；
- marker magnitude / area 是否稳定；
- action smoothness 是否被 guidance 破坏。

这个和当前 server-side force logging 是一致的。

### 建议 F：把调研结论转成后续可执行实验

短期不改变当前 260617-only DP 训练；先把本次 policy 训练充分。训练完成后，按以下顺序做：

1. **稳定性验证**：用同一真机 protocol 比较旧全量 DP、plus_peg DP、260617-only DP，在 baseline 不加 PTG 的情况下先看动作是否稳定。
2. **接触质量闭环**：每次 rollout 保存 server-side force curve 和 marker proxy，统计 `Fz_mean/Fz_p95/|dFz|/marker area/marker smoothness`。
3. **contact-gated PTG**：只在 wiping/contact 阶段打开 TacQuality guidance，approach 阶段关闭或弱化 guidance。
4. **Foresight verifier 对齐**：用 Foresight 预测的未来 marker/latent 计算 TacQuality energy，再和真实 rollout 的 force/marker 指标做相关性检查。
5. **如果真机仍不稳**：优先尝试局部 action tube 或 latent action diffusion，而不是扩大 scorer 分类头数量。

## 近期实验优先级

1. 等 260617-only DP 训练稳定后，保留 `dp_best.pth`。
2. 用同一份真机测试流程比较：
   - 旧全量 DP；
   - 旧 plus_peg DP；
   - 新 260617-only DP；
   - 各自 baseline vs PTG-guided。
3. 用 server-side force curves 评估 baseline/guided 的真实接触质量。
4. 若 guided force 更稳定，再把当前 scorer 固化为论文主线；若 guided 不稳定，则优先调整 contact gate 和 guidance scale，而不是继续堆分类器结构。
