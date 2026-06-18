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

2026-06-18 13:00 监督更新：

- 训练进程仍在运行，PID `1544542`；watcher PID `1563456`。
- 最新到第 120 epoch：
  - 第 105 epoch：`val=0.011152`，当前 best；
  - 第 119 epoch：`train=0.008566`, `val=0.013989`；
  - 第 120 epoch：`train=0.008803`, `val=0.012427`。
- 当前判断：
  - train loss 仍继续下降；
  - val 在 `0.011~0.015` 范围波动，best 未被近期 epoch 稳定刷新；
  - 由于 episode-level validation 只有 8 个 episode，短期 val 波动较大，不能只凭单个 epoch 判定最终过拟合；
  - 最终部署/测试应优先使用 `dp_best.pth`，而不是 `dp_latest.pth`。
- 磁盘：
  - 当前 run 目录约 `16G`；
  - image cache 约 `156G`；
  - `/home` 可用约 `47G`；
  - `dp_latest.pth` 覆盖写，`dp_best.pth` 覆盖写，top-k 只保留 3 个，周期 checkpoint 每 500 epoch 保存一次，空间暂时可控但需要持续监督。

2026-06-18 13:31 监督更新：

- 训练进程仍在运行，PID `1544542`；watcher 由 tmux session `watch_dp260617` 托管。
- 最新完整 epoch：第 170 epoch：
  - `train=0.007784`
  - `val=0.016137`
  - 当前 best 仍为第 105 epoch，`val=0.011152`
- 最近 5 个完整 epoch：
  - 第 166 epoch：`train=0.007494`, `val=0.017629`
  - 第 167 epoch：`train=0.007696`, `val=0.015280`
  - 第 168 epoch：`train=0.008044`, `val=0.015885`
  - 第 169 epoch：`train=0.007734`, `val=0.015785`
  - 第 170 epoch：`train=0.007784`, `val=0.016137`
- 当前判断：
  - 训练 loss 仍在低位继续拟合；
  - 验证 loss 暂时没有刷新第 105 epoch 的 best，说明需要警惕后续过拟合；
  - 但验证集只有 8 个 episode、1024 windows，短期 val 波动较大，当前不重启、不早停；
  - 已设置 watcher：`MIN_EPOCH_BEFORE_EARLY_STOP=1500`, `PATIENCE_EPOCHS=350`，到后期若长时间无收益会自动停止。
- 当前保存状态：
  - `dp_best.pth`：按验证 loss 自动覆盖；
  - `dp_latest.pth`：每个 epoch 覆盖保存，含 optimizer；
  - `dp_topk_*.pth`：保留 3 个训练 loss top-k；
  - `dp_epoch*.pth`：每 500 epoch 保存一次，避免 2.6GB 级 checkpoint 过多占满 `/home`。
- 最新曲线：
  - `/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000/loss_curve.png`
  - `/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000/loss_curve.csv`

2026-06-18 13:34 监督更新与趋势判断：

- 最新完整 epoch：第 178 epoch：
  - `train=0.007074`
  - `val=0.016986`
  - 当前 best 仍为第 105 epoch，`val=0.011152`
- 已更新：
  - `loss_curve.png`
  - `loss_curve.csv`
  - `training_metrics_latest.csv`
- 当前趋势：
  - train loss 继续下降，说明模型仍在拟合训练 windows；
  - val loss 在第 105 epoch 后没有刷新 best，最近 25 个 epoch 的 val mean 约在 `0.016` 附近；
  - 这可能是开始过拟合，也可能是 8 个 validation episode 与训练分布不完全一致导致的高方差。
- 当前不干预的原因：
  - `dp_best.pth` 已保留第 105 epoch 的最佳验证 ckpt；
  - 训练目标是 2000 epoch，目前只到约 9%，过早停止会错过后续 lr 下降后的二次改善；
  - watcher 已启用后期 plateau stop，只有到第 1500 epoch 后且连续 350 epoch 无 best 改善才会停。
- 后续判据：
  - 离线部署优先使用 `dp_best.pth`；
  - 若训练后期 rolling val 长期高于 best 且没有刷新，最终报告中将标记为“best ckpt available, latest overfit”；
  - 真正是否好用仍需真机/离线 rollout 对比，不能只靠 noise prediction val loss 判断擦黑板质量。

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

2026-06-18 12:45 监督更新：

- 训练进程仍在运行，PID `1544542`；watcher PID `1563456`。
- 最新到第 100 epoch：
  - 第 94 epoch：`train=0.009791`, `val=0.011340`，best 更新；
  - 第 96 epoch：`train=0.009805`, `val=0.014953`；
  - 第 100 epoch：`train=0.009731`, `val=0.014058`。
- 当前 best：第 94 epoch，`val=0.011340`。
- 判断：train loss 继续下降，val 在 `0.011~0.015` 之间波动；由于 best 仍在刷新且训练只到 5% 左右，不早停。
- 已更新训练曲线和 CSV：
  - `/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000/training_curve_latest.png`
  - `/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000/training_metrics_latest.csv`

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

## 2026-06-18 真机 force 评估脚本补强

动机：擦黑板的质量标准主要作用在擦拭接触阶段；approach/lift 阶段低力是正常的。如果直接统计整段 episode 的平均力，会把非接触阶段混进去，导致“压力太小/力变化不稳”的判断被稀释。

改动：

- `for_show_xiaomi/eval_board_force_rollouts.py` 新增 contact-phase 自动识别。
- 默认 `--contact_source auto`，优先使用：
  - `left_marker_mag_mean`
  - `right_marker_mag_mean`
  - `ft_f_mag`
  - `left_f_mag`
  - `right_f_mag`
- 阈值采用 robust 规则：

```text
threshold = p10(signal) + contact_threshold_frac * (p90(signal) - p10(signal))
```

- 默认 `contact_threshold_frac=0.25`，`min_contact_fraction=0.05`。
- 输出新增：
  - `contact_source`
  - `contact_threshold`
  - `contact_steps`
  - `contact_fraction`
  - `ft_fz_contact_mean`
  - `ft_fz_contact_p95`
  - `ft_fz_contact_delta_abs_mean`
  - `ft_f_mag_contact_mean`
  - `ft_f_mag_contact_p95`
  - marker contact-phase summary
- Markdown 中新增 `Contact-Phase Group Summary`，用于 baseline/guided 直接比较接触阶段力大小和力平滑性。

验证：

- 用合成 baseline/guided force_trace 做 smoke，通过。
- 合成数据中 approach 段 marker/force 低，wiping 段 marker/force 高，脚本识别 `contact_fraction=0.55`，并正确输出接触段 Fz mean / p95 / dF 指标。

意义：

- 后续真实测试时，`for_show_xiaomi/eval_board_force_rollouts.py --root /home/chenshuai/Project/output/board_force_rollouts/server --tag board_server_all` 会同时给出整段指标和 contact-phase 指标。
- 论文/实验结论应优先看 contact-phase 指标，因为这更贴近擦黑板的质量定义：接触阶段力大小合适且变化平滑。

## 2026-06-18 在线 Board Contact Gate

动机：前面的调研和数据分析都指向同一个问题：擦黑板 approach/lift 阶段触觉/力小是正常现象，只有 wiping/contact 阶段才应该强用 TacQuality scorer。若每个 DP action chunk 都无条件做触觉质量梯度引导，可能在未接触阶段把“低力”误判成坏触觉，从而引入不必要的动作扰动。

设计：

- gate 放在 serving 层，而不是改 scorer 本体；
- scorer 仍只表达“未来触觉后果质量”；
- server 根据当前观测 marker window 决定“现在是否应该启用触觉引导”。

当前实现：

- 文件：`for_show_xiaomi/serve_dp_tac_quality_guided.py`
- board 任务默认开启 contact gate；
- insertion 任务默认不受这个 board marker gate 影响；
- gate metric：当前 marker window 的 mean marker magnitude；
- 默认阈值：
  - `--contact_gate_low 1.8`
  - `--contact_gate_high 2.3`

规则：

```text
metric <= low   -> skip TacQuality guidance, action chunk unchanged
metric >= high  -> full TacQuality guidance
low < metric < high -> linearly scale the guided action delta
```

这相当于在线版本的：

```text
guided_action = base_action + contact_gate * (guided_action_without_gate - base_action)
```

验证：

- `py_compile` 通过。
- contact gate dry-run 输出目录：
  `/home/chenshuai/Project/output/tac_quality_guided_server_packet/contact_gate_smoke/`
- 三档 synthetic marker 测试通过：
  - low contact：`marker_value=0.1`, metric≈`0.141`, gate=`0.0`, 跳过引导，action delta=`0`；
  - mid contact：`marker_value=1.45`, metric≈`2.051`, gate≈`0.501`, 实际 action delta 约为 gate 前的一半；
  - high contact：`marker_value=3.0`, metric≈`4.243`, gate=`1.0`, 完整引导。

意义：

- 这使在线 guidance 与评估口径一致：只在擦拭接触阶段强关注力大小和触觉平滑性；
- 避免 approach 阶段因为低力/低 marker 被 TacQuality scorer 误惩罚；
- 也更符合 AdaVTF / Dream-Tac 中“when to feel / contact-gated tactile fusion”的思想。

## 最近两个月最相关工作

时间窗口按 2026-06-18 往前约两个月筛选，优先选择 tactile / diffusion policy / contact-rich manipulation / guidance 相关工作。

### 总体判断

最近两个月的相关工作正在形成一个清晰趋势：contact-rich manipulation 不能只靠视觉 DP 生成动作，也不能只把触觉当作当前观测特征拼进去；更强的方向是把触觉作为“未来接触后果”的约束信号，在推理时对候选动作进行 steering / editing / refinement。

对本项目最有价值的关键词是：

- `inference-time policy steering`
- `tactile world model`
- `future tactile outcome`
- `contact-gated fusion/guidance`
- `verifier / reward / energy model`
- `bounded action refinement`

因此本项目当前最合理的主线仍是：

```text
DP action proposal
  -> Foresight predicts future tactile outcome
  -> TacQuality scorer/verifier evaluates future contact quality
  -> contact gate decides when tactile guidance is active
  -> trust-region gradient guidance modifies action in a small range
```

这条线比单纯“触觉 DP concat 输入”更有新意，也比 reranking 更贴近用户目标。

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

### 4. TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation

链接：<https://arxiv.org/abs/2606.11184>

提交时间：2026-06-09。

核心思想：面向 contact-rich manipulation 训练 force-guided tactile world model，用动作/力相关信息预测短期未来触觉状态。

和本项目关系：

- 名字和方向都非常接近我们的 Foresight 模块，但我们的最终目标不是只预测未来触觉，而是把预测结果接入 TacQuality energy，再对 DP action 做梯度引导。
- 它能支持我们的核心论点：触觉后果模型本身是接触任务里的关键中间模型，不只是 auxiliary loss。
- 对我们的启发：Foresight 预测输入里可以更系统地利用 force/marker proxy，而不是只依赖 joint action；如果后续真实 rollout 中发现 Foresight 对不同压力模式不敏感，应优先补充 force-conditioned 或 marker-proxy-conditioned 预测分支。

### 5. Multi-Resolution Tactile Imitation Learning

链接：<https://arxiv.org/abs/2606.06281>

提交时间：2026-06-04。

核心思想：融合 RGB、低频 dense tactile（如 GelSight）和高频 event tactile，用 modality-specific stem + transformer fusion，策略用 flow matching。

和本项目关系：

- 我们只有 marker_offset 和 force6d，但同样存在不同时间尺度：marker field 是空间形变，force 曲线是更直接的高频接触强度。
- 对我们的启发：擦黑板评分器最好不要只看 marker 形变大小，还要看力曲线/marker 变化率；短窗口内平滑性是核心质量信号。

### 6. Tube Diffusion Policy

链接：<https://arxiv.org/abs/2604.23609>

提交时间：2026-04-26。

核心思想：传统 action chunking 反应慢；Tube DP 在 diffusion action chunk 外学习 observation-conditioned feedback flow，形成 action tube，可在执行过程中根据 tactile/vision 快速局部修正。

和本项目关系：

- 我们现在是 DP 一次生成 `pred_horizon=16`，执行 `action_horizon=8`，再通过 PTG 做 final clean action refinement。
- Tube DP 支持我们的判断：只做 reranking 不够，接触任务需要局部可微修正或快速反馈。
- 对我们的启发：PTG 可进一步升级成“score-gradient action tube”：不是只改一次完整 action chunk，而是在执行过程中每步根据实时触觉重算局部评分和修正。

### 6b. ContactWorld: What Matters in Vision-Tactile World Models

链接：<https://arxiv.org/abs/2606.13877>

提交时间：2026-06-11。

核心思想：系统研究 vision-tactile world model 对 contact-rich planning 的影响，结论强调 spatially structured、temporally continuous 的表示更适合长程接触规划；触觉不是简单加模态，关键是跨模态表示兼容和长程预测稳定性。

和本项目关系：

- 我们用 TactileVAE latent 和 marker proxy 表示未来触觉后果，正是在做触觉 world representation。
- 这支持一个重要实验设计：评分器/foresight 的评估不能只看单帧分类 accuracy，还要看整段 episode 中预测质量和真实 force/marker 质量指标的相关性。
- 对我们的启发：擦黑板 scorer 应保留空间结构信息或至少保留 contact area/center/spread/smoothness 等 proxy，否则容易只学到力大小而忽略接触稳定性。

### 6c. Ambient Diffusion Policy

链接：<https://arxiv.org/abs/2606.12365>

提交时间：2026-06-10。

核心思想：研究如何从 suboptimal data 中训练 DP，不是简单混合所有数据，而是在不同 diffusion time 上限制低质量数据的贡献，避免 harmful features 污染策略。

和本项目关系：

- 这直接回应“正样本/负样本怎么用”：负样本不一定适合直接混进 DP imitation policy，尤其擦黑板的 `too_high/too_low/oscillate` 负样本可能会污染动作分布。
- 更合理的分工是：高质量/成功数据训练 DP 主策略；负样本主要训练 TacQuality scorer/verifier；如果要混合训练，则需要显式质量权重或 diffusion-time-aware 权重。
- 当前 260617-only DP 只用单独数据训练是合理的 baseline；后续若要加负样本，应优先加到评分器，不应无条件混入策略。

### 7. Latent Diffusion Policy: Shaping Latent Spaces for Diffusion-Based Robotic Manipulation

链接：<https://arxiv.org/abs/2606.08657>

提交时间：2026-06-07。

核心思想：把原始 action 空间中的 diffusion / flow matching 拆成两阶段：先用 observation-conditioned CVAE 把动作压到更集中的 latent 分布，再在 latent 空间做生成；同时用 per-token diffusion forcing 和 staircase inference 缓解训练/推理不一致。

和本项目关系：

- 我们当前 DP 仍是在 joint action chunk 上直接 denoise，视觉理解、触觉历史和轨迹生成都压在同一个 denoising noise predictor 里。
- 如果后续 260617-only 或 plus_peg 数据上出现“训练 loss 降但真机动作不稳定”，可以考虑把 action chunk 改成 latent action token，再用 TacQualityEnergy/Foresight 在 latent 或 decoded action 上做小范围引导。
- 这也呼应之前讨论过的 Foresight 是否要加 CVAE：对策略本体来说，latent action space 可能比直接给 Foresight 加随机潜变量更值得优先尝试。

### 8. HapTile: A Haptic-Informed Vision-Tactile-Language-Action Dataset

链接：<https://arxiv.org/abs/2606.04825>

提交时间：2026-06-03。

核心思想：构建包含视觉、指尖触觉、proprioception、动作轨迹和 haptic feedback 的 contact-rich imitation learning 数据集，并提供 Diffusion Policy 等 baseline。

和本项目关系：

- 它说明最近社区正在把触觉和 haptic feedback 当成数据集核心字段，而不只是 policy 的附加输入。
- 对我们当前 260617 数据很直接：应把每次真机测试的 force curve、接触阶段、是否擦干净、是否提前停止都作为 rollout metadata 保存，后续 scorer 训练和真实评估才能闭环。
- 这支持 server-side force logging 的必要性：不能只保存动作和图像，否则无法判断 PTG 是否真的改善接触质量。

### 9. DreamTacVLA: Learning to Feel the Future

链接：<https://arxiv.org/abs/2512.23864>

说明：该工作初版是 2025-12-29，v3 在 2026-05-06 修订；严格说不属于“近两个月新提交”的主证据，但和“预测未来触觉后果再修正动作”的思路高度相关，可作为背景参考。

核心思想：Think-Dream-Act：先提出草稿动作，再预测该动作导致的未来触觉，最后把真实观测和预测触觉结合起来 refine action。

和本项目关系：

- 它和我们的 `DP action -> Foresight future tactile -> TacQuality score -> action guidance` 几乎是同一个因果结构，只是我们更强调可微质量能量和 bounded gradient refinement。
- 对论文故事有帮助：可以把 Foresight 解释成 task-local tactile consequence model，而不是普通辅助预测头。

## 相关但略超出两个月/非 arXiv 的重要工作

### 10. DPTG: Diffusion Policy with Tactile Feasibility Guidance

链接：<https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1851102/full>

发表时间：2026-06-10，Frontiers。

核心思想：视觉 diffusion policy 负责生成 action，触觉 feasibility classifier 只作为物理可行性约束，不作为并列 action generator；用 feasibility score 自适应调节 guidance 强度，只在接触有信息时启用。

和本项目关系：

- 这是最贴近我们 classifier guidance 的公开工作。
- 它支持一个重要设计选择：DP policy 最好用成功/高质量 demonstrations 训练；坏数据主要用于训练 feasibility / quality scorer，而不是全部混进 policy。
- 对我们当前数据特别重要：260617-only DP 如果是好数据，可作为视觉/触觉 policy；负样本应该主要进入质量分类器/评分器。

### 11. PPGuide: Steering Diffusion Policies with Performance Predictive Guidance

链接：<https://arxiv.org/abs/2603.10980>

提交时间：2026-03-11，略早于两个月窗口，但 classifier guidance 相关性高。

核心思想：用 rollout 中自监督挖掘出来的 success/failure-relevant observation-action chunks 训练 performance predictor，推理时用 predictor gradient 引导 DP 远离失败模式。

和本项目关系：

- 我们现在的评分器标签来自人工定义/规则定义（插孔 bounce，擦黑板力过大/过小/不稳）。PPGuide 提供另一路：从 rollout success/failure 中自动挖关键 chunk。
- 对我们的启发：后续真机 rollout 后，可以把每条擦黑板的 force curve + 完成质量作为 episode label，再自动定位导致失败的 chunk，训练更贴近部署分布的 scorer。

### 12. AdaVTF: Learning When to See and When to Feel

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

### 建议 G：本项目故事可以更明确地写成“可微触觉后果约束”

当前最有说服力的表述不是“给 DP 加一个分类器”，而是：

```text
Diffusion Policy 生成候选动作；
Foresight 预测该动作导致的未来触觉后果；
TacQualityEnergy 把未来触觉后果转成可微质量能量；
Contact gate 判断当前阶段是否应该使用触觉能量；
Trust-region gradient guidance 在小范围内修正动作。
```

这和最近工作相比的差异点：

- 相比纯 tactile imitation：我们不是只把触觉作为观测输入，而是在推理时用未来触觉后果主动约束 action；
- 相比 reranking：我们不是只挑候选，而是用可微能量直接改 action；
- 相比普通 classifier guidance：我们的 classifier/scorer 不是直接看当前 obs-action，而是看 Foresight 预测的未来 contact outcome；
- 相比单一二分类：擦黑板质量可拆成力过小、力过大、不稳定和专家接触，再组合成连续能量，便于解释和调权。

## 评分器/引导是否“合适”的验证链路

仅有分类准确率不够。一个适合 DP gradient guidance 的触觉质量模型需要同时满足以下条件。

### 1. 标签和任务目标一致

插孔：

```text
positive = 无 bounce 的成功插入/正常插入阶段
negative = bounce 前后导致碰外壁/失败接触的阶段
```

擦黑板：

```text
expert     = wiping/contact 阶段力大小合适，marker/force 变化平滑
too_small  = 接触不足/压力太小，擦不干净
too_large  = 压力过大，风险高
unstable   = 力或 marker 忽大忽小，不柔顺
```

注意：擦黑板的标签只应主要作用在 wiping/contact 阶段，approach/lift 阶段低力不是坏样本。

### 2. 评估必须 episode-level split

frame-level 随机划分会把同一个 episode 的相邻帧同时放进 train/test，导致严重泄漏。评分器评估应优先用：

```text
GroupKFold(group = episode_id)
```

至少报告：

- binary AUC / AP / balanced accuracy；
- multi-class macro-F1；
- 每个 failure reason 的 recall；
- calibration / reliability；
- score decile 单调性。

### 3. 分数要能排序，而不是只会分类

guidance 需要的是连续可微的方向，所以要看分数排序是否符合质量：

```text
score_high -> 更稳定/更合适的接触
score_low  -> too small / too large / unstable / bounce risk
```

需要报告：

- Spearman(score, quality target)；
- top decile vs bottom decile 的真实质量差；
- score bins 中 good rate 是否单调；
- 在擦黑板中，score 和 contact-phase `Fz_mean/Fz_p95/|dFz|/marker_smoothness` 的相关性。

### 4. 分数必须对 action 有有效梯度

适合 classifier guidance 的 scorer 不能只是离散判别器，还必须满足：

```text
candidate action
  -> Foresight predicted tactile
  -> scorer score
  -> d score / d action 有限、非零、方向稳定
```

至少做以下 smoke：

- `finite_grad_rate` 接近 1；
- `positive_grad_rate` 不为 0；
- action delta 被 trust region 限制；
- guidance 后 scorer 分数上升；
- guidance 后动作变化不超过安全阈值。

### 5. 必须验证 Foresight 分数和真实结果一致

即使 scorer 在 GT tactile 上分类很好，也不代表它能引导 DP。因为部署时 scorer 看到的是：

```text
Foresight(action) 预测出来的 future tactile
```

所以要额外验证：

```text
score(Foresight(action)) 与真实 rollout 的接触质量指标相关
```

如果相关性差，优先修 Foresight 或 contact gate，不应盲目加大 guidance scale。

### 6. 最终真机评价看 contact-phase force/marker

擦黑板任务最终要看：

- contact 阶段 `Fz_mean` 是否在目标范围；
- `Fz_p95/max` 是否不过大；
- `|dFz|` 是否降低；
- marker magnitude/area 是否稳定；
- guided 是否破坏动作轨迹平滑性；
- 真实擦拭效果是否提升。

因此 DP loss、scorer accuracy、Foresight MSE 都只是中间证据，不是真正最终结论。

## 近期实验优先级

1. 等 260617-only DP 训练稳定后，保留 `dp_best.pth`。
2. 用同一份真机测试流程比较：
   - 旧全量 DP；
   - 旧 plus_peg DP；
   - 新 260617-only DP；
   - 各自 baseline vs PTG-guided。
3. 用 server-side force curves 评估 baseline/guided 的真实接触质量。
4. 若 guided force 更稳定，再把当前 scorer 固化为论文主线；若 guided 不稳定，则优先调整 contact gate 和 guidance scale，而不是继续堆分类器结构。

## 下一轮 scorer 实验清单

当前正式包内已有 `TFAC_V5/tac_quality_energy/eval_score_calibration.py`，它能检查已训练 scorer 的 score ordering / decile monotonicity。但这还不能完全证明 scorer 训练本身没有数据泄漏，也不能证明 Foresight-predicted score 能对应真实接触结果。因此下一轮实验应补齐以下内容。

### Experiment 1: episode-level scorer retrain/eval

目标：用 `GroupKFold(group=episode_id)` 重新评估插孔和擦黑板 scorer，避免 frame/window 随机划分泄漏。

输出：

- per-task binary AUC / AP / balanced accuracy；
- board 4-class macro-F1；
- reason recall: `expert / too_small / too_large / unstable`；
- score calibration / decile monotonicity；
- 每折保存 train/test episode 列表。

判定：

- 若 frame-level 高、GroupKFold 明显下降，说明之前有泄漏或 episode 特异性；
- 后续论文/报告只使用 GroupKFold 结果。

2026-06-18 13:50 已完成一版正式评估：

- 脚本：`TFAC_V5/tac_quality_energy/eval_groupkfold_scorers.py`
- 输出目录：`/home/chenshuai/Project/output/tac_quality_groupkfold_eval`
- 输出文件：
  - `groupkfold_scorer_eval.md`
  - `groupkfold_scorer_eval.json`
  - 每个 section 的 `group_splits.json`
  - 每个模型的 fold metrics CSV 和 decile plot
- 特征缓存已确认包含 episode/group 信息：
  - insertion cache: 36447 samples, 162 groups
  - PTG mixed cache: 36762 samples, 242 groups
- 评估处理：
  - `binary=-1` 视为中性/未定义 good-bad，不参与二分类训练/评估；
  - `binary=0/1` 参与 good-bad；
  - 所有样本仍参与 quality score 排序/decile 分析；
  - reason 多类分类独立评估。

GroupKFold 结果摘要：

| section | best model | AUC | AP | bACC | binary F1 | reason F1 | Spearman(q) | q decile step | q top-bottom gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| insertion_proxy | HGB | 0.9453 | 0.9820 | 0.8633 | 0.8706 | 0.6750 | 0.4256 | 0.7778 | 0.7216 |
| ptg_board | LogReg | 0.9781 | 0.9514 | 0.9100 | 0.9106 | 0.8709 | 0.8858 | 1.0000 | 0.8397 |
| ptg_insertion | HGB | 0.9720 | 0.9915 | 0.9085 | 0.9136 | 0.7220 | 0.4368 | 1.0000 | 0.7629 |
| ptg_mixed | HGB | 0.9774 | 0.9895 | 0.9233 | 0.9313 | 0.7569 | 0.4588 | 0.8889 | 0.7150 |

解释：

- 黑板 `ptg_board` 的跨 episode 结果最干净：LogReg 已有 AUC `0.9781`、bACC `0.9100`、reason F1 `0.8709`、quality Spearman `0.8858`，且 score decile 的 quality 单调率为 `1.0`。这说明当前黑板质量 proxy 的可分性和可排序性很强，不只是 frame-level 泄漏。
- 插孔和 mixed 的二分类也强，但 quality Spearman 只有 `0.42~0.46`。这说明 good/bad 边界比较可分，但连续质量排序没有黑板任务稳定；用于 guidance 时更适合用 logit/energy/margin，而不是过度解释为精细质量标尺。
- 这个结果仍不是最终真机结论：它证明 scorer 特征跨 episode 有泛化证据，但还需要通过 `d score / d action` 和 `score(Foresight(action)) -> real force/marker quality` 两个环节。

### Experiment 2: guidance-gradient audit

目标：验证 scorer 不只是能分类，还能给 action 提供稳定梯度。

输入：

```text
candidate action chunk
  -> multistep Foresight
  -> predicted future marker
  -> scorer score
  -> d score / d action
```

输出：

- `finite_grad_rate`
- `nonzero_grad_rate`
- score before/after trust-region refinement
- action delta norm / max per joint
- guidance 是否被 contact gate 正确关闭/打开

判定：

- score 应在 accepted refinement 后上升；
- action delta 必须小于安全 trust region；
- approach/lift 阶段 contact gate 应抑制不必要 guidance。

2026-06-18 13:57 已完成 board/default_guided 真实 Foresight 链路 audit：

- 脚本：`TFAC_V5/tac_quality_energy/eval_guidance_gradient_audit.py`
- 输出目录：`/home/chenshuai/Project/output/tac_quality_guidance_gradient_audit`
- 输出文件：
  - `guidance_gradient_audit.md`
  - `guidance_gradient_audit.json`
- 输入：
  - task: `board`
  - arm: `default_guided`
  - scorer: `PTGProxyScorerV2Runtime`
  - score mode: `profile`
  - Foresight: `/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload`
  - dataset: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/wipe_pos_straight_z124_125_150_20260609`
  - samples: 24 real board windows

结果：

| metric | value |
|---|---:|
| pass | True |
| finite grad rate mean | 1.0000 |
| positive grad rate mean | 1.0000 |
| accept rate mean | 1.0000 |
| improved rate mean | 1.0000 |
| trust region pass rate | 1.0000 |
| score delta mean | 0.000063 |
| action delta norm mean | 0.000801 |

解释：

- `action -> real multistep Foresight -> predicted marker -> PTG board scorer -> d score / d action` 全链路有有限、非零梯度；
- trust-region refinement 后 scorer 分数稳定上升，且 action delta 很小；
- 这证明当前 scorer/Foresight/refiner 链路具备“可微引导”条件，不只是一个离线分类器；
- score delta 数值较小，说明当前默认 board action step 很保守，适合先做安全真机测试；如果真实 rollout 改善弱，优先调 guidance scale/action_step/contact gate，而不是先换 scorer。

边界：

- 该结果不证明真机擦拭质量提升；
- 它只证明推理时的梯度链路可用且受限；
- 下一步必须做 `score(Foresight(action))` 与真实 contact-phase force/marker quality 的相关性验证。

### Experiment 3: Foresight-score vs real rollout quality

目标：验证部署时真正会用到的 `score(Foresight(action))` 是否和真实接触质量一致。

流程：

1. 用 DP 在离线 episode 或真机 rollout 中生成 action；
2. Foresight 预测未来 tactile；
3. scorer 对预测 tactile 打分；
4. 对比真实 rollout 中 contact-phase force/marker 指标。

指标：

- Spearman(score, -`|dFz|`)
- Spearman(score, target-range force quality)
- score decile 对应真实 contact smoothness 是否单调
- guided vs baseline 的真实 force curve 改善

判定：

- 如果 GT tactile scorer 很准，但 Foresight-score 和真实质量不相关，问题在 Foresight 或 action-conditioned prediction；
- 如果相关但真机 guided 不提升，优先调 guidance scale/contact gate/trust region；
- 如果相关且 guided 提升，当前 TacQuality guidance 主线成立。

2026-06-18 14:03 已完成一版 offline Foresight-score alignment：

- 脚本：`TFAC_V5/tac_quality_energy/eval_foresight_score_alignment.py`
- 输出目录：`/home/chenshuai/Project/output/tac_quality_foresight_score_alignment`
- 输出文件：
  - `foresight_score_alignment.md`
  - `foresight_score_alignment.json`
  - `foresight_score_alignment_samples.csv`
  - `foresight_score_alignment.png`
- 数据：
  - positive: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/wipe_pos_straight_z124_125_150_20260609`
  - too_small: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/z_too_high`
  - too_large: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_low`
  - oscillate: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_oscillate`
- 采样：
  - 120 contact-phase windows；
  - score mode: `profile`；
  - 对每个窗口同时计算：
    - `score(Foresight(action))`
    - `score(GT future marker)`
    - predicted marker vs GT future marker MAE
    - future force magnitude / force delta proxy

结果摘要：

| metric | value |
|---|---:|
| predicted-score AUC(good) | 0.5622 |
| GT-future-score AUC(good) | 0.3250 |
| predicted vs GT score Spearman | 0.5932 |
| predicted vs GT score Pearson | 0.5496 |
| predicted score vs -marker MAE Spearman | 0.2434 |
| predicted score vs -force abs Spearman | -0.2266 |
| predicted score vs -force delta Spearman | 0.0969 |
| marker MAE mean | 0.2582 |

按类别均值：

| label | pred score mean | GT score mean | marker MAE mean | future force abs mean | future force delta abs mean |
|---|---:|---:|---:|---:|---:|
| oscillate | 2.1346 | 2.1142 | 0.3631 | 5.8414 | 0.1344 |
| positive | 2.1552 | 2.1053 | 0.2976 | 10.6113 | 0.2605 |
| too_large | 2.1516 | 2.1089 | 0.2102 | 12.9496 | 0.3024 |
| too_small | 2.1661 | 2.1634 | 0.1328 | 7.6861 | 0.0759 |

关键解释：

- `score(Foresight(action))` 与 `score(GT future marker)` 有中等相关性，Spearman `0.5932`，说明 Foresight-score 不是随机的，预测触觉后果能保留一部分 scorer 排序。
- 但 predicted-score 不能很好区分 collection-regime good/bad，AUC 只有 `0.5622`；更关键的是 GT future score 自己对 good/bad 的 AUC 也只有 `0.3250`。
- 这说明问题不只是 Foresight，而是当前 board `profile` scorer 本身更偏向 marker 稳定/接触形变模式，没有显式把“目标力大小区间”编码成质量目标。
- 具体表现：`too_small` 的 force abs mean 约 `7.69`，marker MAE 最低，pred/GT scorer 分数最高；但按照任务定义它仍是负样本，因为压力偏小/擦不干净。当前 scorer 会误把这类稳定但力偏小的触觉当成好。
- `too_large` 和 positive 的分数接近，也说明没有明确的 force-band penalty。

设计结论：

- 当前 TacQuality guidance 链路具备可微性，但 board scorer 需要加入显式“目标力区间/force-band”质量头，不能只依赖 marker proxy；
- 更合理的 board energy 应从：

```text
profile = quality_logit + binary_margin
```

升级为：

```text
board_energy =
  contact_gate *
  (
    w_marker * marker_quality
  + w_smooth * smoothness_quality
  + w_force  * force_band_quality
  - w_heavy  * too_large_penalty
  - w_light  * too_small_penalty
  )
```

- 如果在线没有真实 force prediction，则至少需要从 marker proxy 中学习/蒸馏 force-band target，或让 Foresight/score 输入包含 predicted/observed force proxy；
- 这也是下一版 scorer 的创新点：不是单一 good/bad classifier，而是“contact-aware force-band energy + marker smoothness energy”的可微质量模型。

### Experiment 4: policy data mixing ablation

目标：验证负样本是否应该进入 DP policy 训练，还是只用于 scorer。

比较：

- high-quality/positive-only DP；
- 260617-only DP；
- plus_peg/full mixed DP；
- 如果使用负样本，增加 quality-weighted 或 diffusion-time-aware data weighting。

判定：

- 如果 mixed DP 的 val loss 更低但真机 force 更差，说明 policy 被负样本/分布混合污染；
- 更合理方案是 positive/high-quality 数据训练 policy，负样本训练 scorer/verifier。

## 2026-06-18 14:13 training supervision update

260617-only DP 训练仍在运行：

- PID: `1544542`
- watcher tmux: `watch_dp260617`
- 最新解析到第 `238/2000` epoch：
  - `train=0.006125`
  - `val=0.019737`
  - 当前 best: 第 `105` epoch, `val=0.011152`
- GPU: RTX 4090, 显存约 `14.7GB/24.6GB`, utilization 约 `94%`
- 磁盘：
  - run dir: 约 `12G`
  - `/home`: 剩余约 `51G`
- 已更新：
  - `/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000/loss_curve.png`
  - `/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000/loss_curve.csv`

当前判断：

- 训练 loss 仍在下降，说明模型还在继续拟合 260617 数据；
- val loss 自第 105 epoch 后没有刷新 best，并且第 230-238 epoch 基本在 `0.018~0.025`，明显高于 best；
- 这说明 `dp_latest.pth` 已经有过拟合倾向，但 `dp_best.pth` 正常保留；
- 继续跑 2000 epoch 的原因是后期 learning rate 下降后仍可能出现新的低谷，同时这次训练目标是充分训练；
- 真机/离线测试优先使用：
  - `/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000/dp_best.pth`

## Recent arXiv scan: 2026-04-18 to 2026-06-18

调研范围：

- tactile robot manipulation
- force/contact-aware robot policy
- vision-tactile world model
- diffusion policy guidance / steering / policy optimization
- suboptimal data imitation learning

筛选标准：

- 最近两个月 arXiv；
- 和本项目的 tactile DP、Foresight、TacQuality scorer、gradient guidance 相关；
- 优先关注能支持“触觉后果评分器 + 去噪过程梯度引导”的工作。

### Closest works

1. **ViTaL: Inference-time Policy Steering via Vision and Touch** (`arXiv:2606.14981`)

   相关性：

   - 也是 inference-time steering；
   - 强调只看视觉不足以处理接触任务，需要 tactile/touch 参与动作验证；
   - 与我们当前的 `DP action -> Foresight tactile consequence -> TacQuality score -> guidance` 思路高度接近。

   对本项目启发：

   - 我们应明确区分两件事：
     - reranking / candidate verification；
     - differentiable guidance。
   - 当前项目更强调后者：在 DP denoising 过程中用 scorer gradient 直接更新 action。
   - 论文叙事可以对比：ViTaL 类方法证明 tactile steering 有意义，但我们进一步把 tactile consequence score 作为可微能量接入 action denoising。

2. **TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation** (`arXiv:2606.11184`)

   相关性：

   - 关键词几乎和我们的方向一致：force-guided tactile world model；
   - 强调 global force 与 local tactile 的非对称时空作用；
   - 这正好对应我们在 board scorer 里发现的问题：只看 marker smoothness 会把 too-small 稳定接触误判为好。

   对本项目启发：

   - 下一版 Foresight 不应只预测 marker latent/marker field；
   - 应增加 force head 或 force-band head：
     - 预测未来 `force_abs`
     - 预测未来 `force_delta`
     - 或预测 `too_small / proper / too_large / oscillate`
   - 这样 TacQuality scorer 的 force-band 能量可以来自 Foresight，而不是只靠 marker proxy 间接猜。

3. **ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation** (`arXiv:2606.13877`)

   相关性：

   - 系统研究 vision-tactile world models 哪些表示对 contact-rich manipulation 真正有用；
   - 与我们现在做的 Level 1/Level 2 验证逻辑一致：先验证 tactile latent 是否有区分信号，再验证 Foresight 是否保留信号。

   对本项目启发：

   - 需要把当前实验流程正式化为三层证据链：
     1. GT tactile latent 可分；
     2. Foresight predicted tactile consequence 仍可分/可排序；
     3. scorer gradient 对 action 有有限、稳定、方向正确的影响。
   - 这不是“gate 机制”，而是论文里的科学证据结构。

4. **Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation** (`arXiv:2606.08737`)

   相关性：

   - 世界模型 + action generation + tactile future prediction；
   - 与我们现在把 Foresight 放在 DP 旁边做后果预测接近。

   对本项目启发：

   - 我们目前是 modular pipeline：
     - DP 负责 action prior；
     - Foresight 负责 tactile consequence；
     - scorer 负责 quality energy；
     - guidance 负责 action refinement。
   - 相比 unified world-action model，模块化优点是更容易解释和做真机安全约束；
   - 缺点是 Foresight 与 DP 不是端到端共同训练，未来可以尝试 joint fine-tuning。

5. **Tube Diffusion Policy: Reactive Visual-Tactile Policy Learning for Contact-rich Manipulation** (`arXiv:2604.23609`)

   相关性：

   - 针对 contact-rich manipulation 中 action chunking 反应慢的问题；
   - 强调视觉-触觉反馈的 reactive policy。

   对本项目启发：

   - 当前 DP `pred_horizon=16`, `action_horizon=8` 有 chunking 延迟；
   - 对擦黑板这类连续接触任务，触觉反馈应以较高频率影响 action；
   - 我们可以保留 chunk DP，但用 TacQuality gradient guidance 在 denoising 内对整段 action chunk 做局部修正，弥补纯 action chunk 的反应迟滞。

6. **Ambient Diffusion Policy: Imitation Learning from Suboptimal Data in Robotics** (`arXiv:2606.12365`)

   相关性：

   - 直接讨论 suboptimal data 如何进入 diffusion policy；
   - 与我们“负样本是否应该混进 DP policy 训练”高度相关。

   对本项目启发：

   - 不应简单把负样本当普通 demonstrations 混入 DP；
   - 更合理的用法：
     - positive/high-quality data 训练 DP prior；
     - bad data 训练 scorer/energy；
     - inference-time 用 energy guidance 避开 bad tactile consequences。
   - 如果要混合训练，应加入 data-quality weighting 或 diffusion-time-aware weighting。

7. **MODIP: Efficient Model-Based Optimization for Diffusion Policies** (`arXiv:2606.10825`)

   相关性：

   - 研究 diffusion policy 的 model-based optimization；
   - 和我们用 Foresight/scorer 改变 denoising 轨迹属于同一类“让 DP 不只是 BC”的方向。

   对本项目启发：

   - 当前 TacQuality guidance 可以被描述为 model-based test-time optimization：
     - Foresight 是 local dynamics / consequence model；
     - TacQuality 是 differentiable objective；
     - trust region 是 safety regularizer。

8. **Sample-Efficient Diffusion-based RL with Critic Guidance** (`arXiv:2605.30056`)

   相关性：

   - 用 critic guidance 引导 diffusion action generation；
   - 和 classifier guidance 的数学位置类似。

   对本项目启发：

   - TacQuality scorer 可以类比 critic，但不是任务成功 reward critic；
   - 它是 tactile consequence critic / energy；
   - 论文表达上可以写成：

```text
score(action | obs)
  = QualityEnergy(Foresight(obs, action))
```

   - 然后在 denoising 中加：

```text
a_t <- a_t + eta * grad_a score(action | obs)
```

9. **Learning from the Best: Smoothness-Driven Metrics for Data Quality in Imitation Learning** (`arXiv:2604.23000`)

   相关性：

   - 用 smoothness/data-quality 指标评价 demonstrations；
   - 和擦黑板“力变化柔顺平稳”一致。

   对本项目启发：

   - smoothness 只能作为 board quality 的一个分量，不能单独决定好坏；
   - 我们的实验已经证明：too-small 负样本可能非常 smooth，但任务质量差；
   - 因此必须使用 `force-band + smoothness` 联合能量。

10. **T-Rex: Tactile-Reactive Dexterous Manipulation** (`arXiv:2606.17055`)

   相关性：

   - 强调 tactile signals 的动态反应能力；
   - 与我们希望推理时根据预测触觉后果动态引导 action 一致。

   对本项目启发：

   - 对连续接触任务，不应把 tactile encoder 当静态 feature；
   - 应建模短窗口变化：
     - marker magnitude
     - marker temporal derivative
     - force magnitude
     - force derivative
     - contact centroid shift

## Architecture/story improvements for this project

### Current story that is already defensible

当前项目可以讲成：

```text
Vision + proprio + tactile history
  -> Diffusion Policy proposes action chunk
  -> Foresight predicts future tactile consequence of that action
  -> TacQuality scorer evaluates predicted tactile consequence
  -> classifier/energy gradient guides denoising action toward better tactile outcome
```

这条主线和 classifier guidance / critic guidance 的关系：

- classifier guidance：用外部分类器的梯度引导生成样本；
- 我们：用触觉后果质量模型的梯度引导 DP 生成 action；
- 关键区别是 scorer 不直接看 action，而是看 `Foresight(obs, action)` 的未来触觉后果，因此梯度包含 action 对未来接触状态的影响。

### Current weakness

board 任务当前最大的弱点不是“梯度不能通”，而是 scorer target 不完整：

- gradient audit 已通过，说明 action guidance 链路成立；
- 但 Foresight-score alignment 显示当前 board scorer 没有正确编码 force-band；
- 当前 scorer 会给 too-small 稳定接触高分，这与“压力太小擦不干净是坏样本”的任务定义冲突。

### Recommended next model: Force-Band Tactile Quality Energy

建议下一版 TacQuality board scorer 使用显式多分量能量：

```text
S_board =
  contact_gate *
  (
    w1 * S_marker_shape
  + w2 * S_marker_smooth
  + w3 * S_force_band
  - w4 * P_too_light
  - w5 * P_too_heavy
  - w6 * P_oscillate
  )
```

其中：

- `S_marker_shape`：marker field 是否处于稳定接触形变区间；
- `S_marker_smooth`：marker temporal delta 是否平稳；
- `S_force_band`：未来/当前力是否处于目标区间；
- `P_too_light`：力太小，擦不干净；
- `P_too_heavy`：力太大，危险/磨损/卡住；
- `P_oscillate`：力变化忽大忽小，不柔顺；
- `contact_gate`：只在擦拭接触阶段生效，避免 approach/free-space 被错误评分。

### Recommended Foresight improvement

Foresight 输出从：

```text
future marker latent / marker field
```

升级为：

```text
future marker field
+ future force proxy
+ future quality proxies
```

最低成本版本：

- 不改 DP；
- 在 Foresight 上新增 force head，预测：
  - future force magnitude mean
  - future force magnitude delta
  - force-band class: too_small / proper / too_large / oscillate

更强版本：

- Foresight 多任务训练：

```text
L = L_marker + lambda_delta * L_marker_delta
  + lambda_force * L_force
  + lambda_band * CE(force_band)
```

注意：

- 这里不是加“主观质量 loss”；
- force-band label 来自数据采集 regime 或 force 曲线阈值，是物理任务定义；
- 更通用，也更容易在论文中解释。

### Recommended DP training strategy

对 board：

- DP policy 训练优先使用 positive/high-quality 或 quality-weighted 数据；
- bad data 不应直接当普通示范混入 policy；
- bad data 应主要用于训练 scorer / force-band classifier；
- 如果要混入 260617 plus bad regimes，应做 ablation：
  1. positive-only DP
  2. 260617-only DP
  3. full mixed DP
  4. quality-weighted mixed DP

评估不能只看 noise-prediction val loss，还要看：

- 真机 force curve 是否落在目标区间；
- force delta 是否平滑；
- marker_offset 是否稳定；
- completion / coverage；
- guided vs baseline 的 paired trajectory 对比。

### Most useful next experiments

1. 继续监督当前 260617-only DP 到 2000 epoch 或 watcher 自动停止。
2. 用 `dp_best.pth` 做 baseline server 测试，不用 latest。
3. 新建 force-band-aware board scorer：
   - 使用 260609/260610 四类 board 数据；
   - label: positive / too_small / too_large / oscillate；
   - eval: episode-level GroupKFold。
4. 给 Foresight 增加 force/force-band head，重新做：
   - predicted score AUC；
   - GT score AUC；
   - predicted-vs-GT score Spearman；
   - gradient audit。
5. 真机测试记录 force curve：
   - baseline；
   - guided；
   - 每条轨迹单独保存；
   - 最后按 baseline/guided 两组画 force magnitude / Fz / delta 曲线。

## 2026-06-18 14:31 force-band board scorer experiment

新增实验：

- script: `TFAC_V5/tac_quality_energy/eval_board_force_band_scorer.py`
- output:
  - `/home/chenshuai/Project/output/tac_quality_board_force_band_eval/board_force_band_scorer_eval.md`
  - `/home/chenshuai/Project/output/tac_quality_board_force_band_eval_mlp/board_force_band_scorer_eval.md`

目标：

- 验证 board 任务是否可以构造比旧 `profile` scorer 更合理的 force-band-aware quality；
- 比较 deployable marker/action features 和 oracle future force features；
- 检查可微 MLP 是否足够接近 RF/HGB，因为最终 DP guidance 需要对 action 反传梯度。

数据：

- positive: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/wipe_pos_straight_z124_125_150_20260609`
- too_small: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/z_too_high`
- too_large: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_low`
- oscillate: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_oscillate`

协议：

- `2652` contact/wiping-phase windows；
- `221` episode groups；
- `GroupKFold=5` by episode；
- window `8`, future horizon `16`, action chunk `16`；
- phase fraction: episode `25%~85%`。

物理质量目标：

```text
quality = 0.62 * force_band
        + 0.25 * force_smooth
        + 0.13 * marker_smooth
```

其中：

- positive force-magnitude center: `11.8385`
- force-band sigma: `3.0232`
- positive force-delta q75: `0.5776`

按类别统计：

| label | n | quality mean | force mag mean | force delta mean | |Fz| mean |
|---|---:|---:|---:|---:|---:|
| positive | 1200 | 0.6398 | 12.1001 | 0.4347 | 8.6427 |
| too_small | 480 | 0.5560 | 7.9478 | 0.1586 | 7.8804 |
| too_large | 480 | 0.3639 | 17.2560 | 0.5124 | 13.5060 |
| oscillate | 492 | 0.4846 | 9.4242 | 0.4239 | 6.7156 |

最佳分类结果：

| feature variant | best | AUC | bACC | reason F1 | Spearman(q) |
|---|---|---:|---:|---:|---:|
| marker_left | MLP | 0.9963 | 0.9796 | 0.9596 | 0.3625 |
| marker_both | RF | 0.9996 | 0.9904 | 0.9922 | 0.4847 |
| marker_action | HGB | 1.0000 | 0.9842 | 0.9957 | 0.4331 |
| force_oracle | HGB | 0.9928 | 0.9333 | 0.9661 | 0.4659 |
| marker_action_force_oracle | HGB | 0.9999 | 0.9821 | 0.9946 | 0.4228 |

可微 MLP 关键结果：

| feature variant | model | AUC | bACC | binary F1 | reason F1 | Spearman(q) | q top-bottom gap |
|---|---|---:|---:|---:|---:|---:|---:|
| marker_action | MLP | 0.9977 | 0.9893 | 0.9893 | 0.9876 | 0.4115 | 0.1939 |
| marker_action_force_oracle | MLP | 0.9972 | 0.9894 | 0.9893 | 0.9889 | 0.4380 | 0.2358 |
| force_oracle | MLP | 0.9901 | 0.9551 | 0.9548 | 0.9664 | 0.4998 | 0.3127 |

关键结论：

- Board 四类数据在 tactile/action proxy 上已经非常可分；
- 只用左手 marker 也能得到 `AUC=0.9963`, `bACC=0.9796`，说明 deployment 可以先用左手 tactile；
- `marker_action` 的可微 MLP 已经足够强：`AUC=0.9977`, `bACC=0.9893`, `reason F1=0.9876`；
- RF/HGB 可以作为 teacher / upper bound，但不能直接用于 DP denoising gradient；
- force oracle 没有显著提升二分类 AUC，但提升了 quality ordering，尤其 `force_oracle/mlp` 的 Spearman(q) `0.4998` 和 q top-bottom gap `0.3127`；
- 这说明下一版 scorer 可以先用 marker/action MLP 接入 guidance，同时保留 force-band head 或 force teacher distillation 来改进 score calibration。

下一步推荐实现：

```text
ForceBandTacQualityEnergy
  input:
    marker proxy, action proxy
    optional predicted force proxy
  heads:
    binary good/bad
    reason: too_small / positive / too_large / oscillate
    continuous force-band quality
    teacher distillation from HGB/RF
    bounded residual energy
  guidance score:
    S = w_bin * good_margin
      + w_reason * positive_vs_bad_reason_margin
      + w_quality * quality_logit
      + w_teacher * teacher_logit
```

部署路径：

```text
DP denoising action
  -> Foresight predicts future marker
  -> marker/action proxy features
  -> ForceBandTacQualityEnergy MLP score
  -> grad(score) wrt action through Foresight
  -> trust-region action update
```

这比旧 board `profile` scorer 更符合用户定义的好坏标准，因为评分目标显式包含“力大小合适”和“力变化平稳”。

## 2026-06-18 14:39 implemented differentiable ForceBandTacQualityEnergy

根据上面的 force-band scorer 实验，已经实现并训练了一个可微 PyTorch scorer：

- model/runtime:
  - `TFAC_V5/tac_quality_energy/force_band_runtime.py`
- trainer:
  - `TFAC_V5/tac_quality_energy/train_board_force_band_energy.py`
- checkpoint dir:
  - `/home/chenshuai/Project/output/board_force_band_tac_quality_energy`
- best checkpoint:
  - `/home/chenshuai/Project/output/board_force_band_tac_quality_energy/force_band_tac_quality_energy_best.pt`
- report:
  - `/home/chenshuai/Project/output/board_force_band_tac_quality_energy/force_band_tac_quality_energy_train.md`

模型：

```text
ForceBandTacQualityEnergy
  input: marker_action proxy, 74 dims
  shared MLP encoder
  heads:
    binary good/bad
    reason: too_small / positive / too_large / oscillate
    continuous force-band quality
    HGB/RF teacher distillation
    free residual energy
```

energy:

```text
E = 0.40 * quality_logit
  + 0.25 * good_margin
  + 0.20 * reason_margin
  + 0.10 * teacher_logit
  + 0.05 * free_energy
```

训练结果：

| metric | held-out val |
|---|---:|
| best epoch | 141 |
| AUC | 0.9942 |
| balanced accuracy | 0.9881 |
| binary macro F1 | 0.9888 |
| reason macro F1 | 0.9894 |
| quality Spearman | 0.9759 |
| energy-quality Spearman | 0.8134 |

Gradient smoke:

| check | value |
|---|---:|
| pass | True |
| marker grad finite | True |
| action grad finite | True |
| marker grad norm | 0.0044 |
| action grad norm | 0.0897 |

独立 runtime 复查：

- `ForceBandTacQualityEnergyRuntime` 可正常 load；
- `score(..., mode='profile')` 可正常输出；
- marker/action 梯度有限且非零；
- 因此该 scorer 已经满足“可用于 classifier guidance 的可微评分器候选”的基本条件。

仍需完成：

- 接真实 board multistep Foresight 做 `score(Foresight(action))` gradient audit；
- 做 predicted-score 与真实 future force-band quality 的 alignment audit；
- 最后才是真机 baseline vs guided force curve 对比。

## 2026-06-18 14:49 real-Foresight gradient audit for ForceBand scorer

已经完成真实 board multistep Foresight 链路下的 gradient audit。

代码改动：

- `TFAC_V5/tac_quality_energy/force_band_runtime.py`
  - `score()` 兼容 `task_id` 参数；
- `TFAC_V5/tac_quality_energy/eval_guidance_gradient_audit.py`
  - 支持 `ForceBandTacQualityEnergyRuntime`；
  - 支持用命令行覆盖 scorer runtime/checkpoint/score_mode。

运行配置：

```text
task: board
scorer: ForceBandTacQualityEnergyRuntime
score mode: profile
scorer checkpoint:
  /home/chenshuai/Project/output/board_force_band_tac_quality_energy/force_band_tac_quality_energy_best.pt
Foresight:
  /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload/foresight_best.ckpt
samples: 24 real board windows
```

输出：

- `/home/chenshuai/Project/output/tac_quality_force_band_guidance_gradient_audit/guidance_gradient_audit.json`
- `/home/chenshuai/Project/output/tac_quality_force_band_guidance_gradient_audit/guidance_gradient_audit.md`

结果：

| metric | value |
|---|---:|
| pass | True |
| finite grad rate | 1.0000 |
| positive grad rate | 1.0000 |
| accept rate | 1.0000 |
| improved rate | 1.0000 |
| trust-region pass rate | 1.0000 |
| score delta mean | 0.001738 |
| action delta norm mean | 0.000798 |

与旧 board PTGProxy scorer 对比：

| scorer | samples | pass | score delta mean | action delta norm mean |
|---|---:|---|---:|---:|
| PTGProxyScorerV2Runtime | 24 | True | 0.0000629 | 0.000801 |
| ForceBandTacQualityEnergyRuntime | 24 | True | 0.001738 | 0.000798 |

解释：

- 在几乎相同的 trust-region action delta 下，ForceBand scorer 的 score improvement 约为旧 PTGProxy scorer 的 `27.6x`；
- 说明新 scorer 不只是离线分类强，也能通过真实 Foresight 链路对 action 产生更强的可优化信号；
- 这支持将 ForceBand scorer 作为 board DP guidance 的下一版候选。

注意：

- 这仍不是“真机擦得更好”的证据；
- 现在证明的是：

```text
action -> Foresight -> predicted tactile marker -> ForceBand score -> gradient wrt action
```

这条链路成立、有限、非零、受 trust-region 约束。

下一步必须做：

- `score(Foresight(action))` 和真实 future force-band quality 的 alignment audit；
- 真机 baseline vs guided 的 force curve 对比。

## 2026-06-18 14:53 ForceBand Foresight-score alignment

完成了 `score(Foresight(action))` 与真实 future marker/force 指标的对齐评估。

代码：

- `TFAC_V5/tac_quality_energy/eval_foresight_score_alignment.py`

输出：

- `/home/chenshuai/Project/output/tac_quality_force_band_foresight_score_alignment/foresight_score_alignment.json`
- `/home/chenshuai/Project/output/tac_quality_force_band_foresight_score_alignment/foresight_score_alignment.md`
- `/home/chenshuai/Project/output/tac_quality_force_band_foresight_score_alignment/foresight_score_alignment_samples.csv`
- `/home/chenshuai/Project/output/tac_quality_force_band_foresight_score_alignment/foresight_score_alignment.png`

结果：

| metric | ForceBand |
|---|---:|
| predicted-score AUC(good) | 0.7321 |
| GT-future-score AUC(good) | 0.7248 |
| predicted vs GT score Spearman | 0.9501 |
| predicted vs GT score Pearson | 0.9552 |
| predicted score vs force-band quality Spearman | -0.1764 |
| predicted score vs -force delta Spearman | -0.5177 |
| force-band quality AUC(good) | 0.6378 |

旧 PTGProxy 对比：

| scorer | pred AUC(good) | GT AUC(good) | pred-GT Spearman |
|---|---:|---:|---:|
| PTGProxyScorerV2Runtime | 0.5622 | 0.3250 | 0.5932 |
| ForceBandTacQualityEnergyRuntime | 0.7321 | 0.7248 | 0.9501 |

解释：

- ForceBand scorer 相比旧 PTGProxy 明显更能区分 positive vs negative collection regime；
- Foresight-predicted score 与 GT future marker score 的一致性很高，说明 Foresight 保留了 scorer 所需的 marker/action pattern；
- 但 predicted score 与当前手工定义的 continuous force-band quality 是弱负相关。

这暴露了一个重要边界：

```text
ForceBand scorer is good at classifying collection regimes
but not yet perfectly aligned with continuous physical force-band quality.
```

可能原因：

- 训练标签里 `positive` 是 collection-level 弱标签，但 positive episode 内部某些窗口 force_mag 过大，按 force-band 公式会被罚低；
- 当前 scorer 的 binary/reason/teacher heads 仍然强，会更偏向区分采集模式，而不是优化连续 force target；
- 当前 board Foresight 只预测 marker，不预测 force，因此 true force-band quality 只能间接从 marker/action 推断。

对下一步的影响：

- 可以把 ForceBand scorer 作为比旧 PTGProxy 更强的 board guidance 候选；
- 但如果论文/部署目标强调“力大小合适 + 力变化柔顺”，下一版应继续优化 force-quality alignment：

```text
Option A: add force/force-band head to Foresight
Option B: retrain scorer with stronger continuous quality loss and weaker teacher/binary dominance
Option C: relabel positive windows by actual force-band quality instead of episode-level positive label
```

推荐路线：

- 短期：用当前 ForceBand scorer 做 cautious guided rollout，因为它的 gradient audit 和 regime discrimination 都显著优于旧 scorer；
- 中期：训练 force-aware Foresight 或 force-quality calibrated scorer；
- 真机评估必须保存 baseline/guided force curves，最终以 force-band tracking 和 smoothness 判断。

## 2026-06-18 15:00 训练监督更新

当前 260617-only DP 训练仍在运行：

- train PID：`1544542`
- 输出目录：`/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000`
- 最新日志：第 `311/2000` epoch 已完成，训练继续。
- 第 `311` epoch：
  - `train=0.005404`
  - `val=0.021822`
- 当前 best：
  - 第 `105` epoch
  - `val=0.011152`
- GPU：
  - RTX 4090；
  - 训练进程显存约 `13.9GB`；
  - GPU 利用率随 dataloader/validation 波动。
- 磁盘：
  - run 目录约 `16G`；
  - image cache 约 `156G`；
  - `/home` 可用约 `47G`。

判断：

- 训练没有崩，loss 有效、ckpt 正常更新；
- 但第 105 epoch 后 validation 暂未刷新，当前 latest 有过拟合趋势；
- 部署/真机测试应默认优先使用 `dp_best.pth`，不是 `dp_latest.pth`；
- 当前不停止训练，因为：
  - 用户目标是充分训练 `2000` epoch；
  - `dp_best.pth` 已保留当前最优；
  - 后期学习率下降后仍可能有新 best；
  - watcher 已设置后期 plateau stop。

## 对本项目架构的改进优先级

结合最近两个月 tactile world model / inference-time steering / diffusion policy 方向的趋势，本项目当前故事线是成立的：

```text
DP 负责生成动作先验
Foresight 负责预测未来触觉后果
TacQualityEnergy 负责把未来触觉变成可微质量分数
Contact gate 决定什么时候启用触觉引导
Trust-region gradient guidance 在小范围内修改 action
```

最值得继续加强的不是 reranking，而是以下四点。

### P0: Force-aware Foresight

当前 board Foresight 主要预测 marker/latent，不直接预测 force 或 force-band quality。

这导致一个问题：擦黑板真正的质量标准是“力大小合适 + 力变化柔顺”，而 scorer 只能从 marker/action 间接推断 force quality。

建议下一版 Foresight 增加多任务输出：

```text
future marker latent
future marker proxy
future force proxy: |F|, Fz, dF, contact probability
future force-band quality
```

Loss 可以保持论文上简洁：

```text
L = L_latent + L_marker + lambda_delta L_delta + lambda_force L_force_proxy
```

不建议现在加入复杂 KL/CVAE，除非明确要建模多模态未来；当前第一目标是预测精确、可用于梯度引导。

### P1: Calibrated TacQuality scorer

当前 ForceBand scorer 的强项是分类 collection regime，弱点是与连续 force-band quality 的相关性还不够好。

下一版 scorer 应把质量定义从“episode-level 正负标签”细化到“window-level 质量”：

```text
score = w1 * force_in_band
      + w2 * force_smoothness
      + w3 * contact_stability
      + w4 * marker_spatial_consistency
      - w5 * action_jerk
```

训练目标可以仍然是多头，但权重应调整：

- binary/reason head：保证能区分正/负采集模式；
- quality head：主导最终 guidance score；
- teacher head：只作为辅助蒸馏，不应支配最终能量；
- residual energy：小权重，用于修正 hand-crafted quality 的盲区。

### P2: Contact-gated guidance

擦黑板只应在 wiping/contact 阶段强引导。

Approach/lift 阶段低触觉、低力是正常状态，不应该被 scorer 惩罚。

当前 server-side contact gate 是合理的：

```text
low marker/contact -> guidance scale 0
middle contact -> partial guidance
stable contact -> full guidance
```

后续可以把 gate 从观测 marker 扩展为 Foresight 预测 contact probability，使引导更提前。

### P3: Real rollout force-curve evaluation

离线 loss、AUC、gradient audit 都不能替代真机结果。

真机测试时必须保存每条轨迹的 force curve，并按 contact phase 统计：

- contact-phase mean Fz / |F|
- contact-phase p95 force
- contact-phase dF/dt
- force-in-band ratio
- marker smoothness
- early stop / unsafe stop

最终报告应比较：

```text
baseline DP vs guided DP
best checkpoint vs latest checkpoint
positive-only/full/260617-only policy variants
```

结论边界：

- 现在可以说：我们有一个可微 scorer，离线区分强，Foresight 梯度链路可用；
- 现在不能说：guidance 已经真实改善擦黑板力曲线；
- 这个结论必须等 server-side force logging 后的真机 rollout 对比。

## 已核验的近期论文依据

以下信息来自 arXiv 页面，按 2026-06-18 往前约两个月筛选。

- ViTaL: Inference-time Policy Steering via Vision and Touch
  - arXiv: <https://arxiv.org/abs/2606.14981>
  - submitted: 2026-06-12
  - 和本项目最相关点：它把 inference-time steering 拆成视觉长程选择与触觉短程 diffusion editing；还用 latent world model 和 verifier 评分 predicted tactile futures。这和我们的 `DP action -> Foresight -> TacQualityEnergy -> trust-region gradient` 是同一类故事。
- Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation
  - arXiv: <https://arxiv.org/abs/2606.08737>
  - submitted: 2026-06-07
  - 和本项目最相关点：联合建模 action、future vision、future tactile dynamics，并使用 contact-gated visuotactile fusion。它支持我们对擦黑板加 contact gate 的设计。
- TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation
  - arXiv: <https://arxiv.org/abs/2606.11184>
  - submitted: 2026-06-09
  - 和本项目最相关点：force-conditioned tactile world model 预测短期 tactile latent dynamics。它直接支持我们下一步做 force-aware Foresight，而不是只预测 marker latent。
- ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation
  - arXiv: <https://arxiv.org/abs/2606.13877>
  - submitted: 2026-06-11
  - 和本项目最相关点：强调 spatially structured、temporally continuous 表示和跨模态兼容性。它支持我们保留 marker spatial proxy、contact area/center/spread/smoothness，而不是只用一个 force scalar。
- Ambient Diffusion Policy: Imitation Learning from Suboptimal Data in Robotics
  - arXiv: <https://arxiv.org/abs/2606.12365>
  - submitted: 2026-06-10
  - 和本项目最相关点：低质量/负样本不应简单混入 DP imitation policy；更合理的是让负样本主要训练 scorer/verifier，或用 diffusion-time/data-quality aware 的训练权重。
- Latent Diffusion Policy: Shaping Latent Spaces for Diffusion-Based Robotic Manipulation
  - arXiv: <https://arxiv.org/abs/2606.08657>
  - submitted: 2026-06-07
  - 和本项目最相关点：将 action sequence 压到 observation-conditioned latent space 再生成，降低直接在 raw action space denoise 的复杂度。它是后续如果当前 DP 动作不够稳时的策略本体升级候选。
- FTP-1: A Generalist Foundation Tactile Policy Across Tactile Sensors for Contact-Rich Manipulation
  - arXiv: <https://arxiv.org/abs/2606.13102>
  - submitted: 2026-06-11
  - 和本项目最相关点：统一不同 tactile sensor 的 latent token。它更偏长期方向，可用于把当前 TactileVAE 从 task-local encoder 升级为多任务 tactile token encoder。
- Tube Diffusion Policy
  - arXiv: <https://arxiv.org/abs/2604.23609>
  - submitted: 2026-04-26
  - 和本项目最相关点：action chunking 在 contact-rich 场景中反应慢，step-wise correction/action tube 更适合触觉反馈。这支持我们做梯度引导而不是单纯 reranking。
- Multi-Resolution Tactile Imitation Learning for Contact-Rich Robotic Manipulation
  - arXiv: <https://arxiv.org/abs/2606.06281>
  - submitted: 2026-06-04
  - 和本项目最相关点：不同时间尺度触觉融合。它支持我们把 marker 空间形变和 force/delta-force 平滑性都纳入 scorer，而不是只看单帧 marker magnitude。
- HapTile: A Haptic-Informed Vision-Tactile-Language-Action Dataset for Contact-Rich Imitation Learning
  - arXiv: <https://arxiv.org/abs/2606.04825>
  - submitted: 2026-06-03
  - 和本项目最相关点：contact-rich 数据集需要同时保存触觉、力反馈、动作轨迹和任务结果。这支持 server-side force curve logging。
- DreamTacVLA: Learning to Feel the Future
  - arXiv: <https://arxiv.org/abs/2512.23864>
  - submitted: 2025-12-29, revised: 2026-05-06
  - 说明：不是近两个月新提交主证据，但可作为“未来触觉预测用于动作修正”的背景参考。
  - 和本项目最相关点：先生成 draft action，再预测未来 tactile，最后 refine action。

当前结论：

- 短期不改训练中的 260617-only DP；
- 论文故事上应强调 inference-time tactile consequence guidance；
- 技术下一步优先级仍是：
  1. force-aware Foresight；
  2. force-quality calibrated TacQualityEnergy；
  3. contact-gated trust-region gradient guidance；
  4. 真机 force curve 闭环评估。

## 2026-06-18 16:05 训练监督更新与已核验调研结论

训练状态：

- 训练进程仍在运行，PID `1544542`。
- 数据路径核查通过：`/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban` 下只有 `peg_in_hole_0617` 一个有效 hdf5 数据目录。
- 数据规模：80 个 hdf5 episode；当前配置记录 `n_train=72`, `n_val=8`，符合 episode-level `val_ratio=0.1`。
- 最新观察到第 415 epoch：
  - 第 413 epoch：`train=0.004358`, `val=0.033736`
  - 第 414 epoch：`train=0.004452`, `val=0.031499`
  - 第 415 epoch：`train=0.004767`, `val=0.036013`
  - 当前 best 仍是第 105 epoch，`val=0.011152`
- 判断：
  - 训练本身健康，GPU 利用率约 `94%~95%`，没有 NaN；
  - `train loss` 持续降低，但 `val loss` 从第 105 epoch 后长期未刷新，当前 latest 已明显不如 best；
  - 这是小数据集大模型训练中典型的过拟合/分布差异信号；
  - 当前不停止，因为用户目标是 2000 epoch，同时 watcher 已设置 1500 epoch 后 plateau early stop；
  - 后续离线/真机测试应优先使用 `dp_best.pth`，不要默认使用 `dp_latest.pth`。

已核验的近两个月 arXiv 条目：

- `2606.14981` ViTaL: Inference-time Policy Steering via Vision and Touch
- `2606.08737` Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation
- `2606.11184` TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation
- `2606.13877` ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation
- `2606.14801` QPILOTS: Efficient Test-Time Q-Steering for Flow Policies
- `2606.08414` PACT: Self-Evolving Physical Safety Alignment for Diffusion Policies in Embodied Manipulation
- `2604.23609` Tube Diffusion Policy: Reactive Visual-Tactile Policy Learning for Contact-rich Manipulation
- `2606.12365` Ambient Diffusion Policy: Imitation Learning from Suboptimal Data in Robotics
- `2606.16447` Training and Evaluating Diffusion Policies with Long Context Lengths
- `2606.17982` LAGO Policy: Latency-Aware Asynchronous Diffusion Policies with Goal-Directed Collision-Free Planning for Smooth Manipulation
- `2605.27886` Tabero: Learning Gentle Manipulation with Closed-Loop Force Feedback from Vision, Touch, and Language
- `2605.23568` TactileReflex: Noise-Statistics-Driven Vision-Tactile Reflex Control for Force-Sensitive Manipulation
- `2606.18959` TactSpace: Learning a Physics-enriched Shared Latent Space for Tactile Sim-to-Real Transfer
- `2606.17055` T-Rex: Tactile-Reactive Dexterous Manipulation

对当前项目最直接的改进方向：

1. 保持主线为 gradient guidance，不回到 reranking。
   - `QPILOTS` 和 `ViTaL` 都支持 test-time steering/guidance 这条故事；
   - 当前项目应表述为：`DP denoised action -> Foresight predicted tactile consequence -> TacQuality differentiable score -> action gradient update`。

2. 给 Foresight 增加 force-aware 或 force-proxy 预测头。
   - `TacForeSight`, `Tabero`, `TactileReflex` 都强调 force/contact quality；
   - 当前 board scorer 的分类能力已经很强，但 continuous force-quality calibration 弱；
   - 只靠 marker latent 预测可能不足以稳定优化“力大小合适 + 力变化柔顺”。

3. Scorer 需要从 regime classifier 升级为 force-quality calibrated scorer。
   - 旧版 ForceBand scorer 的 `quality` mode 对正负采集 regime 有强区分；
   - 但和手工连续 force-band quality 的相关性弱；
   - 下一版应使用 contact-phase force curve 生成 window-level soft label，例如 force band ratio、dF/dt smoothness、marker smoothness，并在 episode-level split 下验证。

4. 引导时加入 contact gate 和 trust region。
   - `Dream-Tac`, `ContactWorld`, `Tube Diffusion Policy` 都支持 contact-aware/reactive 的短程修正；
   - 擦黑板只应在 contact/wiping phase 对 action 做质量梯度，不应在 approach/lift 阶段强拉力分数；
   - trust region 必须限制 action delta，避免 scorer shortcut 或 Foresight 误差导致动作异常。

5. DP 训练数据策略要区分 policy imitation 和 scorer training。
   - `Ambient Diffusion Policy` 提醒低质量数据直接混入 imitation policy 可能有害；
   - 当前 260617-only DP 如果都是同一批新数据，可以继续训练；
   - 对明显负样本或不稳定样本，建议主要用于 scorer/verifier，而不是直接作为 DP 正向模仿目标。

暂时不建议做的方向：

- 不建议把当前目标改成 candidate reranking，因为用户目标是 DP 去噪过程中的梯度引导。
- 不建议只追求离线 noise-prediction val loss；最终必须用真机 force curve、contact-phase force band、平滑度和成功率评估。
- 不建议声称 guidance 已经提升真实擦黑板质量；当前证据只支持离线 scorer、Foresight 梯度链路和训练中的 DP checkpoint。

## 2026-06-18 16:45 训练监督与调研更新

训练状态：

- 训练进程仍在运行，PID `1544542`。
- 最新观察到第 470 epoch：
  - `train=0.004000`
  - `val=0.030129`
  - 当前 best 仍为第 105 epoch，`val=0.011152`
- 最近窗口统计：
  - 最近 10 epoch：`train_mean=0.004051`, `val_mean=0.035285`
  - 最近 25 epoch：`train_mean=0.004140`, `val_mean=0.034790`
  - 最近 50 epoch：`train_mean=0.004255`, `val_mean=0.033185`
  - 最近 100 epoch：`train_mean=0.004431`, `val_mean=0.031951`
- 判断：
  - 训练本身正常，GPU 利用率约 94%，没有 NaN 或进程异常。
  - `train loss` 继续下降，但 `val loss` 从第 105 epoch 后已经 365 个 epoch 未刷新，当前有明显 train/val gap。
  - 这更像是 260617-only 小数据集 + 大容量 DP 的 latest checkpoint 过拟合，而不是训练程序故障。
  - `dp_best.pth` 已保存第 105 epoch 的最佳验证 ckpt，后续测试应固定优先用 `dp_best.pth`，不要用 `dp_latest.pth` 代表最终效果。
  - 当前不停止训练，因为用户指定 2000 epoch，且 watcher 设置为 1500 epoch 后才根据 plateau 自动停；如果后期 learning rate 下降带来二次改善，仍可能刷新 best。

磁盘状态：

- `/home` 当前约剩余 47G，已用 98%。
- 当前 run 目录约 12G。
- raw image fp16 cache 约 156G。
- 由于 checkpoint 单个约 2.6G，必须继续控制保存频率；当前配置 `save_freq=500`, `topk_k=3` 暂时可控。

近两个月最新工作对当前项目的直接启发：

1. `ViTaL` / Inference-time Policy Steering via Vision and Touch, arXiv 2606.14981
   - 提出视觉长程 mode selection + 触觉短程 diffusion editing。
   - 对本项目最直接的启发：我们的故事应明确为 `DP clean action -> Foresight predicted tactile consequence -> TacQuality differentiable score -> trust-region action gradient update`，不是 reranking。

2. `QPILOTS` / Efficient Test-Time Q-Steering for Flow Policies, arXiv 2606.14801
   - 强调不要直接在 noisy intermediate action 上用 critic gradient，而是先估计 final clean action 再计算可用梯度。
   - 对本项目最直接的启发：当前在 DP clean action chunk 后接 TacQuality 梯度更新是合理的；后续若要做到 denoising-step 内部 guidance，也应采用 clean-action projection 或 x0-estimate guidance，而不是直接对 noisy action 评分。

3. `TacForeSight` / Force-Guided Tactile World Model for Contact-Rich Manipulation, arXiv 2606.11184
   - 强调 force-conditioned tactile latent dynamics。
   - 对本项目最直接的启发：擦黑板最终质量定义是力大小合适和力变化柔顺，因此下一版 Foresight 不应只预测 marker latent，最好增加 force-aware 输入或 force/force-proxy 预测头。

4. `Dream-Tac` / Unified Tactile World Action Model, arXiv 2606.08737
   - 联合建模 action、future vision、future tactile dynamics，并使用 contact-gated fusion。
   - 对本项目最直接的启发：擦黑板 scorer/guidance 必须只在接触擦拭阶段强约束，approach/lift 阶段不能用同一套 force-band 评分强拉动作。

5. `ContactWorld` / What Matters in Vision-Tactile World Models, arXiv 2606.13877
   - 强调 spatially structured 和 temporally continuous 表示。
   - 对本项目最直接的启发：TacQuality 评分不能退化成单个 marker magnitude 或单个 force scalar，应保留 marker field、接触面积、中心、扩散范围、时间平滑性等结构化 proxy。

6. `FlowMPC` / Improving Flow Matching policies with World Models, arXiv 2606.16286
   - 用 world model 在 test-time 改善 flow policy。
   - 对本项目最直接的启发：Foresight + scorer 是我们区别于普通 tactile DP 的核心，不应只作为可视化模块，而应成为动作生成时的闭环约束。

7. `LAGO Policy` / Latency-Aware Asynchronous Diffusion Policies, arXiv 2606.17982
   - 关注 chunk 间不连续和低 jerk 执行。
   - 对本项目最直接的启发：擦黑板引导除了 force-band，还应在 action/chunk 层加入 smoothness 或 jerk penalty，避免 scorer 提升但轨迹抖动。

8. `Multi-Resolution Tactile Imitation Learning`, arXiv 2606.06281
   - 使用不同时间尺度的触觉信息。
   - 对本项目最直接的启发：16 帧 tactile history 是合理的，但 scorer 里应同时看短期力变化率和较长窗口稳定接触，而不是只看单帧分类。

阶段性建议：

- 训练继续监督，不干预当前 run。
- 260617-only DP 的最终候选应优先是 `dp_best.pth`。
- 论文/方案故事建议命名为：`Tactile Consequence-Guided Diffusion Policy`。
- 最值得投入的架构改进不是再堆分类头，而是：
  1. force-aware multistep Foresight；
  2. contact-phase force-quality calibrated TacQualityEnergy；
  3. clean-action/x0-estimate 上的 trust-region gradient guidance；
  4. server-side force curve logging 的真机闭环评估。

训练脚本核查：

- 文件：`diffusion/train_dp_tac_concat.py`
- validation split 方式：
  - 先收集 episode 文件列表；
  - 用 `seed=1` 对 episode 列表 shuffle；
  - 按 `val_ratio=0.1` 切出验证 episode；
  - 分别从 train episode 和 val episode 建立 sliding-window dataset。
- 结论：
  - 当前不是 frame-level/window-level random split；
  - 没看到 train/val frame 泄漏；
  - 因此第 105 epoch 后 val 明显变差更应被当作泛化风险，而不是切分错误。
- checkpoint 逻辑：
  - `dp_best.pth` 按 `val_loss` 更新；
  - `dp_latest.pth` 每 epoch 覆盖保存，包含 optimizer；
  - `dp_topk_*.pth` 按 train loss top-k 保存，不代表泛化最优；
  - 因此后续真机测试默认应使用 `dp_best.pth`。

## 2026-06-18 16:56 260617 新数据 ForceBand 分布审计

新增脚本：

- `TFAC_V5/tac_quality_energy/audit_board_260617_forceband_distribution.py`

输出：

- `/home/chenshuai/Project/output/board_260617_forceband_distribution_audit/board_260617_forceband_distribution_audit.md`
- `/home/chenshuai/Project/output/board_260617_forceband_distribution_audit/board_260617_forceband_distribution_audit.json`
- `/home/chenshuai/Project/output/board_260617_forceband_distribution_audit/board_260617_forceband_distribution.png`
- `/home/chenshuai/Project/output/board_260617_forceband_distribution_audit/board_260617_forceband_distribution_windows.csv`

目的：

- 不训练新模型；
- 用当前 `ForceBandTacQualityEnergyRuntime` 检查 260617-only 新数据是否与 260609/260610 的旧正负样本分布对齐；
- 同时计算两种质量：
  - `scorer_quality`：当前神经评分器 quality head；
  - `physical_quality`：用旧 positive force-band 作为中心的手工物理质量分。

关键结果：

| label | n | force_mag mean | force_delta mean | physical_quality mean | scorer_quality mean | p_good mean |
|---|---:|---:|---:|---:|---:|---:|
| new_260617 | 948 | 8.5110 | 0.3945 | 0.5058 | 0.6150 | 0.0521 |
| old positive | 1200 | 12.1084 | 0.4361 | 0.6386 | 0.6379 | 0.9853 |
| old too_small | 480 | 7.9479 | 0.1574 | 0.5529 | 0.5620 | 0.0022 |
| old too_large | 480 | 17.2540 | 0.5084 | 0.3657 | 0.3759 | 0.0177 |
| old oscillate | 492 | 9.4245 | 0.4257 | 0.4813 | 0.5165 | 0.0208 |

260617 相对旧 positive 的分位：

- force magnitude quantile mean: `0.2014`
- force delta quantile mean: `0.4243`
- marker delta quantile mean: `0.4602`
- physical quality quantile mean: `0.3140`
- scorer quality quantile mean: `0.4129`
- p_good quantile mean: `0.0042`

解释：

- 260617 的 `scorer_quality=0.6150` 接近旧 positive 的 `0.6379`，但 `p_good=0.0521` 远低于旧 positive 的 `0.9853`。
- 从物理力带看，260617 的 `physical_quality=0.5058` 也低于旧 positive 的 `0.6386`，主要因为 force magnitude 偏低，接近旧 positive 分布的低分位。
- 这说明当前旧 ForceBand scorer 的二分类/positive reason 头对 260617 有明显分布偏移；不能把 `p_good` 当作 260617 上可靠的好坏概率。
- score-mode 与手工物理质量的相关性进一步支持这个判断：
  - 在 260617 窗口上，`scorer_quality` 与 `physical_quality` 的 Spearman 为 `0.5136`；
  - `profile` 为 `0.4253`；
  - `energy_clipped` 为 `0.3811`；
  - `p_good` 为 `0.2604`；
  - `reason_positive` 为 `0.2613`。
- 因此如果后续用旧 scorer 引导 260617-only DP，优先使用 `quality` 分数并做 contact-gated trust-region，引导强度要保守；不要使用 `p_good/reason_positive` 作为主要梯度分数；最终必须用真机 server-side force curve 判断是否真的改善。
- 更稳妥的下一版是用 260617 的真实力曲线重新校准 board scorer，或者把 force-aware Foresight 加进来，使 score 对 `力大小合适 + 力变化柔顺` 更直接。

## 2026-06-18 17:33 260617-only DP 训练监督更新

当前只关注用户指定的数据集：

- `/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617`

训练仍在运行：

- PID: `1544542`
- 输出目录：`/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000`
- GPU：RTX 4090，显存约 `14.7GB / 24.6GB`
- `/home` 可用空间约 `45G`

最新状态：

- 最新解析 epoch：`549/2000`
- `train=0.003229`
- `val=0.045928`
- best：第 `105` epoch，`val=0.011152`
- 距离 best 已 `444` epoch 未刷新。

最近窗口趋势：

| window | train mean | val mean | val min | val max |
|---:|---:|---:|---:|---:|
| last20 | 0.003649 | 0.040689 | 0.032780 | 0.047764 |
| last50 | 0.003758 | 0.039104 | 0.029753 | 0.047764 |
| last100 | 0.003907 | 0.036859 | 0.028934 | 0.047764 |

判断：

- 训练进程健康，没有 NaN 或崩溃。
- 但当前已经很明显是 `train loss` 继续降低、`val loss` 长期高于 best；这说明 latest checkpoint 更像是在拟合训练 windows，不适合作为默认部署模型。
- 由于用户明确要求 2000 epoch 充分训练，目前不停止；后续如果到后期仍无收益，按 watcher 的 plateau 策略处理。
- 真实测试和 server 启动默认应优先用 `dp_best.pth`，不要用 `dp_latest.pth` 或 train top-k 直接代表泛化效果。

本次监督修正：

- 修正 `scripts/utils/plot_dp_training_log.py` 的日志解析问题。
- 原问题：正则表达式在 `best=val_loss=...` 格式下会漏掉 `val_loss` 列。
- 修正后分别解析 `train / val / best`，已重新生成：
  - `/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000/loss_curve.png`
  - `/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000/loss_curve.csv`
  - `/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000/training_status_latest.txt`

## 2026-06-18 17:35 近两个月 arXiv 调研补充结论

筛选标准：

- 时间范围：2026-04-18 到 2026-06-18 左右；
- 主题：contact-rich manipulation、tactile/force feedback、diffusion/flow policy、world model、inference-time guidance/steering；
- 只记录对当前项目路线有直接启发的内容。

最相关论文和启发：

1. `Inference-time Policy Steering via Vision and Touch` (`arXiv:2606.14981`, 2026-06-12)
   - 方向：用视觉和触觉 verifier 在部署时 steering 预训练生成式策略。
   - 对本项目的意义：直接支持当前路线，即 `DP clean action -> Foresight predicted tactile consequence -> TacQuality score -> gradient guidance`。我们要强调这是 inference-time tactile consequence guidance，不是 reranking。

2. `TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation` (`arXiv:2606.11184`, 2026-06-09)
   - 方向：用 force signal 条件化 tactile latent world model，预测未来触觉动态。
   - 对本项目的意义：擦黑板的好坏标准本质依赖力大小和力变化，因此下一代 Foresight 应做 force-aware；当前 marker-only Foresight 可以作为 baseline，但不是最强故事。

3. `Feedback World Model Enables Precise Guidance of Diffusion Policy` (`arXiv:2605.15705`, 2026-05-15)
   - 方向：world model 在部署时根据真实观测反馈更新 latent feedback state，修正后续预测，并做 action-aware guidance。
   - 对本项目的意义：如果真机擦黑板存在板面、姿态、力带分布偏移，静态 Foresight 容易漂；后续可以用上一轮真实 tactile/force 与预测误差校正下一轮 guidance。

4. `QPILOTS: Efficient Test-Time Q-Steering for Flow Policies` (`arXiv:2606.14801`, 2026-06)
   - 方向：test-time steering 不应直接对 noisy intermediate action 评分，而应在 clean action / projected final action 上算梯度。
   - 对本项目的意义：当前在 DP clean action chunk 后做 trust-region 梯度修正是合理的；如果之后进入 denoising 内部，也应做 `x0-estimate guidance`，而不是直接评分 noisy action。

5. `ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation` (`arXiv:2606.13877`, 2026-06-11)
   - 方向：系统研究 vision-tactile world model 的 representation structure、multimodal compatibility、long-horizon robustness。
   - 对本项目的意义：TacQuality 不应只依赖单个力标量；应保留 marker field、接触面积、接触中心、扩散范围、时间平滑度等结构化 proxy。

6. `Latent Diffusion Policy` (`arXiv:2606.08657`, 2026-06-07)
   - 方向：先学习 observation-conditioned latent action space，再在 latent 上做 diffusion policy，降低原始 action space 学习复杂度。
   - 对本项目的意义：260617-only 只有约 79 个有效 episode，大容量原始 joint-action DP 容易过拟合；后续可以考虑 action-latent DP 或 residual-latent DP。

7. `Multi-Resolution Tactile Imitation Learning` (`arXiv:2606.06281`, 2026-06-04)
   - 方向：融合不同时间分辨率的触觉信息。
   - 对本项目的意义：16 帧 tactile history 是合理的，但评分器应同时看短时变化率和窗口级稳定性；后续有高频 force/torque 时，应作为更强输入。

8. `Tube Diffusion Policy` (`arXiv:2604.23609`, 2026-04-26)
   - 方向：把 diffusion action chunk 与 tube-based feedback correction 结合，提升 contact-rich 场景下的反应性。
   - 对本项目的意义：擦黑板不是只生成一段 open-loop action chunk 就够了，contact 误差需要执行中快速修正；我们的 gradient guidance 和 server-side force logging 可以往 reactive correction 方向扩展。

当前路线收敛：

```text
observation + tactile history
  -> DP proposes clean action chunk
  -> Foresight predicts future tactile/force consequence
  -> TacQualityEnergy scores contact force-band and smoothness
  -> trust-region gradient update on clean action chunk
  -> execute and log force curves
```

最值得做的架构改进优先级：

1. `force-aware multistep Foresight`：让 Foresight 不只预测 marker latent，也能建模 force-band / force proxy。
2. `contact-phase gated scoring`：只在擦拭接触阶段强引导，approach/lift 不强行追求力带。
3. `clean-action/x0-estimate guidance`：保持当前 clean action 后处理路线；进入 denoising 内部时也要先投影到 clean estimate。
4. `action smoothness / jerk penalty`：防止 TacQuality 分数提高但 chunk 间动作抖动。
5. `real rollout force curve evaluation`：baseline/guided 每条轨迹单独记录 force curve，按 contact phase 统计 force mean、std、delta、越界比例、marker smoothness。

当前不能过度声称：

- 只能说当前 DP 训练在跑、已有 best checkpoint；
- 只能说 scorer/Foresight/guidance 链路已有离线 dry-run 和梯度可行性；
- 不能说 guided policy 已真实提升擦黑板质量，必须等真机 force curve 对比。

## 2026-06-18 17:48 with-260617-positive ForceBand scorer guidance audit

目的：

- 前面确认旧 ForceBand scorer 对 260617 positive 的 `p_good` 存在分布偏移；
- 新训练的 `with_260617_positive` ForceBand scorer 修复了 260617 positive 的分类分布；
- 本次检查它是否真的能接入当前 `DP action -> real board Foresight -> predicted tactile -> TacQuality score -> gradient` 链路。

使用的 scorer：

- checkpoint: `/home/chenshuai/Project/output/board_force_band_tac_quality_energy_with_260617_positive_20260618/force_band_tac_quality_energy_best.pt`
- runtime: `ForceBandTacQualityEnergyRuntime`
- score mode: `quality`
- 训练/验证摘要：
  - best epoch: `28`
  - held-out binary AUC: `0.999929`
  - held-out balanced accuracy: `0.997172`
  - held-out reason macro F1: `0.997304`
  - held-out quality Spearman: `0.971293`

真实 Foresight gradient audit：

- 命令脚本：`TFAC_V5/tac_quality_energy/eval_guidance_gradient_audit.py`
- 数据：`/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617`
- Foresight: `/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload/foresight_best.ckpt`
- 输出：
  - `/home/chenshuai/Project/output/tac_quality_force_band_with260617_guidance_gradient_audit_quality/guidance_gradient_audit.json`
  - `/home/chenshuai/Project/output/tac_quality_force_band_with260617_guidance_gradient_audit_quality/guidance_gradient_audit.md`

结果：

| metric | value |
|---|---:|
| samples | 24 |
| pass | true |
| finite grad rate mean | 1.0000 |
| positive grad rate mean | 1.0000 |
| accept rate mean | 1.0000 |
| improved rate mean | 1.0000 |
| trust region pass rate | 1.0000 |
| score delta mean | 0.000054 |
| action delta norm mean | 0.000799 |

解释：

- 这说明新 scorer 在真实 board Foresight 链路上有稳定、有限、正向的 action gradient；
- trust-region 约束有效，action delta 没有越界；
- 这仍然不是机器人 rollout 质量证据，只能证明“可用于梯度引导链路”。

正式服务入口 dry-run：

- 入口：`for_show_xiaomi.serve_dp_tac_quality_guided`
- DP: `/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000/dp_best.pth`
- Foresight: same real multistep board Foresight
- 临时 rollout config：`/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_with260617_scorer_tmp.json`
- 输出：`/home/chenshuai/Project/output/tac_quality_guided_server_packet/with260617_scorer_real_foresight_smoke_20260618.json`

结果：

- `dry_run_guidance_smoke_pass=true`
- `variant=tactile_vae_frozen`
- `obs_cond_shape=[1, 2350]`
- `action_norm_shape=[1, 16, 7]`
- `scorer_runtime=ForceBandTacQualityEnergyRuntime`
- `score_mode=quality`
- `finite_grad_rate=1.0`
- `positive_grad_rate=1.0`
- `improved_rate=1.0`
- `accept_rate=0.25`
- `max_delta_within_trust_region=true`
- `contact_gate_value=1.0`
- `not_reranking=true`
- raw action delta mean: `0.000202`
- normalized action delta mean: `0.000007`
- score delta mean: `0.00000185`

当前结论：

- 新 `with_260617_positive` ForceBand scorer 比旧 scorer 更适合覆盖 260617 positive 数据分布；
- 它已经通过真实 Foresight gradient audit 和正式服务入口 dry-run；
- 但是否替换正式默认 scorer 仍需真机 force-curve 对比验证；
- 如果用户现在要在 260617-only DP 上试 guided rollout，建议使用这个新 scorer 作为候选 arm，同时记录 baseline/guided 每条轨迹 force curve。

当前 DP 训练监督：

- 训练仍运行，最新观察到约第 `570/2000` epoch；
- best 仍为第 `105` epoch，`val=0.011152`；
- latest val 仍明显高于 best，后续测试继续默认用 `dp_best.pth`。

## 2026-06-18 18:20 arXiv API 核验后的短结论

核验方式：

- 用 arXiv API 按 id 查询题名和发布日期；
- 范围控制在最近两个月附近，即 `2026-04-18` 到 `2026-06-18`；
- 只保留和当前路线直接相关的 tactile/contact-rich manipulation、world model、diffusion/flow policy、test-time steering/guidance。

已核验条目：

| arXiv id | published | title | 对当前项目的直接意义 |
|---|---|---|---|
| `2606.14981` | 2026-06-12 | ViTaL: Inference-time Policy Steering via Vision and Touch | 支持部署时用视觉/触觉 verifier/score steering 生成式机器人策略。 |
| `2606.14801` | 2026-06-11 | QPILOTS: Efficient Test-Time Q-Steering for Flow Policies | 支持 test-time guidance 应作用在 clean/projected action 上，而不是直接评分 noisy action。 |
| `2606.11184` | 2026-06-09 | TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation | 支持下一代 Foresight 加 force-aware 预测。 |
| `2606.13877` | 2026-06-11 | ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation | 支持保留结构化 tactile/marker proxy，而不是把触觉压成单个标量。 |
| `2606.08737` | 2026-06-07 | Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation | 支持 world-action model 用预测未来触觉来约束动作。 |
| `2606.16286` | 2026-06-15 | FlowMPC: Improving Flow Matching policies with World Models | 支持 world model 用于 test-time policy improvement。 |
| `2606.17982` | 2026-06-16 | LAGO Policy: Latency-Aware Asynchronous Diffusion Policies | 支持 chunk 间平滑、低延迟和 jerk 控制是 DP 部署关键问题。 |
| `2606.06281` | 2026-06-04 | Multi-Resolution Tactile Imitation Learning for Contact-Rich Robotic Manipulation | 支持多时间尺度 tactile history/quality scoring。 |
| `2605.15705` | 2026-05-15 | Feedback World Model Enables Precise Guidance of Diffusion Policy | 支持用真实反馈修正 world model prediction，再做 guidance。 |
| `2604.23609` | 2026-04-26 | Tube Diffusion Policy: Reactive Visual-Tactile Policy Learning for Contact-rich Manipulation | 支持 contact-rich 任务需要 reactive/tube correction，而不只是 open-loop action chunk。 |

对当前项目的收敛建议：

1. 主线继续保持 `DP clean action -> Foresight predicts tactile/force consequence -> TacQualityEnergy scores quality -> trust-region gradient update`。
2. 不走 reranking；当前目标是 action denoising/clean-action guidance。
3. 擦黑板任务的 scorer 必须 contact-phase aware，只在 wiping 接触阶段强约束 force band 和 smoothness。
4. 下一代 Foresight 最值得加的是 force-aware / proxy-aware 输出，而不是先上复杂 CVAE。
5. 评分器设计应保留可解释质量分量：force band、too-low/too-high、force delta、jerk、marker magnitude、contact area、marker smoothness。
6. 最终效果必须由真实 rollout force curve 证明，离线 scorer AUC、gradient audit 和 dry-run 只能作为 readiness evidence。

当前 260617-only DP 训练状态更新：

- 最新观察到约 `epoch 615/2000`；
- 训练进程正常，GPU 正常占用；
- 当前 best 仍为 `epoch 105`, `val_loss=0.011152`；
- latest val 明显高于 best，后续测试继续使用 `dp_best.pth`。

## 2026-06-18 18:40 260617-only DP 训练监督与路线更新

### 训练状态

- 训练仍在运行，PID `1544542`。
- 输出目录：
  `/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000`
- 最近检查到第 `655/2000` epoch：
  - latest train loss：`0.003165`
  - latest val loss：`0.043750`
  - best val loss：`0.011152 @ epoch 105`
  - epoch 105 后未再刷新 best。
- 最近窗口统计：
  - 最近 10 epoch：train mean `0.00307`, val mean `0.04732`
  - 最近 25 epoch：train mean `0.00319`, val mean `0.04632`
  - 最近 100 epoch：train mean `0.00334`, val mean `0.04328`
- 当前判断：
  - 训练进程健康，GPU 利用率高，checkpoint 正常写入；
  - 后期 train loss 继续下降，但 val loss 明显高于 best，存在清楚的后期过拟合趋势；
  - 后续真机测试和展示应优先使用 `dp_best.pth`，不要默认使用 `dp_latest.pth`。

当前 checkpoint 解释：

- `dp_best.pth`：验证 loss 最优，约 `2.6G`，当前推荐部署版本；
- `dp_latest.pth`：每个 epoch 覆盖保存，含 optimizer，约 `5.1G`，用于恢复训练，不推荐直接展示；
- `dp_epoch500.pth`：周期 checkpoint；
- `dp_topk_*.pth`：train loss top-k，不代表验证泛化最好。

当前磁盘/GPU：

- `/home` 约 `45G` 可用，使用率 `98%`，仍需持续监督；
- GPU 显存约 `14.7/24.6GB`，利用率约 `90%+`。

### 为什么暂时继续训练

用户要求 2000 epoch 并希望训练充分，因此当前不提前停止。继续训练的意义是：

- 观察后期低学习率阶段是否会出现二次改善；
- 形成完整训练曲线，说明为什么最终选择 best 而不是 latest；
- watcher 已设置后期 plateau stop，到较后期若长期无 best 改善会自动停止，避免无意义占用 GPU。

### 调研后的项目主线

近两个月相关论文共同支持这个方向：

```text
image + proprio + tactile history
  -> DP proposes action chunk
  -> Foresight predicts future tactile/force consequence
  -> TacQuality energy scores contact quality
  -> contact gate decides guidance strength
  -> trust-region gradient update improves clean action chunk
```

这条路线的优势：

- 比普通 tactile concat DP 更新颖：触觉不只是输入，而是未来后果约束；
- 比 TouchGuide 式 obs-action 对齐更贴近“好触觉后果”；
- 比 reranking 更符合当前目标：真正对 action 做可微梯度更新；
- 能统一插孔和擦黑板：
  - 插孔坏后果：pre-bounce / bounce；
  - 擦黑板坏后果：too-small force、too-large force、oscillatory contact。

### 论文对应到项目的具体改进

1. ViTaL `<https://arxiv.org/abs/2606.14981>`
   - 支持 inference-time policy steering via vision and touch；
   - 对我们来说，视觉/任务进度适合高层模式，触觉/力适合局部接触 refinement。

2. QPILOTS `<https://arxiv.org/abs/2606.14801>`
   - 支持 test-time guidance 应作用在 clean/projected action 上；
   - 我们当前先在 DP clean action chunk 上做 trust-region update，是稳妥第一版。

3. TacForeSight `<https://arxiv.org/abs/2606.11184>`
   - 支持 force-guided tactile world model；
   - 下一版 Foresight 应增加 force-aware / force-proxy 输出。

4. Dream-Tac `<https://arxiv.org/abs/2606.08737>`
   - 支持 contact-gated tactile fusion；
   - 擦黑板 approach/lift 低力是正常的，只有 wiping/contact 阶段才强引导。

5. ContactWorld `<https://arxiv.org/abs/2606.13877>`
   - 支持保留空间结构和时间连续性；
   - scorer 不应只输出一个分类概率，还应保留 contact area、marker center/spread、smoothness 等 proxy。

6. Feedback World Model `<https://arxiv.org/abs/2605.15705>`
   - 支持用真实执行反馈修正 world model；
   - 后续可把真实 marker/force feedback 用来在线校正 Foresight 误差。

7. Tube Diffusion Policy `<https://arxiv.org/abs/2604.23609>`
   - 支持 contact-rich 任务需要 reactive/tube correction；
   - PTG 后续可从“每个 chunk 一次引导”升级为“执行中滚动短窗口引导”。

8. LAGO Policy `<https://arxiv.org/abs/2606.17982>`
   - 支持 chunk 间连续性和 latency-aware execution；
   - 擦黑板 scorer/guidance 应加入 action jerk / inter-chunk continuity penalty。

9. Multi-Resolution Tactile IL `<https://arxiv.org/abs/2606.06281>`
   - 支持多时间尺度 tactile 表示；
   - marker field 和 force curve 应分别建模再融合。

10. FlowMPC `<https://arxiv.org/abs/2606.16286>`
    - 支持 world model 能改善 imitation policy；
    - 可作为 reranking/MPC 对照，但主方法仍应是梯度引导。

### 当前最需要改进的地方

1. board scorer 不能只追求离线分类 AUC。
   - with-260617 ForceBand scorer 离线 AUC 很高，但 Foresight 链路上 score 动态范围很小；
   - 下一版 scorer 要优先优化 continuous guidance quality：非饱和、可微、和真实 force quality 单调相关。

2. Foresight 应加入 force-aware 目标。
   - 擦黑板质量标准本质是力大小和力变化；
   - 只预测 marker latent 可能不足以区分 too-small/too-large；
   - 建议下一版 Foresight 多头预测 future marker latent、future marker proxy、future force proxy、contact probability。

3. guidance 应只在 contact phase 强生效。
   - approach/lift 阶段低力不是坏；
   - wiping/contact 阶段才评价 force band 和 smoothness；
   - 当前 contact gate 是必要设计，应保留。

4. 评估必须从 frame/window 走向 episode-level rollout。
   - frame-level 随机划分容易泄漏；
   - 评分器要用 episode-level split；
   - 最终比较要看真实 rollout force curve：contact-phase force-in-band ratio、too-low/too-high ratio、force delta/jerk、marker smoothness、是否擦干净/是否中断。

5. DP 策略训练数据和 scorer 训练数据要分工。
   - 高质量/正样本更适合训练 DP imitation policy；
   - 负样本更适合训练 scorer/verifier；
   - 不建议把明显负样本无条件混进 DP 行为克隆，否则可能污染动作分布。

### 当前项目故事

候选名称：

```text
Tactile Consequence-Guided Diffusion Policy
```

核心贡献候选：

1. 训练短时未来触觉/力后果模型 Foresight；
2. 设计可微 TacQuality energy，把任务质量标准转成 action 可导分数；
3. 在 DP clean action chunk 上执行 contact-gated trust-region guidance；
4. 在插孔和擦黑板两个接触任务上验证：
   - 离线：未来触觉后果可区分、score 不饱和、梯度方向有效；
   - 在线：真实 force curve / marker smoothness / 任务成功率改善。

当前证据边界：

- 260617-only DP 仍在训练，尚未完成；
- `dp_best.pth` 当前可用但尚未真机评估；
- board guidance 链路具备离线 readiness，但 with-260617 scorer 的 Foresight-chain score 存在饱和问题；
- 不能声称真实擦黑板效果已提升，必须等 server-side force rollout 数据。

## 2026-06-18 19:15 Board scorer 连续引导信号复查

### BoardProxyEnergy 实验

目的：测试一个非学习的、手工校准的连续 proxy energy 是否可以解决 ForceBand scorer 在 Foresight 链路中饱和的问题。

代码：

- `TFAC_V5/tac_quality_energy/board_proxy_energy.py`
- runtime: `BoardProxyEnergyRuntime`

基础梯度检查：

- marker grad finite: `true`
- action grad finite: `true`
- marker grad norm: `0.004918`
- action grad norm: `0.100802`

Foresight 链路评估：

```bash
conda run --no-capture-output -n TactileACT python \
  TFAC_V5/tac_quality_energy/eval_foresight_score_alignment.py \
  --output_dir /home/chenshuai/Project/output/board_proxy_energy_foresight_alignment_20260618/quality \
  --scorer_runtime BoardProxyEnergyRuntime \
  --score_mode quality \
  --include_260617_positive \
  --max_episodes_per_class 8 \
  --samples_per_episode 2 \
  --max_samples 80 \
  --gpu -1
```

输出：

- `/home/chenshuai/Project/output/board_proxy_energy_foresight_alignment_20260618/quality/foresight_score_alignment.json`
- `/home/chenshuai/Project/output/board_proxy_energy_foresight_alignment_20260618/quality/foresight_score_alignment.md`

结果：

| metric | value |
|---|---:|
| samples | 78 |
| pred AUC(good) | 0.5042 |
| pred/GT Spearman | 0.3036 |
| pred score vs force-band quality Spearman | 0.5104 |
| GT score vs force-band quality Spearman | 0.5702 |
| marker MAE mean | 0.5116 |

解释：

- BoardProxyEnergy 不饱和、可微，并且和 force-band physical proxy 有中等相关；
- 但它几乎不能区分当前采集标签的正负样本，AUC 只有约 `0.50`；
- 因此它只能作为物理连续项/审计 control，不能单独作为最终 DP guidance scorer。

### ForceBand scorer action-window 修复

发现的问题：

- `ForceBandTacQualityEnergyRuntime.proxy_features()` 原来把 action 序列强制截成 marker window 的长度；
- marker window 是 `8`，但训练时 `marker_action` feature 使用的是 `16` 步 action chunk；
- `eval_foresight_score_alignment.py` 里也只传了 `action[:, :window]`，导致 alignment audit 的 action feature 与训练/serving 分布不一致。

修复：

- `TFAC_V5/tac_quality_energy/force_band_runtime.py`
  - action proxy 现在按传入 action 的真实长度计算；
  - 若 serving 传入 16 步 action chunk，就使用 16 步 action proxy。
- `TFAC_V5/tac_quality_energy/eval_foresight_score_alignment.py`
  - `action_score` 从 `action[:, :args.window]` 改成 `action[:, :args.action_chunk]`。

修复后 smoke：

- feature shape: `[3, 74]`
- marker grad finite: `true`, norm `0.000401`
- action grad finite: `true`, norm `0.000344`

### 修复后 ForceBand Foresight 链路结果

输出：

- `/home/chenshuai/Project/output/tac_quality_force_band_with260617_score_mode_sweep_20260618_action16fix/energy_clipped/foresight_score_alignment.json`
- `/home/chenshuai/Project/output/tac_quality_force_band_with260617_score_mode_sweep_20260618_action16fix/profile/foresight_score_alignment.json`

`energy_clipped`:

| metric | before | after action16 fix |
|---|---:|---:|
| pred AUC(good) | 0.6431 | 0.6382 |
| pred/GT Spearman | 0.5331 | 0.5419 |
| pred score vs -marker MAE Spearman | NA | -0.4981 |
| pred score vs force-band quality Spearman | -0.3978 | -0.2484 |

`profile`:

| metric | value |
|---|---:|
| pred AUC(good) | 0.6417 |
| pred/GT Spearman | 0.4241 |
| pred score vs -marker MAE Spearman | -0.5160 |
| pred score vs force-band quality Spearman | -0.2262 |

解释：

- action-window 修复是必要的，因为它消除了一个训练/部署 action feature 不一致；
- 修复后 predicted score 与 GT score 的相关性略改善，score 与 Foresight marker error 的关系也更合理；
- 但核心问题还没有解决：score 和真实 force-band quality 仍是负相关；
- 因此当前 with-260617 ForceBand scorer 仍不能作为最终 board guidance score 过度声称。

### 当前最合理下一步

1. 训练 predicted-domain scorer：
   - 用当前真实 Foresight 把 `action -> predicted future marker` 先跑出来；
   - 用 predicted marker/action feature 训练 scorer，而不是只用 GT future marker；
   - 这样 scorer 的训练输入分布与实际 DP guidance 时一致。
2. 下一代 Foresight 增加 force-aware 输出：
   - 擦黑板的好坏本质依赖力大小和力变化；
   - marker-only Foresight 很难可靠区分 `too_small` 和 `too_large`；
   - 需要预测 force proxy / contact force band / force smoothness。
3. 当前部署建议：
   - 可以继续用 with-260617 ForceBand scorer 做安全的小步 dry-run；
   - 不能把它作为最终论文级 board scorer；
   - 真机测试必须记录 force curve，并用 contact-phase force metrics 做最终判断。

## 2026-06-18 19:26 训练监督与论文来源复核

### 260617-only DP 当前状态

- 训练进程仍在运行：PID `1544542`。
- 当前日志最新完整到第 `729/2000` epoch，训练正在第 730 epoch。
- 当前 best 仍是第 `105` epoch：`val=0.011152`。
- 第 724-729 epoch：
  - epoch 724: `train=0.002792`, `val=0.043198`
  - epoch 725: `train=0.003090`, `val=0.049560`
  - epoch 726: `train=0.002638`, `val=0.045018`
  - epoch 727: `train=0.002714`, `val=0.039339`
  - epoch 728: `train=0.002701`, `val=0.038294`
  - epoch 729: `train=0.002665`, `val=0.058419`
- GPU：约 `14732/24564 MiB`，利用率正常，温度约 `53-65C`。
- 磁盘：`/home` 约 `45G` 可用，`/media/chenshuai/EXTERNAL_USB` 约 `2.3T` 可用。

当前判断：

- 训练进程健康，没有 OOM、NaN、Traceback。
- 训练 loss 继续下降，但验证 loss 长期显著高于 epoch 105 的 best，后期 overfit 趋势已经很明显。
- 由于 `dp_best.pth` 按验证 loss 自动保存，继续跑 2000 epoch 不会覆盖最优部署 checkpoint。
- 后续真机/离线测试默认应使用：
  `/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000/dp_best.pth`
- `dp_latest.pth` 主要用于恢复训练，不建议作为展示/真机默认版本。

### 近两个月论文来源复核

以下条目已通过 arXiv API 或网页检索确认存在，后续文档/论文中可以作为参考方向，但具体实验指标仍应以原文为准。

- `ViTaL: Inference-time Policy Steering via Vision and Touch`, arXiv `2606.14981`, 2026-06-12。该工作把视觉长程选择和触觉短程 diffusion editing 分开，和本项目的 `Foresight + TacQualityEnergy + bounded gradient guidance` 高度相关。
- `Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation`, arXiv `2606.08737`, 2026-06-07。该工作联合建模动作、未来视觉和未来触觉，并提出 contact-gated visuotactile fusion。
- `TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation`, arXiv `2606.11184`, 2026-06-09。该工作强调 force-conditioned tactile latent dynamics，支持本项目后续把 board Foresight 升级成 force-aware / force-proxy-aware。
- `ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation`, arXiv `2606.13877`, 2026-06-11。该工作强调 spatially structured、temporally continuous 表示对接触规划重要，支持保留 marker field/proxy 而不是只看单点力。
- `Ambient Diffusion Policy: Imitation Learning from Suboptimal Data in Robotics`, arXiv `2606.12365`, 2026-06-10。该工作说明低质量/异质数据不应简单混入策略训练，支持本项目将负样本主要用于 scorer/verifier，而不是无条件混入 DP policy。
- `Latent Diffusion Policy: Shaping Latent Spaces for Diffusion-Based Robotic Manipulation`, arXiv `2606.08657`, 2026-06-07。该工作支持后续把 raw joint action diffusion 升级为 latent action diffusion，以降低精细轨迹生成难度。
- `FTP-1: A Generalist Foundation Tactile Policy Across Tactile Sensors for Contact-Rich Manipulation`, arXiv `2606.13102`, 2026-06-11。该工作支持更通用的 tactile token/latent 表示方向。
- `DPTG: Diffusion Policy with Tactile Feasibility Guidance`, Frontiers in Robotics and AI, 2026-06。该工作不是 arXiv，但与 tactile feasibility classifier guidance 很接近，可作为相关工作，不应混写成 arXiv。

API 复核记录：

```text
2606.14981 FOUND Inference-time Policy Steering via Vision and Touch
2606.08737 FOUND Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation
2606.11184 FOUND TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation
2606.13877 FOUND ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation
2606.12365 FOUND Ambient Diffusion Policy: Imitation Learning from Suboptimal Data in Robotics
2606.08657 FOUND Latent Diffusion Policy: Shaping Latent Spaces for Diffusion-Based Robotic Manipulation
2606.13102 FOUND FTP-1: A Generalist Foundation Tactile Policy Across Tactile Sensors for Contact-Rich Manipulation
2604.23609 FOUND Tube Diffusion Policy: Reactive Visual-Tactile Policy Learning for Contact-rich Manipulation
2603.10980 FOUND PPGuide: Steering Diffusion Policies with Performance Predictive Guidance
2604.01414 FOUND Learning When to See and When to Feel: Adaptive Vision-Torque Fusion for Contact-Aware Manipulation
```

对本项目的更新判断：

- 当前最有辨识度的故事仍应是：`DP nominal action -> Foresight predicts future tactile/force consequence -> TacQualityEnergy evaluates contact quality -> contact gate decides when guidance is active -> trust-region gradient modifies action`。
- 这条路线和最新工作一致，但我们需要强调差异：我们不是 reranking，也不是只做触觉 concat；核心是用“预测的未来触觉后果”构造可微质量能量，并在去噪/动作 refinement 中做梯度引导。
- 擦黑板任务下一步最值得改的是 board scorer / Foresight 的 force-aware 表达。当前 marker-only Foresight + ForceBand scorer 在离线分类上强，但在 Foresight 链路上仍有分数饱和和真实 force-quality 对齐不足的问题。
