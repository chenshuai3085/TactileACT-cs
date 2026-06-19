# 2026-06-19 260617-only DP Stable Training and Recent ArXiv Review

## 1. 当前任务

用户要求：

- 先只使用数据 `/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban` 训练擦黑板/接触任务 DP；
- 训练 2000 epoch；
- 训练过程中持续监督，有异常直接处理；
- 监督过程中调研最近两个月内与本项目相关的 arXiv 工作，并思考当前架构和科研故事还能怎么改进。

本记录只对应当前正在运行的 stable 训练，不等同于之前已经早停的 `full_noearly_tmux` run。

## 2. 数据核对

实际 episode 目录：

```text
/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617
```

根目录下只有这一个 episode 子目录。

数据核对结果：

| 项目 | 数值 |
|---|---:|
| HDF5 文件数 | 80 |
| 可读 episode | 79 |
| 不完整 episode | 1 |
| 不完整文件 | `episode_1.hdf5` |
| 缺失键 | `observations/proprio_joint`, `actions/joint_abs`, `observations/tac/left/marker_offset` |
| episode 长度 min / mean / max | 640 / 816.76 / 957 |
| 图像键 | `observations/images/global`, `observations/images/wrist` |
| 原始图像尺寸 | `(T, 200, 266, 3)` uint8 |
| proprio | `observations/proprio_joint`, `(T, 7)` |
| action | `actions/joint_abs`, `(T, 7)` |
| tactile | `observations/tac/left/marker_offset`, `(T, 9, 9, 2)` |

训练脚本会跳过不完整 episode，所以当前训练实际使用 79 个可读 episode。

## 3. 当前训练配置

训练脚本：

```text
scripts/train/train_dp_tac_concat_board_260617_only_stable_e2000.sh
```

底层训练入口：

```text
diffusion/train_dp_tac_concat.py
```

tmux session：

```text
dp260617stable_122234
```

监督 session：

```text
watch260617stable_conservative
```

输出目录：

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr
```

核心参数：

| 参数 | 值 |
|---|---|
| dataset | `peg_in_hole_0617` only |
| camera | `global,wrist` |
| image resize/crop | `200,266` / `200,266` |
| tactile side | left |
| tactile history | 8 |
| TactileVAE | `/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt` |
| TactileVAE norm stats | mean `[-0.3399, -2.9208]`, std `[1.9805, 2.7671]` |
| tactile latent | 144 dim |
| obs horizon | 2 |
| pred horizon | 16 |
| action horizon | 8 |
| action dim | 7 |
| global condition dim | 2350 |
| DP U-Net params | about 315M |
| epochs | 2000 |
| batch size | 64 |
| learning rate | `5e-5` |
| weight decay | `1e-5` |
| warmup steps | 1000 |
| diffusion train/inference steps | 100 / 100 |
| U-Net down dims | `512,1024,2048` |
| EMA | enabled |
| val split | episode-level 0.1 |
| train episodes/windows | 71 / 56989 |
| val episodes/windows | 8 / 2048 |
| max train batches per epoch | 128 |
| val interval | 5 epochs |
| save freq | 50 epochs |
| latest freq | 10 epochs |
| top-k train ckpt | 3 |
| image cache | `/home/chenshuai/Project/output/cache/dp_board_rawimg200x266_fp16` |

## 4. 当前监督状态

截至 2026-06-19 15:00 CST：

| 项目 | 状态 |
|---|---|
| 训练进程 | 正常运行 |
| GPU | RTX 4090 |
| GPU 利用率 | 约 67% 到 95% 波动 |
| 显存 | 约 14.7GB / 24.6GB |
| 外接盘剩余 | 约 2.1TB |
| home 根分区剩余 | 约 42GB |
| 最新监督点 | epoch 235 |
| epoch 235 train | 0.006905 |
| epoch 235 val | 0.017263 |
| 当前 best | val 0.011659 @ epoch 155 |
| 最新整点 ckpt | `dp_epoch200.pth` |
| 当前部署候选 | `dp_best.pth` |

重要 checkpoint：

```text
dp_best.pth      # 当前 best val，epoch 155
dp_epoch50.pth
dp_epoch100.pth
dp_epoch150.pth
dp_epoch200.pth
dp_latest.pth   # 继续训练中的最新状态，不建议作为当前部署结论
```

当前判断：

1. 训练本身健康，没有 NaN、OOM、checkpoint 写入失败或 GPU 异常。
2. 从 epoch 155 之后，train loss 继续下降，但 validation loss 没有刷新 best。
3. 这说明模型在训练 windows 上继续拟合，但泛化暂时没有同步提升。
4. 由于用户要求充分训练 2000 epoch，当前不在早期 epoch 直接停止。
5. 后续部署或 offline 对比应优先使用 `dp_best.pth`，不要把 `dp_latest.pth` 当作当前最优模型。

保守 watcher 的停止条件：

- 至少训练到 epoch 1500；
- 之后 350 epoch 没有刷新 best；
- 最近验证窗口整体差于 best 5% 以上；
- 才允许自动停止。

## 5. 最近两个月相关 arXiv 调研

调研窗口以 2026-04 到 2026-06 为主，重点关注 tactile / force / contact-rich manipulation / diffusion policy / inference-time guidance。

### 5.1 ViTaL: Inference-time Policy Steering via Vision and Touch

- arXiv: https://arxiv.org/abs/2606.14981
- 日期：2026-06-12

核心点：

- 在部署时对预训练 generative robot policy 做 inference-time steering；
- 高层用视觉做长时程行为选择；
- 低层用 tactile-guided diffusion editing 优化短时程接触行为；
- 学习视觉和触觉 verifier，直接对预测的 tactile future latent 打分。

对本项目的启发：

- 这篇最贴近当前故事：不是简单拼接 tactile 到 DP，而是用“预测触觉后果”在推理时修正 action。
- 我们当前 `DP action prior -> Foresight -> TacQualityEnergy -> bounded gradient guidance` 可以定位为 tactile consequence-guided diffusion/action refinement。
- 后续论文表达中，应强调 tactile future verifier/energy，而不是只说 tactile concat baseline。

### 5.2 TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation

- arXiv: https://arxiv.org/abs/2606.11184
- 日期：2026-06-09

核心点：

- 使用高频 wrist force/torque 条件化 tactile world model；
- 预测短时程 tactile latent dynamics；
- 把预测 latent 作为 anticipatory contact prior 给 policy 使用。

对本项目的启发：

- 擦黑板任务里，force 不能只做事后评价；同步力曲线应该逐步进入 Foresight 或 scorer。
- 当前 marker-only Foresight 是第一阶段，下一阶段更合理的是 force-conditioned tactile foresight：

```text
history image/proprio/tactile/action/force -> future tactile marker or latent
```

- 真机测试时 server 侧记录每条轨迹力曲线是必要的，因为它既是评价信号，也可能成为下一版模型输入。

### 5.3 Dream-Tac: A Unified Tactile World Action Model

- arXiv: https://arxiv.org/abs/2606.08737
- 日期：2026-06-07

核心点：

- joint model 同时建模 action、future vision、future tactile dynamics；
- 使用 contact-gated visuotactile fusion 和 contact-aware attention；
- 强调实时部署下的 diffusion 加速。

对本项目的启发：

- 我们现在是模块化版本：

```text
DP proposes action
Foresight predicts tactile consequence
TacQuality scores consequence
gradient guidance edits action
```

- 模块化优点是容易诊断和替换，适合当前阶段；
- 后续如果要增强创新性，可以向 joint world-action model 靠近，但不应过早合并所有模块，否则调试难度会明显增加。

### 5.4 ContactWorld: What Matters in Vision-Tactile World Models

- arXiv: https://arxiv.org/abs/2606.13877
- 日期：2026-06-11

核心点：

- contact-rich world model 需要 spatially structured 和 temporally continuous 的表示；
- tactile 是否有效取决于跨模态表示兼容性，而不是简单增加一个模态；
- 长时程 planning 下 tactile 的作用更明显。

对本项目的启发：

- TactileVAE latent 不能只看一帧重建 MSE，应看接触阶段、力带、平滑度、趋势和 episode-level 后果。
- TacQuality scorer 的输入应该保留 marker field 的空间结构或显式 proxy，而不是只做粗糙全局均值。
- Foresight 的评价也应该从单点 t+16 扩展到完整未来过程 t+1 到 t+16 的趋势评价。

### 5.5 SI-Diff: Force-Domain Diffusion Policy for Search and High-Precision Insertion

- arXiv: https://arxiv.org/abs/2605.12247
- 日期：2026-05-12

核心点：

- 用 force-domain diffusion policy 同时处理 search 和 insertion；
- 通过 mode-conditioning 让同一个模型捕获不同插入阶段的 action pattern；
- 通过 search teacher 生成多样成功轨迹。

对本项目的启发：

- 插座任务不应该只有 good/bad 二分类；更合理是阶段/模式条件：

```text
approach/search/normal insertion/pre-bounce/bounce/recovery
```

- 当前插座的 good-margin scorer 是合理第一版，但后续可把 phase/mode 作为条件输入，避免不同阶段的 score 标准混在一起。

### 5.6 Tube Diffusion Policy

- arXiv: https://arxiv.org/abs/2604.23609
- 日期：2026-04-26

核心点：

- 普通 action chunking 限制了接触任务的反应能力；
- 学习 nominal action chunk 周围的 feedback flow/action tube；
- 用视觉和触觉反馈做快速 reactive correction。

对本项目的启发：

- DP chunk 生成后只执行固定 action 可能不足以处理擦黑板接触扰动。
- 我们的 gradient guidance 可以被表述为一种 bounded action-tube refinement：

```text
base DP chunk + local tactile-quality gradient correction
```

- 这仍然不是 reranking，而是围绕 DP clean action 的小范围可微修正。

### 5.7 ForceFlow

- arXiv: https://arxiv.org/abs/2605.11048
- 日期：2026-05-11

核心点：

- force-aware reactive framework；
- 把任务分为 vision-dominant approach 和 touch/force-dominant contact execution；
- force 作为全局调节信号。

对本项目的启发：

- 擦黑板 guidance 必须 contact-gated：

```text
approach: guidance 弱或关闭
wiping contact: force/tactile quality guidance 强
exit/reset: guidance 弱或关闭
```

- 这能避免在未接触阶段错误优化“压力大小/平滑度”。

### 5.8 AT-VLA 和 Multi-Resolution Tactile IL

- AT-VLA: https://arxiv.org/abs/2605.07308
- Multi-Resolution Tactile IL: https://arxiv.org/abs/2606.06281

核心点：

- tactile 应该在需要接触反馈时自适应注入；
- 多时间尺度 tactile 特征有助于处理快速接触变化。

对本项目的启发：

- TacQualityEnergy 未来应增加多时间尺度统计：

```text
marker magnitude mean/std
marker velocity
marker acceleration
contact dropout ratio
short-window smoothness
long-window force-band occupancy
```

- 这对区分擦黑板的正样本、力过小、力过大、忽大忽小很关键。

### 5.9 Latent Diffusion Policy

- arXiv: https://arxiv.org/abs/2606.08657
- 日期：2026-06-07

核心点：

- 在 shaped latent action space 中做 diffusion/flow，而不是直接在 raw action space 中去噪；
- 用 observation-conditioned CVAE 把动作分布先集中到更平滑的潜空间。

对本项目的启发：

- 当前 DP 在 raw joint action 上 diffusion，模型大、训练慢、可能更容易过拟合。
- 未来可以考虑：

```text
action chunk CVAE latent z_a
DP/flow 在 z_a 上生成
TacQuality guidance 对 z_a 或 decoded action 反传
```

- 但当前阶段不建议马上替换主线，先把 raw-action DP + Foresight + Energy 的链路跑通。

## 6. 对当前项目架构和故事的改进建议

### 6.1 当前最稳故事

当前项目最合理的主线应表述为：

```text
Contact-rich manipulation needs actions with good tactile consequences.
Base DP learns an action prior from demonstrations.
Foresight predicts future tactile/contact consequence for candidate actions.
TacQualityEnergy scores whether the predicted consequence is safe, effective, and smooth.
At inference time, bounded gradient guidance edits the DP action toward better predicted tactile quality.
```

这条线符合用户目标：不是 reranking，而是用评分器/分类器作为可微能量，对 action 产生梯度引导。

### 6.2 当前需要坚持的设计

1. 评分器不能只做 binary good/bad。
   - 插座至少需要 good/bounce risk/reason；
   - 擦黑板至少需要 good/too-light/too-heavy/rough-oscillatory；
   - 连续 quality score 才适合提供平滑梯度。

2. 评估必须用 episode-level split。
   - frame-level random split 会泄漏相邻帧和同一 episode 的统计特征；
   - 现在 scorer 用 GroupKFold 是合理的。

3. guidance 要 bounded。
   - 当前最稳的是 trust-region refinement；
   - 每步 DDPM guidance 可以研究，但必须做 scheduler-aware、step-aware、contact-gated sweep 后才能作为主线。

4. 真实测试必须记录力曲线。
   - board baseline/guided 应分别保存 server 侧 force traces；
   - 真实结论必须看 force-band occupancy、smoothness、过轻/过重比例、任务完成情况。

### 6.3 短期优先改进

1. 当前 260617-only DP 继续训练，但 deploy/对比先用 `dp_best.pth`。
2. 对当前 best/epoch200/后续 epoch250 做 offline action distribution 对比，检查是否出现过拟合式 action 抖动。
3. 在 real rollout server 中确保 baseline 和 guided 的力曲线按类别分别保存。
4. 对 board scorer 加 contact-gating 诊断：
   - 未接触阶段 score 权重应弱；
   - 擦拭接触阶段 score 权重应强。
5. 后续收集更多负样本后，重新训练 board TacQualityEnergy，并固定标签：
   - good；
   - too_light；
   - too_heavy；
   - oscillatory/rough；
   - optional unsafe/spike。

### 6.4 中期创新方向

1. Force-conditioned tactile foresight：

```text
Foresight(obs, action, force history) -> future marker/latent/force proxy
```

2. Contact-gated TacQuality guidance：

```text
score = gate_contact * quality_contact + gate_safety * safety_margin
```

3. Action-tube gradient refinement：

```text
a_guided = a_dp + bounded_delta
delta = argmax quality(Foresight(obs, a_dp + delta))
```

4. Latent action guidance：

```text
z_action -> decoded action -> Foresight -> TacQualityEnergy
```

这比纯 raw-action guidance 更可能稳定，但需要额外训练 action autoencoder/CVAE。

## 7. 当前结论

1. 当前 260617-only DP stable run 已经正确启动，并且只使用用户指定数据目录。
2. 数据存在 1 个不完整 episode，训练已自动跳过，不影响整体训练。
3. 训练目前健康，但 validation best 暂停在 epoch 155；后续 checkpoint 需要和 best 对比，不应默认使用 latest。
4. 最近 arXiv 明确支持本项目的关键方向：预测触觉后果 + tactile/force verifier/energy + 推理时 guidance。
5. 当前最适合继续推进的科研故事是：

```text
Tactile Consequence-Guided Diffusion Policy
with action-conditioned tactile foresight and contact-gated quality energy.
```

6. 真实有效性最终必须由 paired real rollout force traces 验证，不能只凭离线 loss 或 scorer 分数下结论。

## 7.1 2026-06-19 16:45 训练监督更新

截至 2026-06-19 16:45 CST，当前 260617-only stable run 仍在正常训练。

| 项目 | 状态 |
|---|---|
| 最新日志位置 | epoch 407/2000 附近 |
| epoch 400 train / val | 0.005905 / 0.018671 |
| epoch 405 train / val | 0.005506 / 0.021723 |
| 当前 best | val 0.011659 @ epoch 155 |
| 最新整点 ckpt | `dp_epoch400.pth` |
| 当前部署候选 | `dp_best.pth` |
| GPU | RTX 4090，约 14.7GB/24.6GB 显存 |
| 外接盘剩余 | 约 2.1TB |
| 根分区剩余 | 约 42-43GB |

已确认文件：

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_epoch400.pth
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_best.pth
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_latest.pth
```

loss 曲线已重新生成：

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/loss_curve.png
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/loss_curve.csv
```

判断：

1. 训练进程、GPU、磁盘和 checkpoint 写入都正常。
2. train loss 继续下降，但 epoch 400 和 epoch 405 validation 都没有接近 epoch 155 best。
3. 当前不是训练崩溃，而是继续拟合训练 windows、验证泛化没有同步改善。
4. 后续真实测试或 offline 对比优先使用 `dp_best.pth`，除非后续 checkpoint 刷新 best。
5. 下一重点检查 `dp_epoch450.pth`。

## 7.2 2026-06-19 17:12 训练监督更新

截至 2026-06-19 17:12 CST，epoch 450 已完成并保存。

| 项目 | 状态 |
|---|---|
| 最新日志位置 | epoch 452/2000 附近 |
| epoch 450 train / val | 0.005327 / 0.022039 |
| 当前 best | val 0.011659 @ epoch 155 |
| 最新整点 ckpt | `dp_epoch450.pth` |
| 当前部署候选 | `dp_best.pth` |
| GPU | RTX 4090，约 14.7GB/24.6GB 显存 |
| 输出目录大小 | 约 38G |
| 外接盘剩余 | 约 2.1T |
| 根分区剩余 | 约 43G |

近几个验证点：

| epoch | val |
|---:|---:|
| 435 | 0.018519 |
| 440 | 0.020185 |
| 445 | 0.018342 |
| 450 | 0.022039 |

已确认文件：

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_epoch450.pth
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_best.pth
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_latest.pth
```

loss 曲线已重新生成到 epoch 451：

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/loss_curve.png
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/loss_curve.csv
```

判断：

1. 训练进程、GPU、磁盘、watcher 和 checkpoint 写入都正常。
2. epoch 450 validation 比 epoch 155 best 明显更差。
3. `dp_epoch450.pth` 可作为历史 checkpoint 保存，但不是当前部署候选。
4. 当前部署或 offline 对比仍应使用 `dp_best.pth`。
5. 下一重点检查 `dp_epoch500.pth`。

## 7.3 2026-06-19 17:43 训练监督更新

截至 2026-06-19 17:43 CST，epoch 500 已完成并保存。

| 项目 | 状态 |
|---|---|
| 最新日志位置 | epoch 502/2000 附近 |
| epoch 500 train / val | 0.005367 / 0.025795 |
| 当前 best | val 0.011659 @ epoch 155 |
| 最新整点 ckpt | `dp_epoch500.pth` |
| 当前部署候选 | `dp_best.pth` |
| GPU | RTX 4090，约 14.7GB/24.6GB 显存 |
| 输出目录大小 | 约 41G |
| 外接盘剩余 | 约 2.1T |
| 根分区剩余 | 约 43G |

近几个验证点：

| epoch | val |
|---:|---:|
| 480 | 0.019030 |
| 485 | 0.022826 |
| 490 | 0.021393 |
| 495 | 0.025051 |
| 500 | 0.025795 |

已确认文件：

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_epoch500.pth
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_best.pth
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_latest.pth
```

loss 曲线已重新生成到 epoch 502：

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/loss_curve.png
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/loss_curve.csv
```

判断：

1. 训练进程、GPU、磁盘、watcher 和 checkpoint 写入都正常。
2. epoch 500 validation 明显差于 epoch 155 best，且最近 tail validation 均值继续升高。
3. `dp_epoch500.pth` 可作为历史 checkpoint 保存，但不是当前部署候选。
4. 当前部署或 offline 对比仍应使用 `dp_best.pth`。
5. 后续继续训练主要是按用户要求保留充分训练曲线，并观察是否有极晚期回落；不能默认 `dp_latest.pth` 更好。
6. 下一重点检查 `dp_epoch550.pth`。

## 7.4 2026-06-19 18:15 训练监督更新

截至 2026-06-19 18:15 CST，epoch 550 已完成并保存。

| 项目 | 状态 |
|---|---|
| 最新日志位置 | epoch 557/2000 附近 |
| epoch 550 train / val | 0.004410 / 0.019420 |
| 当前 best | val 0.011659 @ epoch 155 |
| 最新整点 ckpt | `dp_epoch550.pth` |
| 当前部署候选 | `dp_best.pth` |
| GPU | RTX 4090，约 14.7GB/24.6GB 显存 |
| 输出目录大小 | 约 43G |
| 外接盘剩余 | 约 2.1T |
| 根分区剩余 | 约 43G |

近几个验证点：

| epoch | val |
|---:|---:|
| 535 | 0.020365 |
| 540 | 0.022955 |
| 545 | 0.018925 |
| 550 | 0.019420 |
| 555 | 0.021355 |

已确认文件：

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_epoch550.pth
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_best.pth
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_latest.pth
```

loss 曲线已重新生成到 epoch 556：

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/loss_curve.png
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/loss_curve.csv
```

判断：

1. 训练进程、GPU、磁盘、watcher 和 checkpoint 写入都正常。
2. epoch 550 validation 仍明显差于 epoch 155 best。
3. `dp_epoch550.pth` 可作为历史 checkpoint 保存，但不是当前部署候选。
4. 当前部署或 offline 对比仍应使用 `dp_best.pth`。
5. 后续继续训练主要是按用户要求保留充分训练曲线，并观察是否有极晚期回落。
6. 下一重点检查 `dp_epoch600.pth`。

## 7.5 2026-06-19 18:44 训练监督更新

截至 2026-06-19 18:44 CST，epoch 600 已完成并保存。

| 项目 | 状态 |
|---|---|
| 最新日志位置 | epoch 606/2000 附近 |
| epoch 600 train / val | 0.004343 / 0.024435 |
| 当前 best | val 0.011659 @ epoch 155 |
| 最新整点 ckpt | `dp_epoch600.pth` |
| 当前部署候选 | `dp_best.pth` |
| GPU | RTX 4090，约 14.7GB/24.6GB 显存 |
| 输出目录大小 | 约 46G |
| 外接盘剩余 | 约 2.1T |
| 根分区剩余 | 约 43G |

近几个验证点：

| epoch | val |
|---:|---:|
| 560 | 0.023542 |
| 565 | 0.021458 |
| 600 | 0.024435 |
| 605 | 0.022597 |

watcher 状态：

| 项目 | 数值 |
|---|---:|
| latest | 600/2000 |
| tail_val_min | 0.018925 |
| tail_val_mean | 0.022392 |

已确认文件：

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_epoch600.pth
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_best.pth
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_latest.pth
```

loss 曲线已重新生成到 epoch 606：

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/loss_curve.png
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/loss_curve.csv
```

判断：

1. 训练进程、GPU、磁盘、watcher 和 checkpoint 写入都正常。
2. epoch 600 validation 明显差于 epoch 155 best，tail validation 均值继续高于 0.02。
3. `dp_epoch600.pth` 可作为历史 checkpoint 保存，但不是当前部署候选。
4. 当前部署或 offline 对比仍应使用 `dp_best.pth`。
5. 如果后续继续训练，应主要视为保留充分训练曲线；泛化改善需要额外数据划分、正则化或早停策略，而不是使用 latest。
6. 下一重点检查 `dp_epoch650.pth`。

## 8. 2026-06-19 15:30 监督更新

当前训练仍在正常运行：

| 项目 | 状态 |
|---|---|
| 最新日志 epoch | 283/2000 附近 |
| 最新完整验证点 | epoch 275 |
| epoch 275 train / val | 0.007284 / 0.014892 |
| 当前 best | val 0.011659 @ epoch 155 |
| GPU | RTX 4090，约 14.7GB/24.6GB 显存，利用率正常波动 |
| 外接盘剩余 | 约 2.1TB |
| 根分区剩余 | 约 43GB |
| 最新整点 ckpt | `dp_epoch250.pth` |
| 下一个重点检查 | `dp_epoch300.pth` 和 epoch 300 validation |

判断：

1. 训练进程、GPU、checkpoint 写入、磁盘空间都正常。
2. 从 epoch 155 之后，训练 loss 继续下降，但 validation loss 未刷新 best。
3. 这不是训练崩溃，而是当前 run 已经出现训练集拟合继续增强、验证泛化暂未同步提升的趋势。
4. 部署或 offline 对比优先使用 `dp_best.pth`，不要使用 `dp_latest.pth` 作为“当前最好”。
5. 按用户要求继续充分训练到 2000 epoch；保守 watcher 仅在 epoch >=1500 且长期没有泛化改进时才允许自动停止。

## 9. 2026-06-19 最新 arXiv spot-check 补充

使用 arXiv API 按 `cs.RO`、`tactile`、`diffusion policy`、`contact/force manipulation`、`guidance diffusion robot` 等关键词检查 2026-06 中旬最新条目。和当前项目最相关的新增条目如下。

### 9.1 Inference-time Policy Steering via Vision and Touch

- arXiv: https://arxiv.org/abs/2606.14981
- 提交日期：2026-06-12
- 相关性：最高。

该工作明确把 inference-time steering、vision/touch verifier、candidate action consequence verification 放在一起。它对当前项目的启发是：我们的主线不应只讲 “DP 拼接触觉”，而应讲成：

```text
DP action prior
-> action-conditioned tactile foresight
-> tactile quality / risk verifier
-> bounded gradient guidance on action
```

这和用户强调的“不是 reranking，而是梯度引导”一致。

### 9.2 ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation

- arXiv: https://arxiv.org/abs/2606.13877
- 提交日期：2026-06-11
- 相关性：高。

对当前项目最重要的提醒是：触觉 world model 不能只看单帧 latent MSE。擦黑板/插孔这种 contact-rich 任务应该评估完整未来过程：

```text
t+1 ... t+16 marker/latent trajectory
contact force band occupancy
too-light / too-heavy / roughness
temporal smoothness
task phase consistency
```

所以后续 Foresight 和 TacQualityEnergy 的可视化与评估应继续保留多步未来过程，而不只展示 t+16。

### 9.3 QPILOTS: Efficient Test-Time Q-Steering for Flow Policies

- arXiv: https://arxiv.org/abs/2606.14801
- 提交日期：2026-06-11
- 相关性：中高。

这类 test-time Q/critic steering 支持我们的技术路线：用可微评价函数在推理时修改生成动作。但它也提醒一点：直接把梯度穿过多步 denoising 可能不稳定。因此当前更稳的实现仍是：

```text
DP 先生成 clean action chunk
在 clean action 附近做 bounded trust-region refinement
accept-only improved action
```

DDPM step 内 guidance 可以作为后续论文增强点，但必须先做 step-aware sweep，不能直接替换主线。

### 9.4 Frequency-Aware Flow Matching for Continuous and Consistent Robotic Action Generation

- arXiv: https://arxiv.org/abs/2606.20135
- 提交日期：2026-06-18
- 相关性：中。

它强调连续、一致的动作生成，和我们擦黑板任务的动作平滑需求一致。对当前项目的启发是：TacQuality 不应该只评估“力是否合适”，还应显式评估动作和触觉后果的平滑度：

```text
action jerk
marker velocity
marker acceleration
force derivative
contact dropout
```

这可以支撑“触觉后果更好”的定义，而不只是二分类。

### 9.5 Ambient Diffusion Policy: Imitation Learning from Suboptimal Data in Robotics

- arXiv: https://arxiv.org/abs/2606.12365
- 提交日期：2026-06-10
- 相关性：中。

这和当前数据结构相关：擦黑板有正样本，也有 too-light / too-heavy / oscillatory 负样本。后续可以把 DP 训练和 TacQuality guidance 分开讲：

1. DP 从全部数据学习可达动作分布；
2. TacQualityEnergy 在推理时把动作推向正样本力带和稳定接触区域；
3. 负样本不一定全部丢弃，而是用于学习“哪些触觉后果不该被引导到”。

## 10. 当前科研故事修正建议

基于今天训练和最新论文，当前最合适的故事不是“训练一个触觉 DP”，而是：

```text
Tactile Consequence-Guided Diffusion Policy
```

核心贡献可以拆成三层：

1. **Action-conditioned tactile foresight**  
   给定当前视觉/本体/触觉历史和候选 action，预测未来多步触觉后果。

2. **Contact-gated tactile quality energy**  
   对预测未来触觉过程评分：插孔看 bounce risk/good margin，擦黑板看 force band、too-light、too-heavy、roughness/smoothness。

3. **Bounded gradient guidance for DP**  
   在推理阶段对 DP 生成的 action chunk 做小范围可微修正，目标是提高预测触觉质量，同时用 trust-region 防止动作偏离示教分布。

短期不建议立刻把主线改成复杂 joint world-action model。原因是当前模块化链路已经能诊断每一环：

```text
DP loss
Foresight prediction quality
TacQuality scorer quality
guidance gradient audit
real rollout force trace
```

论文故事可以强调这种可诊断性和安全性，后续再扩展 force-conditioned foresight 或 latent action guidance。

## 11. 代码现状与故事一致性复核

当前 repo 中实际链路如下。

### 11.1 DP 训练侧

入口：

```text
diffusion/train_dp_tac_concat.py
```

当前 260617-only run 使用的是部署兼容的 concat 版本：

```text
vision features from global/wrist
+ frozen TactileVAE latent from left tactile history
+ proprio
-> obs_cond
-> ConditionalUnet1D diffusion policy
-> future joint action chunk
```

这说明当前训练的 DP 本身仍是 imitation/action prior，不直接包含 TacQualityEnergy。它的作用是学习演示动作分布。

### 11.2 部署/引导侧

入口：

```text
for_show_xiaomi/serve_dp_tac_quality_guided.py
```

文件头部和实现都明确当前不是旧的 reranking 服务器，而是：

```text
DP denoising
-> clean action chunk
-> TacQuality gradient refinement
-> action
```

具体可微链路是：

```text
action_raw
-> Foresight
-> decoded / predicted tactile marker
-> TacQuality score
-> d(score) / d(action_raw)
-> bounded trust-region update
```

实现中还包含：

- `accept_only_improved`：只接受评分提升的更新；
- `max_total_delta` / `action_step`：限制动作偏移；
- board contact gate：擦黑板接触不足时弱化或跳过 TacQuality guidance；
- `--disable_guidance`：同一 serving stack 下可跑 baseline；
- `--guidance_site final_action / denoising_step`：主线是 final-action，denoising-step 保留为 ablation。

所以当前代码和“gradient guidance，不是 reranking”的目标一致。

### 11.3 Foresight 侧

相关入口：

```text
TFAC_V5/pretrain_latent_foresight.py
TFAC_V5/pretrain_latent_foresight_multistep.py
```

多步版本的目标是：

```text
current V/T observation + future action/state chunk
-> future tactile VAE latent sequence z[t+1:t+H]
```

loss 由三部分组成：

```text
L = L_latent + lambda_marker * L_marker + lambda_delta * L_delta
```

这和当前 story 中的 action-conditioned tactile consequence prediction 一致。当前仍缺的是把 force history 显式输入 Foresight；这可以作为下一版增强，而不是当前必须推翻的部分。

## 12. 当前最合理的改进路线

从近期论文和当前代码看，建议按下面顺序推进，而不是立即换大架构。

### 12.1 短期：保持当前模块化链路，补真实评估

目标是证明当前系统真实有效：

```text
baseline DP
vs
DP + TacQuality gradient guidance
```

擦黑板必须记录并比较：

- server 侧每条 rollout 的 force trace；
- force-band occupancy；
- too-light ratio；
- too-heavy ratio；
- force derivative / smoothness；
- 接触 dropout；
- 擦拭覆盖/任务完成情况。

插孔必须记录并比较：

- success；
- bounce；
- retry 次数；
- collision/bounce reason。

这一步比继续堆模型更关键，因为当前缺口不是离线 scorer/guidance smoke，而是真实 paired rollout evidence。

### 12.2 中期：Force-conditioned Foresight

擦黑板的好坏标准本质上与接触力相关。下一版 Foresight 建议改成：

```text
image/proprio/tactile history
+ force history
+ candidate action chunk
-> future tactile marker/latent
+ future force proxy
```

这样 TacQualityEnergy 不只依赖 marker proxy，而是能直接预测“这个 action 会不会造成力过小/过大/忽大忽小”。

### 12.3 中期：Contact-gated multi-head quality energy

擦黑板 scorer 建议保持多头，但更清楚地分工：

```text
goodness head: 正常擦拭质量
too_light head: 力不足/接触弱/擦不干净风险
too_heavy head: 压力过大/安全风险
roughness head: 力或 marker 变化不平滑
contact gate: 只在真实接触/擦拭阶段启用强 guidance
```

最终用于 guidance 的不是单独分类概率，而是连续质量能量：

```text
quality =
  + good_score
  - w_light * too_light_risk
  - w_heavy * too_heavy_risk
  - w_rough * roughness_risk
  - w_delta * action_delta_penalty
```

这样比二分类更适合提供稳定梯度。

### 12.4 后续：Denoising-step guidance 和 latent-action guidance

当前 final-action trust-region 是最稳主线。后续可研究两个增强：

1. **DDPM denoising-step guidance**
   - 更接近 classifier guidance；
   - 但需要 scheduler-aware scaling；
   - 必须做 step sweep，防止中间噪声步梯度破坏动作流形。

2. **latent-action guidance**
   - 用 action CVAE/autoencoder 把 action chunk 压到低维潜变量；
   - diffusion 在 latent action 上生成；
   - TacQuality 梯度通过 decoder 回传；
   - 可能更稳定，但需要额外训练和验证。

当前不建议马上替换主线；更合理的是先把 real rollout evidence 补齐，再用这些作为论文增强/ablation。

## 13. arXiv ID 核对清单

2026-06-19 15:50 使用 arXiv API 对相关条目按 ID 做了二次核对。以下条目均能在 arXiv API 中返回，标题和日期如下。

| arXiv ID | 日期 | 标题 | 对本项目相关性 |
|---|---|---|---|
| 2606.14981 | 2026-06-12 | Inference-time Policy Steering via Vision and Touch | 最高，直接支持 vision/touch verifier 和 inference-time steering |
| 2606.11184 | 2026-06-09 | TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation | 最高，支持 force-conditioned tactile world model |
| 2606.13877 | 2026-06-11 | ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation | 高，支持多步触觉后果和 contact-rich world model 评价 |
| 2606.08737 | 2026-06-07 | Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation | 高，支持 joint world-action/tactile dynamics 方向 |
| 2604.23609 | 2026-04-26 | Tube Diffusion Policy: Reactive Visual-Tactile Policy Learning for Contact-rich Manipulation | 高，支持 action-tube/reactive correction 表述 |
| 2605.11048 | 2026-05-11 | ForceFlow: Learning to Feel and Act via Contact-Driven Flow Matching | 高，支持 contact/force-dominant execution 阶段 |
| 2606.14801 | 2026-06-11 | QPILOTS: Efficient Test-Time Q-Steering for Flow Policies | 中高，支持 test-time critic/Q steering，但提醒多步反传需稳定化 |
| 2606.12365 | 2026-06-10 | Ambient Diffusion Policy: Imitation Learning from Suboptimal Data in Robotics | 中，支持利用 suboptimal/negative data 学习可达分布和质量引导 |
| 2606.20135 | 2026-06-18 | Frequency-Aware Flow Matching for Continuous and Consistent Robotic Action Generation | 中，支持动作平滑和频率一致性指标 |
| 2605.12247 | 2026-05-12 | SI-Diff: A Framework for Learning Search and High-Precision Insertion with a Force-Domain Diffusion Policy | 中高，支持插孔 search/insertion 阶段和 force-domain policy |
| 2606.08657 | 2026-06-07 | Latent Diffusion Policy: Shaping Latent Spaces for Diffusion-Based Robotic Manipulation | 中，支持后续 latent-action guidance |
| 2606.06281 | 2026-06-04 | Multi-Resolution Tactile Imitation Learning for Contact-Rich Robotic Manipulation | 中，支持多时间尺度 tactile 特征 |
| 2605.07308 | 2026-05-08 | AT-VLA: Adaptive Tactile Injection for Enhanced Feedback Reaction in Vision-Language-Action Models | 中，支持接触阶段自适应注入触觉 |
| 2606.13102 | 2026-06-11 | FTP-1: A Generalist Foundation Tactile Policy Across Tactile Sensors for Contact-Rich Manipulation | 中，支持 tactile representation generalization |
| 2606.17055 | 2026-06-15 | T-Rex: Tactile-Reactive Dexterous Manipulation | 中，支持 tactile-reactive manipulation |
| 2606.14862 | 2026-06-12 | TacStyle: Personalizing Tactile Robot Policies using Structured Behavior Representations | 中，支持结构化 tactile behavior preference |
| 2606.20426 | 2026-06-18 | TaCauchy: An Extensible FEM Framework for Vision-Based Tactile Simulation | 中低，更多是 tactile simulation |
| 2606.19161 | 2026-06-17 | HT-Bench: Benchmarking and Learning Dexterous Full-Hand Tactile Representations with Egocentric Vision | 中低，更多是 tactile representation benchmark |
| 2605.27919 | 2026-05-27 | Frequency-Guided Action Diffusion via Sub-Frequency Manifold Traversal | 中，支持动作频率/平滑性控制 |
| 2605.29937 | 2026-05-28 | Fisher-Preserving Guidance: Training-Free Manifold Constraints for Safe Diffusion Control | 中，支持 guidance 不应离开动作流形 |

对当前项目最该优先吸收的不是“换成某一篇论文的完整架构”，而是四个共同趋势：

1. inference-time steering 正在成为 contact-rich generative policy 的重要方向；
2. tactile/force future verifier 比单纯 observation concat 更能解释接触任务改进；
3. guidance 必须保持在动作流形附近，不能为了提高 score 产生不可执行动作；
4. 触觉质量要看完整未来过程和接触阶段，不应只看单帧或全 episode 平均值。

## 14. 基于近期论文的项目改进判断

近期论文和当前代码对齐后的结论如下。

### 14.1 不建议马上换掉当前主线

当前主线：

```text
DP action prior
-> action-conditioned tactile foresight
-> TacQualityEnergy / risk scorer
-> bounded gradient guidance
```

和 ViTaL、TacForeSight、ContactWorld、QPILOTS、Tube Diffusion Policy 的共同趋势一致。现在最不应该做的是把 DP、world model、scorer 全部合成一个难以诊断的大模型。当前模块化链路的优势是每一环都能单独评估：

```text
DP imitation loss
Foresight future tactile prediction
TacQuality classification / regression quality
guidance gradient audit
real rollout force trace
```

### 14.2 最应该补的是真实 rollout 证据

离线训练 loss 和 scorer AUC 只能说明模型内部链路可用，不能证明真机擦得更好。擦黑板任务下一步最重要的是成对比较：

```text
baseline DP
vs
DP + TacQuality gradient guidance
```

每条轨迹必须保存 server 侧 force trace，并计算：

```text
force-band occupancy
too-light ratio
too-heavy ratio
force derivative / smoothness
contact dropout
trajectory success / task completion
```

这会直接回答“引导有没有让触觉后果更好”。

### 14.3 下一版模型改进优先级

优先级 1：force-conditioned Foresight。

```text
obs image/proprio/tactile history
+ force history
+ candidate action chunk
-> future tactile marker/latent
+ future force proxy
```

原因：擦黑板好坏标准本质上依赖接触力，marker-only 只能间接反映力。

优先级 2：contact-gated multi-head TacQualityEnergy。

```text
quality =
  good_score
  - w_light * too_light_risk
  - w_heavy * too_heavy_risk
  - w_rough * roughness_risk
  - w_delta * action_delta_penalty
```

原因：二分类不足以提供稳定、细粒度梯度；多头能区分不同坏法。

优先级 3：latent-action guidance。

```text
action chunk -> action latent z_a
diffusion / flow in z_a
decoder(z_a) -> action chunk
Foresight -> TacQuality -> d(score)/d(z_a)
```

原因：raw joint action guidance 虽然直接，但更容易偏离动作流形；latent action guidance 可能更稳，不过需要额外训练 action autoencoder/CVAE。

## 15. 2026-06-19 19:04 训练监督更新：epoch 640

### 15.1 当前训练状态

当前 run 仍正常运行：

```text
数据:
/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617

脚本:
diffusion/train_dp_tac_concat.py

输出:
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr

TactileVAE:
/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt
```

epoch 640 检查结果：

| 项目 | 数值 |
|---|---:|
| latest epoch | 640 / 2000 |
| train loss | 0.004124 |
| val loss | 0.024099 |
| best val loss | 0.011659 |
| best epoch | 155 |

最近 12 个验证点如下：

| epoch | train | val |
|---:|---:|---:|
| 585 | 0.004481 | 0.022033 |
| 590 | 0.004674 | 0.025091 |
| 595 | 0.003869 | 0.026442 |
| 600 | 0.004343 | 0.024435 |
| 605 | 0.004252 | 0.022597 |
| 610 | 0.004680 | 0.022726 |
| 615 | 0.004116 | 0.023280 |
| 620 | 0.004524 | 0.023983 |
| 625 | 0.004237 | 0.023600 |
| 630 | 0.003880 | 0.026809 |
| 635 | 0.004414 | 0.024344 |
| 640 | 0.004124 | 0.024099 |

### 15.2 checkpoint 与可视化

已重新生成 loss 曲线：

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/loss_curve.png
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/loss_curve.csv
```

checkpoint 状态：

```text
dp_best.pth       epoch 155, val=0.011659
dp_epoch600.pth   latest completed save_freq checkpoint at this check
dp_latest.pth     continuing to update
next expected     dp_epoch650.pth
```

### 15.3 判断

当前训练没有崩溃、没有 NaN、GPU 和磁盘状态正常。但泛化指标仍没有回到 epoch 155 的最好点：

```text
train loss: 继续下降
val loss:   长时间保持在 0.022-0.026 区间
best:       仍是 epoch 155
```

因此当前部署或离线对比仍应使用：

```text
dp_best.pth
```

不应使用：

```text
dp_latest.pth
```

除非后续验证点刷新 best。

## 16. 近期论文核对后对 scorer/guidance 的设计收敛

### 16.1 相关工作对齐

近期工作里，和本项目最相关的方向包括：

| 工作 | 链接 | 对本项目的启发 |
|---|---|---|
| ViTaL / Inference-time Policy Steering via Vision and Touch | https://arxiv.org/abs/2606.14981 | 支持推理时用视觉/触觉 verifier 或 steering 信号修正策略 |
| TacForeSight | https://arxiv.org/abs/2606.11184 | 支持 force-conditioned tactile world model |
| ContactWorld | https://arxiv.org/abs/2606.13877 | 支持用未来触觉后果评价 contact-rich manipulation |
| Dream-Tac | https://arxiv.org/abs/2606.08737 | 支持 tactile world-action model |
| DPTG | https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1851102/full | 直接支持 tactile feasibility guidance 思路 |
| QPILOTS | https://arxiv.org/html/2606.14801v1 | 支持 test-time critic/Q steering，但需要控制引导强度 |
| QGF / Test-Time Gradient Guidance of Flow Policies | https://arxiv.org/abs/2606.11087 | 支持不改 supervised generative policy 主体、只在 test time 用 critic/score 梯度做有限步改进 |
| ForceFlow | https://arxiv.org/abs/2605.11048 | 支持 contact/force dominant 的生成式动作建模 |

### 16.2 当前项目不应转向 reranking

用户已经明确目标是 DP 推理中的梯度引导，不是 reranking。近期工作也更支持如下主线：

```text
DP action prior
-> action-conditioned tactile / force foresight
-> differentiable quality / risk scorer
-> bounded gradient guidance
-> guided action chunk
```

reranking 可以作为 debug baseline，但不应作为论文主线或最终方案。

QPILOTS / QGF 对当前项目还有一个直接提醒：梯度引导要做小幅、受约束的 test-time correction，而不是无限追 scorer。也就是说，论文叙事应强调：

```text
保留 DP 作为 action prior，
用 action-conditioned tactile/force foresight 估计未来接触后果，
再用可微 TacQualityEnergy 对动作做 bounded gradient correction。
```

### 16.3 scorer 设计应从二分类升级为连续能量

插孔和擦黑板的共同需求不是“只判好坏”，而是给 DP 反向传播一个稳定方向。因此更合适的是质量能量：

```text
E_quality = 
  w_bad   * bad_risk
+ w_force * force_band_violation
+ w_smooth * tactile_or_force_roughness
+ w_contact * contact_dropout
+ w_prior * action_deviation_penalty
```

推理时最大化 quality 或最小化 energy：

```text
a <- a - eta * grad_a(E_quality)
```

并用 trust region / norm clipping 保证动作不离开 DP 学到的动作流形。

### 16.4 两个任务的标签标准应保持任务特异但接口统一

插孔：

```text
good:
  成功插入、无 bounce、无外壁碰撞趋势

bad:
  pre-bounce / bounce / recovery risk，尤其来自带 bounce episode 的关键窗口

score:
  good_margin = logit(good) - logit(pre_bounce_or_bounce)
```

擦黑板：

```text
good:
  接触阶段力在合理范围，变化平滑，marker offset 稳定

bad too_light:
  接触太弱、可能擦不干净

bad too_heavy:
  接触太强、安全风险或损伤风险

bad oscillate:
  力/marker 高频波动，擦拭不柔顺
```

统一接口：

```text
score(predicted_future_tactile, predicted_or_observed_force, action_chunk) -> scalar quality
```

这样插孔和黑板可以共用 guidance 机制，但标签头和权重按任务切换。

### 16.5 下一步最有价值的模型改动

短期不建议重写 DP 主体。更高价值的是改 foresight 和 scorer：

1. force-conditioned foresight

```text
obs image/proprio/tactile history
+ force history
+ candidate action chunk
-> future tactile latent / marker
+ future force proxy
```

2. contact-gated TacQualityEnergy

```text
contact gate:
  只在 wiping/contact 阶段强引导

quality heads:
  good
  too_light
  too_heavy
  oscillate / roughness

regularization:
  action_delta_penalty
  guidance_norm_penalty
```

3. paired real rollout force-trace evaluation

```text
baseline DP vs guided DP
same task / same board condition / same start distribution
server side saves force trace for every rollout
evaluate force-band occupancy, too-light ratio, too-heavy ratio, smoothness, dropout
```

这一步是证明 scorer/guidance 真正有效的关键证据。

## 17. 2026-06-19 19:12 训练监督更新：epoch 650

### 17.1 epoch 650 结果

epoch 650 已完成并保存 checkpoint：

```text
dp_epoch650.pth
mtime: 2026-06-19 19:10
size: 约 2.6G
```

指标：

| 项目 | 数值 |
|---|---:|
| epoch | 650 / 2000 |
| train loss | 0.003642 |
| val loss | 0.029504 |
| best val loss | 0.011659 |
| best epoch | 155 |

近 10 个验证点：

| epoch | train | val |
|---:|---:|---:|
| 605 | 0.004252 | 0.022597 |
| 610 | 0.004680 | 0.022726 |
| 615 | 0.004116 | 0.023280 |
| 620 | 0.004524 | 0.023983 |
| 625 | 0.004237 | 0.023600 |
| 630 | 0.003880 | 0.026809 |
| 635 | 0.004414 | 0.024344 |
| 640 | 0.004124 | 0.024099 |
| 645 | 0.003925 | 0.026384 |
| 650 | 0.003642 | 0.029504 |

### 17.2 判断

训练本身健康，checkpoint 和 latest 都在正常保存。但这个保存点不是部署候选：

```text
dp_epoch650.pth: val=0.029504
dp_best.pth:     val=0.011659 at epoch 155
```

当前现象更像是：

```text
train loss 继续下降
validation loss 持续变差
```

所以后续如果要做 deployment/offline comparison，仍应默认使用：

```text
dp_best.pth
```

继续训练的价值是完整观察长程曲线，确认是否存在极晚期 validation 回落；不能因为 epoch 更晚就认为更好。

loss 曲线已更新到 epoch 651：

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/loss_curve.png
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/loss_curve.csv
```

## 18. 当前 scorer / guidance 证据复查

### 18.1 实现链路

当前部署入口：

```text
for_show_xiaomi/serve_dp_tac_quality_guided.py
```

不是 reranking。主链路是：

```text
DP denoising
-> clean action chunk
-> Foresight(action)
-> predicted future marker
-> TacQuality score
-> trust-region gradient ascent on action
-> guided action
```

关键实现文件：

```text
TFAC_V5/tac_quality_energy/serving_guidance.py
TFAC_V5/tac_quality_energy/trust_region.py
TFAC_V5/tac_quality_energy/foresight_bridge.py
TFAC_V5/tac_quality_energy/force_band_runtime.py
TFAC_V5/tac_quality_energy/insertion_runtime.py
```

server 侧真实 rollout 记录：

```text
for_show_xiaomi/server_rollout_logger.py
```

会按 `baseline/` 和 `guided/` 分组，每条 rollout 单独保存：

```text
force_trace.csv
force_trace.npz
force_curve.png
metadata.json
```

### 18.2 插孔 scorer 当前证据

推荐配置：

```text
arm: good_margin_guided
runtime: InsertionRiskScorerRuntime
score mode: good_margin
checkpoint: /home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt
```

episode/group-level CV：

| metric | mean |
|---|---:|
| binary balanced accuracy | 0.9437 |
| binary AUC | 0.9877 |
| reason balanced accuracy | 0.7880 |
| quality corr | 0.7656 |
| quality R2 | 0.5682 |

matched real-Foresight gradient audit：

| metric | value |
|---|---:|
| finite grad rate | 1.0000 |
| positive grad rate | 1.0000 |
| improved rate | 1.0000 |
| trust-region pass rate | 1.0000 |

score-mode ablation 结论：

```text
p_good 是 bounded probability，容易饱和；
good_margin 是 unsaturated binary logit margin，更适合作为梯度引导分数。
```

### 18.3 黑板 scorer 当前证据

推荐配置：

```text
arm: marker_joint_s12_guided
runtime: ForceBandTacQualityEnergyRuntime(marker_joint_action,s12)
score mode: quality
checkpoint: /home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/force_band_tac_quality_energy_best.pt
```

held-out validation：

| metric | value |
|---|---:|
| binary AUC | 1.0000 |
| balanced accuracy | 1.0000 |
| reason macro F1 | 1.0000 |
| quality Spearman | 0.9243 |

260617 作为 positive 加入后的离线评估：

| deployable feature | AUC | bACC | 260617 positive recall | old positive recall |
|---|---:|---:|---:|---:|
| marker_action | 0.9998 | 0.9934 | 0.9926 | 0.9908 |

real-Foresight gradient audit：

| metric | value |
|---|---:|
| finite grad rate | 1.0000 |
| positive grad rate | 1.0000 |
| improved rate | 1.0000 |
| trust-region pass rate | 1.0000 |
| score delta mean | 0.000162 |
| action delta norm mean | 0.000775 |

### 18.4 关键边界

黑板 scorer 的分类/采集 regime 区分很强，但连续物理质量排序仍弱。260617 positive 加入后的 deployable `marker_action` 特征中：

```text
quality Spearman ~= 0.1625
```

所以当前黑板 scorer 可以作为：

```text
接触质量分类器
弱连续能量
安全小步 guidance 候选
```

但不能写成：

```text
已经证明能强优化真实 force curve
```

当前 real rollout precheck 仍显示：

```text
baseline force_trace: missing
guided force_trace: missing
real_rollout_proven: false
```

因此下一步必须做成对真机 rollout：

```text
baseline DP vs DP + TacQuality gradient guidance
```

并用 server 侧 `force_trace.csv` 计算 force-band occupancy、too-light ratio、too-heavy ratio、smoothness 和 dropout。

## 19. 2026-06-19 19:40 训练监督更新：epoch 700

### 19.1 epoch 700 结果

epoch 700 已完成并保存 checkpoint：

```text
dp_epoch700.pth
mtime: 2026-06-19 19:39
size: 约 2.6G
```

指标：

| 项目 | 数值 |
|---|---:|
| epoch | 700 / 2000 |
| train loss | 0.004207 |
| val loss | 0.026389 |
| best val loss | 0.011659 |
| best epoch | 155 |

近 14 个验证点：

| epoch | train | val |
|---:|---:|---:|
| 635 | 0.004414 | 0.024344 |
| 640 | 0.004124 | 0.024099 |
| 645 | 0.003925 | 0.026384 |
| 650 | 0.003642 | 0.029504 |
| 655 | 0.004306 | 0.028255 |
| 660 | 0.004126 | 0.026168 |
| 665 | 0.003878 | 0.025037 |
| 670 | 0.003970 | 0.027623 |
| 675 | 0.004138 | 0.023647 |
| 680 | 0.004177 | 0.025746 |
| 685 | 0.003611 | 0.026311 |
| 690 | 0.003965 | 0.023611 |
| 695 | 0.003878 | 0.025957 |
| 700 | 0.004207 | 0.026389 |

### 19.2 判断

训练和保存机制仍正常，但 epoch 700 没有带来泛化改善：

```text
dp_epoch700.pth: val=0.026389
dp_best.pth:     val=0.011659 at epoch 155
```

当前后期验证损失长期停在 `0.023-0.029` 区间，仍远高于 best。部署或离线对比仍应使用：

```text
dp_best.pth
```

不应使用：

```text
dp_epoch700.pth
dp_latest.pth
```

loss 曲线已更新到 epoch 701：

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/loss_curve.png
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/loss_curve.csv
```

下一重点检查 `dp_epoch750.pth`。如果 750/800 仍无改善，后续监督可以降到每 100 epoch；模型选择仍以 validation best 为准。

## 20. 2026-06-19 19:50 训练监督与 arXiv 核验边界

### 20.1 当前训练状态

当前 260617-only stable run 仍正常运行：

| 项目 | 状态 |
|---|---|
| PID | `3037873` |
| 最新日志位置 | epoch 717/2000 附近 |
| 最新完整验证点 | epoch 715 |
| epoch 710 train / val | 0.003946 / 0.029560 |
| epoch 715 train / val | 0.004006 / 0.026646 |
| 当前 best | val 0.011659 @ epoch 155 |
| GPU | RTX 4090, 约 14.7GB/24.6GB 显存, 利用率约 85% |
| 训练进程 RSS | 约 39GB |
| 输出目录大小 | 约 51GB |
| image cache | 约 156GB |
| 外接盘剩余 | 约 2.1TB |
| 根分区剩余 | 约 42-43GB |

loss 曲线已重新生成到 epoch 711：

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/loss_curve.png
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/loss_curve.csv
```

判断不变：

```text
训练健康；
checkpoint 保存正常；
late epoch 的 val 仍明显差于 epoch 155 best；
当前候选仍是 dp_best.pth，不是 dp_latest.pth 或 late top-k train ckpt。
```

### 20.2 arXiv API 已核验条目

2026-06-19 19:50 再次用 arXiv API 按 ID 拉取元数据。以下条目可返回标题和日期：

| arXiv ID | 日期 | 标题 |
|---|---|---|
| 2604.23609v1 | 2026-04-26 | Tube Diffusion Policy: Reactive Visual-Tactile Policy Learning for Contact-rich Manipulation |
| 2606.12365v1 | 2026-06-10 | Ambient Diffusion Policy: Imitation Learning from Suboptimal Data in Robotics |
| 2606.08737v1 | 2026-06-07 | Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation |
| 2605.11048v1 | 2026-05-11 | ForceFlow: Learning to Feel and Act via Contact-Driven Flow Matching |
| 2606.08657v1 | 2026-06-07 | Latent Diffusion Policy: Shaping Latent Spaces for Diffusion-Based Robotic Manipulation |
| 2606.06281v1 | 2026-06-04 | Multi-Resolution Tactile Imitation Learning for Contact-Rich Robotic Manipulation |
| 2606.11184v1 | 2026-06-09 | TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation |
| 2606.11087v1 | 2026-06-09 | Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning |
| 2606.13877v1 | 2026-06-11 | ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation |
| 2606.20135v1 | 2026-06-18 | Frequency-Aware Flow Matching for Continuous and Consistent Robotic Action Generation |

这组核验结果支持当前项目的技术路线：

```text
DP action prior
-> action-conditioned tactile/force foresight
-> contact/quality/risk energy
-> bounded test-time gradient guidance
```

但需要明确边界：

1. 近期论文只能支持“方向合理”，不能替代本项目自己的实验。
2. 当前 blackboard scorer 的连续物理质量排序仍弱，不能夸大为强 force-curve optimizer。
3. 当前 260617-only DP 的 late checkpoint 没有刷新 validation best，不能把 2000 epoch 训练后的 latest 当作更优模型。
4. 最终是否有效必须靠 paired real rollout 的 server-side force trace 验证。

## 21. 2026-06-19 20:10 训练监督更新：epoch 750

### 21.1 epoch 750 结果

epoch 750 已完成并保存 checkpoint：

```text
dp_epoch750.pth
mtime: 2026-06-19 20:08
size: 约 2.6G
```

指标：

| 项目 | 数值 |
|---|---:|
| epoch 750 train | 0.003983 |
| epoch 750 val | 0.022859 |
| epoch 755 train | 0.003791 |
| epoch 755 val | 0.027480 |
| best val loss | 0.011659 |
| best epoch | 155 |

近 16 个验证点：

| epoch | train | val |
|---:|---:|---:|
| 680 | 0.004177 | 0.025746 |
| 685 | 0.003611 | 0.026311 |
| 690 | 0.003965 | 0.023611 |
| 695 | 0.003878 | 0.025957 |
| 700 | 0.004207 | 0.026389 |
| 705 | 0.004486 | 0.023471 |
| 710 | 0.003946 | 0.029560 |
| 715 | 0.004006 | 0.026646 |
| 720 | 0.003810 | 0.024968 |
| 725 | 0.003615 | 0.023512 |
| 730 | 0.003693 | 0.022191 |
| 735 | 0.004204 | 0.025270 |
| 740 | 0.003757 | 0.027757 |
| 745 | 0.004228 | 0.026487 |
| 750 | 0.003983 | 0.022859 |
| 755 | 0.003791 | 0.027480 |

### 21.2 checkpoint 和系统状态

已确认：

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_best.pth
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_epoch750.pth
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/loss_curve.png
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/loss_curve.csv
```

loss 曲线已重新生成到 epoch 755。

系统状态：

| 项目 | 状态 |
|---|---|
| 训练进程 | PID `3037873`, 正常运行 |
| GPU | RTX 4090, 约 14.7GB/24.6GB, 利用率约 88% |
| 进程 RSS | 约 39GB |
| 外接盘剩余 | 约 2.1TB |
| 根分区剩余 | 约 43GB |

### 21.3 判断

训练没有崩溃，checkpoint 保存正常，但 epoch 750 仍没有泛化改善：

```text
dp_epoch750.pth: val=0.022859
dp_best.pth:     val=0.011659 at epoch 155
```

因此当前结论是：

1. 继续训练符合用户要求，可以作为 2000 epoch 充分训练证据。
2. `dp_epoch750.pth`、`dp_latest.pth` 和 late train-topk checkpoint 不能作为部署最优模型。
3. 当前 260617-only run 的候选仍是 `dp_best.pth`。
4. 下一重点检查 `dp_epoch800.pth`；如果 800/850 仍无改进，后续监督频率可降到每 100 epoch。

## 22. 2026-06-19 20:40 训练监督更新：epoch 800

### 22.1 epoch 800 结果

epoch 800 已完成并保存 checkpoint：

```text
dp_epoch800.pth
mtime: 2026-06-19 20:37
size: 约 2.6G
```

指标：

| 项目 | 数值 |
|---|---:|
| latest log | epoch 804 / 2000 |
| epoch 800 train | 0.003721 |
| epoch 800 val | 0.029729 |
| best val loss | 0.011659 |
| best epoch | 155 |

近 18 个验证点：

| epoch | train | val |
|---:|---:|---:|
| 715 | 0.004006 | 0.026646 |
| 720 | 0.003810 | 0.024968 |
| 725 | 0.003615 | 0.023512 |
| 730 | 0.003693 | 0.022191 |
| 735 | 0.004204 | 0.025270 |
| 740 | 0.003757 | 0.027757 |
| 745 | 0.004228 | 0.026487 |
| 750 | 0.003983 | 0.022859 |
| 755 | 0.003791 | 0.027480 |
| 760 | 0.003663 | 0.025001 |
| 765 | 0.003673 | 0.028802 |
| 770 | 0.003697 | 0.024357 |
| 775 | 0.003553 | 0.030577 |
| 780 | 0.003993 | 0.029230 |
| 785 | 0.003680 | 0.027678 |
| 790 | 0.003707 | 0.028602 |
| 795 | 0.003753 | 0.026867 |
| 800 | 0.003721 | 0.029729 |

### 22.2 checkpoint 和系统状态

已确认：

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_best.pth
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_epoch800.pth
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/loss_curve.png
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/loss_curve.csv
```

loss 曲线已重新生成到 epoch 804。

系统状态：

| 项目 | 状态 |
|---|---|
| 训练进程 | PID `3037873`, 正常运行 |
| GPU | RTX 4090, 约 14.7GB/24.6GB, 利用率约 73% |
| 进程 RSS | 约 39GB |
| 外接盘剩余 | 约 2.1TB |
| 根分区剩余 | 约 43GB |

### 22.3 判断

训练与保存机制仍正常，但 epoch 800 的验证损失更差：

```text
dp_epoch800.pth: val=0.029729
dp_best.pth:     val=0.011659 at epoch 155
```

因此当前结论进一步收敛：

1. 260617-only run 后期训练继续降低 train loss，但没有带来 validation 泛化收益。
2. `dp_epoch750.pth`、`dp_epoch800.pth`、`dp_latest.pth` 和 late train-topk checkpoint 都不适合作为部署候选。
3. 当前部署/离线对比候选仍只推荐 `dp_best.pth`。
4. 后续监督降到每 100 epoch 或异常触发；下一重点检查 `dp_epoch900.pth`，中间只关注 watcher 是否报 NaN、OOM、写盘失败或 best 刷新。
