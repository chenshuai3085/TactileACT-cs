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
