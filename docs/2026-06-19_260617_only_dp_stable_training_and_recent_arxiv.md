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
