# 2026-06-19 260617-only DP Training and Latest ArXiv Follow-up

## 1. 当前训练事实

目标：只使用 260617_v8l_caheiban 数据训练 board wiping 的 tactile DP concat policy，训练目标为 2000 epoch，并持续监督训练状态。

数据：

- 数据目录：`/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617`
- HDF5 episode 数：80
- 训练脚本实际可读 episode：
  - train split：72 个 episode 中跳过 1 个不完整/损坏 episode，实际 71 个 episode
  - val split：8 个 episode
- 训练窗口：
  - train windows：8192
  - val windows：1024
- 读取 key：
  - image：`observations/images/global`, `observations/images/wrist`
  - proprio：`observations/proprio_joint`
  - action：`actions/joint_abs`
  - tactile：`observations/tac/left/marker_offset`

模型与输入：

- 训练脚本：`diffusion/train_dp_tac_concat.py`
- DP 输入：global/wrist RGB + left tactile VAE latent + proprio joint
- image 原始/训练尺寸：200x266
- TactileVAE checkpoint：
  `/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt`
- TactileVAE norm stats：
  - mean = `[-0.3398614526, -2.9208483696]`
  - std = `[1.9804853201, 2.7671177387]`
- tactile history：8
- pred horizon：16
- obs horizon：2
- action horizon：8
- action dim：7
- global condition dim：2350
- policy net 参数量：约 3.15e8

训练配置：

- epochs：2000
- batch size：64
- lr：1e-4
- weight decay：1e-6
- diffusion train timesteps：100
- inference steps：100
- down dims：512,1024,2048
- EMA：enabled
- image cache：
  `/home/chenshuai/Project/output/cache/dp_board_rawimg200x266_fp16`
- 当前稳定 run 使用 `num_workers=0`
  - 原因：第一次后台 nohup 版本在 epoch 1 中途无 traceback 退出；tmux + `num_workers=0` 已稳定跑通 train/val/best ckpt 链路。
  - 后续如果确认 IO 成为瓶颈，可以另开 tmux + `num_workers=4` run；旧 run 证明 `num_workers=4` 在该脚本上能长时间训练。

输出目录：

`/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_full_noearly_tmux`

tmux 会话：

- 训练：`dp260617_only_2000_noearly`
- 只读监控：`dp260617_only_monitor_readonly`

ckpt 策略：

- `dp_best.pth`：按 episode-level validation loss 更新
- `dp_latest.pth`：每 10 epoch 更新，包含 optimizer/scheduler，用于中断恢复参考
- `dp_epoch*.pth`：每 50 epoch 固定保存
- `dp_topk_*.pth`：保留 train loss top-3

当前启动后已确认：

- 第 1 epoch 完成并写出 `dp_best.pth`
- `loss_curve.csv` 和 `loss_curve.png` 已生成
- GPU 约使用 14.7 GB 显存
- 当前训练仍在运行

截至 2026-06-19 08:59 CST 的训练状态：

- latest epoch：10 / 2000
- train loss：0.026821
- val loss：0.025968
- best val loss：0.025968 @ epoch 10
- 状态：healthy，train/val 均快速下降

注意：

- 这里的 val loss 是离线 episode-level validation split 上的 DDPM noise prediction MSE，不等价于真实机器人擦拭成功率。
- 后续部署仍应默认使用 `dp_best.pth`，除非明确要测试 late/overfit checkpoint。
- 真实效果必须通过真机 rollout 的力曲线、轨迹完成度和 baseline/guided 成对评估确认。

## 2. 最近两个月相关论文筛选

时间窗口：约 2026-04-19 到 2026-06-19。

筛选标准：只保留与本项目当前主线直接相关的工作，即 tactile/force world model、diffusion/flow policy guidance、contact-rich manipulation、suboptimal data learning、long context policy。

### 2.1 与当前方案最接近的工作

1. Inference-time Policy Steering via Vision and Touch, arXiv:2606.14981
   - 链接：https://arxiv.org/abs/2606.14981
   - 关键点：部署时用 vision+touch verification/steering 检查候选 action，对 contact-rich manipulation 比 vision-only 更可靠。
   - 对本项目的意义：我们的目标不是 reranking，而是 gradient guidance；但它支持一个核心判断：触觉后果评分器应放在推理时闭环里，而不是只作为训练集标签分析工具。

2. TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation, arXiv:2606.11184
   - 链接：https://arxiv.org/abs/2606.11184
   - 关键点：contact-rich manipulation 需要预测未来触觉/力变化；只用当前触觉反馈不够，必须建模 action-conditioned future contact。
   - 对本项目的意义：和我们的 `DP action -> Foresight predicted tactile consequence -> TacQuality score -> gradient guidance` 主线高度一致。项目故事应强调“未来触觉后果”而不是“当前触觉分类”。

3. Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation, arXiv:2606.08737
   - 链接：https://arxiv.org/abs/2606.08737
   - 关键点：world action model 直接把 anticipated tactile observations 纳入 action generation。
   - 对本项目的意义：我们现在是模块化路线：DP 负责 action prior，Foresight 负责 tactile future，TacQuality 负责能量/评分。相比端到端 world-action model，模块化更容易解释、插入现有 DP、做 ablation。

4. ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation, arXiv:2606.13877
   - 链接：https://arxiv.org/abs/2606.13877
   - 关键点：讨论 vision-tactile world model 中哪些表示性质支撑长时程 contact-rich planning。
   - 对本项目的意义：Foresight 不应只追求单帧 latent MSE，还要评估 contact phase、force band、smoothness、future trend 是否保留。

### 2.2 与梯度引导最相关的工作

5. Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning, arXiv:2606.11087
   - 链接：https://arxiv.org/abs/2606.11087
   - 关键点：在 test-time 对 flow/generative policy 使用梯度引导，把 critic/objective 的梯度注入 action generation。
   - 对本项目的意义：这直接支持“不是 reranking，而是 denoising/flow step 内 gradient guidance”的方向。我们的 TacQualityEnergy 应被表述为 differentiable critic/energy。

6. Sample-Efficient Diffusion-based Reinforcement Learning with Critic Guidance, arXiv:2605.30056
   - 链接：https://arxiv.org/abs/2605.30056
   - 关键点：用 critic guidance 改善 diffusion policy sampling/optimization。
   - 对本项目的意义：TacQuality score 最终应满足两个条件：语义正确、梯度可用。只看分类 accuracy 不够，要额外评估 action-gradient 是否能稳定提高 predicted tactile quality。

7. Fisher-Preserving Guidance: Training-Free Manifold Constraints for Safe Diffusion Control, arXiv:2605.29937
   - 链接：https://arxiv.org/abs/2605.29937
   - 关键点：test-time guidance 容易把采样推离训练流形，需要约束 guidance update。
   - 对本项目的意义：当前 protected DDPM step sweep 里的 trust-region / norm clamp 是必要的；论文层面可以把它解释为“policy prior manifold protection”。

### 2.3 对数据和 DP 架构有启发的工作

8. Ambient Diffusion Policy: Imitation Learning from Suboptimal Data in Robotics, arXiv:2606.12365
   - 链接：https://arxiv.org/abs/2606.12365
   - 关键点：用 principled 方法从 suboptimal robot data 学习，而不是简单把坏数据混进 BC。
   - 对本项目的意义：黑板数据天然包含正样本、力过小、力过大、振荡等等级。仅训练 BC policy 可能学习到坏模式；TacQuality score 可以承担“坏模式排斥项”，也可以用于数据加权训练。

9. Training and Evaluating Diffusion Policies with Long Context Lengths, arXiv:2606.16447
   - 链接：https://arxiv.org/abs/2606.16447
   - 关键点：机器人任务中长上下文历史对需要记忆/状态累积的任务有帮助。
   - 对本项目的意义：当前 DP `obs_horizon=2`、tactile history=8。擦黑板这类连续接触任务可能需要更长 force/tactile context 才能判断“稳定擦拭 vs 慢慢失压/过压”。

10. T-Rex: Tactile-Reactive Dexterous Manipulation, arXiv:2606.17055
    - 链接：https://arxiv.org/abs/2606.17055
    - 关键点：强调 tactile-reactive control，而不是把 tactile 只作为静态输入。
    - 对本项目的意义：当前策略是 open-loop action chunk + tactile conditioning；后续可以把 score/guidance 设计成对接触变化更敏感的 reactive correction。

## 3. 对当前项目架构的判断

当前主线是合理的：

`DP action prior -> Foresight tactile future -> TacQualityEnergy -> bounded gradient guidance`

理由：

- DP 本身提供动作流形和多模态 action prior。
- Foresight 将 action 对未来触觉/力后果的影响显式化。
- TacQualityEnergy 将“好触觉后果”的标准转成可微分 score。
- bounded gradient guidance 在 denoising 过程中小步修正 action，避免完全脱离 DP prior。

但需要避免两个方向错误：

- 不能只做 reranking。Reranking 只能从采样候选里选，不直接改变 denoising trajectory；最终论文和系统目标应强调 gradient guidance。
- 不能只报告分类 accuracy。分类器能分好坏，不代表梯度方向能改进 action。必须增加 gradient audit。

## 4. 下一步建议

### 4.1 训练监督

继续让当前 DP 训练跑，不打断：

- 重点看 `dp_best.pth` 是否继续刷新。
- 每隔一段时间更新 `loss_curve.png` 和 `loss_curve.csv`。
- 如果 val loss 长期反弹，而 train loss 继续下降，仍保留训练，但部署建议用 best ckpt。
- 如果 GPU 空了、训练进程退出、出现 NaN、磁盘不足，立即处理。

当前命令：

```bash
tmux attach -t dp260617_only_2000_noearly
```

当前状态文件：

```bash
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_full_noearly_tmux/training_status_latest.json
```

### 4.2 架构改进优先级

P0：把 TacQualityEnergy 定义为 differentiable critic/energy，而不是普通 classifier。

- 输出包括：
  - force band score
  - smoothness score
  - contact stability score
  - task-specific risk score
- 训练评价不仅看 AUC/accuracy，还要看：
  - score 对人工定义质量标签是否单调
  - 对 predicted future tactile 的梯度是否有限、非零、方向正确
  - protected DDPM step 后 score 是否稳定提升

P1：Foresight 评估从 single latent MSE 扩展到 future contact semantics。

- 需要评估：
  - 预测 marker magnitude trend
  - force band proxy trend
  - delta smoothness
  - contact phase preservation
  - bad-to-good semantic direction 是否保留

P2：更长上下文。

- 当前 `obs_horizon=2` + tactile history=8 可以作为 baseline。
- 对擦黑板，建议后续做 ablation：
  - obs horizon：2 vs 4 vs 8
  - tactile history：8 vs 16
  - Foresight future horizon：8 vs 16 vs 32

P3：坏数据利用方式。

- 不建议简单把所有正负数据混在 BC 中当同质 expert。
- 更合理：
  - DP policy 主体优先学习正样本/高质量样本；
  - suboptimal/negative data 主要用于 TacQualityEnergy 或数据加权；
  - 如果混合训练 DP，需要引入 quality-weighted BC 或 filtered BC。

## 5. 当前结论

260617-only DP 训练已经稳定启动，数据、TactileVAE、image cache、validation、best ckpt 都确认可用。当前 early loss 下降健康，但还不能据此判断真机效果。

最近两个月论文趋势非常支持本项目的核心故事：contact-rich manipulation 需要 tactile/force world model，生成式 policy 需要 inference-time/test-time guidance，而 guidance 必须受到 policy manifold/trust-region 保护。

因此后续最值得强化的创新点不是“又训练一个分类器”，而是：

**面向未来触觉后果的可微质量能量模型 + 受限 DDPM/flow 梯度引导 + 真实力曲线闭环评估。**

## 6. 2026-06-19 09:22 CST 训练监督更新

当前 260617-only DP 训练仍在 tmux 中运行：

- tmux：`dp260617_only_2000_noearly`
- run dir：
  `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_full_noearly_tmux`
- latest epoch：`40 / 2000`
- latest train loss：`0.014244`
- latest val loss：`0.016210`
- best val loss：`0.015936 @ epoch 37`
- trend warning：`healthy`
- GPU：RTX 4090，约 `14.7 GB / 24.6 GB` 显存，训练利用率正常波动
- 外置盘剩余空间：约 `2192 GB`
- home/output 剩余空间：约 `43 GB`

ckpt 状态：

- `dp_best.pth` 已在 epoch 37 附近刷新，mtime `2026-06-19 09:19:29`
- `dp_latest.pth` 已在 epoch 40 附近刷新，mtime `2026-06-19 09:21:50`
- `dp_topk_*.pth` 正常保留 top-3 train-loss checkpoint
- epoch 固定 ckpt 会从 epoch 50 开始按 `save_freq=50` 写出

当前判断：

- train loss 和 val loss 仍处于同一量级；
- latest val 只比 best val 高约 `1.7%`；
- 没有 NaN、进程退出、磁盘不足、ckpt 不更新等异常；
- 继续训练，不需要现在中断。

注意：

- 这仍然是离线 validation 的 DDPM noise-prediction MSE；
- 不能用这个 loss 直接声称真机擦拭更好；
- 部署候选仍应优先看 `dp_best.pth`，并用真实 rollout 力曲线做最终判断。

## 7. arXiv API 复核新增结果

时间窗口：`2026-04-19` 到 `2026-06-19`。

检索方式：使用 arXiv API 按以下关键词复核：

- `tactile AND robot`
- `force AND contact-rich`
- `"diffusion policy" AND guidance`
- `"critic guidance" AND robot`
- `"tactile world model"`
- `"world action model" AND tactile`
- `"diffusion policies" AND "long context"`

除前面已经记录的 ViTaL / TacForeSight / Dream-Tac / ContactWorld / critic guidance 外，这轮应补充关注：

1. `Feedback World Model Enables Precise Guidance of Diffusion Policy`, arXiv `2605.15705`
   - 相关性：直接讨论 world model 如何给 diffusion policy 提供 precise guidance。
   - 对本项目启发：我们的 Foresight 不能只做 open-loop 预测评估，后续应让真实执行中的 force/tactile feedback 校准或修正 foresight/guidance，避免预测模型在分布外误导 DP。

2. `Guided Streaming Stochastic Interpolant Policy`, arXiv `2605.10051`
   - 相关性：inference-time guidance + streaming policy，强调低延迟和反应性。
   - 对本项目启发：擦黑板这种连续接触任务不只需要一段 action chunk 的静态好坏，还需要连续执行时能实时小步修正；后续可以把 TacQuality guidance 做成 streaming/receding-horizon 版本。

3. `LAGO Policy: Latency-Aware Asynchronous Diffusion Policies with Goal-Directed Collision-Free Planning for Smooth Manipulation`, arXiv `2606.17982`
   - 相关性：关注 asynchronous diffusion policy 的 inter-chunk discontinuity、smooth manipulation 和延迟。
   - 对本项目启发：擦黑板的坏触觉不只来自力大小，也可能来自 action chunk 之间的不连续。后续评分器应加入 action-smoothness / chunk-boundary smoothness 审计，而不是只看 tactile latent 分类。

4. `IMPACT: Learning Internal-Model Predictive Control for Forceful Robotic Manipulation`, arXiv `2606.10818`
   - 相关性：forceful/contact-rich manipulation，包含 table wiping 类任务。
   - 对本项目启发：可以把当前 `Foresight + TacQualityEnergy + bounded guidance` 表述为一种学习式 contact consequence MPC：DP 给 prior，Foresight 给内模型，TacQualityEnergy 给接触代价。

5. `WT-UMI: Tactile-based Whole-Body Manipulation via Force-Supervised Contact-Aware Planning`, arXiv `2606.13232`
   - 相关性：force-supervised contact-aware planning。
   - 对本项目启发：黑板评分器应显式保留 force-supervised 物理指标，尤其是 force band、force delta、接触持续性，而不是完全依赖 learned latent。

## 8. 对本项目故事的更新判断

现在项目最清晰的主线应写成：

`Tactile-conditioned DP prior + action-conditioned tactile foresight + force/marker quality energy + protected denoising-step guidance`

相比最近工作，本项目可以强调的差异点：

- 相比 ViTaL 类 steering：我们不是只做候选动作验证/重排，而是把触觉后果质量作为可微能量，进入 DP denoising step 做梯度引导。
- 相比 TacForeSight/Dream-Tac：我们不直接替换 policy，而是把 tactile world model 模块化接到已有 DP 上，便于插入不同任务和做 ablation。
- 相比普通 classifier guidance：我们的分类/评分标准来自接触物理质量，包括力大小、力变化平滑度、接触稳定性、task-specific failure risk。
- 相比只看离线 AUC/accuracy：我们必须报告 gradient usability，包括 score 梯度是否有限、是否非零、是否能在 protected DDPM step 中稳定提升预测质量分。

短期保持当前 260617-only DP 训练继续跑。训练完成或中途出现长期 plateau 后，再用同一真机 protocol 比较：

1. baseline DP，不加 TacQuality guidance；
2. 同一 DP + current board TacQuality guidance；
3. 如果真实力曲线显示过压/欠压明显，再用 260617 force trace 重新校准 board scorer。

## 9. 2026-06-19 09:30 CST epoch 50 保存点检查

当前训练继续正常运行：

- latest epoch：`50 / 2000`
- train loss：`0.013085`
- val loss：`0.016222`
- best val：`0.013598 @ epoch 43`
- trend warning：`healthy`
- GPU：约 `14.7 GB / 24.6 GB` 显存，利用率正常

保存链路检查：

- `dp_epoch50.pth` 已写出，大小约 `2.5 GB`
- `dp_latest.pth` 已在 epoch 50 附近刷新，大小约 `5.0 GB`
- `dp_best.pth` 保持 epoch 43 的 best checkpoint，大小约 `2.5 GB`
- `dp_topk_*.pth` 正常更新

判断：

- `save_freq=50` 的固定 checkpoint 保存链路已验证可用；
- `latest_freq=10` 的可恢复 checkpoint 保存链路已验证可用；
- 当前不需要中断或重启训练。

## 10. 2026-06-19 09:37 CST 中期趋势检查

当前训练继续正常运行：

- latest epoch：`59 / 2000`
- train loss：`0.011673`
- val loss：`0.013041`
- best val：`0.013041 @ epoch 59`
- trend warning：`healthy`

趋势判断：

- epoch 54、58、59 连续刷新或接近刷新 best val；
- train loss 下降的同时，val loss 也从 epoch 43 的 `0.013598` 进一步降到 `0.013041`；
- 这说明当前不是“train 继续降但 val 长期不降”的明显过拟合平台期；
- 继续训练是合理的。

保存链路复查：

- `dp_best.pth` 已在 epoch 59 附近刷新完整，大小约 `2.5 GB`；
- `dp_latest.pth` 在 epoch 60 附近刷新完整，大小约 `5.0 GB`；
- `dp_epoch50.pth` 保留正常。

## 11. 2026-06-19 10:06 CST epoch 100 保存点检查

当前训练继续正常运行：

- latest epoch：`102 / 2000`
- train loss：`0.009431`
- val loss：`0.014299`
- best val：`0.011385 @ epoch 94`
- trend warning：`healthy`

保存链路检查：

- `dp_epoch100.pth` 已写出完整，大小约 `2.5 GB`
- `dp_latest.pth` 已在 epoch 100 附近刷新完整，大小约 `5.0 GB`
- `dp_best.pth` 已在 epoch 94 附近刷新完整，大小约 `2.5 GB`
- `dp_epoch50.pth` 保留正常

趋势判断：

- epoch 82 之后 best val 又从 `0.011432` 小幅刷新到 `0.011385 @ epoch 94`；
- epoch 94 到 102 期间 val 有短期回升，但 monitor 仍判定 `healthy`；
- 当前不能判定强过拟合，只能说进入更慢的改善/波动区间；
- 继续训练，后续重点看 best 是否还能刷新，以及 `epochs_since_best` 是否长期增长。

## 12. 2026-06-19 10:38 CST epoch 150 平台期观察

当前训练继续运行：

- latest epoch：`150 / 2000`
- train loss：`0.007851`
- val loss：`0.015606`
- best val：`0.011385 @ epoch 94`
- trend warning：`watch_plateau_use_best_for_deploy`

保存链路：

- `dp_epoch150.pth` 已写出完整，大小约 `2.5 GB`
- `dp_latest.pth` 已刷新完整，大小约 `5.0 GB`
- `dp_best.pth` 仍是 epoch 94 的 best checkpoint

趋势判断：

- epoch 94 后已有 56 个 epoch 没刷新 best；
- train loss 继续下降，但 val loss 最近 tail20 均值约 `0.01536`，高于 best；
- 这说明模型进入平台期/轻微过拟合观察区；
- 这不是训练程序故障，但部署应明确优先使用 `dp_best.pth`；
- 由于用户要求训练充分，且还未达到强过拟合自动停止阈值，当前继续训练到更后面观察。

## 13. 2026-06-19 11:11 CST epoch 200 后停止训练

停止原因：

- epoch 94 后连续 108 个 epoch 没有刷新 best val；
- latest epoch：`202 / 2000`
- latest train loss：`0.006962`
- latest val loss：`0.019671`
- best val：`0.011385 @ epoch 94`
- latest val / best val：`1.7278`
- tail20 val mean：`0.01797035`
- trend warning：`strong_plateau_or_overfit_use_best`

处理：

- 已在 epoch 200 固定保存点写出完整 `dp_epoch200.pth`；
- 已确认 `dp_best.pth`、`dp_latest.pth`、`dp_epoch50.pth`、`dp_epoch100.pth`、`dp_epoch150.pth`、`dp_epoch200.pth` 均存在；
- 已停止训练 tmux 和 monitor tmux；
- GPU 已释放；
- 已写出：
  - `early_stop_summary.json`
  - `early_stop_summary.md`

最终部署建议：

- 真机测试默认使用：
  `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_full_noearly_tmux/dp_best.pth`
- 不建议使用 `dp_latest.pth` 做默认真机测试，除非明确要观察 late-overfit 行为。

同步更新：

- `docs/2026-06-18_tac_quality_guidance_readiness_matrix.md`
  - 当前推荐已对齐：
    - insertion：`good_margin_guided` / `good_margin`
    - board：`marker_joint_s12_guided` / `quality`
  - DP context 已切到当前 stopped run；
  - board/insertion real-rollout command packet 已补齐 expected arms 和 rollout root。

## 14. 2026-06-19 11:35 CST 最新论文复核与架构判断

复核方式：

- 用 arXiv API 和网页检索复核 2026-04-19 到 2026-06-19 之间与 tactile / force / diffusion policy / guidance 相关的论文。
- 这里记录的是对当前项目设计有直接影响的结论，不把尚未真机验证的内容写成我们自己的实验结论。

### 14.1 最相关的新工作

1. ViTaL: Inference-time Policy Steering via Vision and Touch, arXiv:2606.14981, 2026-06-12
   - 链接：https://arxiv.org/abs/2606.14981
   - 关键点：把 inference-time steering 做成高层视觉采样验证 + 低层 tactile-guided diffusion editing。
   - 对本项目的影响：这直接支持我们从 final clean-action refinement 继续推进到 denoising-step 内的 TacQuality 梯度引导。我们的路线不应停在 reranking 或最后一步小修正。

2. TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation, arXiv:2606.11184, 2026-06-09
   - 链接：https://arxiv.org/abs/2606.11184
   - 关键点：力/力矩提供全局高频接触线索，触觉提供局部接触形态，短时未来 tactile latent prediction 对 contact-rich manipulation 很关键。
   - 对本项目的影响：当前 Foresight 只用 action-conditioned tactile latent prediction 是合理骨架，但黑板任务后续应把 force/torque proxy 或 force-band proxy 纳入 Foresight/score 评估，而不是只看 marker latent MSE。

3. Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation, arXiv:2606.08737, 2026-06-07
   - 链接：https://arxiv.org/abs/2606.08737
   - 关键点：动作、未来视觉、未来触觉在统一 latent 中建模，并用 contact-gated fusion / contact-aware attention bias。
   - 对本项目的影响：我们现在是模块化版本：DP action prior + Foresight + TacQualityEnergy。相比端到端统一模型，优点是能插入现有 DP、可解释、便于 ablation；缺点是 Foresight 和 DP 之间可能存在分布错配。后续可以加 contact gate 和 predicted-contact confidence。

4. ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation, arXiv:2606.13877, 2026-06-11
   - 链接：https://arxiv.org/abs/2606.13877
   - 关键点：空间结构、时间连续性、跨模态兼容性比简单堆模态更重要；长时程 contact planning 中 tactile 更关键。
   - 对本项目的影响：TacQuality 评估不能只看分类准确率。必须继续保留 force-band、smoothness、contact stability、future trend、gradient audit。

5. Critic / gradient guidance for diffusion or flow policies
   - Sample-Efficient Diffusion-based RL with Critic Guidance, arXiv:2605.30056：https://arxiv.org/abs/2605.30056
   - Test-Time Gradient Guidance of Flow Policies in RL, arXiv:2606.11087：https://arxiv.org/abs/2606.11087
   - Fisher-Preserving Guidance, arXiv:2605.29937：https://arxiv.org/abs/2605.29937
   - 对本项目的影响：TacQualityEnergy 应被写成 differentiable critic / energy，而不是普通离线分类器。梯度必须受 trust-region / accept-only / contact gate 保护，避免把动作推离 DP prior 流形。

6. Ambient Diffusion Policy, arXiv:2606.12365, 2026-06-10
   - 链接：https://arxiv.org/abs/2606.12365
   - 关键点：次优数据不能简单和高质量数据等价混训；不同 diffusion time 对不同质量数据的使用方式不同。
   - 对本项目的影响：黑板负样本更适合训练 TacQualityEnergy / 数据权重 / guidance critic，而不应无区分地当 expert BC 数据混进 DP。若要混训 DP，应做 quality-weighted 或 diffusion-time-dependent data usage。

7. Long-context Diffusion Policy, arXiv:2606.16447, 2026-06-15
   - 链接：https://arxiv.org/abs/2606.16447
   - 关键点：长历史对需要记忆、避免重复失败动作、判断状态累积的任务有帮助。
   - 对本项目的影响：擦黑板的“力逐渐偏小/偏大/振荡”不是单帧性质。后续建议做 `obs_horizon=2/4/8`、tactile history `8/16`、Foresight horizon `16/32` 的受控对比。

### 14.2 对当前项目故事的收敛判断

当前最清晰、也最容易写成论文/专利故事的主线是：

```text
DP action prior
  -> action-conditioned tactile/force Foresight
  -> TacQualityEnergy differentiable critic
  -> contact-gated, trust-region, accept-only diffusion-step guidance
  -> real force-curve rollout evaluation
```

这个故事和最新工作一致，但要保留我们的区别：

- ViTaL 偏 inference-time steering / verification；我们要强调 TacQualityEnergy 通过 Foresight 对 action 产生可微梯度，不只是候选验证。
- TacForeSight / Dream-Tac 偏 tactile world/action model；我们要强调 scorer/energy 是任务质量标准，能把“力过大、力过小、不稳定、碰撞风险”等可解释目标注入 DP。
- ContactWorld 强调 representation；我们要把 marker field、force-band proxy、smoothness proxy、contact gate 作为结构化 contact representation，而不是黑盒 latent 分类。
- Guidance 类工作提醒必须保护 policy prior；我们当前的 trust-region、accept-only、contact gate 是必要模块，不是额外复杂化。

### 14.3 下一阶段最值得做的改进

P0：把 current TacQuality 从 final clean-action refinement 推到 denoising-step guidance。

- 当前已有 protected DDPM-step sweep 证据，但服务端主路径仍更接近 final clean-action refinement。
- 下一步应该实现真正的每个低噪声 step 调用：
  `action_t -> pred x0/action -> Foresight -> TacQuality -> grad -> constrained update -> scheduler step`
- 评估仍要看：
  - score delta 是否稳定非负；
  - action delta 是否在 trust region 内；
  - force curve 是否更接近目标 band；
  - 是否降低 too-low / too-high / oscillatory 比例。

P1：黑板任务加入 force-aware Foresight / score calibration。

- 当前 board scorer 已用 force-band 标准构造质量，但 Foresight 仍主要依赖 marker latent。
- 推荐增加：
  - force-band proxy head；
  - marker smoothness / delta head；
  - contact gate confidence；
  - predicted future quality consistency loss。
- 注意：这些是下一阶段建议，不应写成当前已经验证过的结果。

P2：DP 训练数据策略不要简单混负样本。

- 260617-only DP 的验证集在 epoch 94 后明显过拟合，说明小数据 DP 继续训练会记忆训练集。
- 后续黑板负样本应优先用于 TacQualityEnergy 和 guidance，而不是直接等价加入 BC。
- 若混入 DP，建议做：
  - 正样本主训练；
  - 负样本只在高噪声/低噪声 diffusion time 提供有限约束；
  - 或者按质量分数做 sample weight。

P3：长上下文 ablation。

- 当前 `obs_horizon=2`、tactile history=8 是可部署 baseline。
- 对擦黑板这种持续接触任务，后续应验证更长历史是否改善 force stability：
  - `obs_horizon=4/8`
  - tactile history=16
  - Foresight future horizon=32

当前结论：

- 260617-only DP 训练已经充分到出现强平台/过拟合信号，继续同配置训练不合理；
- 真机测试应使用 `dp_best.pth @ epoch 94`；
- current board guidance 应使用 `marker_joint_s12_guided`；
- current insertion guidance 应使用 `good_margin_guided`；
- 项目下一步创新重点不是再堆普通分类器，而是把 TacQualityEnergy 做成 contact-gated、force-aware、denoising-step 可微质量 critic，并用真实力曲线闭环验证。

## 15. 2026-06-19 12:10 CST 服务端 denoising-step guidance 实现与 smoke

本次把 TacQuality 从“只在 DP 采样结束后 refinement”推进到可选的“denoising-step 内 classifier guidance”：

```text
action_t
  -> DP noise_pred
  -> predicted clean action x0
  -> Foresight(action x0)
  -> TacQualityEnergy score
  -> grad wrt action_t
  -> unit-gradient trust-region update
  -> accept-only check
  -> scheduler step
```

代码位置：

- `for_show_xiaomi/serve_dp_tac_quality_guided.py`

新增参数：

- `--guidance_location final_action|denoising_step`
- `--ddpm_guidance_steps`
- `--ddpm_guidance_scale`
- `--ddpm_max_delta_norm`
- `--ddpm_sample_clip`
- `--disable_ddpm_accept_only`
- `--disable_ddpm_x0_clip`

默认仍为 `final_action`，所以当前稳定真机命令不受影响。只有显式加 `--guidance_location denoising_step` 才会进入新路径。

服务端命令表已补充实验块：

- board denoising-step guided：`for_show_xiaomi/guide_forshow.sh`，端口 `8768`
- insertion denoising-step guided：`for_show_xiaomi/guide_forshow.sh`，端口 `8788`

Smoke 结果：

1. Board / `marker_joint_s12_guided`
   - evidence：
     `/home/chenshuai/Project/output/tac_quality_guided_server_packet/board_s12_denoising_step_smoke_20260619/guided_server_dry_run_smoke.json`
   - pass：`true`
   - guidance location：`inside DP denoising loop on predicted clean action x0`
   - score delta：`+0.001405`
   - accept rate：`1.0`
   - finite grad rate：`1.0`
   - positive grad rate：`1.0`
   - update norm：`0.001`

2. Insertion / `good_margin_guided`
   - evidence：
     `/home/chenshuai/Project/output/tac_quality_guided_server_packet/insertion_good_margin_denoising_step_smoke_20260619/guided_server_dry_run_smoke.json`
   - pass：`true`
   - guidance location：`inside DP denoising loop on predicted clean action x0`
   - score delta：`+0.062713`
   - accept rate：`1.0`
   - finite grad rate：`1.0`
   - positive grad rate：`1.0`
   - update norm：`0.001`

解释：

- 这证明服务端已经具备真正 denoising-step TacQuality gradient guidance 的可运行路径；
- smoke 使用合成观测/合成 Foresight，只验证代码路径、梯度链路、accept-only 和输出形状，不等价于真实机器人效果；
- 真机默认仍建议先跑 final-action baseline/guided，再用 denoising-step 端口做小规模 A/B；
- 真实结论仍需要 server-side force traces 和 paired rollout evaluation。
