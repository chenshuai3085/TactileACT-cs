# 2026-06-21 260617-only DP 训练核验与近期论文调研

## 训练核验

用户要求：先只用 `/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban` 训练擦黑板 DP，按 2000 epoch 训练，并监督过程。

实际核验结果：同一数据、同一目标配置的训练已经在本机完成，不重复启动浪费 GPU。

补充核验：本地存在两个完整 2000 epoch 的 260617-only 稳定训练 run。二者都只使用
`peg_in_hole_0617`，但随机种子不同。按 episode-level validation loss 选择模型时，
`20260619_stable_fullwindow_slowlr` 更好，应作为当前优先候选。

最近一次已完成 run：

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun
```

当前推荐 run：

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr
```

关键配置：

- 数据：`/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617`
- episode：80 个文件，其中训练脚本有效计入 79 个；`episode_1.hdf5` 缺少 `observations/proprio_joint`
- episode split：训练 72，验证 8
- 图像输入：`global,wrist`，raw `200x266`，不 resize/crop 改尺寸
- tactile：left `marker_offset`，history 8
- tactile encoder：`/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt`
- DP：`train_dp_tac_concat.py` concat 版本
- horizon：`pred_horizon=16`，`obs_horizon=2`，`n_action_steps=8`
- 训练：`epochs=2000`，`batch_size=64`，`lr=5e-5`，`weight_decay=1e-5`
- 验证：episode-level split，`val_ratio=0.1`，`val_interval=5`
- 保存：`save_freq=50`，`latest_freq=10`

训练状态：

- 已跑满：`2000/2000`
- GPU 当前空闲
- loss 曲线：
  - `loss_curve.png`
  - `loss_curve.csv`
- 状态文件：
  - `training_status_latest.json`
  - `metrics.json`

最近一次 run 关键指标：

| epoch | train loss | val loss |
|---:|---:|---:|
| 50 | 0.013188 | 0.015548 |
| 85 | 0.011576 | **0.014062** |
| 100 | 0.011680 | 0.015453 |
| 500 | 0.005230 | 0.032898 |
| 1000 | 0.003301 | 0.047194 |
| 1500 | 0.002599 | 0.055993 |
| 2000 | 0.002152 | 0.058560 |

判断：

1. 训练 loss 持续下降，说明模型继续拟合训练集。
2. 验证 loss 在 epoch 85 最好，之后长期升高，epoch 2000 已约为 best 的 4.16 倍。
3. 这是明确的 episode-level 验证过拟合信号，不应把 `dp_final.pth` 当作最佳部署模型。

推荐 checkpoint：

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_best.pth
```

推荐依据：

| run | best epoch | best val loss | final val loss | 备注 |
|---|---:|---:|---:|---|
| `20260619_stable_fullwindow_slowlr` | 155 | **0.011659** | 0.038709 | 当前推荐 |
| `20260620_rerun` | 85 | 0.014062 | 0.058560 | 已完成，但验证集略差 |
| home 旧 run | 105 | 0.011152 | 0.041854 at epoch 830 | 被监控提前停在 830，不是完整 2000 epoch run |

解释：

1. 如果严格要求“跑满 2000 epoch”，优先用 `20260619_stable_fullwindow_slowlr/dp_best.pth`。
2. 如果只看目前见过的最低 validation loss，home 旧 run 的 epoch 105 略低，但它在 epoch 830
   被监控提前停止，不是这次要求的完整 2000 epoch 结果。
3. 所有 260617-only run 都显示后期 train loss 继续下降、val loss 升高，所以不能使用
   `dp_final.pth` 或 train-loss top-k ckpt 作为默认部署模型。

保留但不推荐作为默认部署：

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_final.pth
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_epoch2000.pth
```

证据边界：

- 这是离线训练/验证核验。
- 没有新增真实机器人擦拭 rollout。
- 真机效果仍需 baseline/guided paired rollout 和 force trace 评估。

## 近期 arXiv 调研

调研时间：2026-06-21。筛选范围优先看最近约两个月内与触觉、力、contact-rich manipulation、diffusion policy、guidance/steering/world model 直接相关的 arXiv 工作。

高相关论文：

1. Tube Diffusion Policy: Reactive Visual-Tactile Policy Learning for Contact-rich Manipulation, arXiv:2604.23609, 2026-04-26.
   - 重点：批评 action chunking 在 contact-rich 场景反应慢，提出更 reactive 的 visual-tactile diffusion policy。
   - 对本项目启发：当前 `action_horizon=8` 和 chunk execution 可能限制触觉反馈闭环速度；真机部署时应比较 `action_horizon=4/6/8` 或更频繁 replan。

2. TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance, arXiv:2601.20239, updated 2026-05-13.
   - 重点：预训练视觉策略先给 coarse action，触觉/物理模型在推理时 steering。
   - 对本项目启发：我们方向应继续保持 inference-time gradient guidance，不是 reranking；分类器/评分器要对 action 可微，或经 foresight 把 action 映射到未来触觉再打分。

3. Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation, arXiv:2606.08737, 2026-06-07.
   - 重点：联合建模 action、未来视觉、触觉动态；contact-gated fusion 和 contact-aware attention。
   - 对本项目启发：现在的 Foresight 只预测 tactile latent，后续可加入 contact gate，让模型在接触阶段更依赖触觉/力，非接触阶段更依赖视觉/状态。

4. TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation, arXiv:2606.11184, 2026-06-09.
   - 重点：force-conditioned tactile world model，强调全局 force 与局部 tactile 的不对称作用。
   - 对本项目启发：擦黑板评分器应保留 force-aware 分支：力大小 band、力变化平滑性、marker 接触面积/形变共同作为质量能量。

5. Inference-time Policy Steering via Vision and Touch, arXiv:2606.14981, 2026-06-12.
   - 重点：visuo-tactile inference-time steering，把候选动作验证/引导作为双层优化。
   - 对本项目启发：我们的创新故事可以表述为“触觉后果预测 + 质量能量的可微推理时 steering”，但需要强调不是简单候选重排，而是作用在 denoising 梯度上。

6. ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation, arXiv:2606.13877, 2026-06-11.
   - 重点：系统研究 vision-tactile world model 的 representation properties。
   - 对本项目启发：评估不应只看单步 MSE，应加入 contact phase、force band、smoothness、future tactile separability 这类任务相关指标。

7. Set-Supervised Diffusion Policy: Learning Action-Chunking Diffusion through Corrections, arXiv:2606.01865, 2026-06-01.
   - 重点：利用 undesired action 与 corrective action 的配对监督，不只拟合正样本。
   - 对本项目启发：插座 bounce 和黑板 too-small/too-large/oscillate 都是有价值的负信号；未来可把“坏 action set”转成能量 margin loss，而不是只删掉坏数据。

8. Multi-Resolution Tactile Imitation Learning for Contact-Rich Robotic Manipulation, arXiv:2606.06281, 2026-06-04.
   - 重点：多时间分辨率 tactile fusion。
   - 对本项目启发：擦黑板质量既看瞬时力大小，又看一段时间内平滑性；评分器和 foresight 应保留短窗/长窗两种 temporal feature。

9. IMPACT: Learning Internal-Model Predictive Control for Forceful Robotic Manipulation, arXiv:2606.10818, 2026-06-09.
   - 重点：forceful manipulation 中显式建模 force/torque 与控制。
   - 对本项目启发：擦黑板任务不能只学 joint action BC，至少评估层必须看 force curve；更进一步可以在 action 表示中考虑 impedance/force-aware residual。

10. FTP-1 / T-Rex / HapTile / TacCoRL 等 2026-06 触觉基础模型与数据工作。
    - 重点：触觉数据规模、跨传感器 token 化、触觉反应式策略。
    - 对本项目启发：目前最现实的短期改进不是直接换 VLA，而是把本项目的 marker/force 表征做成可复用 tactile token，并做跨任务 scorer。

## 对本项目架构和故事的建议

当前最合理主线：

```text
DP 生成动作
  -> Foresight 预测未来触觉/力后果
  -> TacQualityEnergy 评估接触质量
  -> 对 denoising action 做 classifier/energy guidance
```

建议保留的核心点：

1. 不做 reranking 主线。
   - reranking 可以作为审计/可视化辅助，但论文和实现主线应是 gradient guidance。

2. 评分器要从“分类正确”升级为“可引导”。
   - 离线 AUC/F1 只能证明能分好坏。
   - 还必须评估 guidance gradient 是否把 action 推向更低风险/更高质量区域。

3. 擦黑板任务的质量标准应保持多头/多指标：
   - force magnitude band：过小/合适/过大；
   - force smoothness：变化是否平滑；
   - marker contact proxy：接触面积、形变强度、中心稳定性；
   - phase awareness：只在 wiping/contact 阶段强约束。

4. 插座任务的坏样本定义仍然清晰：
   - pre-bounce/bounce 是负；
   - stable insertion 是正；
   - 评分形式优先用 good-bad margin logit，而不是饱和概率。

5. 训练 DP 时，260617-only 当前已有过拟合。
   - 单靠 2000 epoch 不提升泛化。
   - 下一步要么选 `dp_best.pth` 做真机对比，要么扩充/混合更多正负擦黑板数据，而不是继续拉长 epoch。

短期可执行改进：

1. 使用 `dp_best.pth` 跑真机 paired baseline/guided，保存 force trace。
2. 对擦黑板 score/guidance 做真实指标：
   - Fz mean 是否进入目标 band；
   - Fz std / jerk 是否降低；
   - 接触中断比例是否降低；
   - 轨迹完成覆盖是否保持。
3. 在 foresight/scorer 评估里加 contact-phase-only 指标，避免 approach 阶段稀释结果。
4. 做 `action_horizon` ablation：8 vs 4/6，验证更 reactive 是否改善擦黑板接触稳定。
5. 若要进一步创新，优先做 force-aware tactile foresight + energy guidance，而不是单纯换更大的 DP。
