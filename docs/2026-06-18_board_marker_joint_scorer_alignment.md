# 2026-06-18 Board Marker-Joint TacQuality Scorer Alignment

## 背景

目标是为擦黑板 DP classifier guidance 找到一个能同时满足三点的触觉质量评分器：

1. 离线 good/bad 和 failure reason 分类准确；
2. 部署链路中 `score(Foresight(action))` 有有效动态范围；
3. 对 action 有稳定、有限、受 trust region 限制的梯度。

之前的 predicted-domain ForceBand scorer 在缓存特征上指标很高，但进入真实 Foresight 链路后分数几乎饱和：

| old score mode | pred AUC | pred-GT Spearman | force-band Spearman | 现象 |
|---|---:|---:|---:|---|
| energy_clipped | 0.5271 | 0.6022 | -0.0438 | 类别均值约 `3.395`，动态范围极小 |
| profile | 0.5187 | 0.6086 | -0.0254 | 类别均值约 `5.45`，动态范围极小 |
| quality | 0.4174 | 0.5996 | 0.0359 | 不适合引导 |

## 发现的问题

旧 predicted-domain feature cache 使用：

```text
predicted marker proxy + joint action proxy + dataset eef_abs proxy
```

但真实 serving/guidance contract 是：

```text
DP joint action chunk
  -> Foresight predicts marker
  -> scorer sees predicted marker + candidate joint action
```

服务端的 `ForesightTacQualityBridge` 只能从 candidate joint action 中派生 `eef_action_seq = action_raw[..., :6]`，并没有未来真实 `actions/eef_abs`。因此旧 scorer 在训练/验证时使用了部署时不可获得的 `eef_abs` 特征，造成 train/eval cache 很强但 Foresight guidance 链路分数饱和。

## 改动

新增部署一致特征：

```text
marker_joint_action = predicted marker proxy(54) + joint action proxy(10)
left_marker_joint_action = predicted left marker proxy(18) + joint action proxy(10)
```

代码改动：

- `TFAC_V5/tac_quality_energy/add_board_deploy_feature_variants.py`
  - 从已有 predicted-domain cache 派生 joint-only deploy feature，不重新跑 Foresight。
- `TFAC_V5/tac_quality_energy/build_board_predicted_domain_features.py`
  - 后续新建 cache 时直接保存 `marker_joint_action` 和 `left_marker_joint_action`。
- `TFAC_V5/tac_quality_energy/force_band_runtime.py`
  - runtime 支持 `marker_joint_action` / `left_marker_joint_action` checkpoint。
- `TFAC_V5/tac_quality_energy/train_board_force_band_energy.py`
  - 训练脚本允许选择新的 deploy feature variant。
- `TFAC_V5/tac_quality_energy/eval_foresight_score_alignment.py`
  - 默认采样限定为擦拭阶段 `phase_start_frac=0.25`, `phase_end_frac=0.85`；
  - scorer 调用显式传入 serving-style `eef_action_seq=action[..., :6]` 和 `joint_action_seq=action`；
  - 输出 JSON 记录 phase 采样范围。

## 新特征 Cache

输入：

```text
/home/chenshuai/Project/output/board_predicted_domain_force_band_features_20260618/board_predicted_domain_force_band_features.npz
```

输出：

```text
/home/chenshuai/Project/output/board_predicted_domain_force_band_features_20260618_deploy/board_predicted_domain_force_band_features_deploy.npz
```

新增维度：

| feature | shape |
|---|---:|
| marker_joint_action | 1800 x 64 |
| left_marker_joint_action | 1800 x 28 |
| gt_marker_joint_action | 1800 x 64 |

## 新 Scorer 训练结果

训练命令：

```bash
conda run --no-capture-output -n TactileACT python TFAC_V5/tac_quality_energy/train_board_force_band_energy.py \
  --features /home/chenshuai/Project/output/board_predicted_domain_force_band_features_20260618_deploy/board_predicted_domain_force_band_features_deploy.npz \
  --output_dir /home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618 \
  --feature_variant marker_joint_action \
  --device cpu \
  --epochs 300 \
  --min_epochs 50 \
  --patience 40 \
  --batch_size 128 \
  --hidden 192 \
  --log_interval 10
```

Held-out GroupKFold validation:

| metric | value |
|---|---:|
| best epoch | 51 |
| binary AUC | 0.9997 |
| balanced accuracy | 0.9828 |
| binary macro F1 | 0.9803 |
| reason macro F1 | 0.9703 |
| quality Spearman | 0.9239 |
| energy-quality Spearman | 0.8414 |

Gradient smoke:

| metric | value |
|---|---:|
| pass | true |
| marker grad norm | 0.6465 |
| action grad norm | 0.6832 |

Checkpoint:

```text
/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/force_band_tac_quality_energy_best.pt
```

## Foresight Alignment

数据：

- old positive: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/wipe_pos_straight_z124_125_150_20260609`
- 260617 positive: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617`
- too small: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/z_too_high`
- too large: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_low`
- oscillate: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_oscillate`

采样：

- contact-only
- `phase_start_frac=0.25`
- `phase_end_frac=0.85`
- `n=120`

Score-mode sweep:

| mode | pred AUC | GT AUC | pred-GT Spearman | force-band Spearman | force-mag-band Spearman | force-delta Spearman |
|---|---:|---:|---:|---:|---:|---:|
| energy_clipped | 0.9994 | 0.8926 | 0.6540 | 0.1759 | 0.2053 | -0.3232 |
| profile | 0.9994 | 0.8855 | 0.6316 | 0.1907 | 0.2184 | -0.3096 |
| quality | 0.9991 | 0.8986 | 0.6130 | 0.4733 | 0.4545 | 0.1005 |
| p_good | 0.9994 | 0.8770 | 0.6723 | 0.0970 | 0.1438 | -0.4138 |
| reason_good | 0.9994 | 0.8727 | 0.6670 | 0.0959 | 0.1356 | -0.4110 |

按 label 的 `quality` score 均值：

| label | pred score mean |
|---|---:|
| positive_260617 | 0.8932 |
| positive | 0.8389 |
| too_small | 0.2103 |
| oscillate | 0.1784 |
| too_large | 0.1053 |

解释：

- 新 scorer 解决了旧 scorer 在 Foresight 链路中 score 饱和的问题；
- `energy/profile/p_good/reason_good` 最适合做 good/bad 分类；
- `quality` 模式和 force-band physical quality 的相关性最好，更适合作为擦黑板连续质量引导；
- 当前推荐擦黑板 guidance 使用 `score_mode=quality`，而不是默认 `energy_clipped/profile`。

## Guidance Gradient Audit

命令：

```bash
conda run --no-capture-output -n TactileACT python TFAC_V5/tac_quality_energy/eval_guidance_gradient_audit.py \
  --task board \
  --arm default_guided \
  --dataset_dir /media/chenshuai/EXTERNAL_USB/pih_dataset/260609/wipe_pos_straight_z124_125_150_20260609 \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload/foresight_best.ckpt \
  --output_dir /home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/guidance_gradient_audit_quality \
  --scorer_runtime ForceBandTacQualityEnergyRuntime \
  --scorer_checkpoint /home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/force_band_tac_quality_energy_best.pt \
  --score_mode quality \
  --max_episodes 8 \
  --samples_per_episode 4 \
  --max_samples 24 \
  --gpu -1
```

结果：

| metric | value |
|---|---:|
| pass | true |
| finite grad rate | 1.0000 |
| positive grad rate | 1.0000 |
| accept rate | 1.0000 |
| improved rate | 1.0000 |
| trust-region pass rate | 1.0000 |
| score delta mean | 0.001320 |
| action delta norm mean | 0.000791 |

## 当前结论

当前最合理的擦黑板 scorer 是：

```text
Board ForceBand TacQualityEnergy, feature_variant=marker_joint_action, score_mode=quality
```

它满足：

- episode-level held-out 分类/排序强；
- Foresight 链路中不饱和；
- `score(Foresight(action))` 与 force-band quality 有中等正相关；
- 对 action 有稳定可微梯度，并且 trust-region 约束有效。

## 证据边界

- 这仍是离线链路验证，不是真机 rollout 结果。
- 不能声称它已经让真实擦黑板力曲线更好。
- 下一步需要在 guided server 里接入该 checkpoint 和 `score_mode=quality`，用 server-side force curves 比较 baseline vs guided 的 contact-phase `Fz_mean/Fz_p95/|dFz|/marker smoothness`。
