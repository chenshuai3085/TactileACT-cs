# 2026-06-18 插孔任务 TacQuality 梯度引导链路验证

## 目的

验证插孔任务当前 scorer 是否真的能用于 DP classifier guidance / energy guidance：

```text
DP clean action
  -> insertion Foresight(global, wrist, gelsight marker)
  -> predicted future tactile marker
  -> InsertionRiskScorerRuntime score
  -> d score / d action
  -> trust-region action update
```

这不是 reranking，也不是真机 rollout 指标。它只验证“预测触觉后果评分能否把可用梯度传回 action”。

## 代码修复

文件：

- `TFAC_V5/tac_quality_energy/eval_guidance_gradient_audit.py`

修复内容：

- 原 audit 脚本只给 Foresight 传入 `gelsight marker_window`。
- 插孔 Foresight `latent_foresight_full` 的 `camera_names=["global","wrist","gelsight"]`，需要真实视觉输入。
- 已新增：
  - ImageNet 图像预处理；
  - 按 `fs_cfg["camera_names"]` 自动加载 `observations/images/global` 和 `observations/images/wrist`；
  - 按 Foresight/TactileVAE checkpoint 的 marker norm stats 归一化 marker；
  - 将 `global/wrist` tensor 和 `gelsight marker_window` 一起传入 `ForesightTacQualityBridge`。
- `profile` score mode 在当前服务端逻辑中，如果 runtime 没有 `weighted_energy_score`，会退回 `energy_clipped`。audit 已保持同样语义。

## 实验设置

数据：

- `/home/chenshuai/data/dataset/0414`

Foresight：

- `/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full`
- checkpoint: `foresight_best.ckpt`
- 输入：`global`, `wrist`, `gelsight`

Scorer：

- `InsertionRiskScorerRuntime`
- checkpoint: `/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt`

Refiner：

- `refine_steps=4`
- `action_step=0.02`
- `max_total_delta=0.08`
- `accept_only_improved=true`

## 结果

### 默认服务端路径：profile

输出：

- `/home/chenshuai/Project/output/insertion_guidance_gradient_audit_real_foresight_profile_20260618`

结果：

| metric | value |
|---|---:|
| samples | 24 |
| pass | true |
| finite grad rate | 1.0000 |
| positive grad rate | 1.0000 |
| accept rate | 0.9688 |
| improved rate | 1.0000 |
| score delta mean | 0.266473 |
| action delta norm mean | 0.074925 |
| trust-region pass rate | 1.0000 |

说明：

- 这是当前 `default_guided` 插孔配置的默认路径。
- 对 `InsertionRiskScorerRuntime` 来说，服务端 `profile` fallback 到 `energy_clipped`。
- 结果证明：真实 `global/wrist/gelsight` Foresight 链路下，score 对 action 有有限且非零的梯度，trust-region 更新后 score 稳定提升。

### risk_guidance 小样本对比

输出：

- `/home/chenshuai/Project/output/insertion_guidance_gradient_audit_real_foresight_risk_guidance_small_20260618`

结果：

| metric | value |
|---|---:|
| samples | 8 |
| pass | true |
| finite grad rate | 1.0000 |
| positive grad rate | 1.0000 |
| accept rate | 1.0000 |
| improved rate | 1.0000 |
| score delta mean | 0.297445 |
| action delta norm mean | 0.074835 |
| trust-region pass rate | 1.0000 |

说明：

- `risk_guidance = quality_score + 0.35*log_p_good - 0.5*risk_prob`。
- 小样本也通过，说明它可以作为后续插孔 score mode ablation。
- 但当前默认配置仍保持 `profile/energy_clipped`，因为它已有 24 样本链路验证和旧 full-chain evidence。

### smoke

输出：

- `/home/chenshuai/Project/output/insertion_guidance_gradient_audit_real_foresight_smoke_20260618`

结果：

| metric | value |
|---|---:|
| samples | 4 |
| pass | true |
| finite grad rate | 1.0000 |
| positive grad rate | 1.0000 |
| accept rate | 1.0000 |
| improved rate | 1.0000 |
| score delta mean | 0.388338 |
| action delta norm mean | 0.076821 |

## 当前结论

- 插孔任务当前 `InsertionRiskScorerRuntime + latent_foresight_full + trust-region action update` 可以形成真实可微的 action guidance 链路。
- 当前默认插孔 guidance 候选保持：
  - scorer: `InsertionRiskScorerRuntime`
  - score mode: `profile` in config, runtime fallback equals `energy_clipped`
  - refiner: final clean action trust-region refinement
- `risk_guidance` 是有潜力的 ablation score mode，但还需要更多样本和真实 rollout 对比后才能替换默认。

## 证据边界

- 这证明的是离线梯度链路可用。
- 这不证明真机插孔成功率提升。
- 这不证明擦黑板任务力曲线改善。
- 真正上线结论仍需要：
  - baseline vs guided 真机 rollout；
  - server-side force/marker/action 轨迹记录；
  - contact-phase 指标对比。
