# TacQuality Guidance Gap Audit

Generated: `2026-06-20T14:34:59`

## Current State

| item | value |
|---|---:|
| offline_scorer_ready | `True` |
| gradient_guidance_ready | `True` |
| server_rollout_schema_ready | `True` |
| real_paired_rollout_complete | `False` |
| missing real rollout records | `12` |

## Board

| metric | value |
|---|---:|
| clean-action score_delta mean | `0.00016205` |
| clean-action action_delta_norm mean | `0.00077498` |
| Foresight pred-vs-GT score Spearman | `0.5291` |
| pred score vs force-band quality Spearman | `0.3988` |
| force-band quality AUC good | `0.4953` |

| gap | risk | evidence | next action |
|---|---|---|---|
| real paired board force evidence missing | `high` | rollout_status_counts={'missing': 12} | Run paired baseline/guided board rollouts and evaluate force_trace.csv. |
| board clean-action guidance magnitude is small | `medium` | score_delta_mean=0.00016205, action_delta_norm_mean=0.00077498 | Sweep board score modes / step sizes inside DDPM-step guidance and require nontrivial score/action deltas under trust-region limits. |
| board force-band continuous alignment is only moderate | `medium` | pred_score_vs_force_band_quality_spearman=0.3988, pred_gt_spearman=0.5291 | Train or audit a force-aware Foresight head or force-proxy head so the score targets force-band quality more directly. |
| force-band quality alone does not explain good-label AUC | `medium` | force_band_quality_auc_good=0.4953 | Keep semantic labels and force metrics separate in the paper; do not claim force-band metric alone defines board success. |

## Insertion

| metric | value |
|---|---:|
| matched 0401 score_delta mean | `0.1401` |
| matched 0401 action_delta_norm mean | `0.0799` |

| gap | risk | evidence | next action |
|---|---|---|---|
| real paired insertion success/bounce evidence missing | `high` | rollout_status_counts={'missing': 12} | Run paired baseline/good_margin_guided insertion rollouts with success, bounce_count, retry_count, and stopped_early metadata. |
| probability heads saturate; use logit margin for guidance | `low` | p_good/log_p_good deltas are zero in score-mode ablation; good_margin improve_rate=0.9375. | Keep good_margin as default and treat p_good as a reporting metric, not guidance objective. |
| insertion reason classifier is weaker than binary classifier | `medium` | reason_macro_f1=0.7894, binary_auc=0.9877 | For paper claims, emphasize good-vs-risk guidance; use reason labels mainly for interpretation unless reason F1 improves. |

## Priority Experiments

| rank | experiment | why | success criterion |
|---:|---|---|---|
| 1 | paired real rollout evaluation | It is the only missing evidence level and blocks the final claim. | At least 3 complete baseline/guided pairs per task; board force metrics and insertion outcomes improve without safety regressions. |
| 2 | board DDPM-step guidance sweep with stronger but bounded settings | Current board gradient is finite but very small in clean-action audit. | Positive score delta with meaningful action_delta_norm, trust_region_pass_rate>=0.999, and no final-score regression after accept filtering. |
| 3 | force-aware board Foresight / force-proxy scorer | Board quality is physically force-band based, but current runtime relies on marker/action proxies and marker-only Foresight. | Improve pred_score_vs_force_band_quality_spearman beyond 0.50 and preserve held-out episode-level classification. |
| 4 | insertion reason-head refinement | Binary guidance is strong; reason labels are less reliable but useful for interpretability. | Improve GroupKFold reason_macro_f1 without reducing binary AUC or good_margin gradient quality. |

## Claim Boundary

Can claim now:
- The current offline task-specific scorers are strong enough for controlled real rollout tests.
- The current Foresight-to-score-to-action gradient path is finite and trust-region bounded.
- The implementation path is gradient guidance, not offline reranking.

Cannot claim yet:
- Real robot improvement over baseline has not been proven.
- Board force-curve improvement has not been proven.
- Insertion success/bounce/retry improvement has not been proven.
- The board scorer is not yet a direct force-prediction scorer; it is a marker/action proxy scorer trained from force-band labels.

## Evidence Paths
- scorer_audit: `/home/chenshuai/Project/output/tac_quality_current_scorer_audit/current_tac_quality_scorer_audit.json`
- scorecard: `/home/chenshuai/Project/output/tac_quality_current_scorecard/current_tac_quality_scorecard.json`
- evidence_bundle: `/home/chenshuai/Project/output/tac_quality_evidence_bundle/current_tac_quality_evidence_bundle.json`
- rollout_config: `/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_20260619.json`
