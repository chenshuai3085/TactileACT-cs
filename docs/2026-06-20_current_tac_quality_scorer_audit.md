# Current TacQuality Scorer Audit

Generated: `2026-06-20T12:28:30`

## Current Recommendation

| task | arm | runtime | score mode | role |
|---|---|---|---|---|
| insertion | `good_margin_guided` | `InsertionRiskScorerRuntime` | `good_margin` | unsaturated good-vs-risk logit margin |
| board | `marker_joint_s12_guided` | `ForceBandTacQualityEnergyRuntime` | `quality` | four-class force-band tactile quality |

This replaces the older interpretation that the distilled/manual-board scorer is the main board candidate.  The distilled scorer remains an ablation; the current board default is the four-class ForceBand scorer.

## Pass Summary

| category | pass |
|---|---:|
| insertion_offline | `True` |
| insertion_gradient | `True` |
| board_offline | `True` |
| board_foresight | `True` |
| board_gradient | `True` |
| overall offline guidance ready | `True` |
| production validated by real rollouts | `False` |

## Key Metrics

| task | binary AUC | bACC | reason F1 | quality metric | gradient evidence |
|---|---:|---:|---:|---:|---|
| insertion | 0.9877 | 0.9437 | 0.7894 | corr 0.7656 | 0209 improve 0.9167, 0401 improve 1.0000 |
| board | 1.0000 | 1.0000 | 1.0000 | rho 0.9239 | improve 1.0000, finite 1.0000 |

## Insertion Good-Margin Cross-Score Ablation

- rows: `32`
- final accept rate: `1.0000`
- guided action delta norm mean: `0.00010238`
- own good-margin delta mean: `0.00115377`
- good-margin improve rate: `0.9375`
- profile delta mean: `0.00009344`
- energy delta mean: `0.00039385`
- quality-logit delta mean/min: `0.00055696` / `0.00000000`

## Board Four-Class Coverage

| label | count | quality mean |
|---|---:|---:|
| oscillate | 492 | 0.4846 |
| positive | 1200 | 0.6398 |
| too_large | 480 | 0.3639 |
| too_small | 480 | 0.5560 |

Board labels:

- `positive`: proper contact force band and smooth wiping.
- `too_small`: pressure too small / insufficient wiping.
- `too_large`: pressure too large.
- `oscillate`: unstable force/contact transition.

## Foresight-Chain Board Alignment

- pred AUC good: `1.0000`
- GT AUC good: `0.8513`
- predicted-vs-GT score Spearman: `0.5291`
- predicted score vs force-band quality Spearman: `0.3988`

## Evidence Boundary

- Offline scorer and Foresight-gradient evidence are ready for controlled real rollout tests.
- No formal paired baseline-vs-guided robot rollout gate is present here.
- Do not claim real success-rate, bounce-rate, or wiping-force improvement until paired robot logs are evaluated.

## Next Required Evidence

- Insertion paired baseline vs good_margin_guided rollouts with success/bounce/retry metadata.
- Board paired baseline vs marker_joint_s12_guided rollouts with server-side force curves.
- Use the same DP checkpoint per task when comparing guided vs unguided.

## Evidence Paths

- insertion_eval: `/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_eval.json`
- insertion_score_ablation: `/home/chenshuai/Project/output/tac_quality_score_mode_ablation/insertion_0401_profile_pgood_energy_goodmargin_cross_score_20260619/insertion_score_mode_ablation.json`
- insertion_grad_0209: `/home/chenshuai/Project/output/insertion_guidance_gradient_audit_0209_matched_20260619/guidance_gradient_audit.json`
- insertion_grad_0401: `/home/chenshuai/Project/output/insertion_guidance_gradient_audit_0401_matched_20260619/guidance_gradient_audit.json`
- board_eval: `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/train_result.json`
- board_alignment: `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/foresight_alignment_quality/foresight_score_alignment.json`
- board_grad: `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/guidance_gradient_audit_quality/guidance_gradient_audit.json`
- board_features: `/home/chenshuai/Project/output/tac_quality_board_force_band_eval_mlp/board_force_band_scorer_eval.json`
- rollout_config: `/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_20260619.json`
