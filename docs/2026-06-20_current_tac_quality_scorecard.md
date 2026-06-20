# Current TacQuality Scorecard

Generated: `2026-06-20T10:46:57`

## Evidence Levels

| level | status |
|---|---:|
| `offline_scorer_ready` | `True` |
| `gradient_guidance_ready` | `True` |
| `real_evidence_pipeline_ready` | `True` |
| `real_paired_rollout_complete` | `False` |
| `goal_complete` | `False` |

## Current Recommended Scorers

| task | arm | runtime | score mode | ready for real rollout |
|---|---|---|---|---:|
| insertion | `good_margin_guided` | `InsertionRiskScorerRuntime` | `good_margin` | `True` |
| board | `marker_joint_s12_guided` | `ForceBandTacQualityEnergyRuntime` | `quality` | `True` |

## Key Metrics

| task | classifier metric | quality metric | Foresight/guidance metric |
|---|---|---|---|
| insertion | AUC `0.9877`, bACC `0.9437` | corr `0.7656` | 0401 improve `1.0000`, good-margin improve `0.9375` |
| board | AUC `1.0000`, bACC `1.0000` | rho `0.9239` | pred-vs-GT rho `0.5291`, guidance improve `1.0000` |

## Real Rollout Coverage

- planned_counts: `{"board": {"baseline": 3, "guided": 3}, "insertion": {"baseline": 3, "guided": 3}}`
- observed_counts: `{}`
- status_counts: `{"missing": 12}`
- real_rollout_evidence_complete: `False`

## Board DP Checkpoint Policy

- recommended: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_best.pth`
- avoid as default: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_final.pth`
- best val epoch/loss: `155` / `0.011659`
- final epoch/val loss: `2000` / `0.038709`

## Innovation Story

- Do not present tactile concat DP as the main novelty; treat it as the action prior.
- Main novelty: differentiable tactile/force consequence scoring for DP classifier guidance.
- Insertion uses an unsaturated good-margin risk scorer over good insert vs pre-bounce/impact modes.
- Board uses a four-class force-band quality energy covering proper, too-small, too-large, and oscillatory contact.
- Both tasks use bounded trust-region guidance through Foresight rather than offline reranking.

## Evidence Boundary

Can claim now:
- Offline scorer quality is strong for both tasks.
- Foresight-gradient guidance path is ready for real rollout tests.
- The command and manifest pipeline is ready for paired real robot evidence collection.

Cannot claim yet:
- No real paired baseline-vs-guided improvement has been proven.
- No board force-curve improvement should be claimed without force_trace.csv pairs.
- No insertion success/bounce/retry improvement should be claimed without outcome metadata.

## Next Required Evidence

- Run board block 1 vs block 2 in guide_forshow.sh and collect server-side force_trace.csv.
- Run insertion block 3 vs block 4 and fill success/stopped_early/bounce_count/retry_count metadata.
- Rerun audit_real_rollout_coverage.py until both tasks have at least three complete pairs.
- Only then run eval_tac_quality_real_rollouts.py for final real robot evidence.

## Input Paths

- scorer_audit: `/home/chenshuai/Project/output/tac_quality_current_scorer_audit/current_tac_quality_scorer_audit.json`
- guidance_state: `/home/chenshuai/Project/output/tac_quality_guidance_state_audit/tac_quality_guidance_state_audit.json`
- coverage: `/home/chenshuai/Project/output/tac_quality_real_rollout_coverage/current_s12_good_margin_coverage/tac_quality_real_rollout_coverage.json`
- dp_status: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/training_status_latest.json`
- rollout_config: `/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_20260619.json`
