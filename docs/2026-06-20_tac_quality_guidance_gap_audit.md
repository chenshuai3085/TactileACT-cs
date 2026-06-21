# TacQuality Guidance Gap Audit

Generated: `2026-06-21T11:58:35`

## Current State

| item | value |
|---|---:|
| offline_scorer_ready | `pass` |
| gradient_guidance_ready | `pass` |
| server_rollout_schema_ready | `pass` |
| insertion_config_consistent | `pass` |
| force_aware_board_config_consistent | `pass` |
| real_paired_rollout_complete | `missing` |
| goal_complete | `missing` |
| main missing real rollout records | `12` |
| force-aware board missing real rollout records | `6` |

## Insertion Good-Margin

| metric | value |
|---|---:|
| guidance signal strong | `pass` |
| matched/offline score_delta mean | `0.1401` |
| matched/offline action_delta_norm mean | `0.0799` |

| gap | risk | evidence | next action |
|---|---|---|---|
| real paired insertion success/bounce evidence missing | `high` | rollout_status_counts={'missing': 12} | Run paired baseline/good_margin_guided insertion rollouts with success, bounce_count, retry_count, and stopped_early metadata. |
| probability heads saturate; use logit margin for guidance | `low` | config_pass=True, score_mode=good_margin, good_margin_improve_rate=0.9375 | Keep good_margin as default and treat p_good as a reporting metric, not guidance objective. |
| insertion reason classifier is weaker than binary classifier | `medium` | reason_macro_f1=0.7894, binary_auc=0.9877 | For paper claims, emphasize good-vs-risk guidance; use reason labels mainly for interpretation unless reason F1 improves. |

## Board Deploy Candidate

This is the deployable marker/action scorer, not the preferred scientific board scorer.

| metric | value |
|---|---:|
| guidance signal strong | `missing` |
| clean-action score_delta mean | `0.00016205` |
| clean-action action_delta_norm mean | `0.00077498` |
| Foresight pred-vs-GT score Spearman | `0.5291` |
| pred score vs force-band quality Spearman | `0.3988` |
| force-band quality AUC good | `NA` |

| gap | risk | evidence | next action |
|---|---|---|---|
| deploy board real force evidence missing | `high` | main_rollout_status_counts={'missing': 12} | Run the marker_joint_s12 board baseline/guided pairs only if this deployable arm remains a candidate. |
| deploy board guidance signal is too weak for the current objective | `high` | score_delta_mean=0.00016205, action_delta_norm_mean=0.00077498 | Do not present marker_joint_s12 as the scientific board solution; keep it as a deployable baseline unless stronger real force traces prove otherwise. |
| deploy board score is not a direct force-consequence scorer | `medium` | pred_score_vs_force_band_quality_spearman=0.3988, pred_gt_spearman=0.5291 | Use the force-aware Foresight branch for the main board guidance story. |
| force-band quality alone does not explain good-label AUC | `medium` | force_band_quality_auc_good=NA | Keep semantic labels and force metrics separate in the paper; do not claim force-band metric alone defines board success. |

## Board Force-Aware Scientific Candidate

This is the current preferred research candidate for board wiping because it scores predicted force/contact consequences.

| metric | value |
|---|---:|
| guidance signal strong | `pass` |
| real-window serving ready | `pass` |
| offline score_delta mean | `3.5201` |
| offline action_delta_norm mean | `0.0513` |
| weight-sweep best score_delta mean | `4.2904` |
| real-HDF5-window serving score_delta mean | `1.7634` |
| real rollout complete | `missing` |

| gap | risk | evidence | next action |
|---|---|---|---|
| force-aware board real rollout evidence missing | `high` | force_aware_rollout_status_counts={'missing': 6} | Run three paired baseline/force_aware_guided board trials and evaluate force_trace.csv. |
| force-aware board is offline/serving-ready but not robot-proven | `medium` | config_consistent=True, serving_real_window_ready=True, score_delta_mean=1.7634 | Claim only preflight readiness until real paired force traces show improvement. |
| force-aware score currently selects margin-only objective | `low` | best_preset=margin_only, sweep_score_delta_mean=4.2904, score_weights={'band_margin': 1.0, 'contact_logprob': 0.0, 'force_center': 0.0, 'force_smooth': 0.0, 'action_smooth': 0.0} | Keep smooth/contact penalties as hypotheses for real-force evaluation; do not assume they improve deployment before force traces. |

## Rollout Config

| item | value |
|---|---|
| recommended_board_arm | `marker_joint_s12_guided` |
| recommended_insertion_arm | `good_margin_guided` |
| force_aware_board_arm_present | `True` |
| board_score_mode | `quality` |
| force_aware_score_preset | `margin_only` |
| insertion_score_mode | `good_margin` |

## Priority Experiments

| rank | experiment | why | success criterion |
|---:|---|---|---|
| 1 | force-aware board paired real rollout evaluation | It is the strongest board scientific candidate and the missing evidence is real force_trace improvement. | At least 3 complete baseline/force_aware_guided pairs; Fz band occupancy and smoothness improve without task/safety regression. |
| 2 | insertion paired real rollout evaluation | Good-margin insertion is config-consistent and has strong offline/serving signal; it still lacks success/bounce/retry evidence. | At least 3 complete baseline/good_margin_guided pairs; success increases or bounce/retry decreases with complete metadata. |
| 3 | board action_horizon/reactivity ablation | Recent work emphasizes reactive tactile policies; current action_horizon=8 may be slow for contact correction. | Compare action_horizon 4/6/8 under identical force-aware scoring; select the shortest horizon that preserves trajectory completion and improves force smoothness. |
| 4 | force-aware score penalty validation | Offline sweep favored margin_only, but smoothness/contact penalties are still physically meaningful hypotheses. | Use real force traces to decide whether margin_only, margin_smooth, or margin_contact best improves Fz smoothness and contact continuity. |

## Claim Boundary

Can claim now:
- Offline scorer quality is strong for both tasks.
- Foresight-gradient guidance path is ready for real rollout tests.
- Force-aware board consequence scorer has passed offline held-out gradient audit.
- Force-aware board consequence scorer has an optional serving arm whose dry-run smoke passed.
- Force-aware board serving arm has passed a stratified real-HDF5-window audit across five board labels.
- Force-aware board paired real-rollout manifest is prepared for three baseline/guided board pairs.
- Server-side rollout log schema is ready for force/action/guidance evaluation.
- The command and manifest pipeline is ready for paired real robot evidence collection.

Cannot claim yet:
- No real paired baseline-vs-guided improvement has been proven.
- No board force-curve improvement should be claimed without force_trace.csv pairs.
- No insertion success/bounce/retry improvement should be claimed without outcome metadata.

## Evidence Paths
- scorer_audit: `/home/chenshuai/Project/output/tac_quality_current_scorer_audit/current_tac_quality_scorer_audit.json`
- scorecard: `/home/chenshuai/Project/output/tac_quality_current_scorecard/current_tac_quality_scorecard.json`
- evidence_bundle: `/home/chenshuai/Project/output/tac_quality_evidence_bundle/current_tac_quality_evidence_bundle.json`
- rollout_config: `/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_forceaware_board_20260621.json`
- main_rollout_coverage: `/home/chenshuai/Project/output/tac_quality_real_rollout_coverage/current_s12_good_margin_coverage/tac_quality_real_rollout_coverage.json`
- force_aware_rollout_coverage: `/home/chenshuai/Project/output/tac_quality_real_rollout_coverage/board_force_aware_coverage/tac_quality_real_rollout_coverage.json`
