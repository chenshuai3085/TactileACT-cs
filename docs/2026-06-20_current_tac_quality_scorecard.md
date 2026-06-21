# Current TacQuality Scorecard

Generated: `2026-06-21T14:57:35`

## Evidence Levels

| level | status |
|---|---:|
| `offline_scorer_ready` | `True` |
| `gradient_guidance_ready` | `True` |
| `insertion_guidance_signal_strong` | `True` |
| `insertion_config_consistent` | `True` |
| `board_deploy_guidance_signal_strong` | `False` |
| `force_aware_board_guidance_signal_strong` | `True` |
| `force_aware_board_gradient_audit_ready` | `True` |
| `force_aware_board_serving_smoke_ready` | `True` |
| `force_aware_board_real_window_serving_ready` | `True` |
| `force_aware_board_config_consistent` | `True` |
| `force_aware_board_rollout_manifest_ready` | `True` |
| `force_aware_board_real_rollout_complete` | `False` |
| `server_rollout_schema_ready` | `True` |
| `real_evidence_pipeline_ready` | `True` |
| `real_paired_rollout_complete` | `False` |
| `goal_complete` | `False` |

## Current Recommended Scorers

| task | arm | runtime | score mode | ready for real rollout |
|---|---|---|---|---:|
| insertion | `good_margin_guided` | `InsertionRiskScorerRuntime` | `good_margin` | `True` |
| board | `marker_joint_s12_guided` | `ForceBandTacQualityEnergyRuntime` | `quality` | `True` |

Board research candidate: `force_aware_foresight_quality_energy` (offline gradient audit ready: `True`, serving smoke ready: `True`, real-window serving ready: `True`, paired rollout manifest ready: `True`).

Scientific board preference: `force_aware_guided` / `ForceAwareForesightGuidanceRuntime` with score preset `margin_only` (preferred_research_candidate_not_real_robot_proven). Board quality is explicitly force-band and smoothness based; force_aware_guided has much stronger guidance signal than the deployable marker_joint_s12 scorer.  The current offline weight sweep selects the force-band good-vs-risk margin as the strongest bounded guidance score; extra contact/center/smooth penalties are kept as hypotheses for real force-trace validation rather than assumed improvements.

## Key Metrics

| task | classifier metric | quality metric | Foresight/guidance metric |
|---|---|---|---|
| insertion | AUC `0.9877`, bACC `0.9437` | corr `0.7656` | 0401 improve `1.0000`, good-margin improve `0.9375` |
| board | AUC `1.0000`, bACC `1.0000` | rho `0.9239` | pred-vs-GT rho `0.5291`, guidance improve `1.0000` |
| board force-aware candidate | band bACC `0.9736`, contact acc `0.9207` | good/bad AUC `1.0000` | finite grad `1.0000`, improve `0.9409`, score delta `3.5201`; smoke score delta `0.4879`, real-window score delta `1.7634` |

## Guidance Signal Strength

This separates classification quality from whether the scorer provides a nontrivial denoising guidance signal.

| scorer | status | score delta mean | action delta mean | threshold |
|---|---|---:|---:|---|
| insertion good-margin | `strong` | `0.140125` | `0.079864` | score >= `0.05`, action >= `0.02` |
| board marker_joint_s12 deployable | `weak_or_unproven` | `0.000162` | `0.000775` | score >= `0.01`, action >= `0.005` |
| board force-aware audit | `strong` | `3.520149` | `0.051284` | score >= `0.25`, action >= `0.01` |
| board force-aware real-window serving | `strong` | `1.763424` | `0.017770` | score >= `0.25`, action >= `0.005` |

Interpretation: the deployable `marker_joint_s12_guided` board scorer remains useful for real A/B testing because it is integrated, but its current gradient update is numerically weak.  The force-aware scorer is the better scientific candidate for the final TacQuality guidance story because it produces a stronger bounded action update and directly scores force/contact consequences.

## Insertion Config Consistency

- audit path: `/home/chenshuai/Project/output/insertion_config_consistency/20260621_113544/insertion_config_consistency.json`
- pass: `True`
- expected arm/runtime/score mode: `good_margin_guided` / `InsertionRiskScorerRuntime` / `good_margin`
- ablation good-margin delta mean: `0.001154`
- ablation p_good delta mean: `0.000000`
- DDPM final-score improve rate: `0.9375`
- final-action smoke score delta mean: `0.2717`
- denoising-step smoke score delta mean: `0.0627`
- evidence boundary: This is a consistency/preflight audit for offline and serving artifacts. It proves that insertion guidance is configured as the selected good_margin logit-margin scorer and that final-action and denoising-step serving paths load the same runtime/score mode. It does not prove real robot insertion improvement.

## Force-Aware Score Weight Sweep

- sweep path: `/home/chenshuai/Project/output/force_aware_score_weight_sweep/20260621_104503/force_aware_score_weight_sweep.json`
- best preset: `margin_only`
- best weights: `{'band_margin': 1.0, 'contact_logprob': 0.0, 'force_center': 0.0, 'force_smooth': 0.0, 'action_smooth': 0.0}`
- best ranking score: `0.9958`
- best improve / score delta / action delta: `0.9919` / `4.2904` / `0.0594`

| rank | preset | ranking | improve | score delta | action delta | raw delta |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `margin_only` | `0.9958` | `0.9919` | `4.2904` | `0.0594` | `0.6717` |
| 2 | `margin_smooth` | `0.9958` | `0.9919` | `4.2855` | `0.0594` | `0.6712` |
| 3 | `margin_action_smooth` | `0.9958` | `0.9919` | `4.2899` | `0.0593` | `0.6705` |
| 4 | `margin_force_action_smooth` | `0.9958` | `0.9919` | `4.2850` | `0.0594` | `0.6709` |
| 5 | `margin_center` | `0.9910` | `0.9677` | `4.2437` | `0.0587` | `0.6625` |

Interpretation: on the full validation sweep, the plain force-band good-vs-risk margin is the strongest offline guidance score. Contact, force-center, force-smooth, and action-smooth penalties remain useful design hypotheses, but they did not improve the current offline guidance ranking and must be justified by paired real force_trace rollouts before becoming the default.

## Force-Aware Config Consistency

- audit path: `/home/chenshuai/Project/output/force_aware_config_consistency/20260621_111101/force_aware_config_consistency.json`
- pass: `True`
- expected preset: `margin_only`
- expected weights: `{'band_margin': 1.0, 'contact_logprob': 0.0, 'force_center': 0.0, 'force_smooth': 0.0, 'action_smooth': 0.0}`
- serving score delta mean: `1.7634`
- serving normalized action delta mean: `0.0178`
- evidence boundary: This is a consistency/preflight audit for offline and serving artifacts. It proves that the force-aware board research arm is configured with the selected margin_only score and that the latest serving-window audit loaded the same weights. It does not prove real robot improvement.

## Real Rollout Coverage

- planned_counts: `{"board": {"baseline": 3, "guided": 3}, "insertion": {"baseline": 3, "guided": 3}}`
- observed_counts: `{}`
- status_counts: `{"missing": 12}`
- real_rollout_evidence_complete: `False`

## Force-Aware Board Rollout Manifest

- manifest ready: `True`
- real rollout complete: `False`
- planned board pairs/trials: `3` / `6`
- arms: baseline `baseline`, guided `force_aware_guided`
- ports: baseline `8765`, guided `8769`
- rollout root: `/home/chenshuai/Project/output/board_force_rollouts/260617_only_force_aware_scorer`
- coverage status: `{"missing": 6}`
- precheck detail: No board force_trace.csv found yet; run baseline/guided robot tests first.

## Server Rollout Log Schema

- schema_pass: `True`
- n_trials: `1`
- synthetic_count: `1`
- real_count: `0`
- note: Schema smoke proves server log format only; synthetic logs are not real robot evidence.

## Board DP Checkpoint Policy

- recommended: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_best.pth`
- avoid as default: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_final.pth`
- best val epoch/loss: `155` / `0.011659`
- latest logged epoch/train loss: `2000` / `0.002315`
- latest validation epoch/loss: `2000` / `0.038709`
- training running at status timestamp: `False` (pid `None`, status `2026-06-20 09:30:00 CST`)

## Innovation Story

- Do not present tactile concat DP as the main novelty; treat it as the action prior.
- Main novelty: differentiable tactile/force consequence scoring for DP classifier guidance.
- Insertion uses an unsaturated good-margin risk scorer over good insert vs pre-bounce/impact modes.
- The deployable board arm currently uses a four-class marker_joint_action force-band energy.
- Do not treat classification accuracy alone as sufficient; guidance signal strength must be nontrivial.
- The stronger board research candidate is force-aware Foresight consequence energy because it directly scores predicted force band, contact, and smoothness.
- Both tasks use bounded trust-region guidance through Foresight rather than offline reranking.

## Evidence Boundary

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

## Next Required Evidence

- Optionally run board force_aware_guided vs baseline after deciding to evaluate the new research arm.
- Run board block 1 vs block 2 in guide_forshow.sh and collect server-side force_trace.csv.
- Run insertion block 3 vs block 4 and fill success/stopped_early/bounce_count/retry_count metadata.
- Rerun audit_real_rollout_coverage.py until both tasks have at least three complete pairs.
- Only then run eval_tac_quality_real_rollouts.py for final real robot evidence.

## Input Paths

- scorer_audit: `/home/chenshuai/Project/output/tac_quality_current_scorer_audit/current_tac_quality_scorer_audit.json`
- guidance_state: `/home/chenshuai/Project/output/tac_quality_guidance_state_audit/tac_quality_guidance_state_audit.json`
- coverage: `/home/chenshuai/Project/output/tac_quality_real_rollout_coverage/current_s12_good_margin_coverage/tac_quality_real_rollout_coverage.json`
- schema_audit: `/home/chenshuai/Project/output/tac_quality_server_rollout_schema_audit/current_schema_smoke/tac_quality_server_rollout_schema_audit.json`
- dp_status: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/training_status_latest.json`
- rollout_config: `/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_forceaware_board_20260621.json`
- insertion_config_consistency: `/home/chenshuai/Project/output/insertion_config_consistency/20260621_113544/insertion_config_consistency.json`
- force_aware_board_audit: `/home/chenshuai/Project/output/force_aware_foresight_guidance_audit/20260621_090725/audit_results.json`
- force_aware_board_smoke: `/home/chenshuai/Project/output/tac_quality_guided_server_packet/board_force_aware_guided_smoke_20260621/guided_server_dry_run_smoke.json`
- force_aware_serving_real_window: `/home/chenshuai/Project/output/force_aware_serving_real_window_audit/20260621_110440/force_aware_serving_real_window_audit.json`
- force_aware_rollout_manifest: `/home/chenshuai/Project/output/tac_quality_real_rollout_manifest/board_force_aware_manifest/tac_quality_rollout_manifest.json`
- force_aware_rollout_coverage: `/home/chenshuai/Project/output/tac_quality_real_rollout_coverage/board_force_aware_coverage/tac_quality_real_rollout_coverage.json`
- force_aware_rollout_eval: `/home/chenshuai/Project/output/tac_quality_real_rollout_eval/board_force_aware_tac_quality_precheck/tac_quality_real_rollout_eval.json`
- force_aware_weight_sweep: `/home/chenshuai/Project/output/force_aware_score_weight_sweep/20260621_104503/force_aware_score_weight_sweep.json`
- force_aware_config_consistency: `/home/chenshuai/Project/output/force_aware_config_consistency/20260621_111101/force_aware_config_consistency.json`
