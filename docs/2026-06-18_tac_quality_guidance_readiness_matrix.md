# 2026-06-18 TacQuality Guidance Readiness Matrix

Generated at: `2026-06-18 23:30:07`

## Scope

This document tracks the current TacQuality classifier/energy scorers for DP classifier guidance:

```text
DP clean action
  -> task Foresight predicts future tactile consequence
  -> TacQuality scorer gives a differentiable score
  -> trust-region gradient update on the action chunk
```

This is clean-action classifier/energy guidance. It is not reranking.

## Current Recommendation

| task | recommended arm | scorer | checkpoint | score mode | rollout readiness |
|---|---|---|---|---|---|
| insertion | `default_guided` | `InsertionRiskScorerRuntime` | `/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt` | `profile` | true |
| board | `marker_joint_guided` | `ForceBandTacQualityEnergyRuntime(marker_joint_action)` | `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/force_band_tac_quality_energy_best.pt` | `quality` | true |

## Offline Scorer Evidence

| task | protocol | AUC | bACC | reason F1 | quality corr / Spearman | evidence |
|---|---|---:|---:|---:|---:|---|
| insertion | GroupKFold over insertion windows | 0.9877 | 0.9437 | 0.7894 | 0.7656 | `/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_eval.json` |
| board | grouped held-out deploy features, `marker_joint_action` | 0.9997 | 0.9828 | 0.9703 | 0.9239 | `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/train_result.json` |

Interpretation:

- Insertion has strong binary risk separation and usable continuous quality correlation.
- Board uses deploy-aligned features: Foresight-predicted marker proxy plus candidate joint-action proxy. It does not use unavailable future `eef_abs`.

## Foresight-Chain Alignment

| task | score mode | samples | pred AUC(good) | GT AUC(good) | pred/GT Spearman | score vs force quality | evidence |
|---|---|---:|---:|---:|---:|---:|---|
| insertion | `profile` | NA | NA | NA | NA | NA | gradient audit below |
| board | `quality` | 120 | 0.9991 | 0.8986 | 0.6130 | 0.4733 | `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/foresight_alignment_quality/foresight_score_alignment.json` |

The board Foresight-chain score is no longer saturated: positive labels score much higher than too-small / too-large / oscillatory contact in `quality` mode.

## Guidance Gradient Evidence

| task | samples | pass | finite grad | positive grad | improved | accept | trust-region | score delta mean | action delta norm mean | evidence |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| insertion | 24 | true | 1.0000 | 1.0000 | 1.0000 | 0.9688 | 1.0000 | 0.2665 | 0.0749 | `/home/chenshuai/Project/output/insertion_guidance_gradient_audit_real_foresight_profile_20260618/guidance_gradient_audit.json` |
| board | 24 | true | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0013 | 0.0008 | `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/guidance_gradient_audit_quality/guidance_gradient_audit.json` |

Interpretation:

- Both tasks have finite, non-zero action gradients through Foresight and the scorer.
- Board uses a deliberately small trust-region step, so score/action deltas are much smaller than insertion.
- These audits prove differentiability and bounded refinement. They do not prove real robot improvement.

## Server Entrypoint Smoke

| task | pass | scorer runtime | score mode | contact gate | score delta | evidence |
|---|---|---|---|---|---:|---|
| insertion | true | `InsertionRiskScorerRuntime` | `profile` | NA | 0.0153 | `/home/chenshuai/Project/output/tac_quality_guided_server_packet/auto_discovered/insertion_guided_server_real_foresight_smoke.json` |
| board | true | `ForceBandTacQualityEnergyRuntime` | `quality` | 1.0000 | 0.0012 | `/home/chenshuai/Project/output/tac_quality_guided_server_packet/marker_joint_board_real_foresight_smoke_20260618/guided_server_dry_run_smoke.json` |

## Active 260617-only Board DP Context

- Active run: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260618_ext`
- Home symlink: `/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260618_ext`
- Recommended checkpoint for real tests: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260618_ext/dp_best.pth`
- Recommended checkpoint exists: `true`
- Latest epoch: `181/2000`
- Latest train/val: `0.007726` / `0.016040`
- Best epoch/val: `105` / `0.011387`
- Trend warning: `watch_plateau_use_best_for_deploy`
- Epochs since best: `76`

Deployment/testing should use `dp_best.pth`, not `dp_latest.pth`, unless a later epoch refreshes the best validation checkpoint.

## Board Real-Rollout Command Packet

Current copy-paste command sheet:

- `for_show_xiaomi/guide_forshow.sh`

Current board rollout config:

- `/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_marker_joint_20260618.json`
- guided arm: `marker_joint_guided`
- baseline guidance flag: `--disable_guidance`
- guided scorer runtime: `ForceBandTacQualityEnergyRuntime`
- guided score mode: `quality`
- expected server-side rollout root: `/home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_scorer`

Expected real-rollout layout:

```text
/home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_scorer/
  baseline/<trial>/force_trace.csv
  baseline/<trial>/force_trace.npz
  baseline/<trial>/force_curve.png
  baseline/<trial>/metadata.json
  guided/<trial>/force_trace.csv
  guided/<trial>/force_trace.npz
  guided/<trial>/force_curve.png
  guided/<trial>/metadata.json
```

After real robot trials, evaluate with:

```bash
conda run --no-capture-output -n TactileACT python for_show_xiaomi/eval_board_force_rollouts.py \
  --root /home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_scorer \
  --tag board_260617_marker_joint_scorer
```

## Remaining Real-Rollout Evidence Gap

- Board real force rollout ready: `false`
- Insertion real force rollout ready: `false`
- Overall real rollout evidence complete: `false`

Missing evidence:

- insertion: baseline vs guided real rollouts with success/bounce/retry outcomes;
- board: matched baseline/guided real rollouts with server-side `force_trace.csv`;
- board contact-phase metrics: force-in-band ratio, too-low/too-high ratio, force derivative, marker smoothness, and task completion/coverage.

## Current Gates

| gate | status |
|---|---|
| offline scorer quality | true |
| Foresight gradient readiness | true |
| real rollout improvement proven | false |

Bottom line: insertion and board scorers are ready for controlled real-rollout testing, but the full project goal is not proven until matched real robot results show improved contact outcomes.

## Source Inputs

- `evidence`: `/home/chenshuai/Project/output/tac_quality_evidence_audit_20260618/tac_quality_evidence_audit.json`
- `state`: `/home/chenshuai/Project/output/tac_quality_guidance_state_audit/tac_quality_guidance_state_audit.json`
- `real_rollout`: `/home/chenshuai/Project/output/tac_quality_real_rollout_eval/current_tac_quality/tac_quality_real_rollout_eval.json`
- `board_train`: `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/train_result.json`
- `board_alignment`: `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/foresight_alignment_quality/foresight_score_alignment.json`
- `board_gradient`: `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/guidance_gradient_audit_quality/guidance_gradient_audit.json`
- `board_smoke`: `/home/chenshuai/Project/output/tac_quality_guided_server_packet/marker_joint_board_real_foresight_smoke_20260618/guided_server_dry_run_smoke.json`
- `insertion_eval`: `/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_eval.json`
- `insertion_gradient`: `/home/chenshuai/Project/output/insertion_guidance_gradient_audit_real_foresight_profile_20260618/guidance_gradient_audit.json`
- `insertion_smoke`: `/home/chenshuai/Project/output/tac_quality_guided_server_packet/auto_discovered/insertion_guided_server_real_foresight_smoke.json`
- `rollout_config`: `/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_marker_joint_20260618.json`
- `dp_run`: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260618_ext`
