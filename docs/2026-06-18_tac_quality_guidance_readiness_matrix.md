# 2026-06-18 TacQuality Guidance Readiness Matrix

Generated at: `2026-06-19 03:45:15`

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

Board note: a stronger offline s12 candidate exists at
`/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/force_band_tac_quality_energy_best.pt`.
It improves held-out predicted-domain metrics, but the 20260618 scorer remains the conservative default because it has stronger pred/GT score consistency on the same include-260617 alignment audit.
See `docs/2026-06-19_board_scorer_s12_predicted_domain_comparison.md`.
A simple old/s12 ensemble sweep found a tiny offline gain for rank-normalized `0.85*old + 0.15*s12`, but the gain is too small to justify deployment complexity before real rollouts.
See `docs/2026-06-19_board_scorer_ensemble_sweep.md`.

## Offline Scorer Evidence

| task | protocol | AUC | bACC | reason F1 | quality corr / Spearman | evidence |
|---|---|---:|---:|---:|---:|---|
| insertion | GroupKFold over insertion windows | 0.9877 | 0.9437 | 0.7894 | 0.7656 | `/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_eval.json` |
| board | grouped held-out deploy features, `marker_joint_action` | 0.9997 | 0.9828 | 0.9703 | 0.9239 | `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/train_result.json` |
| board s12 candidate | grouped held-out predicted-domain deploy features, `marker_joint_action` | 1.0000 | 1.0000 | 1.0000 | 0.9239 | `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/train_result.json` |

Interpretation:

- Insertion has strong binary risk separation and usable continuous quality correlation.
- Board uses deploy-aligned features: Foresight-predicted marker proxy plus candidate joint-action proxy. It does not use unavailable future `eef_abs`.

## Foresight-Chain Alignment

| task | score mode | samples | pred AUC(good) | GT AUC(good) | pred/GT Spearman | score vs force quality | evidence |
|---|---|---:|---:|---:|---:|---:|---|
| insertion | `profile` | NA | NA | NA | NA | NA | gradient audit below |
| board | `quality` | 120 | 0.9991 | 0.8986 | 0.6130 | 0.4733 | `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/foresight_alignment_quality/foresight_score_alignment.json` |
| board old default + 260617 | `quality` | 180 | 0.9994 | 0.9205 | 0.6785 | 0.3861 | `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/foresight_alignment_quality_include260617_sameset/foresight_score_alignment.json` |
| board s12 candidate + 260617 | `quality` | 180 | 1.0000 | 0.8513 | 0.5291 | 0.3988 | `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/foresight_alignment_quality/foresight_score_alignment.json` |

The board Foresight-chain score is no longer saturated: positive labels score much higher than too-small / too-large / oscillatory contact in `quality` mode.

## Guidance Gradient Evidence

| task | samples | pass | finite grad | positive grad | improved | accept | trust-region | score delta mean | action delta norm mean | evidence |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| insertion | 24 | true | 1.0000 | 1.0000 | 1.0000 | 0.9688 | 1.0000 | 0.2665 | 0.0749 | `/home/chenshuai/Project/output/insertion_guidance_gradient_audit_real_foresight_profile_20260618/guidance_gradient_audit.json` |
| insertion matched 0209 | 24 | true | 1.0000 | 1.0000 | 0.9167 | 0.8438 | 1.0000 | 0.1067 | 0.0671 | `/home/chenshuai/Project/output/insertion_guidance_gradient_audit_0209_matched_20260619/guidance_gradient_audit.json` |
| insertion matched 0401 | 24 | true | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.1401 | 0.0799 | `/home/chenshuai/Project/output/insertion_guidance_gradient_audit_0401_matched_20260619/guidance_gradient_audit.json` |
| board | 24 | true | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0013 | 0.0008 | `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/guidance_gradient_audit_quality/guidance_gradient_audit.json` |
| board old default + 260617 | 24 | true | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0004 | 0.0008 | `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/guidance_gradient_audit_quality_include260617_sameset/guidance_gradient_audit.json` |
| board s12 candidate | 24 | true | 1.0000 | 1.0000 | 1.0000 | 0.9688 | 1.0000 | 0.0002 | 0.0008 | `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/guidance_gradient_audit_quality/guidance_gradient_audit.json` |

Interpretation:

- Both tasks have finite, non-zero action gradients through Foresight and the scorer.
- Board uses a deliberately small trust-region step, so score/action deltas are much smaller than insertion.
- These audits prove differentiability and bounded refinement. They do not prove real robot improvement.

## Noisy-Action Robustness Audit

This audit perturbs recorded action chunks by fractions of the Foresight action standard deviation, then checks whether the scorer/Foresight chain still gives finite positive gradients and locally improves the score.

It is evidence for noisy-action guidance readiness, but it is still not a true DDPM-step guidance benchmark and not real robot evidence.

| task | samples | noise levels(action std) | overall pass | Foresight kind | missing / unexpected keys | per-noise improve/delta | evidence |
|---|---:|---|---|---|---:|---|---|
| insertion | 4 | `[0.0, 0.05, 0.1, 0.2, 0.4]` | true | `single_step` | 100 / 0 | 0.0:improve=1.0000,delta=0.3883; 0.05:improve=1.0000,delta=0.0825; 0.1:improve=1.0000,delta=0.1037; 0.2:improve=1.0000,delta=0.0618; 0.4:improve=1.0000,delta=0.1043 | `/home/chenshuai/Project/output/tac_quality_noisy_action_guidance_audit/insertion_profile_current_fast4/noisy_action_guidance_audit.json` |
| insertion matched 0209 | 4 | `[0.0, 0.05, 0.1, 0.2, 0.4]` | true | `single_step` | 0 / 0 | 0.0:improve=1.0000,delta=0.1673; 0.05:improve=1.0000,delta=0.0704; 0.1:improve=1.0000,delta=0.0764; 0.2:improve=1.0000,delta=0.0726; 0.4:improve=1.0000,delta=0.0336 | `/home/chenshuai/Project/output/tac_quality_noisy_action_guidance_audit/insertion_0209_matched_fast4/noisy_action_guidance_audit.json` |
| insertion matched 0401 | 4 | `[0.0, 0.05, 0.1, 0.2, 0.4]` | true | `single_step` | 0 / 0 | 0.0:improve=1.0000,delta=0.0702; 0.05:improve=1.0000,delta=0.0446; 0.1:improve=1.0000,delta=0.0514; 0.2:improve=1.0000,delta=0.0461; 0.4:improve=1.0000,delta=0.0356 | `/home/chenshuai/Project/output/tac_quality_noisy_action_guidance_audit/insertion_0401_matched_fast4/noisy_action_guidance_audit.json` |
| board | 4 | `[0.0, 0.05, 0.1, 0.2, 0.4]` | true | `multistep` | 0 / 0 | 0.0:improve=1.0000,delta=0.0028; 0.05:improve=1.0000,delta=0.0003; 0.1:improve=1.0000,delta=0.0001; 0.2:improve=1.0000,delta=0.0000; 0.4:improve=1.0000,delta=0.0004 | `/home/chenshuai/Project/output/tac_quality_noisy_action_guidance_audit/board_marker_joint_current_fast4/noisy_action_guidance_audit.json` |

Interpretation:

- Board passes all tested perturbation levels with the deploy-aligned `marker_joint_guided` scorer, but score deltas are intentionally tiny because the trust-region step is small.
- Insertion now has matched 0209 and 0401 Foresight audits with 0 missing / 0 unexpected keys; the older `latent_foresight_full` audit remains historical caveat evidence only.
- These results support moving from final clean-action refinement toward denoising-time guidance, but a true DP denoising-step implementation still needs its own audit.

## DDPM-Step Guidance Audit

This audit inserts the current board TacQuality scorer into the DP denoising loop and scores the predicted clean action estimate `x0` through Foresight.

| task | setting | samples | final improve | final score delta | per-step score delta | finite grad | action delta norm | evidence |
|---|---|---:|---:|---:|---:|---:|---:|---|
| board | `4inf/2guide/scale=0.001` | 1 | 0.0000 | -0.031705 | -0.001477 | 1.0000 | 0.077360 | `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_real_chain_smoke/ddpm_step_guidance_audit.json` |
| board | `4inf/1guide/scale=0.001` | 1 | 1.0000 | 0.000070 | 0.000070 | 1.0000 | 0.000061 | `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_t0_s001_seed1/ddpm_step_guidance_audit.json` |
| board | `4inf/1guide/scale=0.0005` | 1 | 1.0000 | 0.000034 | 0.000034 | 1.0000 | 0.000030 | `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_t0_s0005_seed1/ddpm_step_guidance_audit.json` |
| board | `8inf/1guide/scale=0.001` | 1 | 1.0000 | 0.000690 | 0.000690 | 1.0000 | 0.000043 | `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_steps8_t0_s001_seed1/ddpm_step_guidance_audit.json` |
| insertion | `2inf/1guide-t0/scale=0.001` | 1 | 1.0000 | 0.005425 | 0.005425 | 1.0000 | 0.000130 | `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/insertion_0401_default_t0_s001_cpu_smoke_20260619/ddpm_step_guidance_audit.json` |

Interpretation:

- The current board scorer has usable gradients inside the sampler, but guidance timing matters.
- In the 260617 smoke sample, guiding the last two denoising steps reduced final score; guiding only the final `t=0` step produced small positive score gains.
- The insertion matched-Foresight scorer also has a passing minimal DDPM-step smoke on one held-out 0401 episode/frame, with `0 missing / 0 unexpected` Foresight keys and a positive final-score delta.
- Current recommendation: keep production on final clean-action trust-region guidance, and treat true DDPM-step guidance as experimental until a larger sweep confirms late-step-only settings across tasks, episodes, seeds, and guidance timings.

## Server Entrypoint Smoke

| task | pass | scorer runtime | score mode | contact gate | score delta | evidence |
|---|---|---|---|---|---:|---|
| insertion | true | `InsertionRiskScorerRuntime` | `profile` | NA | 0.0365 | `/home/chenshuai/Project/output/tac_quality_guided_server_packet/insertion_0401_default_guided_smoke_20260619/guided_server_dry_run_smoke.json` |
| board | true | `ForceBandTacQualityEnergyRuntime` | `quality` | 1.0000 | 0.0012 | `/home/chenshuai/Project/output/tac_quality_guided_server_packet/board_260617_marker_joint_guided_smoke_20260619/guided_server_dry_run_smoke.json` |

Board contact-gate skip check:

| pass | marker metric | gate value | skipped | raw action delta | evidence |
|---|---:|---:|---|---:|---|
| true | 0.1414 | 0.0000 | true | 0.0000 | `/home/chenshuai/Project/output/tac_quality_guided_server_packet/current_marker_joint_board_contact_gate_skip_20260619/guided_server_dry_run_smoke.json` |

## 260617-only Board DP Context

Latest stopped run:

- Run: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619`
- Run status: `early_stopped_by_agent`
- Dataset: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617`
- Last complete epoch: `197/2000`
- Last train/val: `0.007725` / `0.019584`
- Best validation loss reported by training: `0.010671`
- Best epoch reported/inferred: `94`
- Epochs since best: `103`
- Latest val / best val: `1.835`
- Tail-20 val mean: `0.017727`
- Stop reason: validation plateau / overfitting after more than 100 epochs without refreshing best while train loss kept decreasing.
- Recommended checkpoint from this run: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619/dp_best.pth`
- Early-stop summary: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619/early_stop_summary.json`
- Final curve: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619/loss_curve.png`

Earlier stopped run:

- Run: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260618_ext`
- Run status: `stopped`
- Home symlink: `/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260618_ext`
- Recommended checkpoint for real tests: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260618_ext/dp_best.pth`
- Recommended checkpoint exists: `true`
- Stop reason: `early_stop_by_agent_strong_plateau_or_overfit_use_best`
- Stopped at: `2026-06-19 02:21:40`
- Last complete epoch: `523/2000`
- Last complete train/val: `0.004056` / `0.034266`
- Best epoch/val: `105` / `0.011387`
- Epochs since best: `418`
- Early-stop summary: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260618_ext/early_stop_summary.json`

Deployment/testing should use `dp_best.pth`, not `dp_latest.pth`, unless intentionally testing late-overfit behavior.

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
- `real_rollout`: `/home/chenshuai/Project/output/tac_quality_real_rollout_eval/current_tac_quality_pre_rollout_20260619/tac_quality_real_rollout_eval.json`
- `board_train`: `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/train_result.json`
- `board_alignment`: `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/foresight_alignment_quality/foresight_score_alignment.json`
- `board_gradient`: `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/guidance_gradient_audit_quality/guidance_gradient_audit.json`
- `board_s12_train`: `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/train_result.json`
- `board_s12_alignment`: `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/foresight_alignment_quality/foresight_score_alignment.json`
- `board_s12_gradient`: `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/guidance_gradient_audit_quality/guidance_gradient_audit.json`
- `board_old_include260617_alignment`: `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/foresight_alignment_quality_include260617_sameset/foresight_score_alignment.json`
- `board_old_include260617_gradient`: `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/guidance_gradient_audit_quality_include260617_sameset/guidance_gradient_audit.json`
- `board_smoke`: `/home/chenshuai/Project/output/tac_quality_guided_server_packet/board_260617_marker_joint_guided_smoke_20260619/guided_server_dry_run_smoke.json`
- `insertion_eval`: `/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_eval.json`
- `insertion_gradient`: `/home/chenshuai/Project/output/insertion_guidance_gradient_audit_real_foresight_profile_20260618/guidance_gradient_audit.json`
- `insertion_gradient_0209`: `/home/chenshuai/Project/output/insertion_guidance_gradient_audit_0209_matched_20260619/guidance_gradient_audit.json`
- `insertion_gradient_0401`: `/home/chenshuai/Project/output/insertion_guidance_gradient_audit_0401_matched_20260619/guidance_gradient_audit.json`
- `insertion_smoke`: `/home/chenshuai/Project/output/tac_quality_guided_server_packet/insertion_0401_default_guided_smoke_20260619/guided_server_dry_run_smoke.json`
- `board_gate_skip_smoke`: `/home/chenshuai/Project/output/tac_quality_guided_server_packet/current_marker_joint_board_contact_gate_skip_20260619/guided_server_dry_run_smoke.json`
- `board_noisy_action_audit`: `/home/chenshuai/Project/output/tac_quality_noisy_action_guidance_audit/board_marker_joint_current_fast4/noisy_action_guidance_audit.json`
- `insertion_noisy_action_audit`: `/home/chenshuai/Project/output/tac_quality_noisy_action_guidance_audit/insertion_profile_current_fast4/noisy_action_guidance_audit.json`
- `insertion_noisy_action_audit_0209`: `/home/chenshuai/Project/output/tac_quality_noisy_action_guidance_audit/insertion_0209_matched_fast4/noisy_action_guidance_audit.json`
- `insertion_noisy_action_audit_0401`: `/home/chenshuai/Project/output/tac_quality_noisy_action_guidance_audit/insertion_0401_matched_fast4/noisy_action_guidance_audit.json`
- `board_ddpm_step_audits`: `['/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_real_chain_smoke/ddpm_step_guidance_audit.json', '/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_t0_s001_seed1/ddpm_step_guidance_audit.json', '/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_t0_s0005_seed1/ddpm_step_guidance_audit.json', '/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_steps8_t0_s001_seed1/ddpm_step_guidance_audit.json']`
- `insertion_ddpm_step_smoke`: `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/insertion_0401_default_t0_s001_cpu_smoke_20260619/ddpm_step_guidance_audit.json`
- `rollout_config`: `/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_marker_joint_20260618.json`
- `dp_run`: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260618_ext`
