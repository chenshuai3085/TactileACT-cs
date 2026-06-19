# 2026-06-19 TacQuality Guidance Readiness Matrix

Generated at: `2026-06-19 15:19:26`

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
| insertion | `good_margin_guided` | `InsertionRiskScorerRuntime` | `/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt` | `good_margin` | true |
| board | `marker_joint_s12_guided` | `ForceBandTacQualityEnergyRuntime(marker_joint_action,s12)` | `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/force_band_tac_quality_energy_best.pt` | `quality` | true |

Board note: a stronger offline s12 candidate exists at
`/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/force_band_tac_quality_energy_best.pt`.
It improves held-out predicted-domain metrics and now also has stronger semantic bad-to-good guidance geometry plus a stronger protected DDPM-step sweep.
It is the next board A/B candidate, but still not a real-robot improvement claim.
See `docs/2026-06-19_board_scorer_s12_predicted_domain_comparison.md`.
A simple old/s12 ensemble sweep found a tiny offline gain for rank-normalized `0.85*old + 0.15*s12`, but the gain is too small to justify deployment complexity before real rollouts.
See `docs/2026-06-19_board_scorer_ensemble_sweep.md`.
Semantic direction evidence is summarized in `docs/2026-06-19_tac_quality_semantic_direction_audit.md`.

## Semantic Direction Evidence

This audit checks whether score gradients point from bad tactile outcomes toward good tactile outcomes, not just whether the classifier separates labels.

| task | deployed mode | semantic recommended mode | correction pass | strict pass | best bad-to-good projection | evidence |
|---|---|---|---:|---:|---:|---|
| insertion | `profile` | `p_good` | true | false | 0.8255 | `/home/chenshuai/Project/output/tac_quality_semantic_direction_audit/tac_quality_semantic_direction_audit.json` |
| board_default | `quality` | `quality` | false | false | 0.6354 | `/home/chenshuai/Project/output/tac_quality_semantic_direction_audit/tac_quality_semantic_direction_audit.json` |
| board_s12 | `quality` | `quality` | true | false | 0.7969 | `/home/chenshuai/Project/output/tac_quality_semantic_direction_audit/tac_quality_semantic_direction_audit.json` |

Interpretation:

- Insertion `p_good` has better semantic direction geometry than `profile`, but the DDPM-step sweep below shows it saturates at score 1.0 and gives no sampler improvement.
- A follow-up insertion cross-score ablation shows that the unsaturated `good_margin` logit margin avoids this saturation and is the stronger next insertion A/B candidate.
- Board s12 `quality` passes bad-to-good correction geometry and is a stronger board A/B candidate than the old/default scorer.
- Strict pass is still false, so accept-only and final fallback remain required.

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
| insertion | `good_margin` | NA | NA | NA | NA | NA | gradient audit below |
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
| board | 3 | `[0.0, 0.05, 0.1, 0.2, 0.4]` | true | `multistep` | 0 / 0 | 0.0:improve=1.0000,delta=0.0001; 0.05:improve=1.0000,delta=0.0004; 0.1:improve=1.0000,delta=0.0003; 0.2:improve=1.0000,delta=0.0001; 0.4:improve=1.0000,delta=0.0001 | `/home/chenshuai/Project/output/tac_quality_noisy_action_guidance_audit/board_marker_joint_s12_260617_fast4/noisy_action_guidance_audit.json` |

Interpretation:

- Board passes all tested perturbation levels with the deploy-aligned `marker_joint_s12_guided` scorer, but score deltas are intentionally tiny because the trust-region step is small.
- Insertion now has matched 0209 and 0401 Foresight audits with 0 missing / 0 unexpected keys; the older `latent_foresight_full` audit remains historical caveat evidence only.
- These results support moving from final clean-action refinement toward denoising-time guidance, but a true DP denoising-step implementation still needs its own audit.

## Insertion DDPM-Step Multi-Episode Sweep

This sweep evaluates matched `latent_foresight_0401` late-step `t=0` guidance across multiple real insertion observations.

| task | eval points | rows | improve | score delta mean | score delta min | step accept | final accept | finite grad | action delta norm | evidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| insertion | 16 | 32 | 0.9375 | 0.000108 | 0.000000 | 0.9375 | 1.0000 | 1.0000 | 0.000110 | `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/insertion_0401_default_protected_multiep8_start2_seed2_t0_s001/insertion_ddpm_step_guidance_sweep.json` |

Interpretation:

- Matched insertion Foresight/scorer gradients are finite across all tested rows, and most late-step updates improve the scorer.
- This protected sweep uses step-level accept-only updates plus final fallback to the base action when the scorer would get worse.
- With the protected setting, the final score delta minimum is non-negative. It is still offline sampler evidence, not robot outcome evidence.

## Insertion p_good DDPM-Step Ablation

This ablation tests the score mode recommended by the semantic direction audit.

| mode | eval points | rows | improve | score delta mean | score delta min | step accept | final accept | finite grad | action delta norm | evidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| p_good | 16 | 32 | 0.0000 | 0.000000 | 0.000000 | 1.0000 | 1.0000 | 1.0000 | 0.000112 | `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/insertion_0401_p_good_protected_multiep8_start2_seed2_t0_s001/insertion_ddpm_step_guidance_sweep.json` |

Interpretation:

- `p_good` has good offline semantic geometry but saturates in the matched DDPM/Foresight chain: base scores are already near 1.0 and final score deltas are exactly zero.
- Therefore `p_good` is not recommended as the current insertion DDPM-step guidance score, despite the semantic direction audit.
- Keep insertion DDPM-step evidence on the protected `profile` sweep unless a less-saturated calibrated score is trained.

## Insertion Score-Mode Cross-Score Ablation

This ablation uses each insertion score mode as the DDPM-step guidance objective, then re-scores the same base/guided actions with all candidate heads. This avoids judging a mode only by the score it optimized.

| guidance mode | rows | final accept | action norm | own delta | own improve | profile delta | energy delta | good margin delta | quality logit delta | min quality delta | evidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `profile` | 32 | 1.0000 | 0.000110 | 0.000108 | 0.9375 | 0.000108 | 0.000500 | 0.001160 | 0.000768 | -0.000054 | `/home/chenshuai/Project/output/tac_quality_score_mode_ablation/insertion_0401_profile_pgood_energy_goodmargin_cross_score_20260619/insertion_score_mode_ablation.json` |
| `p_good` | 32 | 1.0000 | 0.000112 | 0.000000 | 0.0000 | 0.000094 | 0.000379 | 0.001245 | 0.000508 | -0.000584 | `/home/chenshuai/Project/output/tac_quality_score_mode_ablation/insertion_0401_profile_pgood_energy_goodmargin_cross_score_20260619/insertion_score_mode_ablation.json` |
| `energy` | 32 | 1.0000 | 0.000110 | 0.000500 | 0.9375 | 0.000108 | 0.000500 | 0.001160 | 0.000768 | -0.000054 | `/home/chenshuai/Project/output/tac_quality_score_mode_ablation/insertion_0401_profile_pgood_energy_goodmargin_cross_score_20260619/insertion_score_mode_ablation.json` |
| `good_margin` | 32 | 1.0000 | 0.000102 | 0.001154 | 0.9375 | 0.000093 | 0.000394 | 0.001154 | 0.000557 | 0.000000 | `/home/chenshuai/Project/output/tac_quality_score_mode_ablation/insertion_0401_profile_pgood_energy_goodmargin_cross_score_20260619/insertion_score_mode_ablation.json` |

Interpretation:

- `p_good` remains saturated: own-score delta is exactly zero under the matched DDPM/Foresight chain.
- `good_margin` is the strongest unsaturated insertion candidate: it gives the largest own-score gain while keeping `profile`, `energy`, and `quality_logit` non-negative in this protected sweep.
- This does not replace real robot evidence; it only upgrades the next insertion A/B candidate from bounded probability `p_good` to logit-margin `good_margin`.

## DDPM-Step Guidance Audit

This audit inserts the current board TacQuality scorer into the DP denoising loop and scores the predicted clean action estimate `x0` through Foresight.

| task | setting | samples | final improve | final score delta | per-step score delta | finite grad | action delta norm | evidence |
|---|---|---:|---:|---:|---:|---:|---:|---|
| board | `4inf/1guide/scale=0.001` | 4 | 1.0000 | 0.000052 | 0.000052 | 1.0000 | 0.000107 | `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_20260619_ep2_s80_t0_s001_seed1_4/ddpm_step_guidance_audit.json` |
| board | `4inf/1guide/scale=0.001` | 4 | 1.0000 | 0.000050 | 0.000050 | 1.0000 | 0.000116 | `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_20260618ext_ep2_s80_t0_s001_seed1_4/ddpm_step_guidance_audit.json` |
| board | `4inf/2guide/scale=0.001` | 1 | 0.0000 | -0.031705 | -0.001477 | 1.0000 | 0.077360 | `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_real_chain_smoke/ddpm_step_guidance_audit.json` |
| board | `4inf/1guide/scale=0.001` | 1 | 1.0000 | 0.000070 | 0.000070 | 1.0000 | 0.000061 | `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_t0_s001_seed1/ddpm_step_guidance_audit.json` |
| board | `4inf/1guide/scale=0.0005` | 1 | 1.0000 | 0.000034 | 0.000034 | 1.0000 | 0.000030 | `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_t0_s0005_seed1/ddpm_step_guidance_audit.json` |
| board | `8inf/1guide/scale=0.001` | 1 | 1.0000 | 0.000690 | 0.000690 | 1.0000 | 0.000043 | `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_steps8_t0_s001_seed1/ddpm_step_guidance_audit.json` |

Interpretation:

- The current board scorer has usable gradients inside the sampler, but guidance timing matters.
- In the 260617 smoke sample, guiding the last two denoising steps reduced final score; guiding only the final `t=0` step produced small positive score gains.
- Current recommendation: keep production on final clean-action trust-region guidance, and treat true DDPM-step guidance as experimental until a larger sweep confirms late-step-only settings.

## Board DDPM-Step Multi-Episode Sweep

This sweep reuses one loaded DP/Foresight/scorer stack and evaluates late-step `t=0` guidance across multiple real 260617 board observations.

| task | eval points | rows | improve | score delta mean | score delta min | step accept | final accept | finite grad | action delta norm | contact gate mean | evidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| board | 12 | 24 | 1.0000 | 0.001762 | 0.000025 | 1.0000 | 1.0000 | 1.0000 | 0.000253 | 0.9810 | `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_20260619_protected_multiep6_start2_seed2_t0_s001/board_ddpm_step_guidance_sweep.json` |

Interpretation:

- The multi-episode sweep is stronger than the single-frame smoke: it covers 6 valid episodes, 12 contact-phase start points, and 24 seed/start rows.
- This protected sweep uses step-level accept-only updates plus final fallback; all tested rows had finite gradients and positive final score deltas under late-step `t=0` guidance.
- This supports the scorer as a stable local gradient source, but it is still offline sampler evidence, not real robot improvement.

## Board s12 DDPM-Step Multi-Episode Sweep

This sweep uses the semantic-direction-favored `marker_joint_s12_guided` board scorer.

| task | eval points | rows | improve | score delta mean | score delta min | step accept | final accept | finite grad | action delta norm | contact gate mean | evidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| board s12 | 12 | 24 | 1.0000 | 0.003919 | 0.000013 | 1.0000 | 1.0000 | 1.0000 | 0.000227 | 0.9810 | `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_s12_260617_20260619_protected_multiep6_start2_seed2_t0_s001/board_ddpm_step_guidance_sweep.json` |

Interpretation:

- Board s12 improves every tested row and has larger mean score gain than the old/default board protected sweep, with similar or smaller action update norm.
- This makes s12 the better next board A/B candidate, but still only offline sampler evidence.

## Server Entrypoint Smoke

| task | pass | scorer runtime | score mode | contact gate | score delta | finite grad | positive grad | accept | evidence |
|---|---|---|---|---|---:|---:|---:|---:|---|
| insertion profile | true | `InsertionRiskScorerRuntime` | `profile` | NA | 0.0365 | 1.0000 | 1.0000 | 1.0000 | `/home/chenshuai/Project/output/tac_quality_guided_server_packet/insertion_0401_default_guided_smoke_20260619/guided_server_dry_run_smoke.json` |
| insertion good_margin | true | `InsertionRiskScorerRuntime` | `good_margin` | NA | 0.1856 | 1.0000 | 1.0000 | 1.0000 | `/home/chenshuai/Project/output/tac_quality_guided_server_packet/current_20260619_insertion_good_margin_cpu_smoke/guided_server_dry_run_smoke.json` |
| board | true | `ForceBandTacQualityEnergyRuntime` | `quality` | 1.0000 | 0.0012 | 1.0000 | 1.0000 | 1.0000 | `/home/chenshuai/Project/output/tac_quality_guided_server_packet/board_260617_20260619_marker_joint_guided_smoke_20260619/guided_server_dry_run_smoke.json` |
| board s12 | true | `ForceBandTacQualityEnergyRuntime` | `quality` | 1.0000 | 0.0001 | 1.0000 | 1.0000 | 1.0000 | `/home/chenshuai/Project/output/tac_quality_guided_server_packet/current_20260619_board_s12_cpu_smoke/guided_server_dry_run_smoke.json` |


Insertion good_margin serving command/config:

- config: `/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_20260619.json`
- arm: `good_margin_guided`
- boundary: dry-run serving smoke only; real insertion success/bounce needs paired robot rollouts.

| shape | value |
|---|---|
| obs_cond | `[1, 2350]` |
| action_norm | `[1, 16, 7]` |
| guided_norm | `[1, 16, 7]` |

Board s12 serving command/config:

- config: `/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_20260619.json`
- arm: `marker_joint_s12_guided`
- boundary: dry-run serving smoke only; real board force improvement needs paired robot rollouts with server-side force traces.

| shape / gate | value |
|---|---|
| obs_cond | `[1, 2350]` |
| action_norm | `[1, 16, 7]` |
| guided_norm | `[1, 16, 7]` |
| contact_gate_metric | `4.2426` |
| contact_gate_value | `1.0000` |


Board contact-gate skip check:

| pass | marker metric | gate value | skipped | raw action delta | evidence |
|---|---:|---:|---|---:|---|
| true | 0.1414 | 0.0000 | true | 0.0000 | `/home/chenshuai/Project/output/tac_quality_guided_server_packet/current_marker_joint_board_contact_gate_skip_20260619/guided_server_dry_run_smoke.json` |

## 260617-only Board DP Context

- Run: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_full_noearly_tmux`
- Run status: `stopped`
- Recommended checkpoint for real tests: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_full_noearly_tmux/dp_best.pth`
- Recommended checkpoint exists: `true`
- Stop reason: `strong_validation_plateau_or_overfit_after_epoch200_use_dp_best`
- Stopped at: `2026-06-19 11:11:33`
- Last complete epoch: `202/2000`
- Last complete train/val: `0.006962` / `0.019671`
- Best epoch/val: `94` / `0.011385`
- Epochs since best: `108`
- Early-stop summary: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_full_noearly_tmux/early_stop_summary.json`

Deployment/testing should use `dp_best.pth`, not `dp_latest.pth`, unless intentionally testing late-overfit behavior.

### 260617 Stable Follow-up DP Run

- Run: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr`
- Run status: `active_or_unknown`
- Candidate checkpoint: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_best.pth`
- Candidate checkpoint exists: `true`
- Deployment status: `candidate_training_run_not_recommended_until_complete_or_validated`
- Latest epoch: `264/2000`
- Latest train/val: `0.006460` / `NA`
- Latest validation epoch/train/val: `260` / `0.006961` / `0.014998`
- Best epoch/val: `155` / `0.011659`
- Trend warning: `strong_plateau_or_overfit_use_best`
- Epochs since best: `109`

The stable follow-up run is a training candidate. It should not replace the stopped run's `dp_best.pth` in robot commands until it has stronger validation/downstream evidence.

## Board Real-Rollout Command Packet

Current copy-paste command sheet:

- `for_show_xiaomi/guide_forshow.sh`

Current board rollout config:

- `/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_20260619.json`
- guided arm: `marker_joint_s12_guided`
- baseline guidance flag: `--disable_guidance`
- guided scorer runtime: `ForceBandTacQualityEnergyRuntime`
- guided score mode: `quality`
- expected server-side rollout root: `/home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_s12_scorer`

Expected real-rollout layout:

```text
/home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_s12_scorer/
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
  --root /home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_s12_scorer \
  --tag board_260617_marker_joint_s12_scorer \
  --expected_baseline_arm baseline \
  --expected_guided_arm marker_joint_s12_guided
```

## Insertion Real-Rollout Command Packet

Current insertion rollout config:

- `/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_20260619.json`
- guided arm: `good_margin_guided`
- baseline guidance flag: `--disable_guidance`
- guided scorer runtime: `InsertionRiskScorerRuntime`
- guided score mode: `good_margin`
- expected server-side rollout root: `/home/chenshuai/Project/output/insertion_rollouts/good_margin_risk_scorer`

Expected real-rollout layout:

```text
/home/chenshuai/Project/output/insertion_rollouts/good_margin_risk_scorer/
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
conda run --no-capture-output -n TactileACT python for_show_xiaomi/eval_insertion_rollouts.py \
  --root /home/chenshuai/Project/output/insertion_rollouts/good_margin_risk_scorer \
  --tag insertion_good_margin_risk_scorer \
  --expected_baseline_arm baseline \
  --expected_guided_arm good_margin_guided
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
- `real_rollout`: `/home/chenshuai/Project/output/tac_quality_real_rollout_eval/current_s12_good_margin_tac_quality/tac_quality_real_rollout_eval.json`
- `board_train`: `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/train_result.json`
- `board_alignment`: `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/foresight_alignment_quality/foresight_score_alignment.json`
- `board_gradient`: `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/guidance_gradient_audit_quality/guidance_gradient_audit.json`
- `board_s12_train`: `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/train_result.json`
- `board_s12_alignment`: `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/foresight_alignment_quality/foresight_score_alignment.json`
- `board_s12_gradient`: `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/guidance_gradient_audit_quality/guidance_gradient_audit.json`
- `board_old_include260617_alignment`: `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/foresight_alignment_quality_include260617_sameset/foresight_score_alignment.json`
- `board_old_include260617_gradient`: `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/guidance_gradient_audit_quality_include260617_sameset/guidance_gradient_audit.json`
- `board_smoke`: `/home/chenshuai/Project/output/tac_quality_guided_server_packet/board_260617_20260619_marker_joint_guided_smoke_20260619/guided_server_dry_run_smoke.json`
- `board_s12_smoke`: `/home/chenshuai/Project/output/tac_quality_guided_server_packet/current_20260619_board_s12_cpu_smoke/guided_server_dry_run_smoke.json`
- `insertion_eval`: `/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_eval.json`
- `insertion_gradient`: `/home/chenshuai/Project/output/insertion_guidance_gradient_audit_real_foresight_profile_20260618/guidance_gradient_audit.json`
- `insertion_gradient_0209`: `/home/chenshuai/Project/output/insertion_guidance_gradient_audit_0209_matched_20260619/guidance_gradient_audit.json`
- `insertion_gradient_0401`: `/home/chenshuai/Project/output/insertion_guidance_gradient_audit_0401_matched_20260619/guidance_gradient_audit.json`
- `insertion_smoke`: `/home/chenshuai/Project/output/tac_quality_guided_server_packet/insertion_0401_default_guided_smoke_20260619/guided_server_dry_run_smoke.json`
- `insertion_good_margin_smoke`: `/home/chenshuai/Project/output/tac_quality_guided_server_packet/current_20260619_insertion_good_margin_cpu_smoke/guided_server_dry_run_smoke.json`
- `board_gate_skip_smoke`: `/home/chenshuai/Project/output/tac_quality_guided_server_packet/current_marker_joint_board_contact_gate_skip_20260619/guided_server_dry_run_smoke.json`
- `board_noisy_action_audit`: `/home/chenshuai/Project/output/tac_quality_noisy_action_guidance_audit/board_marker_joint_s12_260617_fast4/noisy_action_guidance_audit.json`
- `insertion_noisy_action_audit`: `/home/chenshuai/Project/output/tac_quality_noisy_action_guidance_audit/insertion_profile_current_fast4/noisy_action_guidance_audit.json`
- `insertion_noisy_action_audit_0209`: `/home/chenshuai/Project/output/tac_quality_noisy_action_guidance_audit/insertion_0209_matched_fast4/noisy_action_guidance_audit.json`
- `insertion_noisy_action_audit_0401`: `/home/chenshuai/Project/output/tac_quality_noisy_action_guidance_audit/insertion_0401_matched_fast4/noisy_action_guidance_audit.json`
- `insertion_ddpm_step_sweep`: `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/insertion_0401_default_protected_multiep8_start2_seed2_t0_s001/insertion_ddpm_step_guidance_sweep.json`
- `insertion_pgood_ddpm_step_sweep`: `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/insertion_0401_p_good_protected_multiep8_start2_seed2_t0_s001/insertion_ddpm_step_guidance_sweep.json`
- `insertion_score_mode_ablation`: `/home/chenshuai/Project/output/tac_quality_score_mode_ablation/insertion_0401_profile_pgood_energy_goodmargin_cross_score_20260619/insertion_score_mode_ablation.json`
- `board_ddpm_step_audits`: `['/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_20260619_ep2_s80_t0_s001_seed1_4/ddpm_step_guidance_audit.json', '/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_20260618ext_ep2_s80_t0_s001_seed1_4/ddpm_step_guidance_audit.json', '/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_real_chain_smoke/ddpm_step_guidance_audit.json', '/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_t0_s001_seed1/ddpm_step_guidance_audit.json', '/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_t0_s0005_seed1/ddpm_step_guidance_audit.json', '/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_steps8_t0_s001_seed1/ddpm_step_guidance_audit.json']`
- `board_ddpm_step_sweep`: `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_20260619_protected_multiep6_start2_seed2_t0_s001/board_ddpm_step_guidance_sweep.json`
- `board_s12_ddpm_step_sweep`: `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_s12_260617_20260619_protected_multiep6_start2_seed2_t0_s001/board_ddpm_step_guidance_sweep.json`
- `semantic_direction`: `/home/chenshuai/Project/output/tac_quality_semantic_direction_audit/tac_quality_semantic_direction_audit.json`
- `rollout_config`: `/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_20260619.json`
- `good_margin_rollout_config`: `/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_20260619.json`
- `dp_run`: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_full_noearly_tmux`
- `stable_dp_run`: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr`
