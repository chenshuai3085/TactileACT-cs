# 2026-06-19 Board TacQualityEnergy S12 Predicted-Domain Comparison

## Purpose

This experiment tests whether increasing the board scorer training cache from 6 to 12 windows per episode improves the deploy-aligned board TacQualityEnergy scorer.

The tested deploy path is:

```text
joint action chunk -> multistep Foresight -> predicted future marker field -> ForceBandTacQualityEnergy -> d score / d action
```

This is classifier/energy guidance evidence. It is not real robot rollout evidence.

## New Feature Cache

- feature cache: `/home/chenshuai/Project/output/board_predicted_domain_force_band_features_20260619_s12/board_predicted_domain_force_band_features.npz`
- deploy feature cache: `/home/chenshuai/Project/output/board_predicted_domain_force_band_features_20260619_s12_deploy/board_predicted_domain_force_band_features_deploy.npz`
- samples: `3600`
- episode groups: `300`
- labels: `{'oscillate': 492, 'positive_260617': 948, 'positive_old': 1200, 'too_large': 480, 'too_small': 480}`
- feature variant used for deployment: `marker_joint_action` = predicted marker proxy + joint action proxy only

Important Foresight shift observation:

| label | marker MAE mean |
|---|---:|
| positive_old | 0.2444 |
| positive_260617 | 1.8202 |
| too_small | 0.1721 |
| too_large | 0.2637 |
| oscillate | 0.4541 |

The 260617 positive windows have much larger Foresight marker MAE than the old positive windows. This means GT-marker classification alone is not enough; predicted-domain scoring is necessary for guidance.

## Scorer Candidates

| candidate | train cache | n | checkpoint |
|---|---:|---:|---|
| old default | predicted-domain, 6 windows/episode | 1800 | `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/force_band_tac_quality_energy_best.pt` |
| new s12 candidate | predicted-domain, 12 windows/episode | 3600 | `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/force_band_tac_quality_energy_best.pt` |

## Held-Out Episode Validation

| candidate | AUC | bACC | reason F1 | quality Spearman | energy-quality Spearman |
|---|---:|---:|---:|---:|---:|
| old default | 0.9997 | 0.9828 | 0.9703 | 0.9239 | 0.8414 |
| new s12 candidate | 1.0000 | 1.0000 | 1.0000 | 0.9239 | 0.9039 |

The new s12 candidate is clearly stronger as an offline classifier/regressor on the predicted-domain cache.

## Foresight Score Alignment

Both candidates were evaluated on the same include-260617 sampling protocol.

| candidate | samples | pred AUC(good) | pred/GT Spearman | pred vs force-band quality Spearman | GT AUC(good) |
|---|---:|---:|---:|---:|---:|
| old default | 180 | 0.9994 | 0.6785 | 0.3861 | 0.9205 |
| new s12 candidate | 180 | 1.0000 | 0.5291 | 0.3988 | 0.8513 |

Interpretation:

- New s12 candidate has slightly better pred AUC and force-band-quality correlation.
- Old default has stronger pred/GT score consistency on this sample.
- Because online guidance uses `score(Foresight(action))`, pred/GT consistency matters; therefore this result does not justify replacing the default blindly.

## Clean-Action Gradient Audit

| candidate | pass | finite grad | improved | score delta mean | action delta norm mean |
|---|---|---:|---:|---:|---:|
| old default | True | 1.0000 | 1.0000 | 0.000366 | 0.000798 |
| new s12 candidate | True | 1.0000 | 1.0000 | 0.000162 | 0.000775 |

Both candidates provide finite positive gradients through Foresight and pass the trust-region audit.

## Noisy-Action Audit

Each entry reports `final_minus_noisy mean / final_minus_clean mean`.

| candidate | overall pass | per-noise behavior |
|---|---|---|
| old default | True | 0.0: +0.000106 / clean 0.0001; 0.05: +0.000188 / clean -0.1215; 0.1: +0.000140 / clean -0.0919; 0.2: +0.000043 / clean -0.0747; 0.4: +0.000006 / clean -0.0368 |
| new s12 candidate | True | 0.0: +0.000103 / clean 0.0001; 0.05: +0.000407 / clean -0.5174; 0.1: +0.000265 / clean -0.5076; 0.2: +0.000147 / clean -0.5252; 0.4: +0.000074 / clean -0.2371 |

Both pass local noisy-action guidance. However, at nonzero perturbations both refiners improve the noisy score but remain below the clean action score. This means the audit supports local gradient usability, not full DDPM-step recovery.

## Recommendation

Current recommendation: keep the old 20260618 predicted-domain `marker_joint_action` scorer as the conservative default for robot/server testing, and treat the new s12 scorer as a strong candidate for the next ablation.

Reasoning:

- The new s12 scorer has better held-out classifier/regression metrics.
- The old scorer has stronger pred/GT score consistency and slightly more stable noisy-action behavior relative to the clean action.
- Both pass gradient audits, so both are usable for controlled ablation.
- Neither proves real robot force improvement yet.

Recommended next ablation:

1. Run paired real robot board rollouts with the same DP checkpoint and two scorer arms: old default vs new s12 candidate.
2. Save server-side force traces for every trajectory.
3. Evaluate force-in-band ratio, too-low/too-high ratio, force derivative, marker smoothness, task completion, and early stop rate.
4. Promote the new s12 scorer only if it improves real force curves without hurting task success.

## Evidence Boundaries

- Offline held-out metrics prove separability/calibration only.
- Foresight alignment proves predicted-score credibility only under sampled windows.
- Gradient/noisy audits prove local differentiability and bounded action refinement only.
- Final project claim still requires matched baseline/guided real rollouts with force traces.
