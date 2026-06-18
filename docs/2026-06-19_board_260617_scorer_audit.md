# 2026-06-19 Board 260617 TacQuality Scorer Audit

## Purpose

Audit whether the current board-wiping TacQuality scorer evidence is strong enough for DP classifier guidance experiments after adding the 260617 positive board dataset:

`/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617`

This note is evidence bookkeeping. It does not claim real robot improvement.

## Data And Split

Main evaluation file:

`/home/chenshuai/Project/output/tac_quality_board_force_band_with_260617_positive_20260619/board_force_band_with_260617_eval.json`

Protocol:

- 5-fold `GroupKFold` by episode, not frame-level random split.
- 300 episode groups.
- 3600 contact-phase windows.
- Window: 8 tactile frames.
- Horizon/action chunk: 16.
- Wiping/contact phase sampling: episode fraction `0.25` to `0.85`.
- Labels:
  - `positive_old`: 1200 windows
  - `positive_260617`: 948 windows
  - `too_small`: 480 windows
  - `too_large`: 480 windows
  - `oscillate`: 492 windows

This split is appropriate for avoiding obvious temporal/frame leakage.

## Best Classical Scorer Evidence

Best deployable marker/action feature set:

- Feature: `marker_action`
- Feature dim: 74
- Best classical model: RandomForest

Held-out episode-level metrics:

- binary AUC: `0.9998`
- balanced accuracy: `0.9934`
- binary macro F1: `0.9928`
- reason macro F1: `0.9945`
- `positive_260617` recall: `0.9926`
- `positive_old` recall: `0.9908`

Interpretation:

- The new 260617 positive data is classifiable as good contact under marker/action proxy features.
- The four bad/good regimes are highly separable when evaluated by episode-level folds.
- This is good evidence for classification/risk gating.
- Continuous physical quality ordering from the classical score is weaker, so this alone should not be presented as a precise force-quality optimizer.

## Differentiable Scorer Candidate

For DP gradient guidance, the scorer must be differentiable and must operate in the same domain as inference-time predicted tactile consequences.

Most relevant current candidate:

`/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/force_band_tac_quality_energy_best.pt`

Why this candidate is stronger than the older default:

- It is trained on Foresight-predicted marker domain, not only GT marker domain.
- It includes 260617 positive board data.
- It uses `marker_joint_action`, which matches the deployable server-side feature contract more closely than features requiring dataset-only EEF values.
- It is differentiable and has a passing gradient smoke test.

Held-out validation from `train_result.json`:

- samples: 3600
- episode groups: 300
- feature variant: `marker_joint_action`
- feature dim: 64
- best epoch: 110
- binary AUC: `1.0000`
- balanced accuracy: `1.0000`
- binary macro F1: `1.0000`
- reason macro F1: `1.0000`
- quality Spearman: `0.9239`
- energy-quality Spearman: `0.9039`

Gradient smoke:

- pass: `True`
- marker grad finite: `True`
- action grad finite: `True`
- marker grad norm: `0.4377`
- action grad norm: `0.4146`

## Foresight Bridge Evidence

Foresight score alignment:

`/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/foresight_alignment_quality/foresight_score_alignment.md`

Key metrics:

- samples: 180
- predicted-score AUC(good): `1.0000`
- GT-future-score AUC(good): `0.8513`
- predicted vs GT score Spearman: `0.5291`
- predicted vs GT score Pearson: `0.6336`
- predicted score vs force-band quality Spearman: `0.3988`

Guidance gradient audit:

`/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/guidance_gradient_audit_quality/guidance_gradient_audit.md`

Key metrics:

- samples: 24
- finite grad rate mean: `1.0000`
- positive grad rate mean: `1.0000`
- accept rate mean: `0.9688`
- improved rate mean: `1.0000`
- trust-region pass rate: `1.0000`
- score delta mean: `0.000162`
- action delta norm mean: `0.000775`

Interpretation:

- The current guidance chain is differentiable through Foresight.
- Trust-region refinement is bounded and accept-only.
- The score improvement per refinement is small but positive in the offline audit.
- This is a readiness check, not real robot proof.

## Current Method Position

The current board guidance route should be described as:

`DP clean action chunk -> Foresight predicted tactile latent -> TactileVAE decoder -> predicted marker sequence -> TacQualityEnergy score -> trust-region accept-only gradient refinement`

It should not be described as reranking.

The current strongest board scorer story is:

- discrete quality classification is strong;
- bad-contact reason classification is strong;
- predicted-domain differentiable scorer is available;
- Foresight bridge provides finite, bounded gradients;
- real force-trace validation is still required.

## Evidence Boundary

Do not claim:

- real robot improvement;
- reduced force variance;
- reduced force spike;
- better task success;
- final best model.

Those require paired baseline/guided robot rollouts with server-side force traces.

## Recommended Next Use

For board guided serving or offline guidance audits, prefer the predicted-domain 260617-aware checkpoint:

`/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/force_band_tac_quality_energy_best.pt`

Keep the older default only as a stable comparison baseline:

`/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/force_band_tac_quality_energy_best.pt`

