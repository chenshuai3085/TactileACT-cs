# Board Scorer Ensemble Sweep

Date: 2026-06-19

## Purpose

Compare the current board default TacQuality scorer, the newer s12 scorer, and simple old/s12 score ensembles.

The goal is to check whether a lightweight ensemble can keep the old scorer's stronger predicted-vs-GT score consistency while gaining the s12 scorer's slightly better force-quality correlation.

This is an offline ablation only. It does not replace the current deployment default and does not prove real robot improvement.

## Inputs

Current default old scorer alignment CSV:

```text
/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/foresight_alignment_quality_include260617_sameset/foresight_score_alignment_samples.csv
```

s12 scorer alignment CSV:

```text
/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/foresight_alignment_quality/foresight_score_alignment_samples.csv
```

Both files contain the same `180` matched Foresight-alignment samples.

## Method

Script:

```text
TFAC_V5/tac_quality_energy/eval_board_scorer_ensemble_sweep.py
```

For each sample:

```text
ensemble_score = w * old_score + (1 - w) * s12_score
```

Swept:

- `w` from `0.0` to `1.0`
- score normalization modes:
  - raw
  - zscore
  - rank

Selection score:

```text
0.45 * pred_gt_spearman
  + 0.25 * pred_vs_force_quality_spearman
  + 0.20 * pred_auc_good
  + 0.10 * marker_mae_spearman
```

The scalar intentionally prioritizes predicted-vs-GT consistency because online guidance uses `score(Foresight(action))`; force-quality correlation is secondary until real robot force traces exist.

## Results

| candidate | mode | old weight | pred AUC | pred/GT Spearman | pred vs force-quality Spearman | selection score |
|---|---|---:|---:|---:|---:|---:|
| current default old | raw | 1.00 | 0.9994 | 0.6785 | 0.3861 | 0.6017 |
| s12 | raw | 0.00 | 1.0000 | 0.5291 | 0.3988 | 0.5378 |
| best ensemble | rank | 0.85 | 1.0000 | 0.6781 | 0.3919 | 0.6031 |

Top candidates:

| rank | mode | old weight | pred AUC | pred/GT Spearman | force-quality Spearman | selection score |
|---:|---|---:|---:|---:|---:|---:|
| 1 | rank | 0.85 | 1.0000 | 0.6781 | 0.3919 | 0.6031 |
| 2 | zscore | 0.95 | 0.9994 | 0.6794 | 0.3901 | 0.6031 |
| 3 | raw | 0.95 | 0.9994 | 0.6794 | 0.3901 | 0.6031 |
| 4 | rank | 0.90 | 1.0000 | 0.6789 | 0.3892 | 0.6028 |
| 5 | rank | 0.80 | 1.0000 | 0.6754 | 0.3941 | 0.6024 |

Artifacts:

```text
/home/chenshuai/Project/output/board_scorer_ensemble_sweep_20260619/board_scorer_ensemble_sweep.json
/home/chenshuai/Project/output/board_scorer_ensemble_sweep_20260619/board_scorer_ensemble_sweep.csv
/home/chenshuai/Project/output/board_scorer_ensemble_sweep_20260619/board_scorer_ensemble_sweep.md
/home/chenshuai/Project/output/board_scorer_ensemble_sweep_20260619/board_scorer_ensemble_sweep.png
```

## Interpretation

The best ensemble is `rank` normalized with `old_weight=0.85` and `s12_weight=0.15`.

It gives:

- slightly higher force-quality Spearman than old alone:
  - `0.3919` vs `0.3861`
- essentially the same pred/GT Spearman:
  - `0.6781` vs `0.6785`
- slightly higher pred AUC:
  - `1.0000` vs `0.9994`

However, the total gain over current default is tiny:

```text
selection score: 0.6031 vs 0.6017
```

This is not enough to justify adding a two-scorer ensemble into the deployment path before real rollouts.

## Decision

Current deployment default remains unchanged:

```text
ForceBandTacQualityEnergyRuntime(marker_joint_action)
checkpoint: /home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/force_band_tac_quality_energy_best.pt
score_mode: quality
```

The ensemble is kept as an ablation candidate:

```text
rank-normalized ensemble: 0.85 * old + 0.15 * s12
```

It should only be promoted if paired baseline/guided robot rollouts show better force-band and smoothness metrics.

## Evidence Boundary

- Uses saved offline Foresight-alignment samples only.
- Does not retrain any scorer.
- Does not execute robot rollouts.
- Does not prove guided action improves real wiping quality.
