# Board Energy Preference-order Accuracy Validation

Data source: `real_rollout_quality_gate_smoke_input` (4 HDF5 files). Labels from `board_smoke_metadata_min2/real_rollout_episode_metrics.csv`.

## Main Results

| Selection | Score | Correct / Pairs | Accuracy | Mean gap | Median gap |
|---|---:|---:|---:|---:|---:|
| key_marker_expert_margin | expert_margin | 2/3 | 0.6667 | 0.033707 | 0.024794 |
| key_marker_quality_0_100 | quality_0_100 | 2/3 | 0.6667 | 0.000956 | 0.000685 |
| key_force_expert_margin | expert_margin | 2/3 | 0.6667 | -0.012368 | 0.016713 |
| model_best_expert_margin | expert_margin | 2/3 | 0.6667 | 0.060702 | 0.092206 |
| model_best_quality_0_100 | quality_0_100 | 2/3 | 0.6667 | 0.001863 | 0.002820 |
| all_windows_expert_margin | expert_margin | 1090/2436 | 0.4475 | -0.003016 | -0.006750 |
| all_windows_quality_0_100 | quality_0_100 | 1090/2436 | 0.4475 | -0.000048 | -0.000190 |

## Episode Scores

| Group | Episode | Success | Key marker start-end | Key marker quality | Key marker margin | Model-best start-end | Model-best quality | Model-best margin |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | episode_1 | 1 | 24-40 | 0.027075 | -8.214052 | 68-84 | 0.032203 | -8.040528 |
| baseline | episode_10 | 1 | 4-20 | 0.029696 | -8.121606 | 8-24 | 0.032030 | -8.045938 |
| guided | episode_11 | 1 | 36-52 | 0.027979 | -8.181199 | 28-44 | 0.028985 | -8.145860 |
| guided | episode_12 | 0 | 92-108 | 0.027294 | -8.205993 | 32-48 | 0.029209 | -8.138144 |

## Notes

- This is a smoke rollout subset, not a formal success/failure rollout benchmark.
- All scored windows were predicted as `pressure_too_small`; absolute quality is near zero, so the result mainly tests ranking under domain mismatch.
