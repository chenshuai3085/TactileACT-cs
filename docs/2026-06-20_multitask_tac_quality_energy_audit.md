# Multitask TacQualityEnergy Audit

Generated: `2026-06-20T08:58:33`

## Candidate

- name: `manual-board / distilled TacQualityEnergy`
- checkpoint: `/home/chenshuai/Project/output/manual_board_tac_quality_energy/distilled_tac_quality_energy_final.pt`
- runtime: `TFAC_V5.tac_quality_energy.runtime.DistilledTacQualityEnergyRuntime`
- score mode: `energy_clipped`
- action step scale: `0.5`

## Pass Summary

| category | pass |
|---|---:|
| design | True |
| mixed_group_cv | True |
| task_breakdown_group_cv | True |
| runtime_gradient | True |
| action_gradient_sweep | True |
| foresight_bridge_smoke | True |

- offline guidance ready: `True`
- production validated by real rollout gates: `False`

## Key Metrics

| scope | binary AUC | energy AUC | reason macro-F1 | quality corr | energy-quality rho |
|---|---:|---:|---:|---:|---:|
| mixed | 0.9757 | 0.9745 | 0.7881 | 0.7842 | 0.7204 |
| insertion | 0.9457 | 0.9402 | 0.7093 | 0.6157 | 0.6694 |
| board | 1.0000 | 1.0000 | 0.9983 | 0.9970 | 0.4258 |

## Recommended Guidance Setting

- mode: `energy_clipped`
- action step scale: `0.5`
- insertion improved rate: `0.984375`
- board improved rate: `1.0`
- insertion score delta mean: `0.0008082650601863861`
- board score delta mean: `0.00038579168419043225`

## Foresight-Bridge Smoke

| task | finite grad | positive grad | improved rate | score delta mean |
|---|---:|---:|---:|---:|
| insertion | 1.0 | 1.0 | 1.0 | 0.1635155826807022 |
| board | 1.0 | 1.0 | 1.0 | 0.0017673224210739136 |

## Real Rollout Gates

| task | exists | pass | path |
|---|---:|---:|---|
| insertion | False | False | `/home/chenshuai/Project/output/real_rollout_quality_gate/insertion_baseline_vs_guided/real_rollout_quality_gate.json` |
| board | False | False | `/home/chenshuai/Project/output/real_rollout_quality_gate/board_baseline_vs_guided/real_rollout_quality_gate.json` |

## Conclusion

- manual-board / distilled TacQualityEnergy is the strongest unified innovation candidate before real rollout.
- Use as an ablation scorer for DP/Foresight trust-region gradient guidance with score_mode=energy_clipped and action_step_scale=0.5.
- No formal baseline-vs-guided real rollout gate exists for both insertion and board, so true robot improvement remains unproven.

Next required evidence:
- Run paired insertion baseline-vs-guided real rollout gate.
- Run paired board baseline-vs-guided real rollout gate.
- Compare task-default guided vs distilled TacQualityEnergy guided in a three-arm rollout ablation.
