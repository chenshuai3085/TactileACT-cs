# 2026-06-21 Force-Aware Board Score Weight Sweep

## Goal

The board TacQuality scorer is meant for DP classifier/energy guidance, not only offline classification.  This sweep compares force-aware score definitions while keeping the Foresight model, validation split, trust-region refinement, and samples fixed.

Evidence boundary: offline Foresight-gradient audit only.  This does not claim real robot improvement.

## Setup

- Script: `TFAC_V5/tac_quality_energy/sweep_force_aware_score_weights.py`
- Output: `/home/chenshuai/Project/output/force_aware_score_weight_sweep/20260621_104503/force_aware_score_weight_sweep.json`
- Foresight: `/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_forceaware_multistep16_boardvae_e100_bs16_0/foresight_force_best.ckpt`
- Split: `val`
- Samples: `124`
- Trust region: `steps=4`, `action_step=0.02`, `max_total_delta=0.08`

## Result

| rank | preset | ranking | improve | score delta | action delta | raw delta |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `margin_only` | 0.9958 | 0.9919 | 4.2904 | 0.0594 | 0.6717 |
| 2 | `margin_smooth` | 0.9958 | 0.9919 | 4.2855 | 0.0594 | 0.6712 |
| 3 | `margin_action_smooth` | 0.9958 | 0.9919 | 4.2899 | 0.0593 | 0.6705 |
| 4 | `margin_force_action_smooth` | 0.9958 | 0.9919 | 4.2850 | 0.0594 | 0.6709 |
| 5 | `margin_center` | 0.9910 | 0.9677 | 4.2437 | 0.0587 | 0.6625 |
| 6 | `margin_contact` | 0.9816 | 0.9274 | 3.3377 | 0.0483 | 0.5471 |
| 7 | `default_margin_contact_center_smooth` | 0.9760 | 0.9032 | 3.2740 | 0.0474 | 0.5375 |
| 8 | `strong_smooth` | 0.9759 | 0.9032 | 3.2657 | 0.0473 | 0.5365 |
| 9 | `strong_center` | 0.9756 | 0.9032 | 3.2447 | 0.0470 | 0.5333 |
| 10 | `weak_margin_balanced_penalty` | 0.9402 | 0.9032 | 1.6044 | 0.0376 | 0.4316 |

Best current offline score:

```text
S_board_force_aware = logit_good - logsumexp(logit_too_small, logit_too_large, logit_oscillate)
```

Weights:

```json
{
  "band_margin": 1.0,
  "contact_logprob": 0.0,
  "force_center": 0.0,
  "force_smooth": 0.0,
  "action_smooth": 0.0
}
```

## Interpretation

The force-band head already captures the main board quality signal: good contact vs too-small force, too-large force, and oscillation.  Adding contact, force-center, force-smoothness, or action-smoothness penalties did not improve the offline guidance ranking on the current validation sweep.

This does not mean the physical penalty terms are useless.  It means they should remain hypotheses for real force-trace validation instead of being assumed better.  The current scientific recommendation is:

1. Keep `force_aware_guided` as the preferred research candidate for board wiping.
2. Use `margin_only` as the current best offline score preset.
3. Do not claim this is the final real-robot score until paired baseline/guided force traces confirm better force magnitude and smoothness.

## Scorecard Integration

The current scorecard now records:

- scientific board preference: `force_aware_guided`
- score preset: `margin_only`
- sweep path: `/home/chenshuai/Project/output/force_aware_score_weight_sweep/20260621_104503/force_aware_score_weight_sweep.json`

## Serving Config Integration

The force-aware board rollout config has been regenerated from source so that
`force_aware_guided` explicitly uses the selected `margin_only` score preset:

- Builder: `TFAC_V5/tac_quality_energy/build_force_aware_board_rollout_config.py`
- Config: `/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_forceaware_board_20260621.json`
- Weights loaded by serving:

```json
{
  "band_margin": 1.0,
  "contact_logprob": 0.0,
  "force_center": 0.0,
  "force_smooth": 0.0,
  "action_smooth": 0.0
}
```

Serving verification on real HDF5 windows:

- Output: `/home/chenshuai/Project/output/force_aware_serving_real_window_audit/20260621_110440/force_aware_serving_real_window_audit.json`
- Windows: `32`
- Labels: `positive_old=8`, `positive_260617=8`, `oscillate=8`, `too_large=4`, `too_small=4`
- pass: `true`
- score delta mean: `1.7634`
- normalized action delta mean: `0.0178`
- runtime checks: `ForceAwareForesightGuidanceRuntime`, trust-region adapter, not reranking

This verifies that the deployment helper actually loads `margin_only`.  It is
still an offline/serving-window check, not a real robot improvement claim.
