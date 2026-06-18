# 2026-06-19 DDPM-Step TacQuality Guidance Parameter Grid

## Goal

Evaluate whether the current TacQuality scorer can be used as a stable
classifier/energy guidance signal inside DP denoising, and identify safe
offline settings for board wiping and insertion.

This is still offline sampler evidence. It does not prove real robot
improvement. Real claims require paired baseline/guided robot rollouts with
server-side force or outcome traces.

## Code

- `TFAC_V5/tac_quality_energy/sweep_board_ddpm_step_guidance.py`
  - refactored to expose `run_sweep(args)` while preserving CLI behavior.
- `TFAC_V5/tac_quality_energy/eval_ddpm_step_guidance_audit.py`
  - added step-level accept-only filtering.
  - added final output fallback: if the guided sample scores lower than the
    unguided sample, return the unguided action for that row.
- `TFAC_V5/tac_quality_energy/sweep_ddpm_guidance_grid.py`
  - new grid wrapper for board/insertion DDPM-step guidance sweeps.
  - writes per-setting JSON and aggregate CSV/JSON/Markdown summaries.

## Experiment Design

Selection rule:

1. Maximize score improve rate.
2. Enforce non-negative final score delta with final fallback.
3. Prefer high final accept rate, not just a high raw score delta.
4. Keep action update norm small.

The grid varies:

- `guidance_steps`: `1`, `2`
- `guidance_scale`: `0.00025`, `0.0005`, `0.001`
- `max_delta_norm`: board `0.005`, `0.01`; insertion `0.005`, `0.01`, `0.02`
- `num_inference_steps`: `4`
- `seeds`: `1,2`
- `max_episodes`: `4`
- `starts_per_episode`: `2`

## Commands

Insertion:

```bash
conda run --no-capture-output -n TactileACT python -u \
  TFAC_V5/tac_quality_energy/sweep_ddpm_guidance_grid.py \
  --tasks insertion \
  --max_episodes 4 \
  --starts_per_episode 2 \
  --seeds 1,2 \
  --num_inference_steps 4 \
  --guidance_steps 1,2 \
  --guidance_scales 0.00025,0.0005,0.001 \
  --max_delta_norms 0.005,0.01,0.02 \
  --output_dir /home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/insertion_guidance_grid_20260619_multiep4_start2_seed2
```

Board:

```bash
conda run --no-capture-output -n TactileACT python -u \
  TFAC_V5/tac_quality_energy/sweep_ddpm_guidance_grid.py \
  --tasks board \
  --max_episodes 4 \
  --starts_per_episode 2 \
  --seeds 1,2 \
  --num_inference_steps 4 \
  --guidance_steps 1,2 \
  --guidance_scales 0.00025,0.0005,0.001 \
  --max_delta_norms 0.005,0.01 \
  --output_dir /home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_guidance_grid_20260619_multiep4_start2_seed2
```

Accept/fallback validation:

```bash
conda run --no-capture-output -n TactileACT python -u \
  TFAC_V5/tac_quality_energy/sweep_ddpm_guidance_grid.py \
  --tasks board,insertion \
  --max_episodes 4 \
  --starts_per_episode 2 \
  --seeds 1,2 \
  --num_inference_steps 4 \
  --guidance_steps 1,2 \
  --guidance_scales 0.001 \
  --max_delta_norms 0.005 \
  --output_dir /home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/accept_final_grid_20260619_g1g2_s001
```

## Outputs

Insertion:

- `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/insertion_guidance_grid_20260619_multiep4_start2_seed2/guidance_grid_summary.csv`
- `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/insertion_guidance_grid_20260619_multiep4_start2_seed2/guidance_grid_summary.json`
- `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/insertion_guidance_grid_20260619_multiep4_start2_seed2/guidance_grid_summary.md`

Board:

- `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_guidance_grid_20260619_multiep4_start2_seed2/guidance_grid_summary.csv`
- `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_guidance_grid_20260619_multiep4_start2_seed2/guidance_grid_summary.json`
- `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_guidance_grid_20260619_multiep4_start2_seed2/guidance_grid_summary.md`

Accept/fallback validation:

- `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/accept_final_grid_20260619_g1g2_s001/guidance_grid_summary.csv`
- `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/accept_final_grid_20260619_g1g2_s001/guidance_grid_summary.json`
- `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/accept_final_grid_20260619_g1g2_s001/guidance_grid_summary.md`

Formal protected sweeps used by the current readiness matrix:

- board:
  `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_20260619_protected_multiep6_start2_seed2_t0_s001/board_ddpm_step_guidance_sweep.json`
- insertion:
  `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/insertion_0401_default_protected_multiep8_start2_seed2_t0_s001/insertion_ddpm_step_guidance_sweep.json`

## Results

### Board Wiping

Best protected offline DDPM-step setting in this grid:

| setting | improve | mean delta | min delta | step accept | final accept | action norm mean |
|---|---:|---:|---:|---:|---:|---:|
| `board_inf4_g1_s0.001_d0.005` | 1.0000 | 0.002298 | 0.000025 | 1.0000 | 1.0000 | 0.000241 |

The same result is obtained with `max_delta_norm=0.01` because the actual update
norm is below the cap.

Unprotected negative result:

| setting family | improve | mean delta | min delta | action norm mean |
|---|---:|---:|---:|---:|
| `guidance_steps=2` | 0.4375 | up to 0.001361 | down to -0.164668 | about 0.09297 |

Protected `guidance_steps=2` result:

| setting | improve | mean delta | min delta | raw min delta | step accept | final accept | action norm mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `board_inf4_g2_s0.001_d0.005` | 0.8750 | 0.015323 | 0.000000 | -0.000367 | 0.7188 | 0.8750 | 0.031663 |

Interpretation:

- Board guidance is most stable when applied only at the final denoising step.
- Step-level accept-only plus final fallback removes negative final outputs in
  the tested `guidance_steps=2` setting, but the action update norm is much
  larger and the final accept rate is lower.
- Current board DDPM-step guidance should therefore default to `g1, scale=0.001`
  for research ablations. Multi-step DDPM guidance should stay behind fallback
  protection.

Formal protected board sweep used for readiness:

| eval points | rows | improve | mean delta | min delta | step accept | final accept | action norm mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 24 | 1.0000 | 0.001762 | 0.000025 | 1.0000 | 1.0000 | 0.000253 |

### Insertion

Best protected offline DDPM-step setting in this grid:

| setting | improve | mean delta | min delta | step accept | final accept | action norm mean |
|---|---:|---:|---:|---:|---:|---:|
| `insertion_inf4_g1_s0.001_d0.005` | 0.9375 | 0.000114 | 0.000000 | 0.9375 | 1.0000 | 0.000112 |

The same result is obtained with `max_delta_norm=0.01` and `0.02` because the
actual update norm is below the cap.

Unprotected negative result:

| setting family | improve | mean delta | min delta | action norm mean |
|---|---:|---:|---:|---:|
| `guidance_steps=2` | 0.3125-0.3750 | negative mean | down to -0.007583 | about 0.00387 |

Protected `guidance_steps=2` result:

| setting | improve | mean delta | min delta | raw min delta | step accept | final accept | action norm mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `insertion_inf4_g2_s0.001_d0.005` | 0.5000 | 0.000413 | 0.000000 | -0.000471 | 0.7500 | 0.5625 | 0.000850 |

Interpretation:

- Insertion guidance is less stable than board guidance.
- With final fallback, the best `g1` insertion setting has no negative final
  output in this grid.
- The `g2` insertion setting requires many fallbacks and only accepts 56.25% of
  final guided samples, so it is not a good default.
- For insertion deployment, prefer final clean-action trust-region guidance or
  at most final-step DDPM guidance with final fallback until larger offline
  sweeps and paired robot tests prove sampler-step guidance is reliable.

Formal protected insertion sweep used for readiness:

| eval points | rows | improve | mean delta | min delta | step accept | final accept | action norm mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 32 | 0.9375 | 0.000108 | 0.000000 | 0.9375 | 1.0000 | 0.000110 |

## Main Conclusion

The current TacQuality scorers are useful as local gradient sources, but the
safe operating region is explicitly trust-region and accept-only:

- Good: final clean-action trust-region refinement.
- Good for research ablation: final DDPM denoising step only, small scale,
  with step-level accept-only and final fallback.
- Possible but not default: multi-step DDPM guidance with final fallback.
- Not recommended now: unprotected multi-step DDPM guidance through the sampler.

This supports the current system story:

`DP action -> Foresight predicted tactile consequence -> TacQuality energy -> bounded local action refinement`.

It does not support aggressive unprotected classifier guidance at many denoising
steps yet.

## Next Evidence Needed

1. Real paired board trials:
   - baseline vs guided
   - same DP checkpoint
   - server-side force traces
   - force-band and smoothness metrics
2. Larger offline sweeps:
   - more episodes/seeds
   - action smoothness checks
   - noised-action scorer training if pursuing true sampler-step guidance
3. Force-aware Foresight/scorer:
   - directly predict or score force-band and force derivative
   - align with board wiping quality definition instead of relying only on
     marker/action proxy features.
