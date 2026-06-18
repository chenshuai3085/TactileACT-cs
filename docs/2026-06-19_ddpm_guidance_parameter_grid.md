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
- `TFAC_V5/tac_quality_energy/sweep_ddpm_guidance_grid.py`
  - new grid wrapper for board/insertion DDPM-step guidance sweeps.
  - writes per-setting JSON and aggregate CSV/JSON/Markdown summaries.

## Experiment Design

Selection rule:

1. Maximize score improve rate.
2. Prefer non-negative worst-case score delta.
3. Keep action update norm small.

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

## Outputs

Insertion:

- `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/insertion_guidance_grid_20260619_multiep4_start2_seed2/guidance_grid_summary.csv`
- `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/insertion_guidance_grid_20260619_multiep4_start2_seed2/guidance_grid_summary.json`
- `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/insertion_guidance_grid_20260619_multiep4_start2_seed2/guidance_grid_summary.md`

Board:

- `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_guidance_grid_20260619_multiep4_start2_seed2/guidance_grid_summary.csv`
- `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_guidance_grid_20260619_multiep4_start2_seed2/guidance_grid_summary.json`
- `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_guidance_grid_20260619_multiep4_start2_seed2/guidance_grid_summary.md`

## Results

### Board Wiping

Best offline DDPM-step setting in this grid:

| setting | improve | mean delta | min delta | action norm mean |
|---|---:|---:|---:|---:|
| `board_inf4_g1_s0.001_d0.005` | 1.0000 | 0.002298 | 0.000025 | 0.000241 |

The same result is obtained with `max_delta_norm=0.01` because the actual update
norm is below the cap.

Important negative result:

| setting family | improve | mean delta | min delta | action norm mean |
|---|---:|---:|---:|---:|
| `guidance_steps=2` | 0.4375 | up to 0.001361 | down to -0.164668 | about 0.09297 |

Interpretation:

- Board guidance is stable when applied only at the final denoising step.
- Two guided denoising steps are unsafe in this offline audit: they can produce
  large negative score deltas and much larger action perturbations.
- Current board DDPM-step guidance should therefore be restricted to late-step
  local guidance if used as a research ablation.

### Insertion

Best offline DDPM-step setting in this grid:

| setting | improve | mean delta | min delta | action norm mean |
|---|---:|---:|---:|---:|
| `insertion_inf4_g1_s0.001_d0.005` | 0.9375 | 0.000110 | -0.000070 | 0.000120 |

The same result is obtained with `max_delta_norm=0.01` and `0.02` because the
actual update norm is below the cap.

Important negative result:

| setting family | improve | mean delta | min delta | action norm mean |
|---|---:|---:|---:|---:|
| `guidance_steps=2` | 0.3125-0.3750 | negative mean | down to -0.007583 | about 0.00387 |

Interpretation:

- Insertion guidance is less stable than board guidance.
- The scorer provides mostly positive local gradients at the final denoising
  step, but even the best setting still has a small negative worst-case delta.
- Insertion DDPM-step guidance should remain experimental and small-scale.
- For insertion deployment, prefer final clean-action trust-region guidance
  until larger offline sweeps and paired robot tests prove sampler-step
  guidance is reliable.

## Main Conclusion

The current TacQuality scorers are useful as local gradient sources, but the
safe operating region is narrow:

- Good: final clean-action trust-region refinement.
- Good for research ablation: final DDPM denoising step only, small scale.
- Not recommended now: multi-step DDPM guidance through the sampler.

This supports the current system story:

`DP action -> Foresight predicted tactile consequence -> TacQuality energy -> bounded local action refinement`.

It does not support aggressive classifier guidance at many denoising steps yet.

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
