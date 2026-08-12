# Guidance Diagnostics Figure Review

Date: 2026-08-12

## Problems in the previous web figure

1. Panels A and B connected per-step aggregate values with lines. The source
   records come from multiple replanning events, so the connected plot visually
   implied a single continuous denoising trajectory that was not the statistical
   unit being evaluated.
2. Panel D summarized only three selected episode/windows across three guided
   steps. This sample is too small to support a general direction-alignment
   claim in the main paper.
3. Panel C averaged signed gradients over 128 held-out windows. Positive and
   negative values can cancel, so this does not directly measure where guidance
   is active in action space.
4. The old caption mixed the 128-window spatial audit with dense denoising logs
   without clearly separating their sample counts and evidence boundaries.

## Revised statistical contract

- Same-state score refinement: 305 paired late-denoising updates from 61
  replanning events in `ddpm_gradient_steps.csv`.
- Score-improvement rate: 96.1% (`post_expert_margin > pre_expert_margin`).
- Finite-gradient rate: 100% over the same 305 paired updates.
- Trust-region rate: 100% at the configured applied update norm of 0.003 in
  normalized action space.
- Spatial sensitivity: mean absolute scaled score gradient over 128 balanced
  held-out windows, indexed by action horizon and joint dimension. Absolute
  values prevent signed cancellation.

No diagnostic panel uses an independent-window index as elapsed time. The
three-window policy/guidance alignment plot is excluded from the paper figure.

## Rollout evidence boundary

The board-wiping RGB frames and force trace come from the recorded replay trial
`caheiban_260609/...episode_6`. The contact-quality curve is an offline scorer
evaluation of that recorded trial. These panels are not a paired comparison of
unguided and guided robot rollouts.

The socket Figure 5 curves remain deterministic illustrative profiles generated
by `paper/plot_force_real.py`; they are not raw per-trial logs. The paper caption
now states this explicitly. The layout can be retained when raw insertion logs
become available.

## Outputs

- `paper/figures/board_wiping_guidance_evidence.{png,pdf}`
- `paper/figures/board_guidance_diagnostics_revised.{png,pdf}`
- `paper/force_real_3panel.{png,pdf}`
- Reproducible builder: `scripts/plot_board_wiping_evidence.py`
