# Figure 2 Multi-Step Tactile Future Asset

## Purpose

Replace the single `Predicted tactile future` marker-field image in ForeTac Figure 2 with a compact temporal sequence that explicitly communicates multi-step tactile prediction.

## Data Contract

The panels are rendered directly from the Board Wiping prediction stored in `paper/figures/foretac_multitask_qpos_preview.npz`. The stored tensor has shape `(16, 9, 9, 2)` and represents one continuous sequence `t+1, t+2, ..., t+16` from the same held-out observation. No future step is skipped in the full assets, and no panel is assembled from a different episode or task.

This source is the existing v8j qpos-conditioned preview. It is suitable as the current Figure 2 visual asset, but it should be regenerated from the final action-conditioned checkpoint if Figure 2 is intended to document the exact final experimental checkpoint.

## Outputs

- `foresight_future_continuous_t1_t6.{png,pdf}`: recommended compact Figure 2 replacement; the first six consecutive future steps are shown without temporal subsampling.
- `foresight_future_continuous_t1_t16_grid.{png,pdf}`: complete 16-step prediction in two rows of eight frames.
- `foresight_future_continuous_t1_t16_row.{png,pdf}`: complete 16-step prediction in one long row for PPT cropping or a wide layout.
- `foresight_pred_tplus01.png` through `foresight_pred_tplus16.png`: independently editable consecutive future panels rendered with one shared color normalization.

For the constrained Figure 2 slot, the consecutive six-frame strip is preferred. The full two-row 16-frame grid should be used when the complete model output must be visible.

Numerical verification confirms that all adjacent predictions are distinct. The 15 adjacent mean marker-vector changes are between 0.005456 and 0.054206, and the mean marker-vector change from `t+1` to `t+16` is 0.177369. The visual changes are subtle because this selected contact sequence evolves smoothly, not because frames were duplicated.
