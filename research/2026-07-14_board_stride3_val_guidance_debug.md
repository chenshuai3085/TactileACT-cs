# 2026-07-14 Blackboard Stride-3 Val Guidance Debug

## Setup

- Foresight: `/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_temporalstride3_e100_bs16_preload/foresight_best.ckpt`
- Scorer: `/home/chenshuai/Project/output/board_latent_energy/ce_margin_temporalstride3_e40/board_latent_energy_best.pt`
- Validation split source: `/home/chenshuai/Project/output/board_latent_energy/ce_margin_temporalstride3_e40/manifest.json`
- Dense denoising logs: `outputs/board_stride3_guidance_gradient_vis/ddpm_gradient_steps.csv`
- Alignment logs: `outputs/board_stride3_ddpm_stage_guidance_audit_20260706/board_ddpm_step_guidance_sweep.json`

The current machine has no CUDA and the original stride-3 DP checkpoint path
`/media/chenshuai/SANDISK ELE/.../dp_best.pth` is not mounted. Therefore:

- scorer validation metrics and action-dimension gradients were recomputed on validation rows;
- score/guidance denoising curves and policy-guidance alignment reuse existing real stride-3 DP denoising logs.

## Main Results

- Scorer final validation set: `acc=1.0`, `macro_f1=1.0`, `expert_vs_negative_auroc=1.0`, `expert_margin_auroc=1.0`, `n=3144`.
- Selected validation rows for gradient heatmap: 16 / 16 succeeded, 4 rows per class.
- Mean validation `||grad score||`: `3.4260`.
- Mean validation `lambda ||grad score||` with `lambda=0.003`: `0.01028`.
- Dense guided denoising step score delta mean: `0.01981`.
- Alignment mean by guided step: `[-0.000013, -0.06123, -0.01245]`.

## Interpretation

The scorer itself separates the validation windows perfectly on the current split. The local guidance gradient is finite and not uniformly spread over all joints. On the sampled validation rows, the dominant action dimensions are `j2` and `j4`, which is useful evidence that the guidance is not pulling every action dimension indiscriminately.

The existing DP denoising logs show that guided steps usually improve score after the first late step, while no-guidance has zero per-step score change by construction. The applied update norm is stable at about `0.003`, because the prior audit used normalized guidance updates.

The policy-guidance alignment is close to zero and slightly negative on average. This suggests the current guidance is mostly a local side correction or mild counter-correction to DP, not a strongly aligned push. It is not obviously exploding, but the alignment evidence is still small-sample because it comes from the existing 3-window audit.

## Artifacts

- Overview figure: `outputs/board_stride3_val_guidance_debug_20260714/board_stride3_val_guidance_debug_overview.png`
- Action-dim heatmap: `outputs/board_stride3_val_guidance_debug_20260714/val_action_dim_guidance_heatmap.png`
- Per-row gradient CSV: `outputs/board_stride3_val_guidance_debug_20260714/val_action_gradient_samples.csv`
- Gradient tensor NPZ: `outputs/board_stride3_val_guidance_debug_20260714/val_action_gradient_tensors.npz`
- Summary JSON: `outputs/board_stride3_val_guidance_debug_20260714/summary.json`

## Next Step

After remounting the original stride-3 DP checkpoint, rerun the same diagnostic with fresh DP sampling on validation rows so A/B/D and C come from exactly the same validation windows.
