# 2026-06-18 With-260617 ForceBand Scorer Foresight Alignment Audit

## Purpose

Check the current board scorer candidate under the actual guidance chain:

- runtime: `ForceBandTacQualityEnergyRuntime`
- checkpoint: `/home/chenshuai/Project/output/board_force_band_tac_quality_energy_with_260617_positive_20260618/force_band_tac_quality_energy_best.pt`
- data: old positive + 260617 positive + too-small + too-large + oscillate
- chain: `action -> multistep Foresight -> predicted future marker -> scorer score`

This experiment asks whether the scorer is suitable for DP classifier/energy guidance, not only whether it classifies offline features well.

## Command

Output root:

```text
/home/chenshuai/Project/output/tac_quality_force_band_with260617_score_mode_sweep_20260618
```

Each score mode used the same settings:

```bash
conda run --no-capture-output -n TactileACT python TFAC_V5/tac_quality_energy/eval_foresight_score_alignment.py \
  --output_dir <ROOT>/<mode> \
  --scorer_runtime ForceBandTacQualityEnergyRuntime \
  --scorer_ckpt /home/chenshuai/Project/output/board_force_band_tac_quality_energy_with_260617_positive_20260618/force_band_tac_quality_energy_best.pt \
  --score_mode <mode> \
  --include_260617_positive \
  --max_episodes_per_class 8 \
  --samples_per_episode 2 \
  --max_samples 80 \
  --force_center 10.339887619018555 \
  --force_sigma 3.5647058646536496 \
  --force_delta_ref 0.5364861339330673 \
  --marker_delta_ref 0.2703184187412262 \
  --gpu -1
```

## Results

| mode | n | pred AUC(good) | GT AUC(good) | pred/GT Spearman | score vs force-quality Spearman | score vs -marker-MAE Spearman | label mean range |
|---|---:|---:|---:|---:|---:|---:|---:|
| `p_good` | 78 | 0.6514 | 0.5486 | 0.8019 | -0.4942 | -0.4742 | 0.00000041 |
| `energy_clipped` | 78 | 0.6431 | 0.4417 | 0.5331 | -0.3978 | -0.5090 | 0.00074194 |
| `profile` | 78 | 0.6431 | 0.4056 | 0.4133 | -0.3665 | -0.5322 | 0.00275540 |
| `quality` | 78 | 0.5604 | 0.3194 | 0.4385 | -0.0011 | -0.3611 | 0.00029819 |
| `reason_good` | 78 | 0.5253 | 0.4191 | 0.6455 | -0.1848 | -0.1874 | 0.00000058 |

By-label mean scores show saturation:

- `p_good` is nearly constant around `0.99989` for every class.
- `reason_good` is nearly constant around `0.999764` for every class.
- `quality` is nearly constant around `0.8953-0.8956`.
- `energy_clipped` and `profile` have slightly larger numeric range, but still weak separation.

## Interpretation

This is an important limiting result.

The with-260617 scorer has strong offline held-out classification metrics:

- AUC `0.999929`
- bACC `0.997172`
- reason macro F1 `0.997304`
- quality Spearman `0.971293`

But when it is used through the actual Foresight guidance chain, all score modes are almost saturated and provide weak ordering. This means:

1. Good offline classification does not guarantee useful continuous gradient guidance.
2. The current `quality` mode should not be treated as a final board guidance score.
3. The runtime score needs better calibration and dynamic range on predicted/GT future marker inputs.
4. Board guidance should move toward a physics-continuous energy, not only classification heads.

## Likely Causes

- The scorer is trained on marker/action proxy features, not direct force.
- 260617 positive and too-small force regimes overlap in force magnitude.
- Force-band quality alone has poor AUC (`0.3674`) against collection labels in this mixed sample.
- Foresight prediction for 260617 positive has high marker MAE in this sample (`positive_260617` mean marker MAE about `1.70`).
- Collection labels and physical force quality are partially conflicting: `too_small` can have high computed force-band quality because it is smooth and near the selected force center, even though the task label is negative.

## Design Consequence

The next board scorer should be a calibrated continuous energy:

```text
E_board = w_label * E_label_margin
        + w_force * E_force_band_proxy
        + w_smooth * E_temporal_smoothness
        + w_contact * E_contact_gate
        + w_action * E_action_jerk_trust_region
```

The deployable guidance score should pass three checks before use:

1. It separates good/bad labels under episode-level splits.
2. It has non-saturated score range on `score(GT future marker)` and `score(Foresight(action))`.
3. It correlates with contact-phase physical quality proxies and, finally, real rollout force curves.

## Evidence Boundary

This is offline Foresight-alignment evidence only. It does not prove real robot guidance failure or success. It does prove that the current with-260617 scorer should be treated as a candidate needing recalibration, not as a final board guidance scorer.
