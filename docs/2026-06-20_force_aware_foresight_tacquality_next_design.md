# Force-Aware Foresight + TacQualityEnergy Next Design

Date: `2026-06-20`

## Motivation

Current board wiping guidance is mechanically implemented as gradient guidance:

```text
candidate action chunk
  -> Foresight predicts future marker sequence
  -> TacQualityEnergy scores predicted tactile consequence
  -> d score / d action
  -> bounded trust-region action update
```

The gap is not the existence of gradients.  The gap is that board quality is
defined by force magnitude and force smoothness, while the current serving-time
Foresight only predicts marker latent / decoded marker fields.  Force is used
offline to create weak labels and quality targets, but it is not directly
predicted in the action-to-score chain.

Current evidence:

- board offline classifier is strong: AUC / balanced accuracy can reach `1.0`
  on the current ForceBand scorer audit.
- board clean-action guidance is weak:
  - `score_delta_mean = 0.00016205`
  - `action_delta_norm_mean = 0.00077498`
- board predicted-score vs force-band quality Spearman is only `0.3988`.

Therefore the next useful model change is force-aware consequence prediction,
not more tactile-concat DP training.

## Proposed Model

Use the current multi-step Foresight as the base and add lightweight heads:

```text
history:
  RGB / optional vision tokens
  qpos
  left tactile marker history

candidate action chunk:
  joint action sequence, horizon 16

shared multi-step Foresight transformer
  -> z_hat[t+1:t+16]                  # tactile latent sequence
  -> marker_hat[t+1:t+16]             # decoded marker sequence through frozen TactileVAE
  -> force_proxy_hat[t+1:t+16]        # continuous force proxy
  -> force_band_logits[t+1:t+16]      # too_small / good / too_large / oscillate
  -> contact_gate_logit[t+1:t+16]     # whether the frame is in contact/wiping phase
```

The first version should avoid CVAE sampling.  For guidance, deterministic and
well-calibrated gradients are more important than diversity.  A stochastic
variant can be an ablation after the deterministic model is validated.

## Targets

Use the available board data with episode-level splits:

- positive / good:
  - `/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/wipe_pos_straight_z124_125_150_20260609`
  - `/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617`
- too small / too light:
  - `/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/z_too_high`
- too large / too heavy:
  - `/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_low`
- oscillatory / unstable:
  - `/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_oscillate`

Per-frame / per-window targets:

- marker latent target: frozen TactileVAE latent sequence.
- marker reconstruction target: decoded marker sequence matched to GT marker.
- delta target: temporal latent and marker deltas.
- force proxy target:
  - `force_mag = norm(force6d[:3])`
  - `fz`
  - `abs(diff(fz))` or local force jerk/smoothness
  - optional normalized force-in-band quality
- force band target:
  - good: force magnitude near positive-set center and low force derivative.
  - too_small: force below good lower band or too little marker/contact evidence.
  - too_large: force above safety/good upper band.
  - oscillate: high force derivative / high marker temporal instability.
- contact gate target:
  - contact if marker magnitude and/or force magnitude is above a robust episode
    threshold.
  - wiping phase should be sampled more heavily than approach/reset.

## Loss

Recommended first loss:

```text
L =
  L_latent_seq
  + w_final * L_latent_final
  + w_marker * L_marker
  + w_delta * L_delta
  + w_force * L_force_proxy
  + w_band * CE(force_band_logits, force_band_label)
  + w_contact * BCE(contact_gate, contact_label)
  + w_smooth * L_temporal_smoothness
```

Initial weights:

- `w_final = 1.0`
- `w_marker = 0.3`
- `w_delta = 0.5`
- `w_force = 0.5`
- `w_band = 0.5`
- `w_contact = 0.2`
- `w_smooth = 0.1`

These are starting values, not final claims.  Select by episode-level validation
and guidance-gradient audits.

## Guidance Score

For board wiping, the guidance score should be multi-horizon and contact-gated:

```text
score_board(action) =
  mean_h contact_gate_h * (
      + a * good_band_logit_h
      - b * too_small_logit_h
      - c * too_large_logit_h
      - d * oscillate_logit_h
      + e * force_in_band_score_h
      - f * force_derivative_penalty_h
      - g * marker_temporal_instability_h
      - h * action_acceleration_penalty_h
  )
```

Important details:

- Use logits / margins, not saturated probabilities, for guidance.
- Keep trust-region projection and accept-only update.
- Apply higher guidance weight only during predicted contact/wiping phase.
- Keep a small action smoothness penalty so guidance does not introduce high
  frequency joint motion.

For insertion, keep the current `good_margin` risk scorer as default because it
already has stronger guidance evidence.

## Evaluation Protocol

Offline checks required before real rollout:

1. Episode-level GroupKFold or held-out-episode validation.
2. Foresight prediction quality:
   - latent sequence loss
   - marker reconstruction loss
   - force proxy MAE / rank correlation
   - force band macro-F1
   - contact gate AUC / balanced accuracy
3. Score calibration:
   - score vs physical force-band quality Spearman
   - score decile monotonicity
   - top-bottom quality gap
4. Guidance-gradient audit:
   - finite gradient rate
   - positive gradient rate
   - score delta mean
   - action delta norm mean
   - trust-region pass rate
   - accept rate
5. Offline DP-chain audit:
   - use the current board DP `dp_best.pth`
   - compare no-guidance vs final-action guidance vs late-DDPM-step guidance
   - require nontrivial score deltas without trust-region violations

Real rollout evidence required before claiming task improvement:

- paired baseline/guided board trials.
- server-side force traces saved per trajectory.
- metrics:
  - percent time in force band
  - mean force magnitude
  - force derivative / jerk
  - contact loss duration
  - action smoothness
  - task completion / visible cleaning quality if available

## Expected Contribution Story

The strongest project story is:

```text
visual/proprio DP action prior
  + tactile/force future consequence model
  + interpretable physical contact-quality energy
  + bounded classifier/scorer gradient guidance
```

This is more novel and defensible than claiming tactile concat alone.  It also
matches recent trends around inference-time tactile steering, contact-aware
world models, and force-supervised contact-rich manipulation.

## Implementation Plan

1. Add a new script instead of modifying the current stable multi-step Foresight:

   `TFAC_V5/pretrain_latent_foresight_multistep_force.py`

2. Add a compatible dataset path that loads:

   - marker history
   - future marker sequence
   - qpos
   - action chunk
   - `observations/tac/left/force6d` or top-level `ft`

3. Add force/head outputs to the model:

   - `force_proxy_head`
   - `force_band_head`
   - `contact_gate_head`

4. Add an evaluation script:

   `TFAC_V5/tac_quality_energy/eval_force_aware_foresight_guidance.py`

5. Success criteria for first useful version:

   - force-band macro-F1 improves over marker-only proxy.
   - predicted score vs physical force-band quality Spearman exceeds `0.50`.
   - board guidance `score_delta_mean` is meaningfully above the current
     `0.00016205` while keeping trust-region pass rate near `1.0`.
   - no real-robot improvement claim until paired force traces exist.

## Current Code Landing Points

Checked on `2026-06-20` while the 260617-only board DP run was still training.

Stable files to build on:

- `TFAC_V5/pretrain_latent_foresight_multistep.py`
  - current clean multi-step marker-latent Foresight training path.
  - currently returns future tactile VAE latent sequence and decoded marker loss only.
  - loss is `L_latent + lambda_marker * L_marker + lambda_delta * L_delta`.
- `TFAC_V5/foresight_multistep.py`
  - current `MultiStepSpatialForesightTransformer`.
  - output is `t_hat_future: (B, H, 9 * latent_dim)`.
  - no force or contact heads yet.
- `TFAC_V5/dataset.py`
  - current `ForesightEpisodicDataset`.
  - loads marker history/future, qpos, and action chunks.
  - does not currently return force windows.
- `TFAC_V5/tac_quality_energy/force_band_runtime.py`
  - deployable current board scorer.
  - differentiable serving input is predicted marker proxy + action proxy.
  - force is used in offline label/target construction, not directly predicted at serving time.
- `TFAC_V5/tac_quality_energy/foresight_bridge.py`
  - current differentiable serving bridge:
    `raw action -> Foresight -> decoded marker -> TacQuality score`.
  - currently exposes `left_marker_seq`, mirrored `right_marker_seq`, and action proxies.
- `TFAC_V5/tac_quality_energy/build_board_predicted_domain_features.py`
  - current predicted-domain feature builder.
  - useful reference because it already materializes Foresight-predicted marker features and force-band labels.

Temporary or audit-heavy files:

- `TFAC_V5/_agent_tmp_scripts_20260609_10/`
  - contains many generated smoke/audit helpers.
  - useful for reference, but not the clean landing place for the next model.

## Minimal Code Path for the Next Version

Do not modify the existing stable marker-only multi-step Foresight first.  Add a
parallel force-aware path:

1. New dataset wrapper or subclass:

   `ForceAwareForesightEpisodicDataset`

   It should reuse `ForesightEpisodicDataset` logic but additionally return:

   - `future_force_window`: `(B, H, 6)` or `(B, H, force_dim)`
   - `force_proxy_target`: force magnitude, `fz`, force delta, and smoothness proxy
   - `force_band_target`: `too_small / good / too_large / oscillate`
   - `contact_gate_target`: contact/wiping mask from marker and force thresholds

   Required HDF5 fallback order:

   ```text
   observations/tac/{side}/force6d
   ft
   zeros fallback only for smoke tests, not for real training
   ```

2. New model wrapper:

   `ForceAwareMultiStepLatentForesightModel`

   Reuse the current encoder and `MultiStepSpatialForesightTransformer`, then
   add small heads on the multi-step future embedding:

   ```text
   force_proxy_head:  (B, H, D_embed) -> (B, H, K_force)
   force_band_head:   (B, H, D_embed) -> (B, H, 4)
   contact_gate_head: (B, H, D_embed) -> (B, H)
   ```

   Current `MultiStepSpatialForesightTransformer.forward` already returns
   `t_embed_future: (B, H, D)`, so the force heads can attach there with minimal
   change.

3. New loss:

   ```text
   L =
     L_latent
     + w_marker * L_marker
     + w_delta * L_delta
     + w_force * SmoothL1(force_proxy_hat, force_proxy_target)
     + w_band * CE(force_band_logits, force_band_target)
     + w_contact * BCE(contact_gate_logit, contact_gate_target)
     + w_smooth * temporal_smoothness
   ```

4. New eval script:

   `TFAC_V5/tac_quality_energy/eval_force_aware_foresight_guidance.py`

   It should report:

   - latent/marker/delta losses
   - force proxy MAE and Spearman
   - force-band macro-F1
   - contact-gate AUC / balanced accuracy
   - predicted score vs physical force-band quality Spearman
   - guidance audit: finite gradient, score_delta, action_delta_norm, trust-region pass rate

5. Serving bridge extension:

   Add a config-gated force-aware bridge path after the model is trained:

   ```text
   raw action
     -> force-aware Foresight
     -> decoded marker + force_proxy + force_band_logits + contact_gate
     -> contact-gated TacQualityEnergy score
     -> trust-region gradient update
   ```

   Keep the current marker-only bridge as the default until the force-aware
   model passes offline and guidance-gradient audits.

## Why This Is the Right Next Engineering Step

The current board scorer is strong offline, but the action-side gradient is
weak because serving-time scoring only sees predicted marker proxies.  Board
quality itself is defined by force magnitude, force band, and force smoothness.
Therefore the missing link is not another tactile-concat DP training run; it is
predicting force/contact quality as part of the differentiable consequence model.

The design keeps three important constraints:

- deterministic gradients for DP classifier/scorer guidance;
- physical interpretability through force-band/contact heads;
- backward compatibility with the current marker-only guidance path.

## 2026-06-21 Implementation Landing

Added the parallel force-aware training path without modifying the stable
marker-only multi-step Foresight:

- `TFAC_V5/pretrain_latent_foresight_multistep_force.py`
- `TFAC_V5/config_pretrain_foresight_board_forceaware_multistep16.json`
- `TFAC_V5/config_pretrain_foresight_board_forceaware_smoke.json`
- `scripts/train/train_foresight_board_forceaware_multistep16.sh`

What the new code implements:

```text
marker history window + qpos + future qpos/action chunk
  -> MultiStepSpatialForesightTransformer
  -> z_pred[t+1:t+16]
  -> marker_pred[t+1:t+16]
  -> force_proxy_pred[t+1:t+16]
  -> force_band_logits[t+1:t+16]
  -> contact_gate_logits[t+1:t+16]
```

Targets loaded from the current board datasets:

- `observations/tac/left/marker_offset`
- `observations/tac/left/force6d`
- `observations/proprio_joint`
- `actions/joint_abs`

Verified dataset roots:

- positive: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/wipe_pos_straight_z124_125_150_20260609`
- positive 260617: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617`
- too small / too light: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/z_too_high`
- too large / too heavy: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_low`
- oscillatory / unstable: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_oscillate`

Force-aware labels:

- `too_small`: low force/contact relative to positive-set robust band.
- `good`: positive-set contact inside the robust force band.
- `too_large`: force above the robust positive-set band.
- `oscillate`: high force-change proxy.
- `contact_gate`: marker or force above robust contact threshold.

Loss implemented:

```text
L =
  L_latent_seq
  + w_final * L_latent_final
  + w_marker * L_marker
  + w_delta * L_delta
  + w_force * SmoothL1(force_proxy)
  + w_band * CE(force_band)
  + w_contact * BCE(contact_gate)
  + w_smooth * SmoothL1(force_delta)
```

Smoke checks completed:

- `python -m py_compile TFAC_V5/pretrain_latent_foresight_multistep_force.py`
- CPU dataset smoke:
  - `marker_hist`: `(8, 9, 9, 2)`
  - `qpos`: `(7,)`
  - `action`: `(16, 7)`
  - `future_marker`: `(16, 8, 9, 9, 2)`
  - `future_force_proxy`: `(16, 6)`
  - `future_force_band`: `(16,)`
  - `future_contact`: `(16,)`
- CPU model forward/backward smoke:
  - `z_pred`: `(2, 16, 144)`
  - `force_proxy_pred`: `(2, 16, 6)`
  - `force_band_logits`: `(2, 16, 4)`
  - `contact_logits`: `(2, 16)`
  - `marker_pred`: `(2, 16, 9, 9, 2)`
  - backward pass completed.
- Command-line entry smoke:
  - forced CPU with `CUDA_VISIBLE_DEVICES=''`.
  - ran `TFAC_V5/pretrain_latent_foresight_multistep_force.py --config /tmp/tactileact_forceaware_smoke_cpu.json`.
  - completed one smoke epoch and wrote outputs under
    `/tmp/tactileact_forceaware_foresight_entry_smoke/entry_smoke_forceaware_foresight`.
  - smoke final metrics were only a wiring check, not a model-quality claim:
    `best_val_total=10.7972`, `force_proxy_mae=0.7552`,
    `contact_acc=0.34375`, `band_balanced_acc=1.0`.

The full GPU training was not started during this update because the
260617-only DP training run was still occupying the GPU.  The intended launch is:

```bash
cd /home/chenshuai/Project/TactileACT-cs
CONFIG=TFAC_V5/config_pretrain_foresight_board_forceaware_multistep16.json \
  scripts/train/train_foresight_board_forceaware_multistep16.sh
```

Next required step after the current DP run releases the GPU:

1. Train this force-aware Foresight.
2. Evaluate force proxy MAE, force-band macro-F1, and contact-gate accuracy on
   held-out episodes.
3. Add a force-aware serving bridge that scores:

   ```text
   contact_gate * (
       good_band_logit
       - too_small_logit
       - too_large_logit
       - oscillate_logit
       - force_smoothness_penalty
   )
   ```

4. Re-run gradient audits and compare score/action delta against the current
   marker-only board guidance path.
