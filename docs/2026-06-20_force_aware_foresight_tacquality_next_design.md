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
