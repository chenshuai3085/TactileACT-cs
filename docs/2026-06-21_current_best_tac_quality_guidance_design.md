# Current Best TacQuality Guidance Design

## Goal

The current objective is not ordinary offline classification.  The target
module must be useful as diffusion-policy classifier/scorer guidance:

```text
DP action prior
  -> candidate action chunk
  -> tactile/force consequence prediction
  -> TacQuality score
  -> bounded gradient update on the action chunk
```

The score must therefore satisfy three requirements:

1. It must classify or rank contact quality correctly under task-specific
   labels.
2. It must provide a useful differentiable gradient with respect to the action
   chunk through the Foresight consequence model.
3. It must stay inside a trust region so score maximization does not push the
   policy off the demonstrated action manifold.

Real robot improvement is not proven yet.  The current evidence supports
offline scorer quality, differentiable guidance readiness, serving integration,
and real-rollout manifest readiness.

## Current Recommendation

| task | recommended arm | status | reason |
|---|---|---|---|
| socket insertion | `good_margin_guided` | current default for real A/B testing | unsaturated good-vs-risk logit margin gives cleaner gradients than saturated `p_good` |
| board wiping | `force_aware_guided` | scientific priority for real A/B testing | scores predicted force/contact consequences directly and has the stronger bounded guidance signal |
| board wiping | `marker_joint_s12_guided` | integrated fallback/comparison | four-class force-band scorer works with deployable marker/action features, but current gradient signal is weak |

## Socket Insertion Scorer

Implementation:

```text
runtime     TFAC_V5/tac_quality_energy/insertion_runtime.py
class       InsertionRiskScorerRuntime
checkpoint  /home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt
arm         good_margin_guided
score       binary_logits[:, good] - binary_logits[:, bad]
```

Input features:

```text
left tactile marker sequence
joint action sequence
marker proxy features
action proxy features
```

Label standard:

```text
good:
  good_insert

bad:
  pre_bounce_risk
  impact_or_recovery

neutral / excluded from binary:
  weak_approach
```

Why `good_margin`:

`p_good` separates good and bad examples offline, but it saturates when the
classifier is confident.  A saturated probability can have a weak gradient even
when the action still needs correction.  The current insertion score is the
binary logit margin:

```text
S_insert = logit_good - logit_bad
```

This keeps the score less saturated and more useful for gradient guidance.

Current evidence:

```text
Group-CV binary AUC              0.9877
Group-CV balanced accuracy       0.9437
reason macro F1                  0.7894
quality correlation              0.7656

matched 0209 gradient improve    0.9167
matched 0401 gradient improve    1.0000
trust-region pass rate           1.0000
good-margin DDPM improve rate    0.9375
```

Evidence boundary:

This is ready for paired real insertion tests, but not yet proven to improve
real insertion success, bounce count, or retry count.

## Board Integrated Fallback/Comparison Scorer

Implementation:

```text
runtime     TFAC_V5/tac_quality_energy/force_band_runtime.py
class       ForceBandTacQualityEnergyRuntime
checkpoint  /home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/force_band_tac_quality_energy_best.pt
arm         marker_joint_s12_guided
score       quality
```

Input features:

```text
left marker proxy
right marker proxy
left/right marker difference proxy
joint action proxy
```

The model predicts several heads:

```text
binary head   good / bad
reason head   too_small / positive / too_large / oscillate
quality head  continuous tactile quality
teacher head  soft teacher target
energy head   residual energy
```

Current default score:

```text
S_board_deploy = sigmoid(quality_logit)
```

Quality label standard:

```text
positive:
  contact force magnitude in a reasonable band
  force changes smoothly
  marker motion is stable

negative:
  too_small    force too light / poor contact
  too_large    force too large
  oscillate    unstable alternating force
```

The offline quality target used for the s12 board scorer is:

```text
quality = 0.62 * force_band + 0.25 * force_smooth + 0.13 * marker_smooth
```

Current evidence:

```text
held-out binary AUC              1.0000
held-out balanced accuracy       1.0000
reason macro F1                  1.0000
quality Spearman                 0.9239

Foresight predicted-score AUC    1.0000
predicted vs GT score Spearman   0.5291
gradient improve rate            1.0000
finite gradient rate             1.0000
trust-region pass rate           1.0000
```

Important caveat:

This board scorer is practical because it can run from predicted marker/action
features and remains useful as an integrated comparison arm.  But board wiping
quality is physically defined by force magnitude and force smoothness, and its
current bounded gradient signal is weak, so it is not the scientific priority
for the final TacQuality guidance story.

## Board Force-Aware Scientific Priority

Implementation:

```text
runtime     TFAC_V5/tac_quality_energy/force_aware_guidance_runtime.py
class       ForceAwareForesightGuidanceRuntime
adapter     ForceAwareBoardGuidanceAdapter
foresight   /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_forceaware_multistep16_boardvae_e100_bs16_0/foresight_force_best.ckpt
arm         force_aware_guided
```

This candidate scores the predicted force/contact consequences directly:

```text
candidate action chunk
  -> force-aware Foresight
  -> future force-band logits
  -> contact logits
  -> force proxy trajectory
  -> TacQuality score
```

Score:

```text
S_force_aware =
    1.00 * good-vs-risk force-band margin
  + 0.20 * contact log probability
  - 0.25 * force-center penalty
  - 0.10 * force-smoothness penalty
```

Where:

```text
force-band margin = logit_good - logsumexp(logit_too_small, logit_too_large, logit_oscillate)
```

This is currently the stronger research story because it matches the board
definition directly:

```text
good board wiping = enough contact + not too much force + smooth force changes
```

Current offline audit:

```text
force-band balanced accuracy     0.9736
contact accuracy                 0.9207
good/bad score AUC               1.0000
finite gradient rate             1.0000
positive gradient rate           1.0000
gradient improved rate           0.9409
trust-region pass rate           1.0000
score delta mean                 3.5201
```

Real-HDF5-window serving audit:

```text
windows                          80
labels                           16 each for positive_old, positive_260617, too_small, too_large, oscillate
finite gradient rate             1.0000
positive gradient rate           1.0000
improved rate                    1.0000
accept rate                      1.0000
trust-region pass rate           1.0000
score delta mean                 1.8718
raw action delta mean            0.0760
normalized action delta mean     0.0175
```

Status:

```text
force_aware_board_gradient_audit_ready       true
force_aware_board_serving_smoke_ready        true
force_aware_board_real_window_serving_ready  true
force_aware_board_rollout_manifest_ready     true
force_aware_board_real_rollout_complete      false
```

The paired real rollout manifest is ready for three board pairs:

```text
baseline arm   baseline, port 8765
guided arm     force_aware_guided, port 8769
rollout root   /home/chenshuai/Project/output/board_force_rollouts/260617_only_force_aware_scorer
coverage       {"missing": 6}
```

Evidence boundary:

This candidate has the best scientific alignment and gradient evidence and is
the board arm to prioritize for paired real tests, but it is not yet a real
robot performance claim because the six planned real `force_trace.csv` files
are still missing.

## Guidance Mechanism

The guidance update is implemented in:

```text
TFAC_V5/tac_quality_energy/trust_region.py
TFAC_V5/tac_quality_energy/serving_guidance.py
```

The core update is:

```text
a_base = DP denoised clean action chunk
score(a) = TacQuality(Foresight(obs, a))
grad = d score / d a
a_proposal = a + step_size * normalize(grad)
a_guided = project_to_trust_region(a_proposal, a_base)
accept only if score(a_guided) > score(a_current)
```

Current trust-region behavior:

```text
accept-only improved score
finite gradient check
positive gradient norm check
max total action delta projection
normalizer maps between DP-normalized action and raw joint action
```

This is classifier/scorer guidance.  It is not reranking because the action
tensor itself is updated by gradients.

## Why This Is the Current Best Design

1. It keeps task-specific quality standards.

   Insertion failure is bounce/risk.  Board failure is too-light, too-heavy, or
   unstable force.  A single generic binary classifier would hide these physical
   differences.

2. It keeps the score differentiable.

   The scorer is useful only if `d score / d action` exists through Foresight.
   All current recommended paths are audited for finite and positive gradients.

3. It avoids probability saturation.

   Insertion uses logit margin instead of `p_good`.  Board force-aware uses
   force-band logit margin rather than only a binary probability.

4. It uses trust-region guidance.

   This protects the DP action prior and prevents arbitrary score hacking.

5. It leaves a clean novelty story.

   The DP is the action prior.  The contribution is outcome-aware
   tactile/force consequence scoring and bounded action-gradient guidance.

## Remaining Required Evidence

Before claiming final success, the following evidence is still missing:

1. Board paired real rollout evidence:

   ```text
   3 baseline force traces
   3 force_aware_guided force traces
   server-side force_trace.csv for every trial
   paired evaluation of force band, smoothness, contact continuity
   ```

2. Insertion paired real rollout evidence:

   ```text
   baseline vs good_margin_guided
   success / stopped_early / bounce_count / retry_count metadata
   at least three complete pairs
   ```

3. Final scorecard completion:

   ```text
   real_paired_rollout_complete = true
   goal_complete = true
   ```

At the current state:

```text
real_paired_rollout_complete = false
goal_complete = false
```
