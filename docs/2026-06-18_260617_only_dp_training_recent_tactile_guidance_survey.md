# 260617-only DP Training and Recent Tactile Guidance Survey

Date: 2026-06-18

## Active Training Run

Task: train tactile Diffusion Policy only on the 2026-06-17 board-wiping collection.

Dataset:

- `/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617`
- `episode_*.hdf5`: 80 files found
- 1 file is missing `observations/proprio_joint` and is skipped by the loader
- Effective split from current run:
  - train: 71 episodes, 8192 windows
  - val: 8 episodes, 1024 windows

Run directory:

- Real output: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260618_ext`
- Home symlink: `/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260618_ext`

Key configuration:

- Script: `diffusion/train_dp_tac_concat.py`
- Input: raw RGB `global,wrist` + left tactile VAE latent + proprio joint
- Image size: `200x266`
- Tactile VAE: `/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt`
- Tactile history: 8
- Prediction horizon: 16
- Observation horizon: 2
- Action execution horizon: 8
- Epochs: 2000
- Batch size: 64
- LR: `1e-4`
- Diffusion timesteps: train 100 / inference 100
- Episode-level validation ratio: 0.1
- Save frequency: every 100 epochs
- Latest checkpoint frequency: every 25 epochs
- Top-k train checkpoints: disabled for this run

Initial loss trend:

| Epoch | Train loss | Val loss | Best val |
|---:|---:|---:|---:|
| 1 | 0.600890 | 0.176095 | 0.176095 |
| 2 | 0.125052 | 0.091765 | 0.091765 |
| 4 | 0.064794 | 0.057585 | 0.057585 |
| 6 | 0.041821 | 0.038877 | 0.038877 |
| 7 | 0.035539 | 0.034096 | 0.034096 |
| 100 | 0.010128 | 0.014296 | 0.012134 |
| 105 | 0.009626 | 0.011387 | 0.011387 |
| 127 | 0.008727 | 0.014520 | 0.011387 |

Status at 2026-06-18 23:04:

- Training process: running.
- Latest parsed epoch: 127 / 2000.
- Current best validation checkpoint: epoch 105, val loss `0.011387`.
- Saved checkpoints observed:
  - `dp_best.pth`
  - `dp_latest.pth`
  - `dp_epoch100.pth`
- GPU: RTX 4090, about 14.7 GB used, high utilization when batches are running.
- Current judgment: continue training. Validation has not improved after epoch 105 yet, but the no-improvement span is still short relative to a 2000-epoch run, and `dp_best.pth` protects the best observed model.

Current evidence boundary:

- This is an offline DP training run.
- Episode-level validation loss is useful for supervision, but it does not prove real board-wiping quality.
- Real quality still requires robot rollouts with saved force traces and board task metrics.

## Recent Related Work

The most relevant recent works point in a consistent direction: tactile should not be treated only as another observation concatenated into the base policy. For contact-rich manipulation, touch is more useful as a predictive contact prior, a feasibility/quality constraint, or a high-frequency correction signal.

### Dream-Tac, June 2026

Paper: `Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation`

Link: https://arxiv.org/html/2606.08737v1

Relevant idea:

- Jointly models action, future visual observations, and future tactile dynamics.
- Uses contact-gated visuotactile fusion and contact-aware attention bias.
- Motivation matches our problem: vision alone is weak for contact-state changes, and tactile events are sparse but critical.

Implication for this project:

- Our current split design, `DP action generator + Foresight future tactile predictor + TacQuality scorer`, is conceptually aligned.
- A reasonable next improvement is to add contact-aware gating so tactile/quality guidance is strongest during wiping contact and weak during approach/reset.

### FAWAM, June 2026

Paper: `FAWAM: Force-Aware World Action Models for Closed-Loop Contact-Rich Manipulation`

Link: https://arxiv.org/abs/2606.08555

Relevant idea:

- Treats force as more than an observation feature.
- Uses force at three levels: perception, future prediction, and online residual correction.
- Jointly predicts future actions and end-effector wrench trajectories, then uses predicted force as a reference for execution-time correction.

Implication for this project:

- This is directly relevant to board wiping because our quality definition is force-band plus smoothness.
- Current DP only consumes tactile latent and images, while the scorer/guidance side uses force-derived labels. A stronger next version should make force trajectory a first-class prediction/constraint:
  - DP or Foresight predicts future marker and optionally future force proxy;
  - TacQuality scores whether predicted future contact stays inside the desired force band;
  - online rollout compares real force against predicted/desired force and logs deviations.
- For the paper story, this supports describing our method as `future contact outcome guidance`, not just tactile concatenation.

### TacForeSight, June 2026

Paper: `TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation`

Link: https://arxiv.org/abs/2606.11184

Relevant idea:

- Predicts short-horizon tactile latent dynamics from current tactile observations conditioned on high-frequency wrist force/torque.
- Uses predicted tactile latents as anticipatory contact priors for policy control.
- Focuses on compact latent prediction rather than high-dimensional video prediction for real-time contact reasoning.

Implication for this project:

- Our Foresight module currently predicts future tactile latent from action/state/vision/tactile.
- For board wiping, adding measured force/torque as an input to Foresight or TacQuality is likely more useful than only marker offsets, because the task definition itself is force-band and smoothness based.
- This is especially relevant for distinguishing `too small force`, `too large force`, and `oscillatory force`.

### ContactWorld, June 2026

Paper: `ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation`

Link: https://arxiv.org/html/2606.13877v1

Relevant idea:

- Provides a benchmark and empirical study for vision-tactile world models across contact-rich tasks.
- The important message for us is evaluation design: world-model quality should be judged by task-relevant contact prediction and downstream control usefulness, not only reconstruction loss.

Implication for this project:

- Our Foresight evaluation should report:
  - future marker/latent prediction error;
  - predicted TacQuality score vs GT-future TacQuality score;
  - predicted score vs real rollout force-band and smoothness metrics.
- This avoids the weak claim "Foresight MSE is low therefore guidance will work." The stronger claim is "predicted contact quality is aligned with measured contact quality."

### DPTG, June 2026

Paper: `DPTG: diffusion policy with tactile feasibility guidance`

Link: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1851102/full

Relevant idea:

- Reformulates tactile sensing as a feasibility constraint for diffusion policy rather than symmetric feature fusion.
- Uses tactile feasibility guidance during policy generation.

Implication for this project:

- This supports our current direction: the scorer should be used as a differentiable constraint/energy during DP denoising, not as reranking only.
- For our board task, the feasibility/quality target should encode:
  - in target force band,
  - low force derivative / smooth contact,
  - stable marker deformation,
  - action smoothness and plausible contact phase.

### TouchGuide, May 2026 version

Paper: `TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance`

Link: https://arxiv.org/html/2601.20239v6

Relevant idea:

- Keeps the base visuomotor policy fixed.
- Uses a task-specific Contact Physical Model to output a feasibility score.
- Applies score gradients during diffusion or flow-matching inference.
- Adds noise-aware training to make the scorer useful on noisy denoising-time actions.

Implication for this project:

- Our TacQuality scorer should be trained not only on clean ground-truth action/marker chunks, but also on noised action chunks matching DP denoising distribution.
- This is likely important if we want stable gradient guidance inside denoising, rather than only final-action refinement.

### ViTaL, June 2026

Paper: `Inference-time Policy Steering via Vision and Touch`

Link: https://arxiv.org/abs/2606.14981

Relevant idea:

- Formulates multimodal inference-time steering as a two-level optimization problem.
- High level: visual sampling and verification handles long-horizon mode selection.
- Low level: tactile-guided diffusion editing refines a short local action horizon to satisfy contact requirements.

Implication for this project:

- This is very close to our intended deployment path:
  - DP provides a plausible action chunk.
  - Foresight predicts the tactile consequence of that chunk.
  - TacQuality edits only the contact-sensitive local chunk through gradients.
- For board wiping, guidance should focus on the wiping contact segment rather than uniformly editing approach/reset.

### LaWAM, June 2026

Paper: `LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies`

Link: https://arxiv.org/html/2606.15768v1

Relevant idea:

- Predicts compact latent future subgoals instead of expensive pixel-level futures.
- Uses latent future prediction to make action generation dynamics-aware while keeping inference efficient.

Implication for this project:

- Our TactileVAE-latent Foresight path is consistent with this: predict compact future contact state rather than full images.
- For the next iteration, it is better to improve latent/contact quality alignment and force conditioning than to switch to heavy video prediction.

### AdaVTF, April 2026

Paper: `Learning When to See and When to Feel: Adaptive Vision-Torque Fusion for Contact-Aware Manipulation`

Link: https://arxiv.org/abs/2604.01414

Relevant idea:

- Studies F/T-vision fusion inside diffusion-based contact-rich manipulation policies.
- Proposes adaptive integration: suppress force/torque influence during non-contact phases and use it strongly during contact.
- Reports a success-rate improvement over their strongest baseline.

Implication for this project:

- TacQuality guidance should have an explicit contact-phase gate.
- For board wiping:
  - approach: guidance near zero;
  - stable wiping contact: force/marker quality guidance active;
  - leaving/reset: guidance decays again.
- This is a cleaner story than always concatenating tactile or force features with the same weight.

### ProgressVLA, March 2026

Paper: `Progress-Guided Diffusion Policy for Vision-Language Robotic Manipulation`

Link: https://arxiv.org/html/2603.27670v1

Relevant idea:

- Uses an explicit progress estimator as classifier guidance for diffusion policy.
- Shows that a scalar score can improve action sampling if the score is meaningful and differentiable.

Implication for this project:

- A board-wiping score can be more than binary good/bad. A scalar quality score is more suitable for gradients:
  - force-band score,
  - smoothness score,
  - contact consistency score,
  - progress/coverage score if we can estimate wiping progress from trajectory or image.

### Multi-Resolution Tactile Imitation Learning, June 2026

Paper: `Multi-Resolution Tactile Imitation Learning for Contact-Rich Robotic Manipulation`

Link: https://arxiv.org/html/2606.06281v1

Relevant idea:

- Tactile signals can carry both local high-frequency contact information and lower-frequency task context.

Implication for this project:

- Current DP uses an 8-frame tactile history encoded into one latent per observation step.
- For wiping, it may be useful to keep both:
  - short-window high-frequency force/marker derivative features for contact smoothness,
  - 16-step future predicted tactile chunks for quality guidance.

### Tube Diffusion Policy, April 2026

Paper: `Tube Diffusion Policy: Reactive Visual-Tactile Policy Learning for Contact-rich Manipulation`

Link: https://arxiv.org/abs/2604.23609

Relevant idea:

- Action chunking alone can be too slow to react to contact disturbances.
- Learns a local feedback flow around nominal diffusion-policy action chunks, forming an action tube for fast correction.

Implication for this project:

- Our current guidance edits a DP action chunk through TacQuality gradients. This is close to an action-tube idea, but our correction signal comes from predicted tactile quality.
- If real rollout shows delayed correction, the next improvement should be step-wise or short-subchunk guidance during execution, not simply a larger classifier.

## Current Architecture Assessment

Current pipeline:

1. TactileVAE compresses marker fields.
2. DP predicts a 16-step joint-action chunk from RGB + tactile latent + qpos.
3. Foresight predicts future tactile latent/marker consequence from candidate action.
4. TacQuality energy/scorer evaluates predicted tactile consequence.
5. Guidance updates action using score gradients.

This is a defensible story:

- DP is the behavior prior.
- Foresight is the tactile consequence model.
- TacQuality is the differentiable task-quality constraint.
- Guidance modifies actions toward better predicted contact outcome.

Main weakness:

- The quality scorer currently depends heavily on marker/latent proxy labels and offline class construction.
- For board wiping, true quality is force-band + force smoothness + stable contact; therefore force traces should become part of scorer training/evaluation as soon as real rollout data is available.
- Current policy training and scorer/guidance are still separated. This is useful scientifically because the policy prior and quality constraint can be analyzed independently, but the deployment proof must show the full chain:
  `DP action -> Foresight prediction -> TacQuality score -> bounded gradient update -> real force curve improvement`.

## Architecture Story After Latest Survey

The most defensible story for this project is:

1. **Base behavior prior**: train DP on high-quality or task-specific demonstrations so it generates plausible board-wiping action chunks.
2. **Predictive contact model**: use TactileVAE + Foresight to predict future tactile/contact consequence of a candidate action.
3. **Differentiable contact-quality energy**: score the predicted future with a task-defined quality function:
   - board: force in target band, low force derivative, stable contact marker field;
   - insertion: low bounce risk, good insertion contact signature.
4. **Contact-gated gradient guidance**: only apply guidance strongly during contact/wiping; keep approach/reset mostly governed by DP.
5. **Real rollout verification**: save server-side force traces and compare baseline vs guided using contact-phase metrics.

This is stronger than plain tactile concatenation because the tactile module is not only an input feature; it becomes a predicted consequence and a differentiable constraint on the action.

## Recommended Next Improvements

Priority 1: train and evaluate the current 260617-only DP to completion or until validation clearly plateaus.

Priority 2: add noised-action scorer training for TacQuality.

- Input clean GT action/marker chunks and noised action chunks sampled with DP scheduler.
- Train scorer to remain calibrated on noisy denoising-time action proposals.
- This directly supports classifier guidance rather than only post-hoc scoring.

Priority 3: add contact-phase gating.

- During approach/non-contact, guidance should be weak or zero.
- During wiping contact, guidance should be strong.
- Gate can be computed from marker magnitude, marker area, or force if available.

Priority 4: add force-conditioned board scorer.

- Good: force inside target band and low derivative.
- Bad classes:
  - pressure too small,
  - pressure too large,
  - pressure oscillatory/unstable.
- Score should be scalar and differentiable:
  - `quality = w_band * force_band + w_smooth * smoothness + w_marker * contact_stability + w_action * action_smoothness`
  - learned model can distill this into a neural energy function.

Priority 5: evaluate with paired real rollouts.

- Offline val loss and scorer AUC are not enough.
- Need baseline/guided paired trajectories with force traces saved per trial.
- Compare force-in-band ratio, force smoothness, mean force error, trajectory completion, and visual task outcome.

## Concrete Next Experiments

1. Let the current 260617-only DP continue. Use `dp_best.pth` for deployment comparison unless a later epoch improves validation loss.
2. After this run, compare three policy priors on real robot:
   - previous full board dataset DP;
   - positive-only board DP;
   - current 260617-only DP.
3. For each policy prior, run baseline and TacQuality-guided mode with server-side force logging.
4. Score each rollout by contact-phase:
   - force-in-band ratio;
   - Fz mean and p95;
   - `|dFz|` mean/p95;
   - marker magnitude stability;
   - whether wiping completes.
5. If TacQuality guidance improves force smoothness but hurts coverage, add a visual/progress verifier. If it improves offline score but not real force curves, debug Foresight alignment and contact gate first.
