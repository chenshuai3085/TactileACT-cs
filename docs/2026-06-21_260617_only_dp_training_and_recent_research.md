# 2026-06-21 260617-only DP training and recent research notes

## Current DP run

Goal: train a deployment-compatible tactile DP concat policy using only the
2026-06-17 board-wiping collection.

Dataset:
- Root requested by user: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban`
- Actual HDF5 directory: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617`
- HDF5 files: 80
- Valid full episodes: 79
- Corrupted/incomplete episode skipped by loader: `episode_1.hdf5`
- Valid frames checked before train: 64,524

Training command:
- Script: `scripts/train/train_dp_tac_concat_board_260617_only_e2000_20260621_codex.sh`
- TactileVAE encoder: `/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt`
- Image cache: `/home/chenshuai/Project/output/cache/dp_board_rawimg200x266_fp16`
- Save dir: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260621_codex`
- tmux training session: `dp260617_codex_e2000`
- tmux monitor session: `dp260617_codex_monitor`

Key hyperparameters:
- Architecture: `train_dp_tac_concat.py`
- Cameras: `global,wrist`
- Image size: raw/cache `200x266`, crop `200x266`
- Tactile side/history: left, 8 frames
- Prediction horizon: 16
- Observation horizon: 2
- Action execution horizon: 8
- Epochs: 2000
- Batch size: 64
- LR: `5e-5`
- Weight decay: `1e-5`
- Warmup: 1000 steps
- Diffusion train/inference steps: 100/100
- Down dims: `512,1024,2048`
- Episode-level val split: 0.1
- Validation interval: 5 epochs
- Save epoch ckpt every 50 epochs
- Save `dp_latest.pth` every 10 epochs
- Keep top-3 train-loss checkpoints

Initial health check:
- Train dataset after split/skips: 71 episodes, 57,886 frames, 56,821 windows
- Val dataset: 8 episodes, 6,638 frames, 2,048 capped val windows
- GPU: RTX 4090, about 14.7 GiB used during train
- First validation points:
  - epoch 1: train `0.829682`, val `0.403277`
  - epoch 5: train `0.076049`, val `0.075261`
  - epoch 10: train `0.042365`, val `0.040457`
  - epoch 25: train `0.018327`, val `0.018601`
  - epoch 45: train `0.014413`, val `0.015797`
  - epoch 50: train `0.013822`, val `0.014964`
- Status at recording: no overfit signal yet; train and val are both falling.
- Checkpoint save verified: `dp_epoch50.pth`, `dp_best.pth`, and
  `dp_latest.pth` exist in the run directory.

Important interpretation:
- This run is a DP policy training run, not a final real-robot quality claim.
- The deployable checkpoint should be selected by `dp_best.pth` unless later
  real rollout evidence shows another checkpoint is safer.
- Because this is only the 260617 collection, it should be compared against:
  1. previous 260617-only stable run,
  2. all-board-data run,
  3. all-board-plus-peg run,
  4. force-guided and non-guided real rollout traces.

## Recent research within roughly the last two months

The most relevant new direction is not plain behavior cloning improvement. The
strongest story for this project is:

> predict future tactile/contact consequences, score their contact quality with a
> differentiable verifier/energy, and inject the gradient into diffusion-policy
> denoising.

Relevant papers checked:

1. ViTaL, "Inference-time Policy Steering via Vision and Touch" (arXiv:2606.14981, 2026-06-12)
   - Link: https://arxiv.org/abs/2606.14981
   - Key idea: high-level visual sampling/verification for mode selection plus
     low-level tactile-guided diffusion editing for local contact requirements.
   - Strongly supports our direction: tactile guidance should act during
     diffusion editing/denoising, not only after policy sampling.

2. TacForeSight, "Force-Guided Tactile World Model for Contact-Rich Manipulation" (arXiv:2606.11184, 2026-06-09)
   - Link: https://arxiv.org/abs/2606.11184
   - Key idea: predict short-horizon tactile latent dynamics conditioned on
     force/torque, then use predicted tactile latents as anticipatory contact
     priors.
   - Supports adding explicit force conditioning to our foresight model and
     making force quality a first-class signal for board wiping.

3. Dream-Tac, "A Unified Tactile World Action Model for Contact-Rich Robot Manipulation" (arXiv:2606.08737, 2026-06-07)
   - Link: https://arxiv.org/abs/2606.08737
   - Key idea: jointly model action, future visual observations, and tactile
     dynamics, with contact-gated visuotactile fusion and contact-aware attention.
   - Supports a future architectural upgrade from "policy + separate foresight"
     to a more unified action/contact world model. For the current codebase,
     the practical next step is contact-gated tactile/force fusion, not a full
     rewrite.

4. Latent Diffusion Policy, "Shaping Latent Spaces for Diffusion-Based Robotic Manipulation" (arXiv:2606.08657, 2026-06-07)
   - Link: https://arxiv.org/abs/2606.08657
   - Key idea: move diffusion from raw action space into a deliberately shaped
     observation-conditioned latent space to simplify the denoising velocity
     field.
   - Useful for our project because our guidance currently acts on raw action
     chunks. A later version could guide a lower-dimensional action latent, which
     may make gradients smoother and less brittle.

5. TouchGuide, "Inference-Time Steering of Visuomotor Policies via Touch Guidance" (arXiv:2601.20239, 2026-01-28)
   - Link: https://arxiv.org/abs/2601.20239
   - Not within two months, but still a key baseline for tactile inference-time
     steering.
   - Main limitation for our specific goal: its contrastive feasibility score is
     useful, but our board and insertion tasks have explicit good/bad tactile
     outcome definitions. A supervised or semi-supervised outcome-quality energy
     is better aligned with gradient guidance.

6. pi0.7, "a Steerable Generalist Robotic Foundation Model with Emergent Capabilities" (arXiv:2604.15483, 2026-04-16)
   - Link: https://arxiv.org/abs/2604.15483
   - This is slightly outside a strict two-month window from 2026-06-21, but
     is included because it was explicitly named as a useful reference.
   - Key idea: steer behavior with richer conditioning such as task details,
     subgoal images and episode metadata.
   - Relevant takeaway: label/metadata quality matters. For this project, this
     maps to explicit contact labels: good force band, too-light contact,
     too-heavy contact, unstable oscillation, pre-bounce/bounce.

## Architecture/story improvements for this project

Near-term, low-risk:
1. Keep the current deployable DP concat policy as the behavior prior.
2. Keep tactile/force quality as a separate differentiable energy used during
   denoising, because this cleanly matches classifier guidance.
3. For board wiping, use force-aware scoring as the primary guided arm:
   pressure too small, pressure too large, and force derivative/oscillation are
   the real task criteria.
4. For insertion, use pre-bounce/bounce vs good-insertion margin as the primary
   guided arm.
5. Always evaluate scorer generalization with episode-level split, not random
   frame split.

Medium-term:
1. Add force-conditioned tactile foresight, inspired by TacForeSight.
2. Add contact-gated tactile fusion, inspired by Dream-Tac. Tactile should not
   dominate when the arm is not in contact.
3. Move from binary good/bad only to multi-head quality:
   - contact band score,
   - stability/smoothness score,
   - too-light score,
   - too-heavy score,
   - task-specific bad-event score.
4. Calibrate the energy into a continuous margin instead of only class labels,
   so gradients are useful even when the classifier is already confident.

High-risk research direction:
1. Latent action diffusion plus tactile-energy guidance in latent action space.
2. Unified tactile world-action model that predicts future tactile and action
   jointly. This is more novel but requires larger code changes and more data.

Current recommendation:
- Do not switch the whole project architecture immediately.
- Finish the 260617-only DP run and real rollout evaluation.
- In parallel, implement the next scorer/foresight upgrade as:
  force-conditioned multi-step tactile foresight + calibrated multi-head
  differentiable quality energy + denoising-time gradient guidance.
