# 260617-only DP ext Run Monitoring and Recent arXiv Review

Date: 2026-06-18

This document records the currently active 260617-only board-wiping DP training
run and the related recent research checked while supervising the run.  It is
separate from the earlier non-`_ext` 260617 run, which had already shown clear
late-stage validation overfitting and was stopped.

## Active Training Run

Goal: train a tactile Diffusion Policy using only:

```text
/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617
```

Current run directory:

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260618_ext
```

Home symlink:

```text
/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260618_ext
```

Run meta:

- Started: `2026-06-18 21:57:51 CST`
- Reason for restart: reduce GB-scale checkpoint writes while keeping frequent
  `latest` and `best` checkpoints.
- Script: `diffusion/train_dp_tac_concat.py`
- Dataset: 260617-only board data above
- Cameras: `global,wrist`
- Images: raw RGB resized/cropped to `200x266`
- Proprio: `observations/proprio_joint`
- Action: `actions/joint_abs`
- Tactile: left marker history encoded by frozen TactileVAE
- TactileVAE:
  `/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt`
- `pred_horizon=16`
- `obs_horizon=2`
- `n_action_steps=8`
- `tac_history=8`
- `batch_size=64`
- `epochs=2000`
- `lr=1e-4`
- `weight_decay=1e-6`
- `warmup_steps=500`
- DP train/inference diffusion steps: `100/100`
- `max_train_windows=8192`
- `max_val_windows=1024`
- Episode validation split: `val_ratio=0.1`
- `save_freq=100`
- `latest_freq=25`
- `topk_k=0`
- Image loading mode: cached fp16 images

Effective split from `config.json`:

- Train episodes: 72
- Validation episodes: 8
- Global condition dimension: 2350
  - vision: `512 * 2`
  - tactile latent: 144
  - qpos: 7
  - repeated over `obs_horizon=2`

## Status Snapshot

Snapshot time: `2026-06-18 23:40 CST`.

Training process was running normally:

- Training PID: `2549103`
- GPU: RTX 4090
- GPU utilization: about 90% during batches
- GPU memory: about 14.7 GB used
- External disk free space: about 2230 GB
- Home disk free space: about 44 GB

Latest parsed status:

| Item | Value |
|---|---:|
| Latest epoch | 203 / 2000 |
| Latest train loss | 0.006961 |
| Latest val loss | 0.020036 |
| Best epoch | 105 |
| Best val loss | 0.011387 |
| Epochs since best | 98 |
| Latest val / best val | 1.75955 |

Checkpoint state:

| Checkpoint | Status |
|---|---|
| `dp_best.pth` | exists, best validation checkpoint from epoch 105 |
| `dp_latest.pth` | exists, refreshed at latest-frequency checkpoints |
| `dp_epoch100.pth` | exists |
| `dp_epoch200.pth` | exists, confirms `save_freq=100` is working |

Current interpretation:

- The training job itself is healthy: process, GPU, disk, data path, and
  checkpoint saving are all normal.
- Train loss continues to decrease, but validation loss has not beaten epoch
  105. This is a warning sign for possible overfitting on the correlated
  sliding-window training data.
- Because the requested run is 2000 epochs and `dp_best.pth` is protected, the
  run should continue unless it crashes, fills disk, or shows a severe system
  issue.
- For deployment or real robot comparison at this stage, use `dp_best.pth`, not
  `dp_latest.pth`.

Evidence boundary:

- This is still offline DP supervision.  Noise-prediction validation loss does
  not prove real board-wiping quality.
- Real task quality must be tested with matched baseline/guided rollouts and
  server-side force traces.

## Recent arXiv Papers Checked

The user asked to use the training supervision time to check recent work from
roughly the last two months.  The entries below were checked via the arXiv API
on 2026-06-18.

### Most Relevant to the Current Project

#### ViTaL: Inference-time Policy Steering via Vision and Touch

- arXiv: https://arxiv.org/abs/2606.14981
- Published: 2026-06-12

Key idea:

- Uses multimodal inference-time steering for robot policies.
- High level: visual verification chooses long-horizon behavior.
- Low level: tactile-guided diffusion editing refines short action sequences to
  satisfy contact requirements.
- Uses predicted tactile futures and verifiers/rewards to steer actions.

Relevance:

- This is very close to our current intended chain:

```text
DP nominal action
  -> Foresight predicts future tactile consequence
  -> TacQualityEnergy scores contact quality
  -> gradient guidance edits the action chunk
```

Main lesson:

- Our direction is aligned with the newest inference-time steering story.
- The important design point is to keep guidance local/contact-sensitive, not
  to rewrite the whole action trajectory uniformly.

#### TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation

- arXiv: https://arxiv.org/abs/2606.11184
- Published: 2026-06-09

Key idea:

- Predicts short-horizon tactile latent dynamics conditioned on high-frequency
  wrist force/torque.
- Uses predicted tactile latents as anticipatory contact priors for policy
  control.

Relevance:

- Our Foresight currently predicts future tactile marker/latent consequences.
- For board wiping, the quality definition is force-band and smoothness, so
  force should eventually be a first-class input or prediction target.

Main lesson:

- Next Foresight version should consider force-conditioned tactile prediction,
  or a force-proxy head, rather than relying only on marker latents.

#### FAWAM: Force-Aware World Action Models for Closed-Loop Contact-Rich Manipulation

- arXiv: https://arxiv.org/abs/2606.08555
- Published: 2026-06-07, updated 2026-06-12

Key idea:

- Uses force at perception, future prediction, and execution correction levels.
- Predicts future actions and wrench trajectories, then uses predicted wrench
  as a reference for online correction.

Relevance:

- Board wiping is fundamentally a force-quality task:
  - force should stay in a target band;
  - force should change smoothly;
  - loss of contact or excessive pressure are both bad.

Main lesson:

- A stronger board pipeline should log and predict force/force-proxy curves,
  not only marker deformation.  The scorer should be evaluated against real
  force traces in contact phase.

#### Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation

- arXiv: https://arxiv.org/abs/2606.08737
- Published: 2026-06-07

Key idea:

- Jointly models actions, future visual observations, and tactile dynamics.
- Uses contact-gated visuotactile fusion and contact-aware attention bias.

Relevance:

- We currently keep DP, Foresight, and TacQuality as separate modules.  That is
  easier to debug scientifically, but Dream-Tac supports the idea that contact
  gating should exist explicitly.

Main lesson:

- TacQuality guidance should be gated by contact phase:
  - approach/reset: weak or zero guidance;
  - wiping contact: strong guidance;
  - lift/recovery: guidance decays.

#### ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation

- arXiv: https://arxiv.org/abs/2606.13877
- Published: 2026-06-11

Key idea:

- Evaluates vision-tactile world models for contact-rich manipulation.
- Emphasizes representation structure, temporal continuity, and downstream
  planning/control usefulness.

Relevance:

- Our Foresight should not be judged only by marker reconstruction loss.
- The important question is whether predicted future contact quality aligns
  with measured future contact quality.

Main lesson:

- Future evaluation should report:
  - future marker/latent error;
  - predicted TacQuality score vs GT-future TacQuality score;
  - predicted quality vs real force-band metrics;
  - downstream baseline vs guided real rollout metrics.

#### Tube Diffusion Policy

- arXiv: https://arxiv.org/abs/2604.23609
- Published: 2026-04-26

Key idea:

- Action chunks can be too open-loop for contact-rich tasks.
- Learns a local feedback flow around nominal diffusion-policy chunks, forming
  an action tube for fast correction.

Relevance:

- Our trust-region TacQuality update is conceptually a local correction around
  the DP action chunk.

Main lesson:

- If final-action guidance is too slow or too coarse, the next version should
  use shorter subchunk/contact-step guidance, not simply a larger scorer.

#### TouchGuide

- arXiv: https://arxiv.org/abs/2601.20239
- Latest checked version: 2026-05-13

Key idea:

- Keeps the base policy fixed and uses a tactile contact model to steer
  diffusion or flow-matching inference.
- Uses a tactile-informed feasibility score for contact constraints.

Relevance:

- This supports our choice of classifier/energy guidance rather than only
  tactile concatenation into the base policy.

Main lesson:

- The scorer should be trained or calibrated on noised action proposals if we
  want stable guidance during diffusion denoising, not just after the final
  clean action is produced.

#### Ambient Diffusion Policy

- arXiv: https://arxiv.org/abs/2606.12365
- Published: 2026-06-10

Key idea:

- Learns from mixed-quality/suboptimal robot data by controlling which diffusion
  timesteps use lower-quality samples.

Relevance:

- Our board data includes positive, too-light, too-heavy, and unstable contact
  modes.  We should not assume all data should be used equally for vanilla BC.

Main lesson:

- For mixed board datasets, train a base DP on good/task-specific data, then
  use bad/suboptimal data primarily to train the quality scorer or noised-time
  constraints.  A future DP training variant can use quality-dependent
  timestep weighting.

#### MODIP

- arXiv: https://arxiv.org/abs/2606.10825
- Published: 2026-06-09

Key idea:

- Uses a world model to improve diffusion policies through model-based
  optimization and supervised fine-tuning targets.

Relevance:

- We currently use Foresight for inference-time gradient guidance.

Main lesson:

- If guidance repeatedly improves actions offline or in real rollouts, those
  guided actions can later be distilled back into DP as supervised targets.
  This would reduce inference overhead.

#### Training and Evaluating Diffusion Policies with Long Context Lengths

- arXiv: https://arxiv.org/abs/2606.16447
- Published: 2026-06-15

Key idea:

- Studies longer observation contexts for diffusion policies and finds that
  longer context can help when conditioned appropriately.

Relevance:

- Our current DP has `obs_horizon=2`, while tactile history inside each
  observation is 8 frames.

Main lesson:

- For tasks with persistent contact states or slow force drift, test a longer
  observation context after the current 260617-only baseline finishes.
  This should be an ablation, not a mid-run change.

#### T-Rex

- arXiv: https://arxiv.org/abs/2606.17055
- Published: 2026-06-15

Key idea:

- Uses high-frequency tactile signals and a variable-rate architecture for
  tactile-reactive manipulation.

Relevance:

- Board wiping contact quality can change faster than a low-frequency action
  chunk update.

Main lesson:

- If real tests show delayed correction, add a high-frequency tactile/force
  correction loop or shorten the guided execution horizon.

## Architecture Assessment for This Project

The current project story remains scientifically reasonable:

```text
Base DP = behavior prior
Foresight = predicts future tactile/contact consequence
TacQualityEnergy = differentiable contact-quality constraint
Trust-region guidance = bounded action improvement during inference
Force trace evaluation = real evidence
```

This is stronger than plain tactile-DP concatenation because tactile is not only
an input.  It becomes a predicted consequence that can produce gradients with
respect to the action.

Current strengths:

- The 260617-only DP is training on the requested data and checkpointing
  correctly.
- The board scorer candidate already has strong held-out offline metrics in the
  existing readiness matrix:
  - AUC about `0.9997`
  - balanced accuracy about `0.9828`
  - quality Spearman about `0.9239`
- The current guidance stack is not reranking; it is a differentiable action
  update through Foresight and TacQuality.

Current weaknesses:

- Offline DP validation is not task-quality proof.
- Board quality is ultimately force-based, but the current DP input uses
  tactile marker latent and images, not force as a first-class predicted
  variable.
- Validation windows are correlated within episodes; episode-level split helps,
  but only 8 validation episodes means validation loss can be noisy.
- The best checkpoint is currently much earlier than the latest checkpoint,
  so the run must be monitored carefully.

## Practical Improvement Plan

Do not change the active 260617-only DP run mid-training.  Use it as a clean
baseline.

After this run:

1. Compare `dp_best.pth` from this 260617-only run against previous full-data
   and positive-only DP checkpoints in real rollouts.
2. Run baseline and TacQuality-guided modes with server-side force logging.
3. Evaluate only the contact/wiping phase for:
   - force-in-band ratio;
   - Fz mean and p95;
   - `|dFz|` mean and p95;
   - marker magnitude stability;
   - completion/coverage if available.
4. If guided mode improves force smoothness but hurts coverage, add a visual or
   progress verifier.
5. If offline TacQuality improves but real force does not, debug Foresight
   alignment and contact gate before changing DP.
6. Next model iteration should prioritize:
   - contact-gated guidance;
   - force-conditioned or force-proxy Foresight;
   - noised-action scorer calibration for denoising-time guidance;
   - longer context or shorter reactive subchunks as separate ablations.

Deployment recommendation at this snapshot:

```text
Use:
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260618_ext/dp_best.pth

Do not use as the default:
dp_latest.pth
```

## Follow-up Snapshot

Snapshot time: `2026-06-18 23:53 CST`.

Latest parsed status:

| Item | Value |
|---|---:|
| Latest epoch | 228 / 2000 |
| Latest train loss | 0.006004 |
| Latest val loss | 0.022471 |
| Best epoch | 105 |
| Best val loss | 0.011387 |
| Epochs since best | 123 |
| Latest val / best val | 1.973391 |

Checkpoint state:

- `dp_epoch200.pth` exists.
- `dp_latest.pth` refreshed at epoch 225 range.
- `dp_best.pth` still points to epoch 105.

Interpretation:

- Training is still running normally.
- The monitor warning is now `strong_plateau_or_overfit_use_best`.
- Continue the 2000-epoch run because the user requested a long run and
  `dp_best.pth` is protected, but current deployment recommendation remains
  `dp_best.pth`, not `dp_latest.pth`.
