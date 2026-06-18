# 260617-only DP Training Supervision and Recent arXiv Review

Date: 2026-06-19

## Current Training Run

Goal: train the tactile concat Diffusion Policy using only the 2026-06-17 board-wiping data:

```text
/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617
```

Run directory:

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260618_ext
```

Home symlink:

```text
/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260618_ext
```

Main configuration:

- Script: `diffusion/train_dp_tac_concat.py`
- Dataset: 260617-only board data above
- Cameras: `global,wrist`
- Image input: raw RGB resized/cropped to `200x266`
- Proprio: `observations/proprio_joint`
- Action: `actions/joint_abs`
- Tactile: left marker history encoded by frozen board TactileVAE
- TactileVAE: `/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt`
- TactileVAE norm stats loaded by run:
  - mean: `[-0.3398614526, -2.9208483696]`
  - std: `[1.9804853201, 2.7671177387]`
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
- Episode-level validation split: `val_ratio=0.1`
- Window caps: `max_train_windows=8192`, `max_val_windows=1024`
- Checkpoint saving: `dp_latest.pth` every 25 epochs, `dp_epoch*.pth` every 100 epochs, `dp_best.pth` on best validation loss
- Image loading: fp16 cached images from `/home/chenshuai/Project/output/cache/dp_board_rawimg200x266_fp16`

## Dataset Integrity

Checked with `h5py` in the `TactileACT` conda environment.

Output saved to:

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260618_ext/dataset_integrity_260617.json
```

Summary:

- `episode_*.hdf5`: 80 files
- Valid episodes: 79
- Bad episode: `episode_1.hdf5`
  - error: missing `observations/proprio_joint`
- Total valid frames: 64,524
- Total possible `pred_horizon=16` windows: 63,339
- Current configured sampled windows: 9,216 total
  - 8,192 train
  - 1,024 val
- Current training uses about `14.55%` of all possible windows per run configuration.

The run log reports:

- Train episodes after split/load: 71
- Validation episodes: 8
- Train windows: 8,192
- Validation windows: 1,024

## Training Status

Snapshot around `2026-06-19 00:28 CST`:

| Item | Value |
|---|---:|
| Latest epoch | 301 / 2000 |
| Latest train loss | 0.005129 |
| Latest val loss | 0.021534 |
| Best epoch | 105 |
| Best val loss | 0.011387 |
| Epochs since best | 196 |
| Latest val / best val | 1.891 |

Checkpoint state:

- `dp_best.pth`: exists, best validation checkpoint from epoch 105
- `dp_latest.pth`: exists, refreshed at epoch 300/301 interval
- `dp_epoch100.pth`: exists
- `dp_epoch200.pth`: exists
- `dp_epoch300.pth`: exists

Latest parsed curve files generated from `train.log`:

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260618_ext/training_curve_reparsed_latest.png
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260618_ext/training_metrics_reparsed_latest.csv
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260618_ext/training_summary_reparsed_latest.json
```

Current judgment:

- Training process is healthy:
  - process alive;
  - GPU utilization around 90%;
  - GPU memory about 14.7 GB / 24.6 GB;
  - external disk still has about 2.2 TB free;
  - checkpoint saving works.
- Validation loss has not improved after epoch 105 and is now consistently worse than best.
- This is a strong plateau/overfitting warning, not a crash.
- Because the user requested a 2000-epoch run and `dp_best.pth` is protected, the current run is being kept alive.
- Any real robot deployment/testing from this run should use `dp_best.pth`, not `dp_latest.pth`, unless intentionally testing late-overfit checkpoints.

Additional snapshot around `2026-06-19 01:05 CST`:

| Item | Value |
|---|---:|
| Latest epoch | 371 / 2000 |
| Latest train loss | 0.005113 |
| Latest val loss | 0.029408 |
| Best epoch | 105 |
| Best val loss | 0.011387 |
| Epochs since best | 266 |
| Latest val / best val | 2.583 |
| Tail-20 val min / mean / max | 0.023870 / 0.027403 / 0.034100 |
| Tail-50 val min / mean / max | 0.021966 / 0.026556 / 0.034100 |

Important validation note:

- `diffusion/train_dp_tac_concat.py` shuffles and splits `episode_*.hdf5` entries before building windows.
- This means the current validation split is episode-level, not random frame/window-level leakage.
- The overfitting warning is therefore meaningful: the model is fitting train episodes better while performance on held-out episodes is worse than the epoch-105 best.
- Because `dp_best.pth` is already protected and the requested 2000-epoch run is still healthy, training is being left running.

Likely causes of validation degradation:

- Only 79 valid episodes are available.
- Sliding-window samples from the same episode are highly correlated.
- Current run samples only 9,216 of 63,339 possible windows.
- Model capacity is large: log reports about `3.15e8` parameters.
- Vision encoder is trained jointly, so the model can memorize visual/action correlations in a small episode set.
- Validation set has only 8 episodes, so val loss is useful but noisy.
- The data comes from one dataset family, so held-out episodes may expose small trajectory/contact variations that a high-capacity image-conditioned DP can overfit.

Recommended next training ablations after this run or when GPU is free:

1. Full-window or larger-window run:
   - remove or raise `max_train_windows` / `max_val_windows`;
   - test whether using more of the 63,339 windows reduces overfitting.
2. Lower-capacity or stronger-regularization run:
   - lower LR, stronger weight decay, or freeze part of the vision encoder;
   - keep board TactileVAE frozen.
3. Data-diverse run:
   - combine 260617 with earlier positive/negative board data once the 260617-only baseline is understood.
4. Real rollout evaluation:
   - use paired baseline/guided tests and server-side force traces;
   - DP validation loss alone does not prove wiping quality.

## Recent Related Work

The latest relevant work from roughly the last two months supports the same high-level direction: contact-rich manipulation should use tactile/force signals as predictive constraints or inference-time guidance, not just concatenate them as another observation.

The first group below is the strict recent-paper set checked around 2026-06-19. The second group contains user-specified or highly relevant references that are still important for method design, but are not all strict "last two months arXiv" entries.

## Strict Recent Papers

### ViTaL: Inference-time Policy Steering via Vision and Touch

- arXiv: https://arxiv.org/abs/2606.14981
- Published: 2026-06-12
- Source checked: arXiv abstract/HTML on 2026-06-19

Relevant idea:

- Uses multimodal inference-time steering.
- High-level visual sampling/verification chooses long-horizon behavior.
- Low-level tactile-guided diffusion editing refines short-horizon action sequences for contact requirements.
- Learns a visuo-tactile latent world model and tactile reward/verifier for predicted tactile futures.

Implication for this project:

- This is the closest recent match to our intended chain:

```text
DP nominal action
  -> Foresight predicts future tactile consequence
  -> TacQualityEnergy scores future contact quality
  -> gradient guidance edits the action chunk
```

- It supports local, contact-sensitive gradient editing rather than global trajectory rewriting.

### TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation

- arXiv: https://arxiv.org/abs/2606.11184
- Published: 2026-06-09
- Source checked: arXiv abstract/HTML on 2026-06-19

Relevant idea:

- Predicts short-horizon tactile latent dynamics.
- Conditions tactile foresight on high-frequency wrist force/torque.
- Uses predicted tactile latents as anticipatory contact priors for control.

Implication for this project:

- Our Foresight should eventually include force/torque or force proxies, especially for board wiping.
- Board quality is naturally force-band plus force-smoothness, so force should not only be used after the fact for evaluation.

### FAWAM: Force-Aware World Action Models for Closed-Loop Contact-Rich Manipulation

- arXiv: https://arxiv.org/abs/2606.08555
- Published: 2026-06-07
- Updated: 2026-06-12
- Source checked: arXiv abstract/HTML on 2026-06-19

Relevant idea:

- Uses force at three levels: perception, future prediction, and closed-loop execution.
- Jointly predicts actions and future wrench trajectories.
- Uses predicted wrench as an execution-time reference for residual correction.

Implication for this project:

- For board wiping, future force or force proxy prediction is a stronger supervision target than marker latent alone.
- The scorer/guidance should be evaluated against real force traces, especially force spikes and variance.

### Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation

- arXiv: https://arxiv.org/abs/2606.08737
- Published: 2026-06-07
- Source checked: arXiv abstract/HTML on 2026-06-19

Relevant idea:

- Jointly models action, future visual observations, and tactile dynamics.
- Uses contact-gated visuotactile fusion and contact-aware attention.

Implication for this project:

- A contact gate is important:
  - weak or off during approach/reset/non-contact;
  - strong during wiping/contact.
- Our scorer/guidance design should not push tactile quality when no contact is expected.

### ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation

- arXiv: https://arxiv.org/abs/2606.13877
- Published: 2026-06-11
- Source checked: arXiv abstract/HTML on 2026-06-19

Relevant idea:

- Studies vision-tactile world models across contact-rich tasks.
- Emphasizes task-relevant contact prediction and downstream control usefulness.

Implication for this project:

- Foresight should not be judged only by latent/marker MSE.
- Better metrics:
  - predicted TacQuality score vs GT-future TacQuality score;
  - predicted force-band score vs measured real force trace;
  - guided vs baseline real rollout outcomes.

### Multi-Resolution Tactile Imitation Learning for Contact-Rich Robotic Manipulation

- arXiv: https://arxiv.org/abs/2606.06281
- Published: 2026-06-04
- Source checked: arXiv abstract/HTML on 2026-06-19

Relevant idea:

- Uses modality-specific tactile encoders and transformer fusion for multi-resolution tactile streams.
- Conditions a flow-matching policy on RGB plus tactile features.
- Reports strong gains over vision-only and single visual-tactile baselines on contact-rich tasks.

Implication for this project:

- Our current single left-hand marker latent is useful, but for wiping it should be treated as one contact signal among several:
  - marker latent/history;
  - marker magnitude/area/smoothness proxy;
  - measured or predicted force/torque.
- A practical near-term version is not to add more sensors immediately, but to add multi-resolution features from the existing marker stream:
  - short window: contact onset/spike;
  - medium window: wiping force smoothness;
  - action chunk window: consequence score for guidance.

### SI-Diff: A Framework for Learning Search and High-Precision Insertion with a Force-Domain Diffusion Policy

- arXiv: https://arxiv.org/abs/2605.12247
- Published: 2026-05-12
- Source checked: arXiv API on 2026-06-19

Relevant idea:

- Uses a force-domain diffusion policy for precision insertion.
- Treats force information as a control-domain signal, not just as an auxiliary observation.

Implication for this project:

- For insertion and board wiping, force-domain objectives can make the story stronger:
  - insertion: reduce collision/bounce risk;
  - board: stay in a desired force band and avoid spikes.
- This supports using force/force-proxy targets in Foresight and TacQualityEnergy rather than relying only on marker latent separability.

### Tube Diffusion Policy: Reactive Visual-Tactile Policy Learning for Contact-rich Manipulation

- arXiv: https://arxiv.org/abs/2604.23609
- Published: 2026-04-26
- Source checked: arXiv API on 2026-06-19

Relevant idea:

- Focuses on reactive visual-tactile policy learning for contact-rich manipulation.
- The relevant point for this project is the emphasis on tactile-conditioned reaction during contact, not only open-loop imitation.

Implication for this project:

- Our DP should remain closed-loop through observation updates, while guidance edits the near-term action chunk.
- For wiping, action quality is best judged over the contact segment of the chunk, not over the whole episode including approach/reset.

### DPTG: diffusion policy with tactile feasibility guidance

- Frontiers Robotics and AI: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1851102/full
- Published: 2026-06

Relevant idea:

- Treats tactile sensing as a physical feasibility constraint for diffusion policy.
- Reports inference-time tactile guidance suppressing force spikes and reducing contact-force variance.

Implication for this project:

- Our method should be framed as future contact quality/feasibility guidance, not just tactile feature fusion.
- Board wiping quality should explicitly measure force spikes and force variance.

## User-Specified and Strongly Related References

### TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance

- arXiv: https://arxiv.org/abs/2601.20239
- Published: 2026-01; updated 2026-05; still highly relevant

Relevant idea:

- Keeps the base diffusion/flow policy fixed.
- Uses a task-specific Contact Physical Model to provide tactile-informed feasibility score.
- Trains the scoring model so it can guide noisy actions during sampling.

Implication for this project:

- TacQualityEnergy should be trained/audited not only on clean expert chunks, but also on noised action chunks matching DP denoising-time states.
- This is important if guidance is applied inside denoising rather than only as final clean-action refinement.

### AdaVTF: Learning When to See and When to Feel

- arXiv: https://arxiv.org/abs/2604.01414
- Published: 2026-04-01

Relevant idea:

- Compares ways of combining vision and force/torque in diffusion policies.
- Proposes adaptive integration:
  - ignore F/T in non-contact phases;
  - use vision and torque during contact.

Implication for this project:

- Supports contact-phase-aware tactile/force guidance.
- For board wiping, this means scoring/guidance should focus on the wiping-contact stage, not approach.

### pi0.7

- arXiv: https://arxiv.org/abs/2604.15483
- Published: 2026-04-16; updated 2026-04-24

Relevant idea:

- Uses rich conditioning and classifier-free guidance to steer behavior style/quality at inference time.

Implication for this project:

- Useful as a high-level analogy for steerable behavior, but our current need is lower-level differentiable tactile/force energy guidance.
- For this project, classifier guidance with TacQualityEnergy is more directly actionable than CFG unless the DP model is retrained with explicit positive/negative conditioning.

## Current Architecture Story

Current project architecture:

```text
Observation:
  RGB(global,wrist) + qpos + tactile marker history

Base policy:
  tactile-concat Diffusion Policy predicts nominal action chunk

Future model:
  Foresight predicts future tactile consequence from state/action/context

Quality model:
  TacQualityEnergy scores whether predicted future contact is good
  board: force band + smoothness + marker/action consistency
  insertion: insertion vs pre-bounce/bounce risk

Guidance:
  use score gradient to edit DP action chunk
  not reranking
```

Most important improvements suggested by the recent work:

1. Make force/torque first-class for board wiping.
   - Record server-side force traces for every real rollout.
   - Add force or force-proxy prediction to Foresight.
   - Score future force band, smoothness, and spikes directly.
2. Train guidance scorer for denoising-time robustness.
   - Add noisy action/chunk augmentation when training TacQualityEnergy.
   - Audit gradients at multiple DP noise levels.
3. Use contact-phase gating.
   - No or weak guidance during approach/reset.
   - Strong guidance during wiping contact.
4. Keep task-specific quality definitions.
   - Insertion: bad = pre-bounce/bounce trajectory; good = stable insertion/contact progress.
   - Board: good = force in desired band and smooth; bad = too light, too heavy, or oscillatory.
5. Report evidence in layers.
   - Offline score/classification: scorer validity.
   - Foresight alignment: predicted quality matches GT future quality.
   - Dry-run guidance: gradients are finite and improve score.
   - Real robot paired rollouts: actual force/outcome improvement.

## Architecture Improvement Priorities for This Project

Priority 1: keep the current DP as the nominal behavior generator.

- The active DP run is useful as a 260617-only behavior baseline.
- It should not be expected to solve force quality by itself, because the DP loss only learns action noise prediction.
- Deployment should compare:
  - baseline DP: no guidance;
  - guided DP: same checkpoint plus TacQuality/Foresight gradient guidance.

Priority 2: make board wiping guidance force-aware.

- Board wiping quality is defined by force band and smoothness, not only by visual progress.
- The current server-side force logging plan is necessary, not optional.
- The next quality model should use real force traces as labels/targets where available:
  - too light;
  - good contact band;
  - too heavy;
  - oscillatory/unstable.
- The most paper-aligned version is a future force/marker consequence model plus a differentiable quality energy, not just a post-hoc binary classifier.

Priority 3: move from clean-action refinement toward denoising-time robustness.

- Current implemented guidance mostly edits a clean action chunk after DP sampling.
- TouchGuide/ViTaL/DPTG-style guidance is stronger if the scorer remains meaningful on noisy candidate chunks during denoising.
- The noisy-action guidance audit is therefore a useful intermediate gate, but it is still not robot evidence.

Priority 4: add contact-phase gating instead of always-on scoring.

- Approach and reset should have weak/no force-quality guidance.
- Wiping contact should have strong quality guidance.
- This matches the board data: the positive/negative distinction is mainly meaningful during the contact/wiping phase.

Priority 5: improve Foresight targets.

- Current marker-latent future prediction is a good base.
- For board wiping, the stronger story is:

```text
state + action chunk
  -> future marker latent / marker proxy / force proxy
  -> contact quality energy
  -> gradient on action chunk
```

- This connects our project directly to TacForeSight/FAWAM without needing to replace the whole DP immediately.

## Immediate Recommendation

For the active 260617-only DP run:

- Continue the run because it was explicitly requested as 2000 epochs and checkpointing is healthy.
- Treat current `dp_best.pth` as the deployable candidate.
- Do not treat `dp_latest.pth` as better just because train loss is lower.
- If validation stays worse through later checkpoints, use this as evidence that the 260617-only dataset is too small/correlated for such a large trainable model under the current sampled-window setup.
