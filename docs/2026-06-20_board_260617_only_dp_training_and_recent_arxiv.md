# 2026-06-20 260617-only Board DP Training and Recent Work Notes

## Current DP Training

Task: train the tactile-concat Diffusion Policy only on:

`/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617`

Run directory:

`/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun`

Launcher:

`scripts/train/train_dp_tac_concat_board_260617_only_20260620_e2000.sh`

Core configuration:

- Model: `diffusion/train_dp_tac_concat.py`
- Vision: global + wrist images, raw collected size `200x266`, no resize/crop change beyond `200,266`
- Tactile: frozen board TactileVAE encoder
- TactileVAE checkpoint: `/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt`
- Tactile side/history: left hand, 8 marker frames
- DP horizon: `pred_horizon=16`, `obs_horizon=2`, `n_action_steps=8`
- Action: `actions/joint_abs`
- Proprio: `observations/proprio_joint`
- Epochs: 2000
- Batch size: 64
- Learning rate: `5e-5`
- Weight decay: `1e-5`
- Validation: episode-level split, `val_ratio=0.1`, `val_interval=5`, `max_val_windows=2048`
- Per-epoch cap: `max_steps_per_epoch=128`; this keeps the 2000-epoch run tractable while sampling shuffled training batches each epoch.
- Checkpoints: `dp_best.pth` updates on best validation loss, `dp_latest.pth` every 10 epochs, `dp_epoch*.pth` every 50 epochs, top-3 train-loss checkpoints retained.

Data check:

- 80 hdf5 files found.
- 79 readable episodes.
- 1 skipped episode: `episode_1.hdf5`, missing `observations/proprio_joint`.
- Usable frames: 64,524.
- Example episode shape:
  - `observations/images/global`: `(701, 200, 266, 3) uint8`
  - `observations/images/wrist`: `(701, 200, 266, 3) uint8`
  - `observations/proprio_joint`: `(701, 7) float32`
  - `observations/tac/left/marker_offset`: `(701, 9, 9, 2) float32`
  - `actions/joint_abs`: `(701, 7) float32`

Startup verification:

- A 1-epoch, 2-batch smoke run completed successfully before launching the full run.
- The first background launch exited before epoch summaries without a Python traceback. It was archived as `train_start_failed_*.log`.
- The run was relaunched using `setsid + nohup`; the second launch is running normally.

Training status observed:

- Epoch 1: train `0.815538`, val `0.388930`
- Epoch 5: train `0.077848`, val `0.073704`
- Epoch 10: train `0.041109`, val `0.040868`
- Epoch 15: train `0.027720`, val `0.024991`
- Epoch 20: train `0.021982`, val `0.022192`
- Epoch 30: train `0.017846`, val `0.019104`
- Epoch 35: train `0.015791`, val `0.018849`
- Epoch 40: train `0.015088`, val `0.018343`
- Epoch 45: train `0.014458`, val `0.018478`
- Epoch 50: train `0.013188`, val `0.015548`
- Epoch 55: train `0.013553`, val `0.015130`
- Epoch 60: train `0.012524`, val `0.015165`
- Epoch 70: train `0.012777`, val `0.014921`
- Epoch 80: train `0.011915`, val `0.014607`
- Epoch 85: train `0.011576`, val `0.014062` best so far
- Epoch 90: train `0.012078`, val `0.014627`
- Epoch 95: train `0.011268`, val `0.015983`
- Epoch 100: train `0.011680`, val `0.015453`
- Epoch 104: train `0.010387`, no validation point
- Epoch 105: train `0.010435`, val `0.015956`
- Epoch 110: train `0.010668`, val `0.016228`
- Epoch 111: train `0.010908`, no validation point
- Epoch 115: train `0.010081`, val `0.015467`
- Epoch 120: train `0.009955`, val `0.014967`
- Epoch 125: train `0.009864`, val `0.018022`
- Epoch 129: train `0.009139`, no validation point
- Epoch 140: train `0.009322`, val `0.016780`
- Epoch 145: train `0.008945`, val `0.015779`
- Epoch 150: train `0.009054`, val `0.017133`
- Epoch 155: train `0.008951`, val `0.016322`
- Epoch 157: train `0.008783`, no validation point
- Epoch 160: train `0.009007`, val `0.016753`
- Epoch 162: train `0.009160`, no validation point
- Epoch 165: train `0.008544`, val `0.017073`
- Epoch 170: train `0.008450`, val `0.016301`
- Epoch 172: train `0.008299`, no validation point
- Epoch 175: train `0.008382`, val `0.018182`
- Epoch 180: train `0.008662`, val `0.019354`
- Epoch 185: train `0.008018`, val `0.018258`
- Epoch 190: train `0.008610`, val `0.016916`
- Epoch 195: train `0.008712`, val `0.019123`
- Epoch 200: train `0.008359`, val `0.018832`

Latest monitored status on 2026-06-20:

- The training process is still running; do not treat any checkpoint as final yet.
- The latest observed best validation checkpoint is epoch 85 with val loss `0.014062`.
- Epoch 95 through 200 did not refresh the best. Epoch 200 is `33.9%` higher than the best validation loss.
- The monitor state is `strong_plateau_or_overfit_use_best`; this is now a sustained plateau/overfit risk after epoch 85.
- `dp_best.pth`, `dp_latest.pth`, `dp_epoch50.pth`, `dp_epoch100.pth`, `dp_epoch150.pth`, `dp_epoch200.pth`, and top-k checkpoints are being saved normally.
- GPU memory is about `14.7GB / 24.6GB`, with high utilization during active batches.

Current interpretation:

- Train and validation losses are both much lower than the startup phase.
- Epoch 85 is the current best validation point. Epoch 95 through 170 all fail to improve validation while training loss continues to edge down.
- Continue training because the requested run is 2000 epochs and the watcher has a conservative stop policy, but deployment/evaluation should strongly prefer `dp_best.pth`.
- Treat post-85 checkpoints as lower-priority candidates unless validation improves again. For real rollout, use epoch-85 `dp_best.pth`, not `dp_latest.pth`.
- A follow-up run should consider lower learning rate, stronger regularization, or fewer effective update steps if the goal is best validation rather than long-run fitting.
- For deployment/evaluation, prefer `dp_best.pth`; `dp_final.pth` should only be used after checking final validation behavior.

2026-06-20 13:43 update:

- Latest parsed epoch: `185/2000`
- Latest train/val: `0.008018` / `0.018258`
- Best checkpoint remains: epoch `85`, val `0.014062`
- Latest val/best ratio: `1.298`
- Status warning: `strong_plateau_or_overfit_use_best`
- Training process is healthy and still using the GPU. The run should continue for the requested 2000-epoch trace, but the current deployable candidate is still `dp_best.pth`.

2026-06-20 13:53 update:

- Latest parsed checkpoint node: epoch `200/2000`
- Latest train/val: `0.008359` / `0.018832`
- Best checkpoint remains: epoch `85`, val `0.014062`
- Latest val/best ratio: `1.339`
- `dp_epoch200.pth` was saved successfully.
- Interpretation: epoch 200 confirms the same trend as epochs 175-195. Training loss is lower than epoch 85, but validation is consistently worse. Keep training for the long-run trace, but do not promote epoch-200/latest checkpoints for deployment unless later validation refreshes best.

2026-06-20 14:04 update:

- Latest parsed monitor node: epoch `216/2000`
- Latest parsed train loss: `0.008025`
- Latest validation node: epoch `215`, train `0.007644`, val `0.021614`
- Best checkpoint remains: epoch `85`, val `0.014062`
- Latest val / best ratio: `1.537`
- Live log had already reached epoch `218` while the monitor JSON was at epoch `216`.
- Process health:
  - training PID `3794700` still running
  - GPU memory about `14.7GB / 24.6GB`
  - GPU utilization about `73%`
  - external output disk free space about `2.0T`
- Interpretation: this is now a sustained overfit/validation-degradation trace after epoch 85. Keep the requested 2000-epoch run running for a complete training curve, but deployment/default testing should continue to use `dp_best.pth` unless a later validation point beats epoch 85.

2026-06-20 14:24 update:

- `dp_epoch250.pth` was saved successfully, size about `2.6G`.
- Epoch 245: train `0.007129`, val `0.020164`, no best refresh.
- Epoch 250: train `0.007350`, val `0.021016`, no best refresh.
- Best checkpoint remains epoch `85`, val `0.014062`.
- Monitor warning remains `strong_plateau_or_overfit_use_best`.
- Interpretation: checkpoint saving is healthy, but validation behavior is now clearly worse than the best by about `43-49%` at recent validation points. Continue the requested long run, but the deployable checkpoint remains `dp_best.pth`.

Monitoring files:

- `train.log`: raw training log
- `training_status_latest.json`: latest parsed status
- `loss_curve.png` / `loss_curve.csv`: periodically refreshed by the monitor
- `monitor_training.log`: monitor process log

## Recent Arxiv Work Relevant to This Project

Search window: papers submitted or updated in roughly the last two months from 2026-06-20. I checked titles, dates, and IDs with the arXiv API. TouchGuide is older by submission date but kept as background because it is a named tactile-guidance reference.

Main trend: the recent direction is not plain tactile concatenation. The stronger direction is future contact prediction, inference-time steering, contact/phase gating, and force-aware quality constraints.

| paper | arXiv | date | most relevant point |
|---|---:|---:|---|
| ViTaL: Inference-time Policy Steering via Vision and Touch | https://arxiv.org/abs/2606.14981 | 2026-06-12 | bi-level vision/touch steering, close to DP + Foresight + tactile score guidance |
| Dream-Tac: A Unified Tactile World Action Model | https://arxiv.org/abs/2606.08737 | 2026-06-07 | contact-gated visuo-tactile fusion and future tactile dynamics |
| ContactWorld | https://arxiv.org/abs/2606.13877 | 2026-06-11 | spatial and temporal contact representations matter for contact-rich planning |
| FAWAM: Force-Aware World Action Models | https://arxiv.org/abs/2606.08555 | 2026-06-07 | force should be used for perception, prediction, and execution correction |
| TacForeSight: Force-Guided Tactile World Model | https://arxiv.org/abs/2606.11184 | 2026-06-09 | global force and local tactile fields have complementary roles |
| SI-Diff: Force-Domain Diffusion Policy | https://arxiv.org/abs/2605.12247 | 2026-05-12 | insertion should be phase/mode-aware, not one flat policy behavior |
| Tube Diffusion Policy | https://arxiv.org/abs/2604.23609 | 2026-04-26 | chunked policies need reactive contact correction |
| SO-TA | https://arxiv.org/abs/2605.20433 | 2026-05-19 | force/pose can structure visuo-haptic attention |
| Latent Diffusion Policy | https://arxiv.org/abs/2606.08657 | 2026-06-07 | shaped latent action spaces may reduce raw-action diffusion difficulty |
| MODIP | https://arxiv.org/abs/2606.10825 | 2026-06-09 | world-model optimization can improve diffusion policies through distillation |
| Feedback World Model | https://arxiv.org/abs/2605.15705 | 2026-05-15 | online feedback can correct world-model prediction drift |
| PACT | https://arxiv.org/abs/2606.08414 | 2026-06-07 | constraint gradients can align diffusion policies with physical safety |
| Fisher-Preserving Guidance | https://arxiv.org/abs/2605.29937 | 2026-05-28 | guidance should avoid pushing diffusion samples off-manifold |
| Test-Time Gradient Guidance of Flow Policies | https://arxiv.org/abs/2606.11087 | 2026-06-09 | policy samples can be improved at test time with critic/value gradients |
| World Pilot / LaWAM / MemoryWAM | https://arxiv.org/abs/2606.12403, https://arxiv.org/abs/2606.15768, https://arxiv.org/abs/2606.20562 | 2026-06 | world-action priors and memory are becoming central for robot policies |

Background reference:

- TouchGuide: https://arxiv.org/abs/2601.20239, submitted 2026-01-28 and updated 2026-05-13. It is relevant because it steers visuomotor policies with tactile feasibility, but it is not in the strict two-month submission window.

## Current Architecture Assessment

The active `260617-only` run is a tactile-concat DP baseline:

```text
global image + wrist image
  -> ResNet18 visual features
left marker history
  -> frozen board TactileVAE latent
qpos
  -> concatenate over obs_horizon=2
  -> ConditionalUnet1D denoises a 16-step joint action chunk
```

This is useful as an action prior, but it should not be presented as the main novelty. The stronger project story is:

```text
DP proposes action chunk
  -> Foresight predicts future tactile / marker / force-proxy consequence
  -> TacQualityEnergy scores future contact quality
  -> bounded classifier / scorer gradient guidance edits action during denoising
```

This remains modular enough to debug on the robot. It also matches the recent literature direction: predicted future contact consequences should constrain policy generation, not only be concatenated as another observation.

## Best Current Story

The current paper/project story should be:

1. Base DP learns the visual/proprio action prior.
2. TactileVAE compresses marker fields into a compact contact latent.
3. Multi-step Foresight predicts future contact consequences from candidate action chunks.
4. TacQualityEnergy turns predicted future contact into a physically interpretable quality score.
5. During denoising, gradient guidance moves the action sample toward better predicted contact outcomes while a trust region keeps the action close to the DP manifold.

This is not reranking. Reranking can be a diagnostic baseline, but the main method is action-gradient guidance.

## Recommended Improvements

1. Keep the active `260617-only` DP run as a baseline and use `dp_best.pth`.

   Reason: validation already plateaued after epoch 85. The 2000-epoch trace is useful evidence, but later checkpoints should not automatically become deployment candidates.

2. Upgrade board Foresight from marker-only to force-aware/contact-aware prediction.

   Reason: board quality is defined by force band and force smoothness. FAWAM and TacForeSight both support force as a first-class future contact signal.

   Minimal version:

   ```text
   marker_history + qpos_history + action_chunk
     -> future marker latent
     -> force proxy / force-band head: too_light / good / too_heavy / oscillatory
     -> temporal smoothness head over predicted future contact
   ```

3. Make guidance contact-gated by default.

   Board approach/reset should not be strongly guided by wiping-force quality. Wiping/contact frames should use higher guidance weight. Insertion should use stronger guidance near contact, pre-bounce, and insertion, but weaker guidance in free-space approach.

4. Make the score explicitly multi-horizon.

   Score `t+1...t+16`, not only one future frame. For board, include force band, contact continuity, marker magnitude, marker temporal smoothness, and spatial contact distribution. For insertion, include margin from pre-bounce/bounce plus sustained insertion likelihood.

5. Add uncertainty/trust control around guidance.

   The current trust-region update is necessary. A stronger version should reduce guidance scale when Foresight residual is high or predicted contact is out of distribution. Fisher-preserving guidance and feedback world-model papers support this concern.

6. Add a fast residual correction layer only after the guidance path is validated.

   DP + TacQualityEnergy handles chunk-level action generation. A small force/tactile residual controller can handle high-frequency force deviations during execution. This is a second layer, not a replacement for the classifier/energy guidance story.

7. Distill guided behavior after real evidence.

   If paired real rollouts show that guided DP improves force curves or insertion success, collect guided action chunks and train a faster distilled DP. MODIP and related world-model optimization work support this as a later step.

## Evaluation Protocol Needed

Offline validation is necessary but not sufficient. Final claims need paired real rollouts:

- same task family and comparable initial conditions
- baseline vs guided under the same rollout protocol
- server-side force/action/guidance logs saved for every trial
- board metrics: mean force, force variance, percent time in target force band, force jerk/smoothness, contact loss duration, task completion if available
- insertion metrics: success/failure, bounce/pre-bounce evidence, force/tactile spike, retries or recovery events

## Current Bottom Line

The current 260617-only DP training should continue under monitoring. The best validation point is still epoch 85 with val `0.014062`; epoch 95 through epoch 215 have not refreshed best and now show sustained validation degradation. The safest checkpoint for rollout remains:

`/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun/dp_best.pth`

For the research story, the strongest version is:

`visual/proprio DP action prior + tactile foresight + physically interpretable tactile quality energy + contact-phase/trust-aware classifier guidance`.

This is more defensible than claiming "tactile concat alone" as the main contribution, because recent work is already converging on future tactile prediction, inference-time steering, force-aware contact modeling, and contact-aware gating.
