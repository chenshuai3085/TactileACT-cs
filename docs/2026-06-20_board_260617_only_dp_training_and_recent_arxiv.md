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

2026-06-20 14:38 update:

- Live log reached epoch `270/2000`: train `0.007050`, val `0.022192`.
- Monitor CSV reached epoch `272/2000`: train `0.006766`, no validation point.
- Best checkpoint remains epoch `85`, val `0.014062`.
- Epoch 270 validation is about `57.8%` worse than the best validation loss.
- Training PID `3794700`, watcher PID `3804063`, and monitor PID `3822906` are all still alive.
- GPU memory remains about `14.7GB / 24.6GB`; GPU utilization is normal during batches.
- Conclusion is unchanged: let the requested 2000-epoch trace continue, but the current rollout candidate remains `dp_best.pth`, not `dp_latest.pth` or later periodic checkpoints.

2026-06-20 14:56 update:

- `dp_epoch300.pth` saved successfully, size about `2.6G`.
- Epoch 300: train `0.006280`, val `0.024982`, no best refresh.
- Best checkpoint remains epoch `85`, val `0.014062`.
- Epoch 300 validation is about `77.7%` worse than the best validation loss.
- The long run is healthy mechanically, but the validation curve is now sustained overfit/validation degradation. Continue the requested 2000-epoch trace, but real rollout/testing should still use `dp_best.pth`.

2026-06-20 15:09 update:

- Latest parsed epoch: `321/2000`.
- Latest validation point: epoch `320`, train `0.006278`, val `0.024674`.
- Best checkpoint remains epoch `85`, val `0.014062`.
- Epoch 320 validation is about `75.5%` worse than the best validation loss.
- Processes remain healthy:
  - training PID `3794700`
  - watcher PID `3804063`
  - monitor PID `3822906`
  - GPU memory about `14.7GB / 24.6GB`, utilization about `70-75%`
- Interpretation is unchanged: the run is mechanically healthy and should continue for the requested long trace, but the deployable checkpoint is still `dp_best.pth`.

2026-06-20 15:25 update:

- `dp_epoch350.pth` saved successfully, size about `2.6G`.
- Epoch 350: train `0.006269`, val `0.029262`, no best refresh.
- Best checkpoint remains epoch `85`, val `0.014062`.
- Epoch 350 validation is about `108.1%` worse than the best validation loss.
- Live log continued into epoch `352`, so training did not stall after checkpoint save.
- Interpretation: the long run remains mechanically healthy, but by epoch 350 validation degradation is severe. The periodic checkpoint is useful for the training trace only; it should not be used as the default rollout candidate.

2026-06-20 15:56 update:

- `dp_epoch400.pth` saved successfully, size about `2.6G`.
- Epoch 400: train `0.005481`, val `0.030581`, no best refresh.
- Best checkpoint remains epoch `85`, val `0.014062`.
- Epoch 400 validation is about `117.5%` worse than the best validation loss.
- Live log continued into epoch `401/402`; training, watcher, monitor, GPU, and disk are still healthy.
- Interpretation: this confirms a sustained long-run overfit trace. Keep training only because the requested experiment is a 2000-epoch run, but do not use `dp_epoch400.pth` or `dp_latest.pth` for rollout unless a later validation point unexpectedly refreshes best.

Monitoring files:

- `train.log`: raw training log
- `training_status_latest.json`: latest parsed status
- `loss_curve.png` / `loss_curve.csv`: periodically refreshed by the monitor
- `monitor_training.log`: monitor process log

## Recent Arxiv Work Relevant to This Project

Search window: papers submitted or updated in roughly the last two months from
2026-06-20. I rechecked the paper IDs and titles with the arXiv API. TouchGuide
is older by first submission date but kept as background because it is a named
tactile-guidance reference and was updated inside this time window.

Main trend: the recent direction is not plain tactile concatenation. The stronger
direction is future contact prediction, inference-time steering, contact/phase
gating, force-aware world models, and trust-region guidance that does not push
the policy far off its action manifold.

Verified recent entries:

| paper | arXiv | date | most relevant point |
|---|---:|---:|---|
| Inference-time Policy Steering via Vision and Touch | https://arxiv.org/abs/2606.14981 | 2026-06-12 | direct support for tactile inference-time steering of generative robot policies |
| Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation | https://arxiv.org/abs/2606.08737 | 2026-06-07 | contact-gated visuo-tactile fusion and future tactile dynamics |
| FAWAM: Force-Aware World Action Models for Closed-Loop Contact-Rich Manipulation | https://arxiv.org/abs/2606.08555 | 2026-06-07 | force should be a first-class future contact prediction signal |
| ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation | https://arxiv.org/abs/2606.13877 | 2026-06-11 | spatially structured and temporally continuous representations help contact-rich planning |
| Tube Diffusion Policy: Reactive Visual-Tactile Policy Learning for Contact-rich Manipulation | https://arxiv.org/abs/2604.23609 | 2026-04-26 | chunked policies need reactive visual-tactile correction during contact |
| SO-TA: Spacetime Optimal-Transport Attention for Visuo-Haptic Imitation Learning of Contact-Rich Manipulation | https://arxiv.org/abs/2605.20433 | 2026-05-19 | force/pose can structure visuo-haptic attention |
| WT-UMI: Tactile-based Whole-Body Manipulation via Force-Supervised Contact-Aware Planning | https://arxiv.org/abs/2606.13232 | 2026-06-11 | force supervision can explicitly regulate contact-aware planning |
| T-Rex: Tactile-Reactive Dexterous Manipulation | https://arxiv.org/abs/2606.17055 | 2026-06-15 | tactile signals should be used reactively, not only as static encodings |
| TactSpace: Learning a Physics-enriched Shared Latent Space for Tactile Sim-to-Real Transfer | https://arxiv.org/abs/2606.18959 | 2026-06-17 | tactile latents should preserve physically meaningful contact information |
| HT-Bench: Benchmarking and Learning Dexterous Full-Hand Tactile Representations with Egocentric Vision | https://arxiv.org/abs/2606.19161 | 2026-06-17 | tactile representations need geometry, cross-modal alignment, and temporal prediction evaluation |
| TaCauchy: An Extensible FEM Framework for Vision-Based Tactile Simulation | https://arxiv.org/abs/2606.20426 | 2026-06-18 | physically grounded force/pressure supervision is becoming important for tactile learning |
| Frequency-Aware Flow Matching for Continuous and Consistent Robotic Action Generation | https://arxiv.org/abs/2606.20135 | 2026-06-18 | action smoothness and frequency consistency are current concerns for generative policies |
| Ambient Diffusion Policy: Imitation Learning from Suboptimal Data in Robotics | https://arxiv.org/abs/2606.12365 | 2026-06-10 | diffusion training can use lower-quality data selectively instead of treating all demos equally |
| Action-Effect Memory Pretraining for Robot Manipulation | https://arxiv.org/abs/2606.12499 | 2026-06-10 | action-conditioned future effects are a useful representation target |
| PACT: Self-Evolving Physical Safety Alignment for Diffusion Policies in Embodied Manipulation | https://arxiv.org/abs/2606.08414 | 2026-06-07 | constraint-style gradients can align diffusion policies with physical safety |
| Fisher-Preserving Guidance: Training-Free Manifold Constraints for Safe Diffusion Control | https://arxiv.org/abs/2605.29937 | 2026-05-28 | guidance should preserve the policy manifold instead of only maximizing score |
| World Pilot / LaWAM / MemoryWAM | https://arxiv.org/abs/2606.12403, https://arxiv.org/abs/2606.15768, https://arxiv.org/abs/2606.20562 | 2026-06 | world-action priors and memory are becoming central for robot policies |
| TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance | https://arxiv.org/abs/2601.20239 | 2026-01-28; updated 2026-05-13 | background reference for tactile feasibility steering at inference time |

I explicitly removed several previously listed unverified title/ID pairs from
this document. They should not be cited unless independently rechecked.

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

Serving-side guidance code review on 2026-06-20:

- The server is not doing reranking for the current TacQuality path.
- `guidance_location=final_action` runs DP denoising first, then applies a bounded accept-only trust-region gradient update to the clean action chunk.
- `guidance_location=denoising_step` applies TacQuality gradients inside the DP sampling loop on predicted clean action `x0` for the final low-noise denoising steps.
- The current real-test recommendation stays conservative: use `final_action` for the main baseline-vs-guided robot comparison, and keep `denoising_step` as an enhancement/ablation arm until real force traces show it is safe and useful.
- Board `marker_joint_s12_guided` is intentionally conservative: `score_mode=quality`, `action_step=0.0002`, `max_total_delta=0.02`.
- Insertion `good_margin_guided` is much stronger by default: `score_mode=good_margin`, `action_step=0.02`, `max_total_delta=0.08`.
- This parameter difference is consistent with the audits: insertion currently has strong score/action deltas; board has strong offline classification but small action-side gradient magnitude.

## Best Current Story

The current paper/project story should be:

1. Base DP learns the visual/proprio action prior.
2. TactileVAE compresses marker fields into a compact contact latent.
3. Multi-step Foresight predicts future contact consequences from candidate action chunks.
4. TacQualityEnergy turns predicted future contact into a physically interpretable quality score.
5. During denoising, gradient guidance moves the action sample toward better predicted contact outcomes while a trust region keeps the action close to the DP manifold.

This is not reranking. Reranking can be a diagnostic baseline, but the main method is action-gradient guidance.

2026-06-20 15:09 literature follow-up:

- Recent arXiv API checks again surface the same direction: tactile robotics work is moving toward inference-time policy steering, tactile/force world models, contact-aware representation, and force-aware future prediction.
- New highly relevant recent entries include:
  - `2606.20426` TaCauchy: FEM framework for vision-based tactile simulation.
  - `2606.19161` HT-Bench: dexterous full-hand tactile representation benchmark.
  - `2606.18959` TactSpace: physics-enriched tactile latent space.
  - `2606.17055` T-Rex: tactile-reactive dexterous manipulation.
  - `2606.14981` Inference-time Policy Steering via Vision and Touch.
- These papers strengthen, rather than change, the current story: tactile concat DP is the action-prior baseline, while the novel part should be future-contact quality energy plus bounded gradient guidance.
- Current project weakness remains evidence/transfer, not offline separability: board guidance has finite gradients, but score-to-action deltas are still small and real paired rollout force logs are still missing.

## Recommended Improvements

1. Keep the active `260617-only` DP run as a baseline and use `dp_best.pth`.

   Reason: validation already plateaued after epoch 85. The 2000-epoch trace is useful evidence, but later checkpoints should not automatically become deployment candidates.

2. Upgrade board Foresight from marker-only to force-aware/contact-aware prediction.

   Reason: board quality is defined by force band and force smoothness. FAWAM, WT-UMI, Dream-Tac, and TaCauchy all support treating force/contact as first-class future prediction or supervision signals.

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

   The current trust-region update is necessary. A stronger version should reduce guidance scale when Foresight residual is high or predicted contact is out of distribution. Fisher-Preserving Guidance and PACT support this concern.

6. Add a fast residual correction layer only after the guidance path is validated.

   DP + TacQualityEnergy handles chunk-level action generation. A small force/tactile residual controller can handle high-frequency force deviations during execution. This is a second layer, not a replacement for the classifier/energy guidance story.

7. Distill guided behavior after real evidence.

   If paired real rollouts show that guided DP improves force curves or insertion success, collect guided action chunks and train a faster distilled DP. MODIP and related world-model optimization work support this as a later step.

8. Treat board denoising-step guidance as the next controlled ablation, not the default claim.

   Reason: the code already supports DDPM-step gradients, but board score-to-action deltas are currently small and the scorer is still proxy/marker-based rather than direct force-prediction based. The cleaner order is:

   ```text
   main real test: final_action baseline vs final_action guided
   ablation: denoising_step guided with the same DP/scorer/Foresight
   next model change: force-aware Foresight or force-proxy head
   ```

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

## 21:26 Epoch 950 Checkpoint

The run reached epoch `950/2000` and saved:

`/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun/dp_epoch950.pth`

Checkpoint state:

- `dp_epoch950.pth`: `2.6G`, written at `2026-06-20 21:26`.
- `dp_latest.pth`: `5.1G`, still updating normally.
- `dp_best.pth`: `2.6G`, still epoch `85` best from `2026-06-20 12:39`.
- training continued after checkpoint writing, so the checkpoint save did not stall the run.

Epoch 950 metrics:

- train loss: `0.003613`
- val loss: `0.042765`
- top-k train best: `0.003023`
- best remains: epoch `85`, val `0.014062`
- epoch 950 val / best val ratio: about `3.04x`

Recent validation progression:

- epoch 910: train `0.003340`, val `0.049196`
- epoch 915: train `0.003362`, val `0.045476`
- epoch 920: train `0.003626`, val `0.047983`
- epoch 925: train `0.003209`, val `0.047004`
- epoch 930: train `0.003859`, val `0.045609`
- epoch 935: train `0.003283`, val `0.048963`
- epoch 940: train `0.003457`, val `0.051306`
- epoch 945: train `0.003406`, val `0.041623`
- epoch 950: train `0.003613`, val `0.042765`

Current process state around this check:

- training PID `3794700` is still running.
- watcher PID `3804063` is still running.
- monitor PID `3822906` is still running.
- GPU around this check: `14.7GB / 24.6GB`, utilization about `68%`, temperature about `53C`.

Interpretation:

- The training job remains mechanically healthy.
- The late training loss is low, but held-out episode validation remains about `3x` worse than the epoch-85 best.
- This checkpoint confirms the same long-run overfit / validation-degradation trend seen at epochs 550, 600, and 800.
- `dp_epoch950.pth` should be kept as a long-run trace checkpoint, not promoted for rollout.
- Default rollout candidate remains:

`/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun/dp_best.pth`

## 20:57 Epoch 900 Checkpoint

The run reached epoch `900/2000` and saved:

`/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun/dp_epoch900.pth`

Checkpoint state:

- `dp_epoch900.pth`: `2.6G`, written at `2026-06-20 20:56`.
- `dp_latest.pth`: `5.1G`, written at `2026-06-20 20:56`.
- `dp_best.pth`: `2.6G`, still written at `2026-06-20 12:39`.
- training continued into epoch `901`, so checkpoint saving did not stall the run.

Epoch 900 metrics:

- train loss: `0.003776`
- val loss: `0.044429`
- top-k train best: `0.003217`
- best remains: epoch `85`, val `0.014062`
- epoch 900 val / best val ratio: about `3.16x`

Recent validation progression:

- epoch 850: train `0.003723`, val `0.039196`
- epoch 855: train `0.003588`, val `0.045827`
- epoch 860: train `0.003695`, val `0.051059`
- epoch 865: train `0.003681`, val `0.044813`
- epoch 870: train `0.003538`, val `0.047467`
- epoch 875: train `0.003599`, val `0.051545`
- epoch 880: train `0.003570`, val `0.052303`
- epoch 885: train `0.003784`, val `0.047865`
- epoch 890: train `0.003719`, val `0.052531`
- epoch 895: train `0.003435`, val `0.047300`
- epoch 900: train `0.003776`, val `0.044429`

Interpretation:

- Epoch 900 confirms the long-run overfit / validation-degradation trace.
- The training job and checkpoint saving remain mechanically healthy.
- `dp_epoch900.pth` is not a default rollout candidate.
- The deployment recommendation remains `dp_best.pth` from epoch `85`.

## 20:28 Epoch 850 Checkpoint

The run reached epoch `850/2000` and saved:

`/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun/dp_epoch850.pth`

Checkpoint state:

- `dp_epoch850.pth`: `2.6G`, written at `2026-06-20 20:27`.
- `dp_latest.pth`: `5.1G`, written at `2026-06-20 20:27`.
- `dp_best.pth`: `2.6G`, still written at `2026-06-20 12:39`.
- training continued into epoch `851`, so checkpoint saving did not stall the run.

Epoch 850 metrics:

- train loss: `0.003723`
- val loss: `0.039196`
- top-k train best: `0.003217`
- best remains: epoch `85`, val `0.014062`
- epoch 850 val / best val ratio: about `2.79x`

Recent validation progression:

- epoch 810: train `0.003719`, val `0.049478`
- epoch 820: train `0.003933`, val `0.046754`
- epoch 825: train `0.003331`, val `0.046229`
- epoch 830: train `0.003589`, val `0.044840`
- epoch 835: train `0.003490`, val `0.044443`
- epoch 840: train `0.003692`, val `0.045580`
- epoch 845: train `0.003857`, val `0.043564`
- epoch 850: train `0.003723`, val `0.039196`

Interpretation:

- Epoch 850 is slightly better than the recent 810-845 validation points, but it
  is still far above the epoch-85 best.
- The training job and checkpoint saving remain mechanically healthy.
- This does not change the deployment recommendation: use `dp_best.pth`, not
  `dp_epoch850.pth` or `dp_latest.pth`, for the default rollout candidate.

## 20:10 arXiv API Recheck and Architecture Implication

I rechecked the main recent-work list through the arXiv API on 2026-06-20.
The following records were returned directly by arXiv and are safe for the
current project notes:

- `2606.14981v1`, 2026-06-12, `Inference-time Policy Steering via Vision and Touch`
- `2606.08737v1`, 2026-06-07, `Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation`
- `2606.08555v2`, 2026-06-07, `FAWAM: Force-Aware World Action Models for Closed-Loop Contact-Rich Manipulation`
- `2606.13877v1`, 2026-06-11, `ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation`
- `2604.23609v1`, 2026-04-26, `Tube Diffusion Policy: Reactive Visual-Tactile Policy Learning for Contact-rich Manipulation`
- `2605.20433v1`, 2026-05-19, `Spacetime Optimal-Transport Attention for Visuo-Haptic Imitation Learning of Contact-Rich Manipulation`
- `2606.13232v1`, 2026-06-11, `WT-UMI: Tactile-based Whole-Body Manipulation via Force-Supervised Contact-Aware Planning`
- `2606.17055v1`, 2026-06-15, `T-Rex: Tactile-Reactive Dexterous Manipulation`
- `2606.18959v1`, 2026-06-17, `TactSpace: Learning a Physics-enriched Shared Latent Space for Tactile Sim-to-Real Transfer`
- `2606.19161v1`, 2026-06-17, `HT-Bench: Benchmarking and Learning Dexterous Full-Hand Tactile Representations with Egocentric Vision`
- `2606.20426v1`, 2026-06-18, `TaCauchy: An Extensible FEM Framework for Vision-Based Tactile Simulation`
- `2606.20135v1`, 2026-06-18, `Frequency-Aware Flow Matching for Continuous and Consistent Robotic Action Generation`
- `2606.12365v1`, 2026-06-10, `Ambient Diffusion Policy: Imitation Learning from Suboptimal Data in Robotics`
- `2606.12499v1`, 2026-06-10, `Action-Effect Memory Pretraining for Robot Manipulation`
- `2606.08414v1`, 2026-06-07, `PACT: Self-Evolving Physical Safety Alignment for Diffusion Policies in Embodied Manipulation`
- `2605.29937v1`, 2026-05-28, `Fisher-Preserving Guidance: Training-Free Manifold Constraints for Safe Diffusion Control`
- `2606.11087v1`, 2026-06-09, `Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning`
- `2606.12403v1`, 2026-06-10, `World Pilot: Steering Vision-Language-Action Models with World-Action Priors`
- `2606.15768v1`, 2026-06-14, `LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies`
- `2606.20562v1`, 2026-06-18, `MemoryWAM: Efficient World Action Modeling with Persistent Memory`
- `2601.20239v6`, 2026-01-28, updated 2026-05-13, `TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance`

Several older notes mention additional titles or IDs from earlier search passes.
Those are not treated as core citations in this document unless they are
separately rechecked.  This is important because the project story should rely
on verified references, not search snippets.

Architecture implication from the verified set:

1. Keep tactile-concat DP as a baseline / action prior, not the main novelty.
2. Make the main method a differentiable future-consequence guidance stack:

   ```text
   visual/proprio DP prior
     -> candidate action chunk
     -> multi-step tactile/force Foresight
     -> physically interpretable TacQualityEnergy
     -> bounded test-time gradient guidance
   ```

3. For board wiping, the next model improvement should be force-aware and
   contact-gated.  The board quality definition is physical: force magnitude,
   force smoothness, contact continuity, and marker-field stability.  A
   marker-only future model is useful but incomplete for this task.
4. For insertion, keep the current good-margin risk guidance as the default
   because insertion has stronger offline and guidance-gradient evidence than
   board wiping.
5. Add trust-region / manifold-preserving guidance as a formal design element,
   not a code afterthought.  This is supported by recent test-time gradient and
   safe diffusion-control work.
6. Do not claim robot-task improvement until paired real rollout force/action
   logs show baseline-vs-guided differences under the same protocol.

For the research story, the strongest version is:

`visual/proprio DP action prior + tactile foresight + physically interpretable tactile quality energy + contact-phase/trust-aware classifier guidance`.

This is more defensible than claiming "tactile concat alone" as the main contribution, because recent work is already converging on future tactile prediction, inference-time steering, force-aware contact modeling, and contact-aware gating.

## 16:10 Training Supervision Update

Active run:

`/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun`

Dataset:

`/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617`

Status:

- training PID `3794700` is still running.
- watcher PID `3804063` is still running.
- monitor PID `3822906` is still running.
- GPU status around this check: `14.7GB / 24.6GB`, utilization around `61%`.
- live log has reached epoch `423/2000` and has entered epoch `424`.
- latest validation line: epoch `420`, train `0.005665`, val `0.030213`, best still `0.014062`.
- previous validation line: epoch `415`, train `0.005377`, val `0.027974`.

Interpretation:

- The run is mechanically healthy: process, GPU, logs, latest checkpoint, monitor, and watcher are all alive.
- The validation loss is still far above best and has not recovered.
- This is a sustained overfit / long-run plateau trace, not a crash or logging issue.
- Continue the 2000-epoch run because it was requested as a sufficient long training trace, but deployment and real rollout should still default to `dp_best.pth` from epoch `85`.

## 16:15 Recent arXiv Scan

Scope:

- queried arXiv through the API for recent `submittedDate:[20260420 TO 20260620]` entries.
- searched around tactile robot manipulation, visuotactile manipulation, force-aware manipulation, diffusion policies, guidance, and robot world models.
- this is a relevance scan, not a full paper-by-paper reproducibility review.

Most relevant recent papers:

1. Inference-time Policy Steering via Vision and Touch, `2606.14981`
   - URL: `https://arxiv.org/abs/2606.14981`
   - Directly relevant because it frames tactile guidance as inference-time steering of generative robot policies.
   - The strongest connection to this project is low-level tactile-guided diffusion editing: our TacQualityEnergy + Foresight guidance is the same broad family, but our score is physically interpretable for board wiping / insertion.

2. ContactWorld, `2606.13877`
   - URL: `https://arxiv.org/abs/2606.13877`
   - Directly relevant because it studies vision-tactile world models for contact-rich manipulation and emphasizes spatial/temporal structure and long-horizon contact robustness.
   - This supports our choice to predict future tactile/contact consequences instead of only concatenating tactile history into the DP observation.

3. Dream-Tac, `2606.08737`
   - URL: `https://arxiv.org/abs/2606.08737`
   - Directly relevant because it jointly models actions, future visual observations, and tactile dynamics with contact-gated visuotactile fusion.
   - This strongly supports a next version of our Foresight: contact-gated tactile/force prediction with multi-step future contact heads.

4. Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning, `2606.11087`
   - URL: `https://arxiv.org/abs/2606.11087`
   - Relevant because it keeps the generative policy fixed and uses a critic/value gradient at test time.
   - This supports our design boundary: do not retrain DP for every scorer; train a stable DP prior and use a differentiable quality score for bounded test-time guidance.

5. T-Rex / Tactile-Reactive Dexterous Manipulation, `2606.17055`
   - URL: `https://arxiv.org/abs/2606.17055`
   - Relevant because it emphasizes tactile reactivity rather than static tactile encoding.
   - This supports adding contact-phase gating and online force/tactile logging to the real rollout pipeline.

6. WT-UMI, `2606.13232`
   - URL: `https://arxiv.org/abs/2606.13232`
   - Relevant because it explicitly treats contact force as a supervised signal for contact-aware planning.
   - This reinforces the need for board wiping to use force-band / force-smoothness evidence, not only marker latent classes.

7. Frequency-Aware Flow Matching, `2606.20135`
   - URL: `https://arxiv.org/abs/2606.20135`
   - Relevant to our DP chunking issue: continuous and temporally consistent action generation is a current concern.
   - This supports evaluating action smoothness and frequency content in real force/action traces.

Architecture implication for this project:

- Keep the current 260617-only tactile-concat DP as a baseline / action prior.
- Do not frame tactile concat alone as the novelty.
- Stronger story:

```text
visual/proprio DP action prior
  -> multi-step tactile/force Foresight predicts future contact consequences
  -> interpretable TacQualityEnergy scores force band, contact continuity, smoothness, and insertion risk
  -> bounded test-time classifier/scorer gradient guidance edits action chunks during or after denoising
  -> server-side real rollout logs verify force/action/outcome improvement
```

Most important next model improvement:

- board: train a force-aware or force-proxy Foresight head, because the current board scorer has strong offline classification but only moderate predicted-score alignment with force-band quality and small clean-action gradient magnitude.
- insertion: keep good-margin guidance as the default because binary/risk guidance is already strong; improve reason labels only for interpretation.
- both tasks: make guidance contact-phase gated and trust-region bounded by default.

## 16:30 Epoch 450 Checkpoint

The run reached epoch `450/2000` and saved:

`/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun/dp_epoch450.pth`

Checkpoint state:

- `dp_epoch450.pth`: `2.6G`, written at `2026-06-20 16:27`.
- `dp_best.pth`: `2.6G`, written at `2026-06-20 12:39`.
- `dp_latest.pth`: `5.1G`, updated at `2026-06-20 16:26`.
- training continued into epoch `451`, so checkpoint saving did not stall the run.

Epoch 450 metrics:

- train loss: `0.005331`
- val loss: `0.032529`
- best remains: epoch `85`, val `0.014062`
- epoch 450 val / best val ratio: about `2.31x`

Current process state:

- training PID `3794700` is still running.
- watcher PID `3804063` is still running.
- monitor PID `3822906` is still running.
- GPU around this check: `14.7GB / 24.6GB`, utilization about `85%`, temperature about `56C`.

Interpretation:

- This checkpoint is mechanically valid and useful as long-run training evidence.
- It should not replace `dp_best.pth` for rollout, because validation has degraded heavily.
- The training/validation split is episode-level: 80 total HDF5 episodes, 72 train episodes, 8 validation episodes.
- The overfit conclusion comes from held-out episode validation, not from frame-level random splitting.

## Architecture Review at Epoch 450

Current DP path:

```text
RGB(global,wrist) + qpos + frozen TactileVAE(left marker history)
  -> obs_cond
  -> ConditionalUnet1D diffusion policy
  -> 16-step joint action chunk
```

This is a reasonable baseline/action prior, but it is not the main novelty.

## 18:30 Epoch 650 Checkpoint and Literature Correction

The run reached epoch `650/2000` and saved:

`/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun/dp_epoch650.pth`

Checkpoint state:

- `dp_epoch650.pth`: about `2.6G`, written at `2026-06-20 18:27`.
- `dp_best.pth`: about `2.6G`, still the epoch `85` best checkpoint.
- `dp_latest.pth`: about `5.1G`, still updating.

Epoch 650 metrics:

- train loss: `0.004319`
- val loss: `0.038270`
- best remains: epoch `85`, val `0.014062`
- epoch 650 val / best val ratio: about `2.72x`

Recent validation points:

- epoch 600: train `0.004665`, val `0.035634`
- epoch 610: train `0.004471`, val `0.038297`
- epoch 620: train `0.004465`, val `0.038352`
- epoch 630: train `0.004299`, val `0.040103`
- epoch 640: train `0.004481`, val `0.036972`
- epoch 645: train `0.004307`, val `0.041321`
- epoch 650: train `0.004319`, val `0.038270`

Process state at this checkpoint:

- training PID `3794700` is still running.
- watcher PID `3804063` is still running.
- monitor PID `3822906` is still running.
- GPU around this check: about `14.7GB / 24.6GB`, utilization about `68%`.

Interpretation:

- Training and checkpoint saving remain mechanically healthy.
- The long-run overfit/validation degradation conclusion is unchanged.
- `dp_epoch650.pth` is useful for the long training trace, but should not replace `dp_best.pth` for rollout.
- Recommended rollout checkpoint remains:

`/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun/dp_best.pth`

Literature correction:

- I rechecked the recent arXiv section with the arXiv API.
- Several earlier title/ID pairs were removed because they were not directly verifiable from the API result.
- The current verified set is enough to support the project direction: tactile/force future-consequence prediction plus bounded gradient guidance is a stronger story than tactile concatenation alone.

Current architecture recommendation after the corrected scan:

1. Keep this 260617-only DP as a baseline/action prior and evaluate with `dp_best.pth`.
2. Treat the existing board TacQuality path as a conservative first real-test arm:
   `action -> Foresight marker prediction -> marker/action proxy score -> bounded score gradient`.
3. Do not claim board guidance is fully solved yet, because the current board scorer does not directly predict future force and its score-to-action gradient is known to be small.
4. Next model change should be a parallel force-aware Foresight path:

```text
marker history + qpos + action chunk
  -> future marker latent / marker field
  -> force proxy head
  -> force-band class head: too_light / good / too_heavy / oscillatory
  -> contact gate head
  -> TacQuality score over t+1...t+16
  -> bounded classifier/scorer guidance on action
```

This is the cleanest way to connect the stated final goal, DP gradient guidance,
to the physical board-quality definition: reasonable force magnitude, stable
contact, and smooth force/marker change.

## 19:00 Epoch 700 Checkpoint

The run reached epoch `700/2000` and saved:

`/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun/dp_epoch700.pth`

Checkpoint state:

- `dp_epoch700.pth`: about `2.6G`, written at `2026-06-20 18:57`.
- `dp_best.pth`: about `2.6G`, still the epoch `85` best checkpoint.
- `dp_latest.pth`: about `5.1G`, still updating.

Epoch 700 metrics:

- train loss: `0.004169`
- val loss: `0.040805`
- best remains: epoch `85`, val `0.014062`
- epoch 700 val / best val ratio: about `2.90x`

Recent validation points after epoch 650:

- epoch 650: train `0.004319`, val `0.038270`
- epoch 655: train `0.004067`, val `0.040885`
- epoch 660: train `0.004681`, val `0.042069`
- epoch 665: train `0.004195`, val `0.038336`
- epoch 670: train `0.003939`, val `0.037263`
- epoch 675: train `0.004092`, val `0.046258`
- epoch 680: train `0.003846`, val `0.044731`
- epoch 685: train `0.003864`, val `0.039010`
- epoch 690: train `0.003949`, val `0.036898`
- epoch 695: train `0.004748`, val `0.036700`
- epoch 700: train `0.004169`, val `0.040805`

Process state at this checkpoint:

- training PID `3794700` is still running.
- watcher PID `3804063` is still running.
- monitor PID `3822906` is still running.
- GPU around this check: about `14.7GB / 24.6GB`, utilization about `73%`.

Interpretation:

- The job is mechanically healthy and continues past epoch `700`.
- Validation has not recovered; the post-650 window still sits around `0.037-0.046`, far above the epoch-85 best.
- `dp_epoch700.pth` should be kept as a long-run checkpoint only.
- The recommended rollout checkpoint remains `dp_best.pth`.

## 19:30 Epoch 750 Checkpoint

The run reached epoch `750/2000` and saved:

`/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun/dp_epoch750.pth`

Checkpoint state:

- `dp_epoch750.pth`: about `2.6G`, written at `2026-06-20 19:27`.
- `dp_best.pth`: about `2.6G`, still the epoch `85` best checkpoint.
- `dp_latest.pth`: about `5.1G`, still updating.

Epoch 750 metrics:

- train loss: `0.003822`
- val loss: `0.045479`
- best remains: epoch `85`, val `0.014062`
- epoch 750 val / best val ratio: about `3.23x`

Recent validation points after epoch 700:

- epoch 700: train `0.004169`, val `0.040805`
- epoch 705: train `0.003489`, val `0.043568`
- epoch 710: train `0.004019`, val `0.043625`
- epoch 715: train `0.003775`, val `0.040154`
- epoch 720: train `0.003903`, val `0.048109`
- epoch 725: train `0.004055`, val `0.043141`
- epoch 730: train `0.003861`, val `0.041605`
- epoch 735: train `0.004058`, val `0.040852`
- epoch 740: train `0.003851`, val `0.039341`
- epoch 745: train `0.004070`, val `0.043655`
- epoch 750: train `0.003822`, val `0.045479`

Process state at this checkpoint:

- training PID `3794700` is still running.
- watcher PID `3804063` is still running.
- monitor PID `3822906` is still running.
- GPU around this check: about `14.7GB / 24.6GB`, utilization about `74%`.

Interpretation:

- The job remains mechanically healthy.
- Validation continues to degrade relative to epoch 85 and is now more than `3x`
  the best validation loss at epoch 750.
- This is further evidence that `dp_best.pth`, not late epoch checkpoints, should
  be used for rollout.
- `dp_epoch750.pth` is kept only as a long-run trace checkpoint.

Current guidance path:

```text
candidate action chunk
  -> Foresight predicts future tactile latent / decoded marker sequence
  -> ForceBandTacQualityEnergy or InsertionRiskScorer scores the predicted consequence
  -> TacQualityTrustRegionRefiner computes d score / d action
  -> bounded accept-only action update
```

This confirms the current implementation is gradient guidance, not reranking.

Important limitation for board wiping:

- The current board runtime uses differentiable marker proxy features and action proxy features.
- Force is used to construct the training labels / force-band target, but force is not directly predicted by Foresight at serving time.
- Existing audits show the board offline classifier is strong, but the gradient pathway is weak:
  - board clean-action `score_delta_mean`: `0.00016205`
  - board clean-action `action_delta_norm_mean`: `0.00077498`
  - board predicted-score vs force-band quality Spearman: `0.3988`
- Therefore the current board guidance is deployable for conservative tests, but the strongest next research improvement is to make Foresight force-aware or force-proxy-aware.

Recommended next architecture:

```text
DP action prior
  -> multi-step Foresight predicts:
       marker latent sequence
       marker delta / smoothness
       force proxy or force-band logits
       contact gate
  -> TacQualityEnergy scores:
       force in-band
       low force / high force / oscillatory contact
       marker smoothness
       action smoothness
       insertion risk margin
  -> trust-region gradient guidance
```

This is also better aligned with the recent arXiv scan: ContactWorld and Dream-Tac support future contact modeling; ViTaL and test-time gradient-guided flow policies support inference-time steering; WT-UMI supports force-supervised contact-aware planning.

## 17:00 Epoch 500 Checkpoint

The run reached epoch `500/2000` and saved:

`/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun/dp_epoch500.pth`

Checkpoint state:

- `dp_epoch500.pth`: `2.6G`, written at `2026-06-20 16:57`.
- `dp_best.pth`: `2.6G`, written at `2026-06-20 12:39`.
- `dp_latest.pth`: `5.1G`, updated at `2026-06-20 16:57`.
- training continued into epoch `501/502`, so checkpoint saving did not stall the run.

Epoch 500 metrics from `train.log`:

- train loss: `0.005230`
- val loss: `0.032898`
- best remains: epoch `85`, val `0.014062`
- epoch 500 val / best val ratio: about `2.34x`

Nearby validation points:

- epoch 475: train `0.004782`, val `0.034969`
- epoch 490: train `0.005094`, val `0.031262`
- epoch 495: train `0.005153`, val `0.035912`
- epoch 500: train `0.005230`, val `0.032898`

Current process state at this check:

- training PID `3794700` is still running.
- watcher PID `3804063` is still running.
- monitor PID `3822906` is still running.
- GPU around this check: `14.7GB / 24.6GB`, utilization about `73%`, temperature about `58C`.

Interpretation:

- The long run remains mechanically healthy.
- The validation trace has not recovered after epoch 85.
- `dp_epoch500.pth` is a valid saved checkpoint for analysis, but it should not replace `dp_best.pth` for real rollout.
- The recommended rollout/evaluation checkpoint remains:

`/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun/dp_best.pth`

## Verified Recent arXiv Boundary

I rechecked recent work through the arXiv API instead of relying on search snippets only. The following IDs were found with valid arXiv records and are safe to cite as background:

- `2606.14981` Inference-time Policy Steering via Vision and Touch
- `2606.13877` ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation
- `2606.08737` Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation
- `2606.08555` FAWAM: Force-Aware World Action Models for Closed-Loop Contact-Rich Manipulation
- `2606.11087` Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning
- `2606.13232` WT-UMI: Tactile-based Whole-Body Manipulation via Force-Supervised Contact-Aware Planning
- `2606.17055` T-Rex: Tactile-Reactive Dexterous Manipulation
- `2606.20135` Frequency-Aware Flow Matching for Continuous and Consistent Robotic Action Generation
- `2606.18959` TactSpace: Learning a Physics-enriched Shared Latent Space for Tactile Sim-to-Real Transfer
- `2606.20426` TaCauchy: An Extensible FEM Framework for Vision-Based Tactile Simulation
- `2606.08414` PACT: Self-Evolving Physical Safety Alignment for Diffusion Policies in Embodied Manipulation
- `2601.20239` TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance

This verified set supports the same project direction:

```text
visual/proprio DP action prior
  + tactile/force future-consequence model
  + interpretable TacQualityEnergy
  + bounded classifier/scorer gradient guidance
```

Do not cite unverified search-only titles in the project story. If a title cannot be found through arXiv or a paper page, keep it as a search lead rather than evidence.

## 17:30 Epoch 550 Checkpoint

The run reached epoch `550/2000` and saved:

`/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun/dp_epoch550.pth`

Checkpoint state:

- `dp_epoch550.pth`: `2.6G`, written at `2026-06-20 17:27`.
- `dp_best.pth`: `2.6G`, still written at `2026-06-20 12:39`.
- `dp_latest.pth`: `5.1G`, updated at `2026-06-20 17:27`.
- training continued into epoch `552/553`, so checkpoint saving did not stall the run.

Epoch 550 metrics:

- train loss: `0.004520`
- val loss: `0.042753`
- best remains: epoch `85`, val `0.014062`
- epoch 550 val / best val ratio: about `3.04x`

Validation progression after epoch 500:

- epoch 500: train `0.005230`, val `0.032898`
- epoch 510: train `0.004678`, val `0.033971`
- epoch 520: train `0.004617`, val `0.031729`
- epoch 530: train `0.004909`, val `0.035000`
- epoch 540: train `0.004730`, val `0.036427`
- epoch 550: train `0.004520`, val `0.042753`

Current process state:

- training PID `3794700` is still running.
- watcher PID `3804063` is still running.
- monitor PID `3822906` is still running.
- GPU around this check: `14.7GB / 24.6GB`, utilization about `84%`, temperature about `57C`.

Interpretation:

- The training process is healthy, but validation degradation is now severe.
- The train loss and top-k train checkpoint continue improving, while held-out episode validation gets worse.
- This is a strong held-out-episode overfit trace. It is not a checkpoint-save issue or a monitor issue.
- `dp_epoch550.pth` should not be used as the default rollout checkpoint.
- Continue the requested 2000-epoch run only as a long-run trace. The recommended checkpoint remains `dp_best.pth` from epoch `85`.

Practical recommendation:

- For real rollout of the 260617-only model, use:

`/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun/dp_best.pth`

- Do not use `dp_latest.pth`, `dp_epoch500.pth`, or `dp_epoch550.pth` unless the goal is specifically to test overfit behavior.

## 18:00 Epoch 600 Checkpoint

The run reached epoch `600/2000` and saved:

`/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun/dp_epoch600.pth`

Checkpoint state:

- `dp_epoch600.pth`: `2.6G`, written at `2026-06-20 17:58`.
- `dp_best.pth`: `2.6G`, still written at `2026-06-20 12:39`.
- `dp_latest.pth`: `5.1G`, updated at `2026-06-20 17:58`.
- training continued into epoch `601/602`, so checkpoint saving did not stall the run.

Epoch 600 metrics:

- train loss: `0.004665`
- val loss: `0.035634`
- best remains: epoch `85`, val `0.014062`
- epoch 600 val / best val ratio: about `2.53x`

Validation progression after epoch 550:

- epoch 550: train `0.004520`, val `0.042753`
- epoch 555: train `0.004154`, val `0.035345`
- epoch 565: train `0.004690`, val `0.034781`
- epoch 575: train `0.004217`, val `0.038904`
- epoch 585: train `0.004710`, val `0.035783`
- epoch 595: train `0.004559`, val `0.038667`
- epoch 600: train `0.004665`, val `0.035634`

Current process state:

- training PID `3794700` is still running.
- watcher PID `3804063` is still running.
- monitor PID `3822906` is still running.
- GPU around this check: `14.7GB / 24.6GB`, utilization about `73%`, temperature about `59C`.

Interpretation:

- The training process and checkpointing remain healthy.
- Validation is slightly better than the worst epoch-550 value, but still far above the best epoch-85 value.
- This does not change the recommendation: `dp_best.pth` remains the only rollout candidate from this run.
- Later checkpoints are useful for documenting the long-run overfit trace, not for deployment.

## 19:58 Epoch 800 Checkpoint

The run reached epoch `800/2000` and saved:

`/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun/dp_epoch800.pth`

Checkpoint state:

- `dp_epoch800.pth`: `2.6G`, written at `2026-06-20 19:57`.
- `dp_latest.pth`: `5.1G`, written at `2026-06-20 19:57`.
- `dp_best.pth`: `2.6G`, still written at `2026-06-20 12:39`.
- training continued into epoch `801`, so validation and checkpoint saving did not stall the run.

Epoch 800 metrics:

- train loss: `0.003953`
- val loss: `0.041427`
- top-k train best: `0.003378`
- best remains: epoch `85`, val `0.014062`
- epoch 800 val / best val ratio: about `2.95x`

Recent validation progression:

- epoch 760: train `0.003996`, val `0.044190`
- epoch 775: train `0.004027`, val `0.040895`
- epoch 780: train `0.003910`, val `0.038762`
- epoch 790: train `0.004037`, val `0.042361`
- epoch 795: train `0.004108`, val `0.040642`
- epoch 800: train `0.003953`, val `0.041427`

Current process state:

- training PID `3794700` is still running.
- watcher PID `3804063` is still running.
- monitor PID `3822906` is still running.
- GPU around this check: `14.7GB / 24.6GB`, utilization about `70%`, temperature about `56C`.

Interpretation:

- The training job, monitor, watcher, and checkpoint saving are mechanically healthy.
- The held-out episode validation loss is still far above the epoch-85 best, while train loss remains low.
- This is still a sustained overfit / validation-degradation trace.
- `dp_epoch800.pth` is useful as a long-run trace checkpoint but should not replace `dp_best.pth` for rollout.
- The default rollout candidate from this run remains:

`/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun/dp_best.pth`
