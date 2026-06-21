# 2026-06-21 260617-only Board DP Completion and Recent Arxiv Architecture Review

## Scope

This note records the DP training requested for:

```text
/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban
```

Only the contained board dataset was used:

```text
/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617
```

The second part summarizes recent arXiv work from roughly the last two months
that is relevant to this project's DP plus tactile/force guidance story.

## DP Training Status

The matching 2000-epoch run has already completed successfully:

```text
run dir:
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun

training script:
scripts/train/train_dp_tac_concat_board_260617_only_20260620_e2000.sh
```

The run was checked from `config.json`, `train.log`, `metrics.json`,
`training_status_latest.json`, and the checkpoint files.

Key configuration:

```text
dataset_dirs:
  /media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617

episodes:
  80 hdf5 files present
  79 valid episodes read
  72 train episodes
  8 validation episodes

policy:
  diffusion/train_dp_tac_concat.py
  image cameras: global,wrist
  proprio: proprio_joint
  action: actions/joint_abs
  tactile side: left
  tactile history: 8
  obs_horizon: 2
  pred_horizon: 16
  n_action_steps: 8

image:
  raw/cached resize: 200 x 266
  crop: 200 x 266
  image cache:
    /home/chenshuai/Project/output/cache/dp_board_rawimg200x266_fp16

tactile encoder:
  /home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt

optimization:
  epochs: 2000
  batch_size: 64
  lr: 5e-5
  weight_decay: 1e-5
  warmup_steps: 1000
  max_steps_per_epoch: 128
  val_ratio: 0.1
  val_interval: 5
  save_freq: 50
  latest_freq: 10
  topk_k: 3
  seed: 20
```

Checkpoint status:

```text
dp_best.pth:
  epoch index in ckpt: 84
  human epoch: 85
  train_loss: 0.0115755695
  val_loss: 0.0140619173
  best_metric_name: val_loss

dp_final.pth:
  epoch: 2000
  final train_loss: about 0.002152
  final val_loss: 0.058560

dp_epoch2000.pth:
  saved at the end of the 2000-epoch run
```

## Training Interpretation

The run completed, but the validation-selected checkpoint is much earlier than
the final checkpoint.

Important observation:

```text
best validation loss:
  epoch 85, val_loss 0.014062

final validation loss:
  epoch 2000, val_loss 0.058560

final / best validation ratio:
  about 4.16x worse
```

This means the run strongly overfits after the early stage.  The training loss
continues decreasing to about `0.002`, while validation loss gets worse.

For deployment/offline comparison, use:

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun/dp_best.pth
```

Do not use `dp_final.pth` as the default unless the goal is only to inspect the
last training state.

Likely reasons for overfitting:

1. The dataset has only 79 valid episodes, while sliding-window training creates
   many highly correlated samples from the same trajectories.
2. The model is large: two ResNet18 image encoders plus a conditional U-Net.
3. Episode-level validation is harder and more honest than random frame/window
   splitting, so validation exposes memorization of trajectory-specific details.
4. The validation split has 8 episodes, which is small but still enough to show
   the train/validation divergence.

This is not a failed run.  It means checkpoint selection by episode-level
validation is necessary.

## Current Architecture Position

The current project architecture is best described as:

```text
1. DP action prior
   image + proprio + tactile latent -> action chunk distribution

2. Tactile/force future predictor
   current state + action chunk -> predicted future contact consequences

3. TacQualityEnergy guidance
   predicted future contact -> differentiable quality score
   -> bounded gradient update on the DP action sample
```

The 260617-only DP is an action-prior/baseline model.  It should not be the main
novelty by itself.

The stronger story is:

```text
Use DP to generate plausible actions, then use an action-conditioned
tactile/force consequence model to score whether those actions will produce good
contact, and use the score gradient to refine the action chunk before execution.
```

This remains gradient guidance, not reranking.

## Recent Arxiv Papers Checked

The following papers were checked through the arXiv API or arXiv pages during
this work session.

| paper | arXiv | date | relevance |
|---|---:|---:|---|
| Inference-time Policy Steering via Vision and Touch | https://arxiv.org/abs/2606.14981 | 2026-06-12 | Direct support for inference-time steering using touch in contact-rich manipulation. |
| Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation | https://arxiv.org/abs/2606.08737 | 2026-06-07 | Supports action-conditioned future tactile dynamics and contact-gated visuotactile modeling. |
| FAWAM: Force-Aware World Action Models for Closed-Loop Contact-Rich Manipulation | https://arxiv.org/abs/2606.08555 | 2026-06-07 | Supports using force for prediction and execution-time correction, not only as an observation. |
| TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation | https://arxiv.org/abs/2606.11184 | 2026-06-09 | Very close reference for force-conditioned tactile foresight. Our distinction should be differentiable quality-energy guidance for DP actions. |
| ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation | https://arxiv.org/abs/2606.13877 | 2026-06-11 | Supports spatially structured and temporally continuous contact representations. |
| PACT: Self-Evolving Physical Safety Alignment for Diffusion Policies in Embodied Manipulation | https://arxiv.org/abs/2606.08414 | 2026-06-07 | Supports post-training physical constraint alignment for diffusion policies. |
| Fisher-Preserving Guidance: Training-Free Manifold Constraints for Safe Diffusion Control | https://arxiv.org/abs/2605.29937 | 2026-05-28 | Supports bounded/manifold-aware guidance instead of unconstrained score maximization. |
| Frequency-Aware Flow Matching for Continuous and Consistent Robotic Action Generation | https://arxiv.org/abs/2606.20135 | 2026-06-18 | Supports temporal smoothness/frequency consistency for robot action generation. |
| Training and Evaluating Diffusion Policies with Long Context Lengths | https://arxiv.org/abs/2606.16447 | 2026-06-15 | Supports increasing observation context when tasks require memory or repeated contact-state reasoning. |
| LAGO Policy: Latency-Aware Asynchronous Diffusion Policies with Goal-Directed Collision-Free Planning for Smooth Manipulation | https://arxiv.org/abs/2606.17982 | 2026-06-16 | Supports smooth inter-chunk execution and guidance-aware action generation. |
| SI-Diff: A Framework for Learning Search and High-Precision Insertion with a Force-Domain Diffusion Policy | https://arxiv.org/abs/2605.12247 | 2026-05-12 | Supports force-domain modeling for insertion and contact-rich assembly. |
| ForceFlow: Learning to Feel and Act via Contact-Driven Flow Matching | https://arxiv.org/abs/2605.11048 | 2026-05-11 | Supports force-aware reactive policies for contact-rich manipulation. |

## Architecture Improvements Suggested by the Literature

1. Keep the DP as a prior, not as the only decision-maker.

   The current DP is good for producing actions that look like demonstrations.
   It is not enough to guarantee good force/contact consequences, especially for
   board wiping where quality depends on force magnitude and force smoothness.

2. Keep the force-aware Foresight branch.

   For board wiping, quality is mainly:

   ```text
   contact exists
   force is not too small
   force is not too large
   force changes smoothly
   action remains close to the demonstrated manifold
   ```

   A marker-only scorer is incomplete.  The force-aware predictor/score should
   stay central.

3. Use multi-horizon energy, not a single-frame score.

   The board score should aggregate over `t+1 ... t+16`, with heavier weight on
   contact-stage frames:

   ```text
   E_good =
     good_force_band_margin
     + contact_continuity
     - force_center_penalty
     - force_smoothness_penalty
     - action_delta_penalty
   ```

4. Use phase/contact gating.

   Approach frames should receive weak or zero wiping-quality guidance.  Contact
   wiping frames should receive strong force-band and smoothness guidance.

5. Keep bounded gradient guidance.

   The guidance should update the action chunk by score gradients, but only
   inside a trust region.  This is consistent with Fisher-preserving and
   physical-safety-alignment style work.

   The intended update is:

   ```text
   a_0 = DP denoised action sample
   score = TacQualityEnergy(Foresight(obs, a_0))
   a_1 = a_0 + eta * normalized_grad(score, a_0)
   accept a_1 only if:
     score(a_1) > score(a_0)
     action_delta_norm is below threshold
     optional smoothness/action-limit constraints pass
   ```

6. Add prediction-error-aware guidance later.

   If recent real rollout tactile/force prediction error is high, guidance
   should be weakened.  This is important because a wrong future model can push
   actions in the wrong direction.

7. Evaluate with real force traces.

   Offline loss is not enough.  The final claim needs paired real rollouts:

   ```text
   baseline DP vs guided DP
   same or matched initial conditions
   server-side force_trace.csv saved per trajectory
   compare force magnitude band, force smoothness, contact continuity, and task success
   ```

## Recommended Current Ckpt for Testing

Use:

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun/dp_best.pth
```

Reason:

```text
It is selected by held-out episode validation loss.
It is much better than the final epoch on validation loss.
```

Avoid using:

```text
dp_final.pth
```

Reason:

```text
It is the last 2000-epoch state and is overfit by validation loss.
```

## Next Useful Work

Short-term:

1. Run real-window audit for the force-aware guided server path.
2. Use `dp_best.pth` from the 260617-only DP run for board testing.
3. Log force traces server-side for every real rollout.
4. Compare baseline and guided rollouts using the same force metrics.

Medium-term:

1. Add long-context or contact-state memory if repeated wiping behavior gets
   stuck or loses context.
2. Add prediction-error gating to reduce guidance strength when Foresight is
   unreliable.
3. Consider stronger action-smoothness/frequency regularization for board
   wiping, especially if real execution has inter-chunk discontinuities.

Evidence boundary:

```text
Completed:
  260617-only DP 2000-epoch training
  validation-based ckpt selection
  recent arXiv architecture review

Not yet proven:
  real robot board wiping improvement from guidance
  paired baseline-vs-guided force curve improvement
```

