# 2026-06-19 260617-only DP Supervision and Recent ArXiv Review

## 1. Current Training Run

Purpose: train a board/caheiban tactile DP concat policy using only the 2026-06-17 dataset requested by chenshuai, while monitoring the run and checking recent tactile/diffusion guidance literature for architectural improvements.

Dataset:

- `/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617`
- HDF5 files found: 80 episodes
- This run uses only this dataset, not the earlier 260609/260610 full board mixture.

Training command:

- script: `diffusion/train_dp_tac_concat.py`
- launcher record: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/run_command.txt`
- output directory:
  `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr`
- training PID at 2026-06-19 23:32 CST: `3037873`

Key configuration:

- epochs: 2000
- batch size: 64
- lr: `5e-5`
- weight decay: `1e-5`
- warmup steps: 1000
- pred horizon: 16
- obs horizon: 2
- action horizon used in deployment: 8
- cameras: `global,wrist`
- image size: resize/crop `200x266`
- tactile side: left
- tactile history: 8 frames
- tactile encoder: frozen board-trained TactileVAE
  `/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt`
- image cache:
  `/home/chenshuai/Project/output/cache/dp_board_rawimg200x266_fp16`
- validation: episode-level split, `val_ratio=0.1`
- save policy:
  - `dp_best.pth` by validation loss
  - `dp_latest.pth` every 10 epochs, with optimizer/scheduler
  - `dp_epoch*.pth` every 50 epochs
  - train-loss top-3 checkpoints

Current status at 2026-06-19 23:32 CST:

- latest epoch: 1090 / 2000
- latest train loss: `0.002896`
- latest validation loss: `0.036328`
- best validation loss: `0.011659` at epoch 155
- GPU memory: about 14.7 GB on RTX 4090
- external disk free: about 2.1 TB
- `/home` filesystem free: about 43 GB, so large checkpoints should stay on the external disk.

Checkpoint re-check at 2026-06-19 23:38 CST:

- epoch 1100 completed
- train loss: `0.003021`
- validation loss: `0.036032`
- best validation loss still: `0.011659` at epoch 155
- confirmed files:
  - `dp_epoch1100.pth`
  - `dp_latest.pth`
  - `dp_best.pth`
- training process continued into epoch 1101 after saving.

Checkpoint re-check at 2026-06-20 00:09 CST:

- epoch 1150 completed
- train loss: `0.002894`
- validation loss: `0.038054`
- best validation loss still: `0.011659` at epoch 155
- confirmed files:
  - `dp_epoch1150.pth`
  - `dp_latest.pth`
  - `dp_best.pth`
- training process continued to epoch 1152 after saving.
- interpretation unchanged: process is healthy, but the validation gap is clear; real tests should prioritize `dp_best.pth`.

Checkpoint re-check at 2026-06-20 00:39 CST:

- epoch 1200 completed
- train loss: `0.002609`
- validation loss: `0.032901`
- best validation loss still: `0.011659` at epoch 155
- confirmed files:
  - `dp_epoch1200.pth`
  - `dp_latest.pth`
  - `dp_best.pth`
- training process continued to epoch 1201 after saving.

Checkpoint re-check at 2026-06-20 01:38 CST:

- epoch 1300 completed
- train loss: `0.002743`
- validation loss: `0.038855`
- best validation loss still: `0.011659` at epoch 155
- confirmed files:
  - `dp_epoch1300.pth`
  - `dp_latest.pth`
  - `dp_best.pth`
- training process continued to epoch 1301 after saving.

Checkpoint re-check at 2026-06-20 03:37 CST:

- epoch 1500 completed
- train loss: `0.002294`
- validation loss: `0.039141`
- best validation loss still: `0.011659` at epoch 155
- confirmed files:
  - `dp_epoch1400.pth`
  - `dp_epoch1450.pth`
  - `dp_epoch1500.pth`
  - `dp_latest.pth`
  - `dp_best.pth`
- training process continued to epoch 1501 after saving.
- conclusion unchanged: later checkpoints continue to fit train windows more tightly, while validation remains much worse than the early best.

Interpretation:

- The run is alive and checkpointing correctly.
- Training loss continues to decrease, but validation loss has been worse than the epoch-155 best for a long time.
- This is a likely overfit/validation-gap pattern. It does not mean the training is broken, but it means the deployment candidate should remain `dp_best.pth`, not `dp_latest.pth`, unless a later validation checkpoint improves.
- The run should continue because chenshuai requested 2000 epochs and checkpoint storage is healthy.

Known configuration caveat:

- The LR scheduler computes total steps as `len(train_loader) * epochs`, but this run uses `--max_steps_per_epoch 128`.
- Because of that, cosine decay is slower than intended for the actually used number of optimizer steps.
- Do not hot-fix this active run. For the next run, the scheduler should use `min(len(train_loader), max_steps_per_epoch) * epochs` when `max_steps_per_epoch` is set.

## 2. Recent ArXiv Review

Search window: roughly the most recent two months from 2026-06-19. The focus was contact-rich manipulation, tactile/force world models, diffusion/flow policy steering, and suboptimal data use.

### 2.1 Most Relevant to Our Guidance Direction

1. Inference-time Policy Steering via Vision and Touch, arXiv:2606.14981

- Link: https://arxiv.org/abs/2606.14981
- Main point: ViTaL uses visuo-tactile inference-time steering. It separates high-level visual sampling/verification from low-level tactile-guided diffusion editing, and scores predicted tactile futures in latent space.
- Relevance: This is very close to our intended story. Our system should be described as tactile future scoring plus gradient guidance inside DP sampling, not as post-hoc reranking.

2. Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning, arXiv:2606.11087

- Link: https://arxiv.org/abs/2606.11087
- Main point: a value/critic gradient can steer a pretrained generative control policy at test time without retraining the policy.
- Relevance: This supports treating TacQualityEnergy as a differentiable critic over predicted tactile consequences.

3. Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation, arXiv:2606.08737

- Link: https://arxiv.org/abs/2606.08737
- Main point: a tactile world action model jointly models actions, future visual observations, and tactile dynamics.
- Relevance: It supports the importance of action-conditioned tactile future prediction. Our route is more modular: DP prior + Foresight + TacQualityEnergy.

4. TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation, arXiv:2606.11184

- Link: https://arxiv.org/abs/2606.11184
- Main point: predicts short-horizon tactile latent dynamics conditioned on force/torque and uses anticipated tactile latents for contact-rich control.
- Relevance: Strongly supports our `action -> future tactile latent/marker -> score` design. It also suggests adding force/torque conditioning to our Foresight when reliable force traces are available.

5. ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation, arXiv:2606.13877

- Link: https://arxiv.org/abs/2606.13877
- Main point: spatially structured and temporally continuous representations help contact-rich planning; tactile usefulness depends on cross-modal compatibility.
- Relevance: Our Foresight evaluation should not stop at latent MSE. It should include contact phase, force-band proxy, marker smoothness, and temporal continuity.

### 2.2 Relevant to Board/Insertion Policy Design

6. SI-Diff: A Framework for Learning Search and High-Precision Insertion with a Force-Domain Diffusion Policy, arXiv:2605.12247

- Link: https://arxiv.org/abs/2605.12247
- Main point: force-domain diffusion policy plus mode conditioning for search and high-precision insertion.
- Relevance: For insertion, the good/bad scorer can be strengthened with phase/mode awareness: search, pre-contact, insertion, pre-bounce, bounce/recovery.

7. Tube Diffusion Policy: Reactive Visual-Tactile Policy Learning for Contact-rich Manipulation, arXiv:2604.23609

- Link: https://arxiv.org/abs/2604.23609
- Main point: action chunking limits reactivity; an action tube allows step-wise corrections around nominal chunks.
- Relevance: Our current DP executes action chunks. Guidance should be framed as a small bounded correction around the DP action prior, similar in spirit to keeping a local action tube.

8. T-Rex: Tactile-Reactive Dexterous Manipulation, arXiv:2606.17055

- Link: https://arxiv.org/abs/2606.17055
- Main point: high-frequency tactile reactivity and temporal tactile encoding improve delicate force-control tasks.
- Relevance: For board wiping, a longer tactile/force history and explicit temporal stability terms may be more important than static tactile classification.

9. FTP-1: A Generalist Foundation Tactile Policy Across Tactile Sensors for Contact-Rich Manipulation, arXiv:2606.13102

- Link: https://arxiv.org/abs/2606.13102
- Main point: cross-sensor tactile representation learning with morphology-aware latent tokens.
- Relevance: Long-term improvement: replace hand-specific marker proxies with a normalized tactile token representation if the project expands across sensors/tasks.

10. Ambient Diffusion Policy: Imitation Learning from Suboptimal Data in Robotics, arXiv:2606.12365

- Link: https://arxiv.org/abs/2606.12365
- Main point: suboptimal data should not simply be mixed into imitation learning as if all demonstrations are equal.
- Relevance: The board data contains positive, too-small-force, too-large-force, and oscillatory negative patterns. Negative data should mainly train the scorer/critic or be used with quality-aware weighting, not naively treated as expert actions.

## 3. Current Architecture Judgment

The current high-level architecture remains reasonable:

```text
DP action prior
  -> action-conditioned Foresight predicts future tactile/marker consequence
  -> TacQualityEnergy scores predicted consequence
  -> bounded gradient update improves action within a trust region
```

What is strong:

- DP gives a learned action prior and keeps actions near the demonstration manifold.
- Foresight provides the action-to-future-contact link required for proactive guidance.
- TacQualityEnergy turns task-specific quality definitions into a differentiable scalar.
- Trust-region refinement protects against large out-of-distribution action updates.

What is still weak:

- Offline classification accuracy alone is not enough.
- A scorer can classify observed marker/force windows well but still give poor gradients through Foresight.
- Real rollout improvement has not been proven until server-side force traces and paired baseline/guided trials are collected.

## 4. Recommended Next Improvements

P0: Add a formal gradient-quality audit for each scorer.

- Evaluate whether score gradients are finite, non-zero, and bounded.
- Starting from noisy or slightly degraded actions, check whether one or a few guidance steps improve:
  - predicted force-band score
  - marker smoothness
  - contact stability
  - insertion good/bad margin
- Report both score improvement and action delta norm.

P1: Strengthen Foresight evaluation beyond latent MSE.

- Add metrics for predicted future:
  - marker magnitude trend
  - marker delta smoothness
  - contact/no-contact phase consistency
  - force-band proxy consistency
  - bad-to-good semantic margin preservation

P2: Make board scorer labels more physically explicit.

- Positive: stable contact, force magnitude in target band, low force/marker jerk.
- Negative categories:
  - too small: weak contact / insufficient wiping force
  - too large: excessive force
  - oscillate: unstable force / large high-frequency variation
- Continuous score:
  - combine force-band closeness and smoothness, not just binary class.

P3: Consider longer context for board wiping.

- Current DP uses `obs_horizon=2` and tactile history 8.
- Board wiping is a continuous-contact task; force drift and oscillation may require longer history.
- Suggested ablation after this run:
  - obs horizon: 2 vs 4
  - tactile history: 8 vs 16
  - Foresight horizon: 16 vs 32

P4: Keep the active 260617-only DP run, but choose checkpoints scientifically.

- Use `dp_best.pth` as the default test checkpoint.
- Keep `dp_epoch*.pth` for ablation if needed.
- Do not prefer `dp_latest.pth` unless validation improves or a real robot test specifically wants to compare late/overfit behavior.

## 5. Immediate Action Items

- Keep monitoring the current 2000-epoch run.
- Do not interrupt unless the process exits, NaN appears, checkpoint writing fails, or disk/GPU state becomes unsafe.
- After the run finishes, collect:
  - `metrics.json`
  - `loss_curve.png`
  - `loss_curve.csv`
  - `dp_best.pth`
  - `dp_final.pth`
  - selected `dp_epoch*.pth`
- For real tests, save server-side force traces separately for baseline and guided rollouts before claiming any physical improvement.
