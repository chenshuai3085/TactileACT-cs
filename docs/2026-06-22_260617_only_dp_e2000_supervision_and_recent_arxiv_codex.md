# 2026-06-22 260617-only Board DP E2000 Supervision And Recent ArXiv Review

## Scope

This note records the active 260617-only board-wiping Diffusion Policy run and a focused recent-paper review for improving the current project story:

```text
vision+tactile DP action prior
-> action-conditioned tactile/force Foresight
-> TacQuality consequence energy
-> bounded gradient guidance inside DP denoising
```

This is a training supervision and design note. It does not claim real robot improvement without paired real baseline/guided rollouts and server-side force traces.

## Active Training Run

- Dataset: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617`
- Run directory: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260621_codex`
- Launch script: `scripts/train/train_dp_tac_concat_board_260617_only_e2000_20260621_codex.sh`
- Policy script: `diffusion/train_dp_tac_concat.py`
- Variant: frozen TactileVAE concat DP
- TactileVAE: `/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt`
- Cameras: `global,wrist`
- Image mode: cached resized raw images at `/home/chenshuai/Project/output/cache/dp_board_rawimg200x266_fp16`
- Image resize/crop: `200x266 / 200x266`
- Tactile input: left hand marker history, `8` frames
- Horizons: `pred_horizon=16`, `obs_horizon=2`, `n_action_steps=8`
- Training length: `2000` epochs
- Batch size: `64`
- Learning rate / weight decay / warmup: `5e-5 / 1e-5 / 1000`
- Train cap: `max_steps_per_epoch=128`
- Validation: episode-level split, `val_ratio=0.1`, `val_interval=5`
- Checkpoints: `dp_best.pth` on validation improvement, `dp_latest.pth` every 10 epochs, `dp_epoch*.pth` every 50 epochs

Important interpretation: one epoch is capped at 128 minibatches, while the dataset reports 888 possible batches. Therefore `2000` epochs is long stochastic minibatch training, not 2000 full passes over every sliding window.

## Dataset Check

The training log confirms the run uses only:

```text
Dataset dirs: ['/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617']
```

Observed dataset details:

- Raw HDF5 files in the directory: `80`
- Train split: `72` requested episodes; `1` corrupted episode skipped during indexing
- Indexed train episodes/windows: `71 / 56821`
- Validation episodes/windows: `8 / 2048`
- TactileVAE norm stats loaded from checkpoint:
  - mean `[-0.3398614525794983, -2.9208483695983887]`
  - std `[1.9804853200912476, 2.767117738723755]`
- DP parameter count: about `3.15e8`
- Global condition dimension: `2350`

## Supervision Snapshot

Snapshot time: `2026-06-22 02:07 CST`

- Latest parsed epoch: `1034/2000`
- Latest train loss: `0.003077`
- Last validation point: epoch `1030`, val loss `0.037629`
- Current best validation: epoch `135`, val loss `0.012777`
- Latest saved periodic checkpoint confirmed: `dp_epoch1000.pth`
- GPU state: `14.7GB / 24.6GB`, utilization about `75%`, temperature about `60-62C`
- External disk: `/media/chenshuai/EXTERNAL_USB` has about `1.8T` free
- Root filesystem is tight: `/home` has about `41G` free, so large future outputs should stay on the external disk

Current judgment:

1. Training process is healthy: tmux session is alive, GPU is active, logs and checkpoints are updating.
2. The validation-best checkpoint is still early: `epoch135`.
3. Later train loss keeps decreasing, but validation has not improved. This is an overfit/validation-gap signal, not a training crash.
4. For deployment or real robot testing, the default candidate remains `dp_best.pth`, not `dp_latest.pth` or a later periodic checkpoint unless validation improves later.
5. The run should continue to `2000` epochs as requested; later checkpoints remain useful for analysis even if they are not recommended for deployment.

## Recent ArXiv Review

Time window: recent papers within roughly the last two months relative to `2026-06-22`. Priority was given to tactile/contact-rich manipulation, diffusion/flow robot policies, inference-time steering, and world/action models.

Verification note: titles, arXiv IDs, and submission dates below were checked with the arXiv API on `2026-06-22`.

### Directly Relevant To This Project

1. ViTaL: Inference-time Policy Steering via Vision and Touch, arXiv `2606.14981`
   - Link: https://arxiv.org/abs/2606.14981
   - Submitted: `2026-06-12`
   - Main point: inference-time steering is framed as visual high-level verification plus tactile low-level diffusion editing.
   - Project implication: strongly supports our direction of steering a pretrained/generative policy during deployment using tactile/contact feedback. Our sharper distinction should be that TacQuality is differentiable and force/tactile-consequence based, so it can guide DP denoising rather than only select samples.

2. Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation, arXiv `2606.08737`
   - Link: https://arxiv.org/abs/2606.08737
   - Submitted: `2026-06-07`
   - Main point: action generation is guided by predicted future visual and tactile dynamics; vision-only world action models are weak in contact-rich manipulation.
   - Project implication: validates our Foresight module. The next architectural improvement should be stronger contact-aware fusion/gating, not just concatenating tactile features.

3. ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation, arXiv `2606.13877`
   - Link: https://arxiv.org/abs/2606.13877
   - Submitted: `2026-06-11`
   - Main point: contact-rich planning depends on multimodal world models and structured temporal/spatial contact representations.
   - Project implication: future scorer should preserve marker-field structure and multi-step temporal consistency instead of relying only on flattened final-frame latent classification.

4. TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation, arXiv `2606.11184`
   - Link: https://arxiv.org/abs/2606.11184
   - Submitted: `2026-06-09`
   - Main point: force and tactile signals play asymmetric roles in contact dynamics: force gives global load changes, tactile gives local contact geometry.
   - Project implication: our board scorer should keep force-band and smoothness terms explicit; the novelty should be action-conditioned force/tactile consequence energy for DP guidance, not simply tactile prediction.

5. Tube Diffusion Policy: Reactive Visual-Tactile Policy Learning for Contact-rich Manipulation, arXiv `2604.23609`
   - Link: https://arxiv.org/abs/2604.23609
   - Submitted: `2026-04-26`
   - Main point: fixed action chunks are brittle in contact-rich settings; action-tube feedback lets execution adjust around a nominal trajectory.
   - Project implication: TacQuality guidance can be described as an inference-time correction around the DP nominal chunk. This is conceptually aligned with action-tube feedback, but our correction is produced by differentiating a predicted consequence score.

6. SI-Diff: Search and High-Precision Insertion with a Force-Domain Diffusion Policy, arXiv `2605.12247`
   - Link: https://arxiv.org/abs/2605.12247
   - Submitted: `2026-05-12`
   - Main point: insertion benefits from force-domain policies and mode conditioning for search vs insertion.
   - Project implication: for peg-in-hole, keep explicit phase/mode labels such as approach/search, stable insertion, pre-bounce risk, impact/recovery. A single good/bad score should be backed by phase-aware labels.

### Useful For Guidance And Evaluation

7. QPILOTS: Efficient Test-Time Q-Steering for Flow Policies, arXiv `2606.14801`
   - Link: https://arxiv.org/abs/2606.14801
   - Submitted: `2026-06-11`
   - Main point: critic-style test-time steering can guide flow/diffusion policy samples using gradients.
   - Project implication: supports our design choice to apply guidance to the estimated clean action `x0` inside denoising, with bounded update norms.

8. TapSampling: Inference-Time Sampling with a Task-Progress-Understanding Verifier for Robotic Manipulation, arXiv `2605.25547`
   - Link: https://arxiv.org/abs/2605.25547
   - Submitted: `2026-05-25`
   - Main point: policy-agnostic inference-time sampling plus verifier can improve actions without policy finetuning.
   - Project implication: useful comparison baseline, but our target remains gradient guidance rather than reranking/sampling.

9. VERITAS: Visual Verification Enables Inference-time Steering and Autonomous Policy Improvement, arXiv `2606.18247`
   - Link: https://arxiv.org/abs/2606.18247
   - Submitted: `2026-06-16`
   - Main point: generator-verifier policy steering and self-improvement with verified rollouts.
   - Project implication: after real robot trials, successful guided rollouts could be fed back as data for a next DP finetune. This should be a future stage, not claimed now.

10. LAGO Policy: Latency-Aware Asynchronous Diffusion Policies, arXiv `2606.17982`
    - Link: https://arxiv.org/abs/2606.17982
    - Submitted: `2026-06-16`
    - Main point: deployment needs smooth asynchronous policy execution under latency.
    - Project implication: force/action jerk and inter-chunk continuity should be reported for board-wiping real tests, because guidance must not create high-frequency action artifacts.

11. Frequency-Aware Flow Matching for Continuous and Consistent Robotic Action Generation, arXiv `2606.20135`
    - Link: https://arxiv.org/abs/2606.20135
    - Submitted: `2026-06-18`
    - Main point: frequency-domain treatment can improve temporal consistency and suppress high-frequency errors.
    - Project implication: later policy training could add action-smoothness/frequency-domain evaluation. For the current run, use it mainly as an evaluation idea: action jerk, force derivative, force jerk.

12. Training and Evaluating Diffusion Policies with Long Context Lengths, arXiv `2606.16447`
    - Link: https://arxiv.org/abs/2606.16447
    - Submitted: `2026-06-15`
    - Main point: short observation histories can fail on memory-dependent manipulation; long context needs careful conditioning/backbones.
    - Project implication: after the current 260617-only baseline, compare `obs_horizon=2` with longer contact history for board wiping.

## Recommended Architecture Story

Do not present the project as "we add tactile input to DP." That story is too weak because many recent papers already use visual-tactile policy learning.

The stronger story is:

```text
DP policy = action prior
Foresight = action-conditioned tactile/force consequence predictor
TacQuality = task-grounded differentiable consequence energy
Guidance = bounded gradient update inside DP denoising on predicted clean action x0
```

For board wiping:

- Positive quality means stable contact, force in a desired band, and smooth force evolution during wiping.
- Negative modes should remain explicit: too small force/contact loss, too large force, oscillatory/unstable contact.
- The score should be evaluated by both label separation and real force-curve improvement.

For insertion:

- Positive quality means stable insertion/contact trajectory.
- Negative modes should remain explicit: pre-bounce risk, impact/bounce, recovery/failed contact.
- The score should remain phase-aware so approach is not incorrectly treated as a bad tactile state.

## Practical Next Improvements

P0: finish and select checkpoint scientifically.

- Let the current 2000-epoch run finish unless it crashes or exhausts disk.
- Keep using `dp_best.pth` as the deployment candidate unless a later validation checkpoint improves.
- Keep later periodic checkpoints for post-hoc analysis only.

P0: collect real paired evidence.

- Board baseline vs force-aware guided should use the same DP checkpoint.
- Server should save a force trace for each trajectory.
- Evaluation should report force-band occupancy, contact retention, mean/std force, force derivative/jerk, and task completion.

P1: strengthen guidance as consequence optimization.

- Keep denoising-step guidance on clean-action estimate `x0`.
- Keep trust-region bounds and accept-only update checks.
- Add contact-gated guidance so the scorer is weak before contact and strong during wiping/contact.

P1: improve scorer beyond classification-only.

- Required scorer checks:
  - episode-level held-out label/quality separation
  - finite nonzero action gradients
  - bounded action deltas
  - real-window serving audit
  - paired real force-curve improvement

P2: later architecture experiments.

- Compare `obs_horizon=2` vs longer observation history.
- Compare flattened tactile latent score vs structured marker/force multi-step score.
- Add phase/contact-conditioned fusion instead of plain concatenation.
- Add action/force smoothness metrics; consider frequency-domain regularization only after the baseline is stable.

## Evidence Boundary

Can say now:

- The 260617-only DP training is running correctly.
- Data path is the intended single 260617 board dataset.
- Checkpointing and validation are functioning.
- Current best validation checkpoint is `epoch135`.
- Recent literature supports inference-time tactile/force consequence guidance as a timely direction.

Cannot say yet:

- The 260617-only DP improves real wiping.
- The guided policy improves force curves on the real robot.
- Later epochs are better than `dp_best.pth`.
- Offline scorer quality alone proves deployment improvement.
