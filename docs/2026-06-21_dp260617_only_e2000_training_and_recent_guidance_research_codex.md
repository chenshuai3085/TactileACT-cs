# 2026-06-21 260617-only DP Training and Recent Guidance Research

## Current Training Run

Purpose: train a tactile DP concat policy only on the 2026-06-17 board-wiping dataset.

Dataset:

```text
/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617
```

Run script:

```text
scripts/train/train_dp_tac_concat_board_260617_only_e2000_20260621_codex.sh
```

Output directory:

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260621_codex
```

Main settings:

```text
policy script       diffusion/train_dp_tac_concat.py
epochs              2000
batch size          64
lr                  5e-5
weight decay        1e-5
pred horizon        16
obs horizon         2
action steps        8
tactile side        left
tactile history     8
image size          raw-compatible 200x266
image cache         /home/chenshuai/Project/output/cache/dp_board_rawimg200x266_fp16
save_freq           50
latest_freq         10
val_interval        5
max_steps_per_epoch 128
max_val_windows     2048
```

Frozen tactile encoder:

```text
/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt
```

Data indexing from `train.log`:

```text
total hdf5 episodes     80
train split             72 episode ids
val split               8 episode ids
skipped train files      1 corrupted/missing required keys
train used episodes      71
train windows            56821
val used episodes        8
val windows              2048
```

The skipped file was expected from dataset inspection: `episode_1.hdf5` is missing required keys for this loader.

## Training Status at 2026-06-21 16:44 CST

Process state:

```text
tmux train session      dp260617_codex_e2000
tmux monitor session    dp260617_codex_monitor
GPU                     RTX 4090
GPU memory              about 14.7 GB / 24.6 GB
GPU utilization          about 70%
```

Latest observed metric:

```text
latest epoch            111 / 2000
latest train loss        0.009971
best val epoch           110
best val loss            0.013394
```

Recent validation trajectory:

```text
epoch 75   val 0.013718
epoch 80   val 0.013495
epoch 85   val 0.014665
epoch 90   val 0.015150
epoch 105  val 0.013418
epoch 110  val 0.013394
```

Interpretation:

The short rise around epochs 85-90 was not enough to call overfitting.  The run refreshed the validation best at epoch 105 and again at epoch 110.  Continue training and keep selecting by `dp_best.pth`, not latest epoch.

Storage:

```text
run dir size             about 18 GB at epoch 111
external disk free       about 1.9 TB
home/root free           about 41 GB
image cache size         about 156 GB under /home/chenshuai/Project/output/cache
```

Because `dp_best.pth` is about 2.6 GB and `dp_latest.pth` stores optimizer state, the current external-disk output location is appropriate.

## Recent Research Checked

Focus: recent work that informs tactile DP, contact-rich world models, and inference-time guidance.  I prioritized papers from roughly April-June 2026 and work directly related to the current project.

Relevant papers:

| work | date | link | relevant point for this project |
|---|---:|---|
| TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation | 2026-06-09 | https://arxiv.org/abs/2606.11184 | Separates global force and local tactile roles; supports using force-aware foresight rather than only marker classification. |
| Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation | 2026-06-07 | https://arxiv.org/abs/2606.08737 | Jointly models action, future vision, and tactile dynamics; supports the action -> future tactile consequence -> score story. |
| ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation | 2026-06-11 | https://arxiv.org/abs/2606.13877 | Emphasizes which world-model representations matter for long-horizon contact-rich planning; supports evaluating foresight beyond one-step reconstruction. |
| Inference-time Policy Steering via Vision and Touch / ViTaL | 2026-06-12 | https://arxiv.org/abs/2606.14981 | Uses multimodal verification/steering at inference time for contact-rich manipulation; close to our DP prior plus tactile-quality guidance direction. |
| TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance | 2026-01, updated 2026 | https://arxiv.org/abs/2601.20239 | Strong baseline idea: steering pretrained diffusion/flow policy at inference time using touch, but not identical because our goal is explicit bad-outcome gradient guidance. |
| Learning When to See and When to Feel: Adaptive Vision-Torque Fusion | 2026-04-01 | https://arxiv.org/abs/2604.01414 | Supports task/phase-dependent weighting of vision vs force/tactile signals. |
| DREAM-Chunk: Reactive Action Chunking with Latent World Model | 2026-06-17 | https://arxiv.org/abs/2606.18589 | Supports using a lightweight latent world model to correct action chunks at test time. |
| MODIP: Efficient Model-Based Optimization for Diffusion Policies | 2026-06-09 | https://arxiv.org/abs/2606.10825 | Shows the broader trend of optimizing diffusion policies through learned models rather than plain behavior cloning only. |
| Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning | 2026-06-09 | https://arxiv.org/abs/2606.11087 | Supports test-time gradient guidance as a valid direction, though not tactile-specific. |

## Implications for the Current Architecture

The current project story should not be framed as only "add tactile features to DP".  The stronger story is:

```text
DP learns the action prior from demonstrations.
Foresight predicts short-horizon tactile/force consequences of candidate actions.
TacQuality scores predicted consequences with task-specific standards.
Trust-region gradient guidance updates the action chunk before execution.
```

This aligns with the most relevant recent work: tactile world models, inference-time steering, and model-based action optimization.

## Recommended Architecture Direction

Priority 1: keep the current 260617-only DP run going.

Reason: the run is healthy and validation best is still improving.  Stop only if validation loss plateaus for a long range or diverges while training loss continues to fall.

Priority 2: for board wiping, keep force-aware guidance as the scientific main branch.

Reason: board quality is physically defined by force magnitude, contact continuity, and force smoothness.  A marker-only classifier can be accurate on labels, but the more defensible guidance score is:

```text
score = good-vs-risk force-band margin
      + contact confidence
      - force-center penalty
      - force-smoothness penalty
```

The current `margin_only` score is a safe first deployment setting because it gives strong gradients without over-constraining secondary penalties.

Priority 3: evaluate policy checkpoints by paired real rollout traces, not only DP validation loss.

Needed metrics for board:

```text
force in target band
contact continuity
force smoothness / jerk
marker stability
task completion / coverage
server-side force_trace.csv per rollout
```

Priority 4: improve foresight evaluation from "future frame looks close" to "future quality ranking is preserved".

The critical metric for guidance is not just marker MSE.  It is whether:

```text
score(Foresight(obs, bad_action)) < score(Foresight(obs, good_action))
```

and whether the gradient step improves this score under a trust region.

Priority 5: consider phase-aware scoring after the current DP run.

The board task has approach and wiping phases.  The scorer should mainly constrain force/contact during wiping, while approach should avoid premature contact.  This can be implemented as a lightweight phase gate or contact gate, not as a heavy new policy architecture.

## Current Evidence Boundary

Do not claim real robot improvement yet.

Allowed claims:

```text
260617-only DP training is running normally.
Validation loss has improved to 0.013394 by epoch 110.
The current architecture is consistent with recent tactile world-model and inference-time steering literature.
Force-aware TacQuality guidance is the most scientifically aligned current board guidance branch.
```

Not yet allowed:

```text
Guided policy improves real board wiping.
Guided policy reduces bad force events on real robot.
The 260617-only checkpoint is better than previous full-data or positive-only checkpoints.
```

Those claims require paired real rollout force traces.
