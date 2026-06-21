# 2026-06-21 260617-only DP 2000 epoch training and recent guidance research

## Goal

User request:

```text
Use only /media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban
to train the board-wiping tactile DP for 2000 epochs.
Supervise the run and handle problems if they appear.
While training, survey very recent arXiv work from roughly the last two months
and identify improvements for the current architecture/story.
```

This note records the active training run and the research implications.  It
does not claim real-robot improvement; that still requires paired rollout force
traces.

## Active training run

Dataset:

```text
/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617
```

Training script:

```text
scripts/train/train_dp_tac_concat_board_260617_only_e2000_20260621_codex.sh
```

Output directory:

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260621_codex
```

Main model and data settings:

```text
policy script        diffusion/train_dp_tac_concat.py
variant              DP + frozen TactileVAE concat
dataset              260617_v8l_caheiban/peg_in_hole_0617 only
cameras              global,wrist
image size           raw-compatible 200x266
proprio              proprio_joint
action               actions/joint_abs
tactile side         left
tactile history      8 frames
pred horizon         16
obs horizon          2
action horizon       8
DP timesteps         100
epochs               2000
batch size           64
learning rate        5e-5
weight decay         1e-5
EMA                  enabled
validation split     episode-level 10 percent
validation interval  every 5 epochs
save_freq            every 50 epochs
latest_freq          every 10 epochs
top-k train ckpt     3
seed                 4
```

Frozen tactile encoder:

```text
/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt
```

Loaded TactileVAE normalization stats:

```text
mean = [-0.3398614525794983, -2.9208483695983887]
std  = [1.9804853200912476, 2.767117738723755]
```

Indexed data from `train.log`:

```text
total HDF5 episodes      80
train split              72 episode ids
validation split          8 episode ids
skipped train episodes    1
train used episodes      71
train windows         56821
validation windows      2048
```

The skipped file is consistent with earlier dataset inspection: one HDF5 file is
missing required loader keys.

## Supervision status

Observed at 2026-06-21 17:21 CST:

```text
tmux train session        dp260617_codex_e2000
tmux monitor session      dp260617_codex_monitor
process pid              430211
GPU                      RTX 4090
GPU memory               about 14.7 GB / 24.6 GB
GPU utilization           about 60-90 percent
external disk free        about 1.9 TB
/home free                about 41 GB
```

Latest parsed training state:

```text
latest epoch              171 / 2000
latest train loss         0.008382
best validation epoch     135
best validation loss      0.012777
latest validation epoch   170
latest validation loss    0.014277
```

Recent validation rows:

| epoch | train loss | val loss | best val |
|---:|---:|---:|---:|
| 130 | 0.010022 | 0.014011 | 0.013394 |
| 135 | 0.009756 | **0.012777** | **0.012777** |
| 140 | 0.009615 | 0.014328 | 0.012777 |
| 145 | 0.009811 | 0.014824 | 0.012777 |
| 150 | 0.009115 | 0.013653 | 0.012777 |
| 155 | 0.008920 | 0.016271 | 0.012777 |
| 160 | 0.009029 | 0.013812 | 0.012777 |
| 165 | 0.008122 | 0.014474 | 0.012777 |
| 170 | 0.008713 | 0.014277 | 0.012777 |

Current interpretation:

1. The run is healthy and still progressing.
2. Training loss is continuing to decrease.
3. Validation loss has not refreshed the epoch-135 best for about 30 epochs, but
   the gap is not yet severe enough to stop or restart.
4. Final deployment should select `dp_best.pth`, not `dp_final.pth` or a
   train-loss top-k checkpoint, unless later validation proves otherwise.

Checkpoint state:

```text
dp_best.pth       updated at epoch 135, about 2.5 GB
dp_latest.pth     updated every 10 epochs, includes optimizer state, about 5.0 GB
dp_epoch50.pth    saved
dp_epoch100.pth   saved
dp_epoch150.pth   saved
```

The run directory is on the external disk, which is appropriate.  `/home` is
nearly full, so large future outputs should stay under `/media/chenshuai/EXTERNAL_USB`.

Live status files:

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260621_codex/train.log
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260621_codex/training_status_latest.json
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260621_codex/loss_curve.png
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260621_codex/loss_curve.csv
```

## Recent arXiv survey

Method: arXiv API and arXiv pages were checked on 2026-06-21.  I kept only
papers whose titles and arXiv ids were verifiable.

Highly relevant papers:

| paper | arXiv date | link | direct implication |
|---|---:|---|---|
| Inference-time Policy Steering via Vision and Touch | 2026-06-12 | https://arxiv.org/abs/2606.14981 | Closest to our deployment story: use visuo-tactile inference-time steering to refine generated action sequences. |
| ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation | 2026-06-11 | https://arxiv.org/abs/2606.13877 | Supports evaluating tactile world models by contact-relevant planning/quality metrics, not only reconstruction MSE. |
| TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation | 2026-06-09 | https://arxiv.org/abs/2606.11184 | Strong support for our force-aware board branch: global force and local tactile should be modeled jointly but not treated as identical. |
| Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation | 2026-06-07 | https://arxiv.org/abs/2606.08737 | Supports the action-to-future-touch consequence story; adds contact-gated visuotactile fusion as a useful future direction. |
| QPILOTS: Efficient Test-Time Q-Steering for Flow Policies | 2026-06-11 | https://arxiv.org/abs/2606.14801 | Supports steering at inference time through predicted clean actions instead of directly trusting noisy intermediate actions. |
| Set-Supervised Diffusion Policy: Learning Action-Chunking Diffusion through Corrections | 2026-06-01 | https://arxiv.org/abs/2606.01865 | Supports using negative/corrective action chunks, not only deleting bad data. |
| Frequency-Aware Flow Matching for Continuous and Consistent Robotic Action Generation | 2026-06-18 | https://arxiv.org/abs/2606.20135 | Relevant because board wiping needs temporally smooth, frequency-consistent actions. |
| Tube Diffusion Policy: Reactive Visual-Tactile Policy Learning for Contact-rich Manipulation | 2026-04-26 | https://arxiv.org/abs/2604.23609 | Points to a limitation of long action chunks in contact-rich tasks; motivates action-horizon/replan-frequency ablations. |
| TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance | 2026-01-28, updated 2026-05-13 | https://arxiv.org/abs/2601.20239 | Important reference for tactile steering; our branch differs by using explicit future tactile/force quality energy for gradient guidance. |

Related tactile representation works:

| paper | arXiv date | link | implication |
|---|---:|---|---|
| T-Rex: Tactile-Reactive Dexterous Manipulation | 2026-06-15 | https://arxiv.org/abs/2606.17055 | Tactile reactivity and high-frequency touch are central, but this is broader VLA/dexterous work. |
| Tac-DINO: Learning Vision-Tactile Features with Patch Alignment | 2026-06-10 | https://arxiv.org/abs/2606.12069 | Supports future tactile representation pretraining, especially local patch-level alignment. |
| TactSpace: Learning a Physics-enriched Shared Latent Space for Tactile Sim-to-Real Transfer | 2026-06-17 | https://arxiv.org/abs/2606.18959 | Supports a physics-aware tactile latent direction, useful if we later need cross-dataset generalization. |

## Architecture implications for this project

The strongest current project story is:

```text
DP learns a demonstration action prior.
Foresight predicts short-horizon tactile/force consequences of candidate actions.
TacQualityEnergy scores those predicted consequences against task quality rules.
Gradient guidance edits the denoising action trajectory under a trust region.
```

This is more defensible than saying only "we concatenate tactile features into
DP."  The tactile DP is the base policy; the novel part is the future-contact
quality guidance.

Recommended near-term architecture focus:

1. Keep the active 260617-only DP training running and choose by episode-level
   validation best.
2. For board wiping, keep `force_aware_guided` as the scientific main branch:
   the real quality criteria are force band, contact continuity, smoothness, and
   marker/contact stability.
3. Treat marker-only classifiers as useful diagnostics, not the primary board
   scorer.  Board wiping quality is force-defined.
4. Keep the guidance location inside the denoising loop on predicted clean
   action `x0`; this matches the recent test-time steering direction better
   than post-hoc reranking.
5. Evaluate the scorer by whether its gradient improves predicted future
   quality under a trust region, not only by binary classification accuracy.
6. Add phase/contact gating in evaluation and guidance: force/smoothness should
   dominate during wiping/contact, while approach should mainly avoid premature
   contact.
7. After this run finishes, compare `action_horizon=8` with shorter horizons
   such as 4 or 6, because contact-rich tactile tasks may need more reactive
   replanning.

## Evidence boundary

Allowed claims today:

```text
The 260617-only tactile DP training is running normally.
The active run reached epoch 171/2000 with best validation loss 0.012777 at epoch 135.
The project direction is well aligned with very recent tactile world-model and inference-time steering papers.
For board wiping, force-aware future-contact quality guidance is more scientifically appropriate than a marker-only classifier.
```

Not allowed yet:

```text
The guided policy improves real robot board wiping.
The new 260617-only DP checkpoint is better than previous real-robot checkpoints.
The final epoch checkpoint is best for deployment.
```

Those require paired real rollouts and saved force traces.
