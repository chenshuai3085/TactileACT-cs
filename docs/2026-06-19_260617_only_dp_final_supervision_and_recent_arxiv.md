# 2026-06-19 260617-only DP Final Supervision and Recent ArXiv Review

## 1. Scope

This note records the requested 260617-only board-wiping DP training supervision and the recent-paper review around the current project direction:

```text
Diffusion Policy action prior
  -> action-conditioned tactile foresight
  -> tactile/force quality energy
  -> bounded gradient guidance during inference
```

The target is gradient guidance for generating actions with better predicted tactile consequences, not candidate reranking as the final mechanism.

## 2. Dataset Check

Requested data root:

```text
/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban
```

Actual training directory used:

```text
/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617
```

Dataset audit:

| item | value |
|---|---:|
| HDF5 files | 80 |
| readable episodes | 79 |
| unreadable/incomplete episodes | 1 (`episode_1.hdf5`, missing required low-dimensional/tactile/action key) |
| frame length min / mean / max | 640 / 816.76 / 957 |
| image keys | `observations/images/global`, `observations/images/wrist` |
| image raw shape | `(T, 200, 266, 3)` uint8 |
| proprio key | `observations/proprio_joint`, `(T, 7)` float32 |
| action key | `actions/joint_abs`, `(T, 7)` float32 |
| tactile key | `observations/tac/left/marker_offset`, `(T, 9, 9, 2)` float32 |

The trainer skipped the incomplete episode and trained on the readable episodes.

## 3. Training Configuration

Training script:

```text
diffusion/train_dp_tac_concat.py
```

Run directory:

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_full_noearly_tmux
```

Core configuration:

| field | value |
|---|---|
| cameras | `global,wrist` |
| image resize/crop | `200,266` / `200,266` |
| tactile side | left |
| tactile history | 8 |
| TactileVAE checkpoint | `/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt` |
| TactileVAE latent | 144 dim (`16 x 3 x 3`) |
| obs horizon | 2 |
| pred horizon | 16 |
| action horizon | 8 |
| action dim | 7 |
| batch size | 64 |
| requested epochs | 2000 |
| learning rate | `1e-4` |
| weight decay | `1e-6` |
| diffusion train/inference steps | 100 / 100 |
| U-Net down dims | `512,1024,2048` |
| EMA | enabled |
| train/val episodes | 72 / 8 before incomplete-episode skipping |
| train/val windows | 8192 / 1024 |
| image cache | `/home/chenshuai/Project/output/cache/dp_board_rawimg200x266_fp16` |

The exact command is saved in:

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_full_noearly_tmux/run_command.txt
```

## 4. Final Training Result

The run was supervised and stopped after epoch 202 because the validation curve showed a strong plateau/overfit pattern:

| metric | value |
|---|---:|
| requested epochs | 2000 |
| stopped after epoch | 202 |
| best epoch | 94 |
| best validation loss | 0.011385 |
| train loss at best | 0.009777 |
| final train loss | 0.006962 |
| final validation loss | 0.019671 |
| final val / best val | 1.7278 |
| epochs since best | 108 |
| tail-20 train mean | 0.007089 |
| tail-20 val mean | 0.017970 |
| tail-50 train mean | 0.007414 |
| tail-50 val mean | 0.017147 |

Interpretation:

- Training itself was healthy: loss dropped rapidly and checkpoints were saved.
- After epoch 94, the training loss kept decreasing but validation loss rose and stayed above the best point.
- Continuing to epoch 2000 would mainly fit the training windows more tightly and is unlikely to improve generalization.
- The correct checkpoint for deployment/offline evaluation is `dp_best.pth`, not `dp_latest.pth`.

Recommended checkpoint:

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_full_noearly_tmux/dp_best.pth
```

Other saved artifacts:

```text
loss_curve.csv
loss_curve.png
early_stop_summary.json
early_stop_summary.md
training_status_latest.json
dp_epoch50.pth
dp_epoch100.pth
dp_epoch150.pth
dp_epoch200.pth
dp_latest.pth
dp_topk_ep194_loss0.0066.pth
dp_topk_ep200_loss0.0068.pth
dp_topk_ep201_loss0.0067.pth
```

Disk usage:

| path | size |
|---|---:|
| run directory | 26G |
| shared image cache | 156G |

Boundary:

This is an offline DP training/validation conclusion. It does not prove real board-wiping improvement. Real claims still require matched baseline/guided rollouts with server-side force traces and wiping outcome checks.

## 5. Recent ArXiv Review

Time window: recent work around 2026-04 to 2026-06, prioritizing tactile, force, contact-rich manipulation, diffusion/flow policy guidance, and suboptimal data usage.

### 5.1 ViTaL: Inference-time Policy Steering via Vision and Touch

- Link: https://arxiv.org/abs/2606.14981
- Date: 2026-06-12

Main point:

ViTaL adapts pretrained generative robot policies at deployment time. It separates long-horizon visual mode selection from short-horizon tactile-guided diffusion editing, and scores predicted tactile futures in latent space.

Project implication:

This is the closest external support for our current story. The useful framing is not "tactile concat improves BC" but "predicted tactile consequences steer the action generation process at inference time." Our `Foresight -> TacQualityEnergy -> bounded denoising guidance` route is aligned with this direction.

### 5.2 TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation

- Link: https://arxiv.org/abs/2606.11184
- Date: 2026-06-09

Main point:

TacForeSight predicts short-horizon tactile latent dynamics conditioned on tactile observations and high-frequency wrist force/torque, then uses those predicted latents as anticipatory contact priors for policy execution.

Project implication:

For board wiping, force should not be treated only as a post-hoc evaluation signal. Once reliable synchronized force traces are available, force history should enter either Foresight conditioning or the TacQuality scorer. This supports upgrading from marker-only future prediction to force-conditioned tactile future prediction.

### 5.3 Dream-Tac: A Unified Tactile World Action Model

- Link: https://arxiv.org/abs/2606.08737
- Date: 2026-06-07

Main point:

Dream-Tac jointly models actions, future visual observations, and tactile dynamics in a unified world-action model. It uses contact-gated visuotactile fusion and contact-aware attention.

Project implication:

Our current architecture is a modular/cascaded version of a tactile world-action model: DP proposes actions, Foresight predicts future tactile consequence, TacQuality scores that future. A future stronger version could move toward joint action/future-tactile training, but the modular version is easier to debug and deploy now.

### 5.4 ContactWorld: What Matters in Vision-Tactile World Models

- Link: https://arxiv.org/abs/2606.13877
- Date: 2026-06-11

Main point:

ContactWorld argues that contact-rich world models need spatially structured and temporally continuous representations. Tactile helps most when cross-modal representations are compatible, not simply because another modality was added.

Project implication:

The marker field and TactileVAE latent should preserve contact structure and temporal continuity. Evaluation should not stop at one-step latent MSE. We should evaluate contact phase, force-band occupancy, smoothness, future trend, and full-episode behavior.

### 5.5 ForceFlow: Learning to Feel and Act via Contact-Driven Flow Matching

- Link: https://arxiv.org/abs/2605.11048
- Date: 2026-05-11

Main point:

ForceFlow treats force as a global regulatory signal in a flow-matching policy and decomposes manipulation into vision-dominant approach and touch/force-dominant contact execution.

Project implication:

For board wiping, this supports phase decomposition:

- approach: vision/proprio/action prior dominates;
- contact/wiping: force/tactile quality dominates;
- exit/reset: guidance should weaken or disable.

It also suggests a future flow-policy backbone if DP sampling latency becomes a bottleneck.

### 5.6 AT-VLA: Adaptive Tactile Injection

- Link: https://arxiv.org/abs/2605.07308
- Date: 2026-05-08

Main point:

AT-VLA injects tactile information adaptively, only when it contributes to contact-rich action generation, and uses a fast tactile stream for rapid feedback.

Project implication:

TacQuality guidance should be contact-gated. Do not apply the same tactile pressure/smoothness gradient during non-contact approach frames. This is especially important for board wiping: pressure quality is meaningful only during wiping contact.

### 5.7 Multi-Resolution Tactile Imitation Learning

- Link: https://arxiv.org/abs/2606.06281
- Date: 2026-06-04

Main point:

Multi-resolution tactile features help contact-rich imitation learning, especially when tactile streams capture fast contact changes.

Project implication:

Our current marker history contains temporal information, but the scorer should explicitly expose short-window temporal statistics:

- marker magnitude mean/std;
- marker velocity/acceleration;
- force proxy derivative;
- contact dropout ratio;
- smoothness over the contact phase.

### 5.8 Ambient Diffusion Policy

- Link: https://arxiv.org/abs/2606.12365
- Date: 2026-06-10

Main point:

Ambient Diffusion Policy studies imitation learning from suboptimal data and argues that naively mixing lower-quality data into BC can make the policy learn harmful features. It uses noise-dependent data usage to extract useful structure without copying bad local behavior.

Project implication:

For blackboard data, force-too-small, force-too-large, and oscillatory data should not simply be treated as expert BC. They are most useful for learning TacQualityEnergy / quality-weighted policy training / bad-mode rejection. If they are mixed into DP training, a quality-aware weighting or diffusion-time-dependent usage strategy should be considered.

### 5.9 TouchGuide v6: Inference-Time Steering via Touch Guidance

- Link: https://arxiv.org/abs/2601.20239
- Updated: 2026-05

Main point:

TouchGuide steers visuomotor policies through touch guidance during denoising or flow matching. The useful point for this project is the action-space / denoising-space steering formulation: the tactile model is not only an observation encoder, but a guidance signal applied during generative action sampling.

Project implication:

This supports the current decision to avoid treating TacQualityEnergy as only a reranker. The scorer should expose gradients with respect to the denoised action chunk, and the update must be bounded by a trust region around the DP prior.

### 5.10 Progress-Guided Diffusion Policy

- Link: https://arxiv.org/abs/2603.27670
- Date: 2026-03, slightly outside the two-month window but directly relevant to guidance design

Main point:

Progress-Guided Diffusion Policy uses an estimator/classifier-style guidance term in latent action space to steer diffusion toward higher task progress.

Project implication:

This is a close analogue of using a learned quality/progress estimator as guidance. For board wiping, the progress signal should not be only geometric progress along the wipe path; it should combine contact validity, force-in-band occupancy, and smooth force evolution.

### 5.11 PPGuide: Performance Predictive Guidance

- Link: https://arxiv.org/abs/2603.10980
- Date: 2026-03, slightly outside the two-month window but directly relevant to classifier-guided DP

Main point:

PPGuide trains a performance predictor/classifier and uses its gradient during DP denoising to move action chunks away from failure modes.

Project implication:

This is the cleanest external precedent for our TacQualityEnergy direction. Our novelty should be that the predictor is contact-consequence aware:

```text
candidate action chunk
  -> predicted tactile/force consequence
  -> task-specific quality energy
  -> bounded gradient update during denoising
```

For insertion, the target is low bounce risk / high good-margin. For board wiping, the target is force band + smoothness during contact.

## 6. Current Architecture Assessment

Current stack:

```text
RGB global/wrist + proprio + left tactile marker history
  -> frozen board TactileVAE latent
  -> DP concat policy action chunk
  -> optional Foresight predicted tactile future
  -> TacQualityEnergy / good-margin score
  -> trust-region gradient update
```

Strong parts:

- Tactile is not only a passive input; it is also used as a predicted outcome for scoring.
- The energy/scorer is differentiable, so it can support classifier/critic guidance.
- Trust-region and accept-only logic are necessary to keep action edits near the DP action prior.
- Board and insertion can share a common story: different task-specific quality energies on predicted future contact.

Weak parts:

- The 260617-only DP run is small: only 79 readable episodes. Overfit after epoch 94 is expected.
- Validation loss is still DDPM noise prediction MSE; it is not equivalent to real wiping success.
- The current board Foresight/scorer stack still needs real server-side force-trace validation.
- Current DP `obs_horizon=2` may be short for continuous contact tasks with slow force drift.

## 7. Recommended Improvements

### P0. Real rollout force-trace evaluation

Before claiming physical improvement, run paired baseline/guided trials and save server-side force traces per trajectory.

Metrics:

- mean contact force during wiping;
- force-band occupancy ratio;
- force jerk / smoothness;
- contact dropout ratio;
- safety violations;
- wipe coverage or task completion if measurable.

### P1. Contact-gated TacQuality guidance

Enable quality gradients only during predicted or observed contact/wiping phase. Use marker magnitude/contact area/force proxy thresholds as a soft gate.

This follows ViTaL / ForceFlow / AT-VLA style separation between global motion and local contact regulation.

### P2. Force-conditioned Foresight

When synchronized force data is reliable, add force/torque history as Foresight input and predict both tactile latent and force-band proxy.

This directly matches the board task definition: good wiping means force magnitude in range and force changes smoothly.

### P3. Quality-aware use of negative data

Keep negative board data for scorer training and guidance evaluation. Avoid treating all negative samples as equal BC expert data.

Possible next route:

```text
positive/high-quality data -> main DP BC prior
all quality-labeled data -> TacQualityEnergy
mixed data -> optional quality-weighted or diffusion-time-aware policy training
```

### P4. Longer context ablation

For board wiping, compare:

- `obs_horizon`: 2 vs 4 vs 8;
- tactile history: 8 vs 16;
- Foresight horizon: 8 vs 16 vs 32;
- force/tactile temporal features with and without smoothness heads.

## 8. Final Status

The requested 260617-only DP training was completed to the point where further training was not useful. Best checkpoint selection is clear:

```text
use dp_best.pth @ epoch 94
do not use dp_latest.pth for normal deployment
```

The recent literature strongly supports the current project story:

```text
predict future contact consequence,
score contact quality,
use bounded gradient guidance to edit generated actions.
```

The main next proof is not another lower validation loss. The main next proof is real paired baseline/guided board-wiping rollouts with saved force curves.

## 9. Follow-up Stable 2000-Epoch Run Started

After the first 260617-only run showed a clear validation-loss rebound after epoch 94, a second 260617-only run was started with a more conservative training recipe. The motivation is to honor the 2000-epoch training request while reducing the risk that the model repeatedly memorizes the same fixed training-window subset.

Training script added:

```text
scripts/train/train_dp_tac_concat_board_260617_only_stable_e2000.sh
```

Run directory:

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr
```

Main differences from the previous `full_noearly_tmux` run:

| item | previous run | stable follow-up run |
|---|---:|---:|
| max epochs | 2000 | 2000 |
| learning rate | `1e-4` | `5e-5` |
| weight decay | `1e-6` | `1e-5` |
| warmup steps | 500 | 1000 |
| train-window handling | fixed `max_train_windows=8192` subset | full train window index, capped by `max_steps_per_epoch=128` |
| val windows | 1024 | 2048 |
| val interval | every epoch | every 5 epochs |
| checkpoint frequency | 50 epoch | 50 epoch |
| seed | 1 | 2 |

The stable run still uses the same deployment-compatible model interface:

```text
RGB(global,wrist) + proprio_joint + left marker history
  -> frozen board TactileVAE latent
  -> concat DP action chunk
```

Initial startup check:

| item | value |
|---|---:|
| HDF5 files | 80 |
| train split | 72 episode entries |
| val split | 8 episode entries |
| train readable episodes | 71 |
| skipped train episodes | 1 |
| train windows indexed | 56,989 |
| val windows | 2,048 |
| GPU | RTX 4090 |
| initial GPU memory | about 14.7 GB |

Artifacts to inspect:

```text
train.log
training_status_latest.json
monitor_training.log
loss_curve.csv
loss_curve.png
dp_best.pth
dp_latest.pth
dp_epoch*.pth
```

Interpretation rule:

- `dp_best.pth` is selected by episode-level validation loss and is the only checkpoint that should be considered for deployment/offline comparison by default.
- `dp_latest.pth` is only a training continuation artifact.
- If the stable run again shows train loss decreasing while val loss rises for a long tail, the conclusion is not "train longer"; the conclusion is that the 260617-only dataset is small and should rely on `dp_best.pth`, more data, stronger regularization, or quality-aware training.

Early stable-run checkpoint:

| epoch | train loss | val loss | status |
|---:|---:|---:|---|
| 1 | 0.819457 | 0.398170 | initial best |
| 5 | 0.077446 | 0.067853 | improved |
| 10 | 0.041592 | 0.041677 | improved |
| 15 | 0.026498 | 0.023322 | improved |
| 20 | 0.021453 | 0.022300 | current best at the time of this note |

Current interpretation at epoch 20:

- The conservative stable run is still improving on validation.
- It has not yet repeated the earlier overfit pattern.
- Continue monitoring later validation points; do not switch deployment commands until there is enough evidence that this stable run beats the previous `full_noearly_tmux/dp_best.pth` checkpoint in downstream/offline or real rollout evaluation.
