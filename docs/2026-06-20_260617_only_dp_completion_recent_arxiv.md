# 2026-06-20 260617-only DP Completion and Recent ArXiv Notes

## 1. Task

User request:

- Ignore other work for now.
- Use only `/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban` to train board-wiping DP.
- Train for 2000 epochs.
- Supervise the run, handle problems directly, and check recent arXiv work from roughly the last two months for possible architecture/story improvements.

## 2. Data

Actual dataset directory used:

```text
/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617
```

Data inspection:

| item | value |
|---|---:|
| hdf5 files | 80 |
| usable episodes | 79 |
| incomplete episodes | 1 |
| incomplete file | `episode_1.hdf5` |
| original image shape | `(T, 200, 266, 3)` uint8 |
| cameras | `global,wrist` |
| proprio | `observations/proprio_joint`, `(T, 7)` |
| action | `actions/joint_abs`, `(T, 7)` |
| tactile | `observations/tac/left/marker_offset`, `(T, 9, 9, 2)` |

The training loader skipped the incomplete episode.

## 3. Training Run

Script:

```text
scripts/train/train_dp_tac_concat_board_260617_only_stable_e2000.sh
```

Run directory:

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr
```

Important configuration:

| item | value |
|---|---:|
| policy | DP tactile concat |
| tactile encoder | frozen board TactileVAE |
| TactileVAE checkpoint | `/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt` |
| tactile side/history | left / 8 |
| obs horizon | 2 |
| pred horizon | 16 |
| action horizon | 8 |
| batch size | 64 |
| epochs | 2000 |
| lr | `5e-5` |
| weight decay | `1e-5` |
| diffusion train/infer steps | 100 / 100 |
| U-Net dims | `512,1024,2048` |
| image size | `200x266` raw-size resize/crop |
| image cache | `/home/chenshuai/Project/output/cache/dp_board_rawimg200x266_fp16` |
| train/val split | episode-level, `val_ratio=0.1` |
| train episodes | 72 in config, 71 usable after skip |
| val episodes | 8 |
| max train steps per epoch | 128 |
| val interval | 5 epochs |
| save freq | 50 epochs |
| latest freq | 10 epochs |

The run was resumed from epoch 1500 to 2000 using `dp_latest.pth`. Final completion was verified from:

- `train_losses.npy`
- `val_losses.npy`
- `dp_latest.pth`
- `dp_best.pth`
- `dp_final.pth`
- `dp_epoch2000.pth`
- `train_resume_1500_to_2000.log`

## 4. Results

| metric | value |
|---|---:|
| final epoch | 2000 / 2000 |
| best val epoch | 155 |
| best val loss | `0.0116586018` |
| best train loss at epoch 155 | `0.0084222523` |
| final train loss | `0.0023153078` |
| final val loss | `0.0387085451` |
| min train loss | `0.0017991070` @ epoch 1989 |
| val entries | 401 |

Checkpoint selection:

```text
Use dp_best.pth for deployment/offline comparison.
Do not use dp_final.pth as the default conclusion checkpoint.
```

Reason:

- Train loss kept decreasing until very late training.
- Validation loss reached its best around epoch 155 and later rose to about `0.0387`.
- This is a clear train/val gap on this 260617-only dataset.
- The 2000-epoch run is complete and useful as an overfitting/long-training record, but the practical checkpoint is `dp_best.pth`.

Key artifacts:

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_best.pth
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_final.pth
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_epoch2000.pth
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/loss_curve_full_2000.png
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/loss_curve_full_2000.csv
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/training_status_latest.json
```

## 5. Recent ArXiv Notes

Search window: mainly 2026-04 to 2026-06. Source: arXiv API on 2026-06-20.

### Closest to Our Direction

| paper | date | source | relevance |
|---|---:|---|---|
| ViTaL: Inference-time Policy Steering via Vision and Touch | 2026-06-12 | [arXiv:2606.14981](https://arxiv.org/abs/2606.14981) | Very close to tactile verifier / inference-time steering. Supports our story of using predicted tactile consequences to guide a generative policy. |
| TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation | 2026-06-09 | [arXiv:2606.11184](https://arxiv.org/abs/2606.11184) | Strong support for force-conditioned tactile foresight. Suggests board wiping should eventually include force as model input, not only as evaluation. |
| ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation | 2026-06-11 | [arXiv:2606.13877](https://arxiv.org/abs/2606.13877) | Supports spatially structured and temporally continuous tactile representations for long-horizon contact. |
| Feedback World Model Enables Precise Guidance of Diffusion Policy | 2026-05-15 | [arXiv:2605.15705](https://arxiv.org/abs/2605.15705) | Supports closing the loop between predicted and observed consequences during diffusion policy guidance. |
| Fisher-Preserving Guidance | 2026-05-28 | [arXiv:2605.29937](https://arxiv.org/abs/2605.29937) | Supports trust-region / manifold-preserving guidance rather than unconstrained score-gradient steps. |
| POTR: Prior-Corrected Orthogonal Trust-Region Guidance | 2026-05-23 | [arXiv:2605.24433](https://arxiv.org/abs/2605.24433) | Relevant to action-chunk smoothness and trust-region correction during inference. |
| Action-Prior Denoising / Soft RTC | 2026-05-25 | [arXiv:2605.25537](https://arxiv.org/abs/2605.25537) | Relevant to action chunk overlap and smoother real-time chunking. |
| SI-Diff | 2026-05-12 | [arXiv:2605.12247](https://arxiv.org/abs/2605.12247) | Force-domain diffusion policy for insertion; supports using force/tactile quality modes in contact-rich tasks. |
| Tube Diffusion Policy | 2026-04-26 | [arXiv:2604.23609](https://arxiv.org/abs/2604.23609) | Highlights action chunking reactivity limits in contact-rich manipulation. |
| TactSpace | 2026-06-17 | [arXiv:2606.18959](https://arxiv.org/abs/2606.18959) | Supports physics-enriched tactile latent spaces and sim-to-real tactile representation alignment. |
| TaCauchy | 2026-06-18 | [arXiv:2606.20426](https://arxiv.org/abs/2606.20426) | Simulation-side support for physically grounded tactile/force supervision. |

The closest entries were re-checked against the arXiv API on 2026-06-20:

- [arXiv:2606.14981](https://arxiv.org/abs/2606.14981): published 2026-06-12.
- [arXiv:2606.11184](https://arxiv.org/abs/2606.11184): published 2026-06-09.
- [arXiv:2606.13877](https://arxiv.org/abs/2606.13877): published 2026-06-11.
- [arXiv:2605.15705](https://arxiv.org/abs/2605.15705): published 2026-05-15.

## 6. Architecture Implications

Current project story is still coherent:

```text
DP action prior
  -> Foresight predicts future tactile consequence
  -> TacQualityEnergy scores tactile consequence
  -> bounded gradient guidance edits action
```

Recommended improvements:

1. Keep DP concat as the behavior prior baseline, but do not present concat as the main novelty.
2. Main novelty should be consequence-aware tactile/force energy guidance.
3. Add force into board-wiping scorer/foresight when enough clean force curves are available:

```text
history image/proprio/tactile/action/force -> future tactile/force consequence -> quality energy
```

4. Use trust-region guidance as a first-class design constraint:

```text
score gradient + action delta clamp + smoothness clamp + optional manifold projection
```

5. For 260617-only DP, use `dp_best.pth` for real rollout. If retraining is needed, prefer stronger regularization / smaller model / more data mixing instead of simply increasing epochs.

## 7. Next Practical Checks

Before claiming policy improvement:

1. Run real board wiping with `dp_best.pth`.
2. Record server-side force curves per trajectory.
3. Compare baseline vs guided runs using the same ckpt and same episode protocol.
4. Report force mean/range/smoothness/contact-loss rate and task completion, not just visual impression.

## 8. 2026-06-20 Codex Verification Update

This section records the follow-up verification after the user asked to focus only on the 260617 dataset and continue supervision.

### Training state

Current process check showed no active DP training process. The 260617-only run has already finished 2000/2000 epochs.

The practical model remains:

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_best.pth
```

Do not use `dp_final.pth` as the default deployment checkpoint for this run. The final model is useful as a long-training artifact, but validation loss is much worse than the best checkpoint:

| epoch range | train trend | validation trend |
|---|---|---|
| 1-155 | train loss drops to `0.008422` | val loss reaches best `0.011659` |
| 156-600 | train keeps dropping | val starts rising |
| 601-1500 | train keeps dropping below `0.003` | val mostly `0.026-0.044` |
| 1501-2000 | train reaches min `0.001799` | final val `0.038709` |

Interpretation: on 260617-only data, the model memorizes the train windows after early training. The correct checkpoint selection is validation-based `dp_best.pth`.

Storage footprint at verification time:

```text
run dir: 126G
image cache: 156G
```

### Verified recent arXiv references

The following entries were checked through the arXiv API on 2026-06-20.

| arXiv | date | title | implication for this project |
|---|---:|---|---|
| [2606.14981](https://arxiv.org/abs/2606.14981) | 2026-06-12 | Inference-time Policy Steering via Vision and Touch | Closest support for tactile-conditioned inference-time steering. Our DP + Foresight + TacQuality gradient guidance story is aligned with this direction. |
| [2606.11184](https://arxiv.org/abs/2606.11184) | 2026-06-09 | TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation | Strong support for adding force-conditioned tactile foresight, especially for board wiping. |
| [2606.13877](https://arxiv.org/abs/2606.13877) | 2026-06-11 | ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation | Supports treating tactile prediction as a world-model problem, not just a classifier feature. |
| [2605.15705](https://arxiv.org/abs/2605.15705) | 2026-05-15 | Feedback World Model Enables Precise Guidance of Diffusion Policy | Suggests closing the loop between predicted consequence and newly observed tactile/force feedback during rollout. |
| [2605.29937](https://arxiv.org/abs/2605.29937) | 2026-05-28 | Fisher-Preserving Guidance | Supports bounded/manifold-preserving gradient guidance rather than unconstrained classifier gradients. |
| [2605.24433](https://arxiv.org/abs/2605.24433) | 2026-05-23 | Smoother Action Chunking Flow Policy via Prior-Corrected Orthogonal Trust-Region Guidance | Supports trust-region and smoothness constraints for action chunks. |
| [2605.25537](https://arxiv.org/abs/2605.25537) | 2026-05-25 | Action-Prior Denoising for Smooth Real-Time Chunking | Relevant to stable real-time chunk execution and action prior preservation. |
| [2605.12247](https://arxiv.org/abs/2605.12247) | 2026-05-12 | SI-Diff: Learning Search and High-Precision Insertion with a Force-Domain Diffusion Policy | Supports force-domain diffusion for contact-rich insertion, relevant to the insertion branch. |
| [2604.23609](https://arxiv.org/abs/2604.23609) | 2026-04-26 | Tube Diffusion Policy | Highlights chunk reactivity limits in contact-rich tactile policy learning. |
| [2604.15483](https://arxiv.org/abs/2604.15483) | 2026-04-16 | pi_0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities | Relevant as a CFG/steerable policy reference; supports presenting our guidance as a tactile-quality steering signal. |
| [2606.12365](https://arxiv.org/abs/2606.12365) | 2026-06-10 | Ambient Diffusion Policy | Relevant when mixing positive and imperfect board-wiping data rather than training only on clean demonstrations. |
| [2606.19998](https://arxiv.org/abs/2606.19998) | 2026-06-18 | Tri-Info: Generalizable, Interpretable Failure Prediction for VLA Models via Information Theory | Suggests adding rollout-level failure signatures beyond per-frame classifier scores. |
| [2606.20562](https://arxiv.org/abs/2606.20562) | 2026-06-18 | MemoryWAM | Supports using persistent memory/history for non-Markovian contact tasks. |
| [2606.20135](https://arxiv.org/abs/2606.20135) | 2026-06-18 | Frequency-Aware Flow Matching | Supports continuous/frequency-aware action generation to reduce chunk discontinuity. |
| [2606.18959](https://arxiv.org/abs/2606.18959) | 2026-06-17 | TactSpace | Supports physics-enriched tactile latent spaces and sim-to-real tactile representation alignment. |
| [2606.20426](https://arxiv.org/abs/2606.20426) | 2026-06-18 | TaCauchy | Simulation-side reference for physically grounded tactile/force supervision. |
| [2606.08737](https://arxiv.org/abs/2606.08737) | 2026-06-07 | Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation | Supports the "world action model + future tactile dynamics" framing. Our differentiator should be explicit tactile/force quality energy and gradient guidance. |
| [2606.01027](https://arxiv.org/abs/2606.01027) | 2026-05-31 | tau_0-WM: A Unified Video-Action World Model for Robotic Manipulation | Supports future consequence prediction plus action evaluation. It is broader and vision-heavy; our story should emphasize contact-rich tactile/force scoring. |
| [2605.21976](https://arxiv.org/abs/2605.21976) | 2026-05-21 | TacO: Benchmarking Tactile Sensors for Object Manipulation | Supports task-specific tactile evaluation instead of generic policy loss only. |
| [2606.06281](https://arxiv.org/abs/2606.06281) | 2026-06-04 | Multi-Resolution Tactile Imitation Learning for Contact-Rich Robotic Manipulation | Supports multi-timescale tactile features. For us this suggests combining short-window force smoothness with longer-window tactile consequence scores. |

### Architecture recommendation after the latest survey

The most defensible near-term architecture is still:

```text
DP action prior
  + differentiable Foresight consequence predictor
  + tactile/force TacQualityEnergy scorer
  + bounded trust-region gradient guidance during denoising or final action refinement
```

For the board task, the next meaningful improvement is not another 260617-only 2000-epoch DP run. The better research/engineering direction is:

1. Use `dp_best.pth` from the completed 260617-only run for real tests.
2. Keep score guidance separate from behavior cloning, so the policy prior stays close to demonstrations.
3. Add force-conditioned foresight/scoring once reliable server-side force traces are collected.
4. Add trust-region constraints to every guided action update: action delta clamp, smoothness clamp, and optionally prior/manifold preservation.
5. For future training, prefer mixed data or stronger regularization/early stopping over simply increasing epochs.

This keeps the story aligned with recent work while preserving the core novelty: tactile-consequence quality guidance for contact-rich DP.

### TacQuality guidance audit after verification

Command:

```bash
conda run --no-capture-output -n TactileACT python for_show_xiaomi/audit_tac_quality_guidance_state.py
```

Outputs:

```text
/home/chenshuai/Project/output/tac_quality_guidance_state_audit/tac_quality_guidance_state_audit.json
/home/chenshuai/Project/output/tac_quality_guidance_state_audit/tac_quality_guidance_state_audit.md
```

Audit result:

| item | value |
|---|---:|
| insertion scorer ready for real rollout | `true` |
| board scorer ready for real rollout | `true` |
| denoising-step serving path ready | `true` |
| real evidence pipeline ready | `true` |
| real rollout evidence complete | `false` |
| overall goal complete | `false` |

Interpretation:

- Current insertion and board scoring/guidance chains are ready to test on the robot.
- The serving path supports actual gradient guidance inside DP denoising or final action refinement; this is not reranking.
- The paired manifest/metadata pipeline is ready, with board and insertion each prepared as 3 baseline/guided pairs.
- No final performance claim should be made yet because the true robot baseline-vs-guided force/outcome evidence is still missing.

For the 260617-only DP branch:

```text
Use:
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_best.pth

Avoid as default:
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_final.pth
```
