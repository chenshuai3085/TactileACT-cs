# TactileVAE V2 Intensity/Ranking Supervision Ablation

Date: 2026-08-20

## Objective

Evaluate whether the intensity and pairwise-ranking losses in TactileVAE V2
provide useful structure beyond last-frame marker reconstruction. The study
compares the deployed V1 encoder, V2 without intensity/ranking supervision,
and Full V2 under matched board-wiping data and action-conditioned Foresight
contracts.

## Models and controlled variables

The two V2 models use the same architecture, data split, normalization,
optimizer, seed, and training duration. The ablated model changes only
`intensity_weight` and `rank_weight` from `0.1/0.05` to `0/0`.

- Data: 221 left-marker streams from the 260609/260610 board force-band data.
- Conditions: stable wiping, insufficient pressure, excessive pressure, and
  oscillatory contact.
- Tactile input: eight consecutive `9x9x2` marker-displacement frames.
- V2 training windows: 97,449, stride 2, seed 42.
- V2 training: 150 epochs, batch 512, AdamW, learning rate `1e-4`.
- Shared latent shape: `16x3x3` (144D).
- Reconstruction evaluation: the same 9,744 validation windows.
- Smoothness evaluation: statistics fitted on 198 training episodes and
  evaluated on 23 held-out episodes.
- Foresight condition: `actions/joint_abs[t:t+16]`.
- Foresight target: tactile latents/markers at `t+1:t+16`.
- Foresight evaluation: 20,055 exhaustive windows from the same 23 held-out
  episodes.

V1 and V2 latent losses are not numerically comparable because the latent
coordinate systems differ. Decoded marker-space metrics are therefore the
primary cross-model Foresight comparison.

## Unified results

| Metric | V1 | V2 without intensity/ranking | Full V2 |
|---|---:|---:|---:|
| Raw marker reconstruction MAE | 0.085213 | **0.046094** | 0.047801 |
| Marker reconstruction cosine | 0.998538 | **0.999586** | 0.999554 |
| Marker-activity pair-order accuracy | -- | 43.61% | **98.53%** |
| Tactile $F_z$ pair-order accuracy | -- | 33.96% | **78.63%** |
| Whitened latent step RMS | 0.085917 | 0.069044 | **0.065861** |
| Whitened latent curvature RMS | 0.139771 | 0.111036 | **0.104453** |
| 16-step Foresight raw-marker MAE | 0.211360 | 0.190987 | **0.185056** |
| 16-step Foresight raw-marker RMSE | 0.249144 | 0.219022 | **0.213316** |
| 16-step Foresight marker cosine | 0.992932 | 0.993642 | **0.994082** |
| Scorer preference accuracy | -- | 100% | 100% |

The ablated V2 has slightly better last-frame reconstruction than Full V2:
its raw marker MAE is 3.57% lower. This confirms that the semantic losses are
not merely reconstruction regularizers. Their effect appears in latent
organization and downstream prediction.

## Semantic ordering

Full V2 assigns the first latent channel an explicit contact-intensity role.
On the shared validation windows, its spatially averaged intensity channel
orders normalized marker activity correctly for 98.53% of sampled pairs and
orders measured tactile $F_z$ correctly for 78.63%. Removing the intensity and
ranking losses reduces these values below chance, to 43.61% and 33.96%.

This supports a **supervised intensity head** claim. It does not establish
strict intensity-pattern disentanglement: prior linear-probe analysis showed
that the remaining 135-D pattern code still retains substantial magnitude and
force information. The architecture and reconstruction objective permit this
information to remain distributed across channels.

## Latent temporal smoothness

Per-dimension whitening is fitted only on training episodes before temporal
derivatives are evaluated on held-out episodes. Relative to the ablated V2,
Full V2 reduces mean whitened step RMS from 0.069044 to 0.065861 (4.61%) and
mean whitened curvature RMS from 0.111036 to 0.104453 (5.93%). Relative to V1,
the reductions are 23.35% and 25.27%, respectively.

These are average smoothness improvements, not a uniform dominance statement:
the V1 and Full V2 95th-percentile derivative values are close. Temporal
attention is also strongly last-frame weighted. The mean weights for Full V2
are `[0.0063, 0.0170, 0.0948, 0.8818]`, versus
`[0.0062, 0.0166, 0.0842, 0.8930]` without supervision. The temporal module
therefore primarily selects the latest latent while retaining a small amount
of recent history.

## Action-conditioned Foresight

All three Foresight models are evaluated on identical held-out episodes and
all valid windows. Full V2 obtains raw-marker MAE 0.185056, improving by 12.44%
over V1 and 3.11% over V2 without intensity/ranking supervision. Full V2 has
the lowest raw-marker MAE at every future horizon from `t+1` through `t+16`.

| Horizon | V1 | V2 without intensity/ranking | Full V2 |
|---:|---:|---:|---:|
| 1 | 0.167192 | 0.139556 | **0.127541** |
| 4 | 0.189051 | 0.162202 | **0.155235** |
| 8 | 0.216212 | 0.193484 | **0.189383** |
| 12 | 0.226727 | 0.210969 | **0.206645** |
| 16 | 0.247281 | 0.238278 | **0.233290** |

The supervision therefore provides a measurable downstream benefit even
though it slightly worsens standalone reconstruction. This is the strongest
current evidence for retaining the intensity/ranking objectives.

## Scorer preference experiment

Separate tactile-only latent scorers were trained using identical splits for
Full V2 and the ablated V2. The validation set contains 2,963 windows from 20
positive and 25 negative episodes. Strict positive-greater-than-negative
comparison produces 2,130,072 pairs.

| Encoder | Correct / pairs | Preference accuracy | 95% episode-bootstrap CI | Mean margin gap |
|---|---:|---:|---:|---:|
| V2 without intensity/ranking | 2,130,072 / 2,130,072 | 100% | [100%, 100%] | 20.2428 |
| Full V2 | 2,130,072 / 2,130,072 | 100% | [100%, 100%] | 20.2696 |

This experiment is saturated and does not show a meaningful Full V2 advantage.
The labels correspond to separate collection-condition classes, so the scorer
can exploit differences that are easier than the final matched-rollout
preference problem. These values must not be presented as evidence that Full
V2 improves real-rollout preference ordering. A definitive comparison requires
independently selected positive/negative key-contact pairs from completed real
rollouts, matched by task and interaction phase.

## Conclusion and next decision

1. Full V2 strongly organizes the designated intensity channel and produces a
   modestly smoother latent trajectory.
2. Full V2 improves action-conditioned 16-step Foresight over both V1 and the
   supervision ablation, including every evaluated future horizon.
3. Reconstruction alone would select the wrong ablation: the unsupervised V2
   reconstructs slightly better but loses semantic ordering and downstream
   Foresight accuracy.
4. The current evidence supports using Full V2 and describing it as a
   supervised intensity-aware representation, not a strictly disentangled one.
5. The scorer experiment is inconclusive because both encoders saturate the
   collection-condition split.

The planned three-seed extension is not triggered yet. The predefined rule
required Full V2 to improve both Foresight error and scorer preference
accuracy; only the Foresight criterion improved. The next useful experiment is
to build a harder matched key-contact preference set before spending compute
on additional seeds.

## Artifacts

- Full V2 checkpoint:
  `/home/chenshuai/Project/output/tactile_vae_v2_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae_v2.pt`
- V2 without intensity/ranking checkpoint:
  `/home/chenshuai/Project/output/tactile_vae_v2_no_intensity_rank_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae_v2.pt`
- VAE ablation metrics and figure:
  `/home/chenshuai/Project/output/v2_intensity_rank_ablation_20260820/vae/`
- Three-model Foresight metrics and figure:
  `/home/chenshuai/Project/output/v2_intensity_rank_ablation_20260820/foresight/`
- Scorer checkpoints and evaluation:
  `/home/chenshuai/Project/output/v2_intensity_rank_scorer_ablation_20260820/`
- V1/Full V2 latent smoothness:
  `/home/chenshuai/Project/output/tactile_vae_v2_board_260609_260610_left_tw8_ld16_s2_e150/comparison/latent_smoothness.json`
