# 2026-06-21 Current Architecture and Recent Research Improvement Notes

## Current Project Position

The current system has three layers:

```text
1. DP action prior
   image + proprio + left tactile latent -> diffusion action chunk

2. Future contact predictor
   action chunk + current state/tactile -> future tactile consequence

3. TacQuality guidance
   predicted future contact -> quality score -> bounded gradient update on action
```

The active `260617-only` DP run is useful as the action prior / baseline.
It should not be framed as the main novelty by itself.  Recent work points to a
stronger story: future tactile/force consequences should be predicted and used
to guide diffusion-policy action generation at inference time.

## Relevant Recent Research Direction

Most relevant recent papers checked through arXiv API / arXiv pages:

| paper | arXiv | main implication |
|---|---|---|
| Inference-time Policy Steering via Vision and Touch | https://arxiv.org/abs/2606.14981 | supports tactile inference-time steering of generative policies |
| Dream-Tac | https://arxiv.org/abs/2606.08737 | supports action-conditioned future tactile/world dynamics |
| FAWAM | https://arxiv.org/abs/2606.08555 | force should be predicted and used for closed-loop correction, not only concatenated as observation |
| ContactWorld | https://arxiv.org/abs/2606.13877 | spatially structured and temporally continuous contact representations matter |
| Feedback World Model | https://arxiv.org/abs/2605.15705 | world-model guidance should be corrected by observed prediction error |
| PACT | https://arxiv.org/abs/2606.08414 | diffusion policies can be aligned with physical constraints via post-training / gradient-style updates |
| Fisher-Preserving Guidance | https://arxiv.org/abs/2605.29937 | guidance should be bounded/manifold-aware to avoid off-policy actions |
| Frequency-Aware Flow Matching | https://arxiv.org/abs/2606.20135 | action smoothness/frequency consistency is important for stable robot execution |

## What This Means for Our Architecture

The most defensible method story is:

```text
image/proprio/tactile DP prior
  -> sample action chunk
  -> multi-step tactile + force-aware Foresight predicts contact consequences
  -> TacQualityEnergy evaluates future contact quality
  -> bounded classifier/scorer gradient guidance edits the action
  -> server logs force/action/guidance traces for real rollout evaluation
```

This is not reranking. Reranking can remain a diagnostic baseline, but the core
method should be gradient guidance on the action sample.

## Highest-Value Improvements

1. Make board Foresight force-aware.

   Board quality is defined by force magnitude and force smoothness. Marker-only
   prediction is incomplete. The next useful predictor should output:

   ```text
   future marker latent / marker field
   future force proxy: |F|, Fz, delta |F|, delta Fz, torque magnitude, jerk proxy
   force-band logits: too_light / good / too_heavy / oscillatory
   contact gate logits
   ```

2. Use multi-horizon scoring.

   Board wiping is a trajectory-quality problem, not a single-frame problem.
   Score `t+1...t+16` by force band, contact continuity, temporal smoothness, and
   marker/force stability.

3. Keep contact-phase gating.

   Approach/reset should receive weak or zero wiping-quality guidance. Wiping
   contact frames should receive stronger force-band and smoothness guidance.

4. Keep trust-region / accept-only guidance.

   The serving code already has bounded updates and accept-only checks. This is
   important because unconstrained score maximization can push actions off the DP
   manifold.

5. Add prediction-error awareness after force-aware Foresight.

   If the predicted force/marker residual is high on recent real rollouts,
   guidance should be weakened. This matches the Feedback World Model direction.

## Current Evidence Boundary

What is supported now:

- The `260617-only` DP 2000-epoch run completed and checkpointed normally.
- The run strongly overfits after epoch 85 by held-out episode validation loss.
- The validation-selected rollout candidate remains:

  `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun/dp_best.pth`

- Server-side TacQuality guidance code supports final-action and denoising-step
  guidance, contact gating, and force-trace logging.
- Force-aware board Foresight training has started and reached the first formal
  checkpoint interval.  At epoch 25, both `foresight_force_best.ckpt` and
  `foresight_force_epoch_25.ckpt` loaded successfully, with:

  ```text
  val_total   0.4173
  force_loss  0.0368
  band_loss   0.0706
  contact     0.2069
  band_bacc   0.977
  contact_acc 0.908
  ```

  This is a training-monitoring result only; the guidance value still needs
  offline gradient audit and paired real rollout validation.

What is not proven yet:

- Real board wiping improvement from guidance.
- Force curve improvement versus baseline under paired real rollout conditions.
- That marker-only Foresight is enough for board force-quality guidance.

## Recommended Next Experiment After DP Run

The most useful next experiment is now running: force-aware Foresight.

```bash
cd /home/chenshuai/Project/TactileACT-cs
CONFIG=TFAC_V5/config_pretrain_foresight_board_forceaware_multistep16.json \
  scripts/train/train_foresight_board_forceaware_multistep16.sh
```

Success criteria should include:

- future marker/latent reconstruction error
- force proxy MAE
- force-band macro-F1 or balanced accuracy
- contact gate accuracy
- gradient audit: finite gradient rate, score delta, action delta norm,
  trust-region pass rate
- real rollout comparison: baseline vs guided force traces and task result

Only after paired real rollout evidence should we claim that the scorer/guidance
improves actual board wiping behavior.

## 04:13 Verified Recent Arxiv Notes

I re-checked the recent-paper list with the arXiv API to avoid recording
uncertain titles as evidence.  The most relevant confirmed papers from the last
two months are:

| paper | arXiv | date | direct implication for this project |
|---|---:|---:|---|
| TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance | 2601.20239 | 2026-01-28 | Not within the last two months, but it remains the closest prior for inference-time tactile steering. Our difference should be outcome/quality-aware contact scoring rather than pure touch-action compatibility. |
| SI-Diff: A Framework for Learning Search and High-Precision Insertion with a Force-Domain Diffusion Policy | 2605.12247 | 2026-05-12 | Insertion benefits from force-domain modeling; supports keeping insertion and board quality heads physically grounded. |
| ForceFlow: Learning to Feel and Act via Contact-Driven Flow Matching | 2605.11048 | 2026-05-11 | Contact-rich policies should model force/contact evolution, not only image/proprio actions. |
| Tabero: Learning Gentle Manipulation with Closed-Loop Force Feedback from Vision, Touch, and Language | 2605.27886 | 2026-05-27 | Board wiping quality should explicitly include gentle/stable force feedback and closed-loop force traces. |
| Fisher-Preserving Guidance: Training-Free Manifold Constraints for Safe Diffusion Control | 2605.29937 | 2026-05-28 | Guidance updates need a trust region/manifold constraint; this supports our bounded/accept-only gradient updates. |
| PACT: Self-Evolving Physical Safety Alignment for Diffusion Policies in Embodied Manipulation | 2606.08414 | 2026-06-07 | Physical constraints can be enforced after pretraining; supports keeping DP as prior and adding a safety/quality score at inference/post-training time. |
| FAWAM: Force-Aware World Action Models for Closed-Loop Contact-Rich Manipulation | 2606.08555 | 2026-06-07 | Strong support for force-aware Foresight: force should appear in prediction and execution-time correction, not only as observation. |
| Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation | 2606.08737 | 2026-06-07 | Strong support for action-conditioned future tactile/world dynamics; this matches DP action chunk -> Foresight -> score. |
| TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation | 2606.11184 | 2026-06-09 | Very close recent prior for force-conditioned tactile latent prediction. It supports our force-aware Foresight design, but their predicted tactile latents are used as anticipatory policy features; our intended novelty should be converting predicted tactile/force consequences into a differentiable quality energy for DP action guidance. |
| ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation | 2606.13877 | 2026-06-11 | Supports evaluating representation properties and temporal contact continuity, not just single-step prediction loss. |
| Inference-time Policy Steering via Vision and Touch | 2606.14981 | 2026-06-12 | Directly supports the inference-time steering framing. |
| DREAM-Chunk: Reactive Action Chunking with Latent World Model | 2606.18589 | 2026-06-17 | Supports using a latent world model to correct action chunks at test time. |
| Frequency-Aware Flow Matching for Continuous and Consistent Robotic Action Generation | 2606.20135 | 2026-06-18 | Supports adding action/force smoothness and frequency consistency checks, especially for board wiping. |

Design consequence for our current codebase:

- Keep `diffusion/train_dp_tac_concat.py` as the action prior training path for now.
- Treat the `260617-only` 2000-epoch DP as a baseline/action-prior run, not the main novelty.
- The main research contribution should be a differentiable tactile/force
  consequence scorer:

  ```text
  action chunk from DP
    -> multi-step force-aware Foresight
    -> marker/force/contact-quality heads
    -> bounded classifier/scorer gradient guidance
    -> server-side force/action/guidance trace logging
  ```

- For board wiping, a marker-only future predictor is not enough because the
  positive/negative definition is force magnitude plus force smoothness during
  the contact wiping phase.
- `TacForeSight` is an important positioning reference.  It makes force-guided
  tactile foresight a timely and defensible direction, but it also means the
  paper story should not claim that force-conditioned tactile prediction itself
  is the main novelty.  The stronger distinction is:

  ```text
  recent tactile world models:
    force/tactile history -> future tactile latent -> policy feature

  our target method:
    DP action sample -> force-aware future tactile/force consequence
    -> differentiable contact-quality energy
    -> bounded action-gradient guidance
  ```

  This keeps the contribution on outcome-aware guidance for diffusion policies,
  not only on adding another tactile predictor.
- The next high-value GPU job after the DP run remains:

  ```bash
  cd /home/chenshuai/Project/TactileACT-cs
  CONFIG=TFAC_V5/config_pretrain_foresight_board_forceaware_multistep16.json \
    scripts/train/train_foresight_board_forceaware_multistep16.sh
  ```

Evidence boundary:

- The above is a research/design conclusion from verified recent papers and the
  current codebase structure.
- It is not yet a real-robot performance claim.
- Real improvement must be judged by paired baseline-vs-guided board wiping
  rollouts with server-side force traces.
