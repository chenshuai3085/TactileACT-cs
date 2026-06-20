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

- DP training is running and checkpointing.
- The active `260617-only` DP run strongly overfits after epoch 85 by held-out
  episode validation loss.
- The validation-selected rollout candidate remains:

  `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun/dp_best.pth`

- Server-side TacQuality guidance code supports final-action and denoising-step
  guidance, contact gating, and force-trace logging.

What is not proven yet:

- Real board wiping improvement from guidance.
- Force curve improvement versus baseline under paired real rollout conditions.
- That marker-only Foresight is enough for board force-quality guidance.

## Recommended Next Experiment After DP Run

After the GPU frees, the most useful next experiment is force-aware Foresight:

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
