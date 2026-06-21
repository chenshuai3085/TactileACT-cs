# 2026-06-21 260617-only Board DP Training And Recent Research Notes

## Scope

This note records the current 260617-only board-wiping DP training run and a focused two-month arXiv survey for improving the current project story:

`vision+tactile DP action prior -> action-conditioned tactile/force Foresight -> differentiable quality score -> bounded gradient guidance during DP denoising`

This is a work record and design note. It does not claim real robot improvement unless paired real rollout force traces are available.

## Active Training Run

- Dataset: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617`
- Run directory: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260621_codex`
- Launch script: `scripts/train/train_dp_tac_concat_board_260617_only_e2000_20260621_codex.sh`
- Tactile encoder: `/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt`
- Policy script: `diffusion/train_dp_tac_concat.py`
- Variant: frozen TactileVAE concat DP
- Cameras: `global,wrist`
- Image mode: cached raw-image resize/normalize cache at `/home/chenshuai/Project/output/cache/dp_board_rawimg200x266_fp16`
- Image size: resize/crop `200x266`
- Tactile side/history: `left`, `8` frames
- Pred horizon / action horizon / obs horizon: `16 / 8 / 2`
- Batch size: `64`
- Epochs: `2000`
- LR / weight decay / warmup: `5e-5 / 1e-5 / 1000`
- Validation: episode-level split, `72` train episodes and `8` validation episodes
- Val interval: every `5` epochs
- Save policy: `dp_best.pth` on validation improvement, `dp_latest.pth` every `10` epochs with optimizer, `dp_epoch*.pth` every `50` epochs
- Train cap: `max_steps_per_epoch=128`

Important interpretation: because `max_steps_per_epoch=128`, one epoch is not a full pass over all sliding windows. Each epoch shuffles and trains on 128 batches. The 2000 epoch setting is therefore long stochastic minibatch training, not 2000 full dataset sweeps.

## Monitoring Snapshot

Snapshot time: `2026-06-21 20:05:47 CST`

- Latest logged epoch: `431/2000`
- Latest train loss: `0.005461`
- Latest validation epoch/loss: `430 / 0.024090`
- Best validation epoch/loss: `135 / 0.012777`
- `dp_epoch400.pth` exists and checkpoint writing is healthy
- GPU/training process/monitor process are alive
- External disk free space is sufficient, about `1.9T`
- Home root is tight but the active run outputs checkpoints to the external disk

Current judgment:

- Training is healthy in the sense that the process, GPU, logging, and checkpoint writes are normal.
- Validation has not improved since epoch 135. This is a plateau/overfit signal, not an execution failure.
- For real robot testing, use `dp_best.pth` by default unless a later validation checkpoint improves.
- `dp_latest.pth` is mainly a resumable training checkpoint because it includes optimizer/scheduler state; it is not the default deployment checkpoint.

## Current Guidance Direction

The current board guidance priority is still the force-aware Foresight quality energy:

- Runtime: `ForceAwareForesightGuidanceRuntime`
- Primary score preset: `margin_only`
- Score:

```text
S = logit_good_force_band - logsumexp(logit_too_small, logit_too_large, logit_oscillate)
```

The strongest current serving command applies the score inside the DP denoising loop on predicted clean action `x0` with bounded trust-region updates:

- `--guidance_location denoising_step`
- `--ddpm_guidance_steps 1`
- `--ddpm_guidance_scale 0.001`
- `--ddpm_max_delta_norm 0.01`

This remains gradient guidance, not reranking. Real claims still require paired baseline/guided rollouts with saved force traces.

## Recent ArXiv Survey: Last Two Months

Search window: approximately `2026-04-21` to `2026-06-21`.

### Most Relevant

1. ViTaL: Inference-time Policy Steering via Vision and Touch, arXiv `2606.14981`
   - Link: `http://arxiv.org/abs/2606.14981`
   - Key idea: inference-time steering with a high-level visual sampler/verifier and low-level tactile-guided diffusion editing.
   - Relevance: very close to our direction. It supports the story that a generative robot policy can be adapted at inference time using predicted tactile futures and tactile rewards.
   - Difference/opportunity for us: keep our contribution focused on differentiable force/tactile consequence scoring for DP denoising, especially explicit force-band and smoothness criteria for board wiping and pre-bounce risk for insertion.

2. Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation, arXiv `2606.08737`
   - Link: `http://arxiv.org/abs/2606.08737`
   - Key idea: jointly model action, future visual observations, and tactile dynamics; uses contact-gated visuotactile fusion and contact-aware attention bias.
   - Relevance: confirms that contact-aware gating and future tactile modeling are timely.
   - Project implication: add contact gating more systematically in policy conditioning and guidance, so guidance is weak before contact and stronger during detected wiping/contact windows.

3. ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation, arXiv `2606.13877`
   - Link: `http://arxiv.org/abs/2606.13877`
   - Key idea: spatially structured and temporally continuous representations are important for contact-rich planning; tactile usefulness depends on cross-modal compatibility.
   - Relevance: supports preserving marker-field structure and multi-step temporal continuity instead of relying only on flattened frame-level classifiers.
   - Project implication: our future scorer should prefer structured marker/force sequences and multi-step consistency metrics.

4. TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation, arXiv `2606.11184`
   - Link: `http://arxiv.org/abs/2606.11184`
   - Key idea: force-conditioned tactile latent dynamics for real-time proactive manipulation.
   - Relevance: strongly aligned with force-conditioned tactile foresight.
   - Project implication: our novelty must be positioned carefully: not merely "predict future tactile", but using action-conditioned force/tactile future quality as a differentiable DP guidance energy.

5. Tube Diffusion Policy: Reactive Visual-Tactile Policy Learning for Contact-rich Manipulation, arXiv `2604.23609`
   - Link: `http://arxiv.org/abs/2604.23609`
   - Key idea: learn an action tube around nominal chunks for reactive visual-tactile feedback.
   - Relevance: points out the weakness of fixed action chunking in contact-rich tasks.
   - Project implication: our bounded guidance can be framed as a lightweight inference-time correction around a nominal DP chunk.

### Secondary But Useful

6. SI-Diff: Search and High-Precision Insertion with a Force-Domain Diffusion Policy, arXiv `2605.12247`
   - Link: `http://arxiv.org/abs/2605.12247`
   - Relevance: force-domain diffusion and mode conditioning for insertion.
   - Project implication: insertion should keep explicit phase/mode definitions: approach/search, good insert, pre-bounce risk, bounce/recovery.

7. Spacetime Optimal-Transport Attention for Visuo-Haptic Imitation Learning, arXiv `2605.20433`
   - Link: `http://arxiv.org/abs/2605.20433`
   - Relevance: tri-modal fusion and interpretable phase-dependent diagnostics.
   - Project implication: later DP policy backbone can improve beyond simple concatenation by using phase/contact-conditioned attention.

8. LAGO Policy: Latency-Aware Asynchronous Diffusion Policies, arXiv `2606.17982`
   - Link: `http://arxiv.org/abs/2606.17982`
   - Relevance: smoothness and inter-chunk consistency for deployed diffusion policies.
   - Project implication: add executed-action smoothness and inter-chunk continuity to evaluation; keep guidance trust-region bounded.

9. Frequency-Aware Flow Matching for Continuous and Consistent Robotic Action Generation, arXiv `2606.20135`
   - Link: `http://arxiv.org/abs/2606.20135`
   - Relevance: action smoothness and suppressing high-frequency errors.
   - Project implication: use force/trajectory jerk metrics in board rollout evaluation; consider DCT/frequency-domain regularization later, not during the current training run.

10. Training and Evaluating Diffusion Policies with Long Context Lengths, arXiv `2606.16447`
    - Link: `http://arxiv.org/abs/2606.16447`
    - Relevance: challenges the assumption that short observation context is always enough.
    - Project implication: for board wiping and contact recovery, test longer observation contexts after the current 260617-only baseline is stable.

## Recommended Project Improvements

P0: Finish the current training and use validation-selected checkpoints.

- Let the 260617-only run continue to 2000 epochs unless it crashes or fills disk.
- Do not deploy `dp_latest.pth` by default.
- Deploy `dp_best.pth` unless validation improves later.

P0: Complete real paired rollout evidence.

- Board: run baseline `8765` vs force-aware guided `8769`.
- Save server-side force traces for every trajectory.
- Evaluate force-band occupancy, contact retention, force magnitude mean/std, force derivative/jerk, and task completion.
- This is necessary before claiming real improvement.

P1: Keep the main novelty as differentiable consequence guidance.

- DP concat policy is the action prior, not the main novelty.
- Foresight predicts future tactile/force consequences conditioned on candidate actions.
- The scorer maps predicted consequences to a quality energy.
- DP denoising uses the gradient of that energy under a trust region.

P1: Improve board scorer evaluation from classification-only to outcome-quality alignment.

- Continue reporting label separation, but do not treat classification accuracy alone as sufficient.
- Required scorer tests:
  - held-out episode-level label separation
  - finite and positive action gradients
  - bounded action delta
  - real-window serving audit
  - paired real force-curve improvement

P2: Architectural next steps after this run.

- Structured tactile/force representation: keep marker field and multi-step force proxy rather than only flattened final latent.
- Contact-aware gating: guide mainly during wiping/contact windows.
- Long context: compare obs horizon 2 vs longer history for board wiping.
- Smoothness-aware rollout metrics: force jerk, action jerk, inter-chunk continuity.
- Phase/mode-conditioned insertion scorer: approach/search vs good insert vs pre-bounce/bounce.

## Current Decision

Do not change the active training process. Continue monitoring. If no later validation improvement occurs, keep `dp_best.pth` as the recommended checkpoint for real tests.
