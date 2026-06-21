# 2026-06-21 260617-only DP Training and Recent ArXiv Survey

## Current training

Task: train the tactile-concat Diffusion Policy only on:

`/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617`

Run directory:

`/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260621_codex`

Training command script:

`scripts/train/train_dp_tac_concat_board_260617_only_e2000_20260621_codex.sh`

Main configuration:

- policy: `diffusion/train_dp_tac_concat.py`
- tactile side: left
- tactile encoder: frozen board TactileVAE
- TactileVAE checkpoint: `/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt`
- image input: `global,wrist`, raw deployment-compatible resize/crop `200x266`
- image cache: `/home/chenshuai/Project/output/cache/dp_board_rawimg200x266_fp16`
- pred horizon: 16
- obs horizon: 2
- action horizon: 8
- epochs: 2000
- batch size: 64
- learning rate: `5e-5`
- weight decay: `1e-5`
- validation: episode-level split, `val_ratio=0.1`, `val_interval=5`
- checkpointing: `dp_latest.pth` every 10 epochs, epoch checkpoints every 50 epochs, best checkpoint by validation loss

Status at 2026-06-21 18:18:

- latest epoch: 259 / 2000
- latest train loss: 0.006352
- latest validation epoch: 255
- latest validation loss: 0.019188
- current best validation epoch: 135
- best validation loss: 0.012777
- GPU: RTX 4090 active
- output disk: external USB, enough space
- `/home` remains tight, so large checkpoints are intentionally written to external USB.

Interpretation:

- The run is healthy and still training.
- The training loss continues to decrease.
- The episode-level validation loss stopped improving after epoch 135 and later rose, so the model is showing overfitting relative to the current held-out episodes.
- For real testing, prefer `dp_best.pth`; keep `dp_latest.pth` and epoch checkpoints only for diagnostics or ablation.

## Recent papers checked

The survey window is 2026-04-21 to 2026-06-21. The most relevant recent arXiv papers are below.

1. DPTG: Diffusion Policy with Tactile Guidance for Contact-rich Manipulation, arXiv:2606.06281.
   - Main relevance: tactile feedback is used as a physical feasibility constraint to guide the diffusion denoising process.
   - Architectural implication: tactile should not only be concatenated as another observation; it should shape the generated action distribution through a guidance signal.
   - Fit to this project: directly supports the current goal of using a tactile/force quality scorer for gradient guidance rather than reranking.

2. Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation, arXiv:2606.08737.
   - Main relevance: jointly models action, future visual observations, and tactile dynamics for contact-rich manipulation.
   - Architectural implication: action-conditioned future tactile prediction is a defensible module, not an auxiliary toy experiment.
   - Fit to this project: supports the current Foresight direction and the idea that action quality should be judged through predicted tactile consequences.

3. TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation, arXiv:2606.11184.
   - Main relevance: predicts short-horizon tactile latent dynamics conditioned on high-frequency wrist force/torque.
   - Architectural implication: force and tactile should play asymmetric roles; force can stabilize or condition tactile foresight.
   - Fit to this project: strongly supports a force-aware board-wiping scorer/Foresight instead of marker-only scoring.

4. ViTaL: Inference-time Policy Steering via Vision and Touch, arXiv:2606.14981.
   - Main relevance: multimodal inference-time steering for contact-rich manipulation.
   - Architectural implication: use vision for long-horizon mode selection and touch for short-horizon local contact refinement.
   - Fit to this project: supports the current direction of DP action prior + tactile/force future scoring + gradient refinement.

5. ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation, arXiv:2606.13877.
   - Main relevance: contact-rich world models need spatially structured and temporally continuous representations.
   - Architectural implication: marker field / force field structure should be preserved rather than collapsed too early.
   - Fit to this project: supports keeping marker-field latent prediction and temporal smoothness checks in Foresight.

6. WT-UMI: Tactile-based Whole-Body Manipulation via Force-Supervised Contact-Aware Planning, arXiv:2606.13232.
   - Main relevance: explicit force prediction/reference improves contact-rich execution.
   - Architectural implication: board wiping should not rely on marker-only quality; force-aware supervision is central.
   - Fit to this project: supports force-band and force-smooth quality heads for board wiping.

7. Ambient Diffusion Policy, arXiv:2606.12365.
   - Main relevance: suboptimal data can be useful if the model controls when and how it uses it.
   - Architectural implication: do not blindly throw away bad board demonstrations; use them to learn quality/risk boundaries and optionally diffusion-time-dependent training or guidance.
   - Fit to this project: supports maintaining positive, too-small, too-large, and oscillatory contact modes as labeled quality data for the scorer.

8. QPILOTS: Efficient Test-Time Q-Steering for Flow Policies, arXiv:2606.14801.
   - Main relevance: test-time gradient steering of generative robot policies.
   - Architectural implication: guidance should act on clean action estimates or scheduler-aware intermediate states, with trust-region control.
   - Fit to this project: supports using TacQualityEnergy as an action-gradient signal instead of a reranking-only module.

9. SI-Diff: A Framework for Learning Search and High-Precision Insertion with a Force-Domain Diffusion Policy, arXiv:2605.12247.
   - Main relevance: high-precision insertion can be modeled in force-domain diffusion policy space, with mode conditioning for search vs insertion.
   - Architectural implication: insertion and wiping should not share a single undifferentiated quality head; task phase/mode matters.
   - Fit to this project: supports insertion-specific risk scoring and phase-aware guidance, especially for pre-bounce vs insertion.

10. Frequency-Aware Flow Matching for Continuous and Consistent Robotic Action Generation, arXiv:2606.20135.
   - Main relevance: action chunks need temporal consistency and low high-frequency artifacts.
   - Architectural implication: board wiping quality should include smooth action/force frequency penalties or action delta constraints.
   - Fit to this project: supports adding frequency-domain or derivative penalties to DP/guidance for wiping smoothness.

11. Training and Evaluating Diffusion Policies with Long Context Lengths, arXiv:2606.16447.
   - Main relevance: longer observation context can improve tasks that require memory.
   - Architectural implication: current obs horizon 2 is deployment-compatible but may be short for board wiping contact-state memory.
   - Fit to this project: motivates an ablation with longer tactile/force history for scorer/Foresight first, then DP if deployment latency allows.

12. T-Rex: Tactile-Reactive Dexterous Manipulation, arXiv:2606.17055.
   - Main relevance: high-frequency tactile streams and temporal tactile encoders improve reactive manipulation.
   - Architectural implication: dynamic tactile encoders may be more useful than static latent snapshots for guidance.
   - Fit to this project: supports temporal tactile VAE/Foresight and short-window contact phase modeling.

13. TactSpace, arXiv:2606.18959, and TaCauchy, arXiv:2606.20426.
   - Main relevance: physics-enriched tactile representation and tactile simulation.
   - Architectural implication: a physics-aware tactile latent could improve transfer and robustness.
   - Fit to this project: useful as future sim/augmentation direction, not required for the current 260617-only DP run.

## Architecture recommendation

The current best story should stay:

`vision+tactile DP action prior -> action-conditioned Foresight -> force/tactile quality energy -> gradient guidance during denoising or clean-action refinement`

Key points:

- This is not simple tactile concatenation.
- Tactile concatenation helps the policy condition on current contact.
- The guidance scorer evaluates predicted future contact consequences of candidate actions.
- The gradient should push the generated action toward better predicted force/contact outcomes.
- For board wiping, the most defensible quality definition is force-aware:
  - good: stable contact, force in target band, low force jerk/oscillation, marker response stable.
  - bad-too-small: weak/no contact, insufficient wiping force.
  - bad-too-large: excessive force, safety and wear risk.
  - bad-oscillate: alternating force/contact instability.
- For insertion, the quality definition remains:
  - good: successful insertion/contact pattern.
  - bad: pre-bounce risk and impact/recovery states.

## Suggested next technical improvements

1. Keep this 260617-only DP training running, but deploy from `dp_best.pth` unless validation later improves.

2. Add paired real rollout force-trace validation before claiming robot improvement:
   - baseline DP vs guided DP
   - same task setup and comparable episodes
   - server-side force trace saved per rollout
   - compare Fz band score, smoothness, too-small/too-large rates, stopped-early, and task success.

3. For board quality guidance, keep `margin_only` as the current stable default:

   `S = logit_good - logsumexp(logit_too_small, logit_too_large, logit_oscillate)`

   Physical penalty terms should remain ablation candidates until validated with real rollout traces.

4. The most promising innovation direction is a contact-phase-aware quality energy:
   - apply force/marker quality only during detected wiping contact phase;
   - avoid penalizing approach/lift phases;
   - make the score differentiable with respect to action through Foresight.

5. A strong follow-up ablation is guidance timing:
   - final clean-action trust-region refinement;
   - late denoising steps only;
   - scheduler-aware every-step guidance.

   The current evidence supports gradient guidance as a research path, but real rollout evidence is still required for production claims.

## Sources

- https://arxiv.org/abs/2606.14981
- https://arxiv.org/abs/2606.06281
- https://arxiv.org/abs/2606.08737
- https://arxiv.org/abs/2606.11184
- https://arxiv.org/abs/2606.13877
- https://arxiv.org/abs/2606.13232
- https://arxiv.org/abs/2606.12365
- https://arxiv.org/abs/2606.14801
- https://arxiv.org/abs/2605.12247
- https://arxiv.org/abs/2606.20135
- https://arxiv.org/abs/2606.16447
- https://arxiv.org/abs/2606.17055
- https://arxiv.org/abs/2606.18959
- https://arxiv.org/abs/2606.20426
