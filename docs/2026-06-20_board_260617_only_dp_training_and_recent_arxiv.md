# 2026-06-20 260617-only Board DP Training and Recent Work Notes

## Current DP Training

Task: train the tactile-concat Diffusion Policy only on:

`/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617`

Run directory:

`/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun`

Launcher:

`scripts/train/train_dp_tac_concat_board_260617_only_20260620_e2000.sh`

Core configuration:

- Model: `diffusion/train_dp_tac_concat.py`
- Vision: global + wrist images, raw collected size `200x266`, no resize/crop change beyond `200,266`
- Tactile: frozen board TactileVAE encoder
- TactileVAE checkpoint: `/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt`
- Tactile side/history: left hand, 8 marker frames
- DP horizon: `pred_horizon=16`, `obs_horizon=2`, `n_action_steps=8`
- Action: `actions/joint_abs`
- Proprio: `observations/proprio_joint`
- Epochs: 2000
- Batch size: 64
- Learning rate: `5e-5`
- Weight decay: `1e-5`
- Validation: episode-level split, `val_ratio=0.1`, `val_interval=5`, `max_val_windows=2048`
- Per-epoch cap: `max_steps_per_epoch=128`; this keeps the 2000-epoch run tractable while sampling shuffled training batches each epoch.
- Checkpoints: `dp_best.pth` updates on best validation loss, `dp_latest.pth` every 10 epochs, `dp_epoch*.pth` every 50 epochs, top-3 train-loss checkpoints retained.

Data check:

- 80 hdf5 files found.
- 79 readable episodes.
- 1 skipped episode: `episode_1.hdf5`, missing `observations/proprio_joint`.
- Usable frames: 64,524.
- Example episode shape:
  - `observations/images/global`: `(701, 200, 266, 3) uint8`
  - `observations/images/wrist`: `(701, 200, 266, 3) uint8`
  - `observations/proprio_joint`: `(701, 7) float32`
  - `observations/tac/left/marker_offset`: `(701, 9, 9, 2) float32`
  - `actions/joint_abs`: `(701, 7) float32`

Startup verification:

- A 1-epoch, 2-batch smoke run completed successfully before launching the full run.
- The first background launch exited before epoch summaries without a Python traceback. It was archived as `train_start_failed_*.log`.
- The run was relaunched using `setsid + nohup`; the second launch is running normally.

Training status observed:

- Epoch 1: train `0.815538`, val `0.388930`
- Epoch 5: train `0.077848`, val `0.073704`
- Epoch 10: train `0.041109`, val `0.040868`
- Epoch 15: train `0.027720`, val `0.024991`
- Epoch 20: train `0.021982`, val `0.022192`
- Epoch 30: train `0.017846`, val `0.019104`
- Epoch 35: train `0.015791`, val `0.018849`
- Epoch 40: train `0.015088`, val `0.018343`
- Epoch 45: train `0.014458`, val `0.018478`
- Epoch 50: train `0.013188`, val `0.015548`
- Epoch 55: train `0.013553`, val `0.015130`
- Epoch 60: train `0.012524`, val `0.015165`
- Epoch 70: train `0.012777`, val `0.014921`
- Epoch 80: train `0.011915`, val `0.014607`
- Epoch 85: train `0.011576`, val `0.014062` best so far
- Epoch 90: train `0.012078`, val `0.014627`
- Epoch 95: train `0.011268`, val `0.015983`
- Epoch 100: train `0.011680`, val `0.015453`
- Epoch 104: train `0.010387`, no validation point
- Epoch 105: train `0.010435`, val `0.015956`
- Epoch 110: train `0.010668`, val `0.016228`
- Epoch 111: train `0.010908`, no validation point
- Epoch 115: train `0.010081`, val `0.015467`
- Epoch 120: train `0.009955`, val `0.014967`
- Epoch 125: train `0.009864`, val `0.018022`
- Epoch 129: train `0.009139`, no validation point
- Epoch 140: train `0.009322`, val `0.016780`
- Epoch 145: train `0.008945`, val `0.015779`
- Epoch 150: train `0.009054`, val `0.017133`
- Epoch 155: train `0.008951`, val `0.016322`
- Epoch 157: train `0.008783`, no validation point
- Epoch 160: train `0.009007`, val `0.016753`
- Epoch 162: train `0.009160`, no validation point

Latest monitored status on 2026-06-20:

- The training process is still running; do not treat any checkpoint as final yet.
- The latest observed best validation checkpoint is epoch 85 with val loss `0.014062`.
- Epoch 95 through 160 did not refresh the best. Epoch 160 is `19.1%` higher than the best validation loss.
- The monitor state remains `watching_no_recent_best`; this is now a sustained plateau/overfit risk after epoch 85.
- `dp_best.pth`, `dp_latest.pth`, `dp_epoch50.pth`, and top-k checkpoints are being saved normally.
- GPU memory is about `14.7GB / 24.6GB`, with high utilization during active batches.

Current interpretation:

- Train and validation losses are both much lower than the startup phase.
- Epoch 85 is the current best validation point. Epoch 95 through 160 all fail to improve validation while training loss continues to edge down.
- Continue training because the requested run is 2000 epochs and the watcher has a conservative stop policy, but deployment/evaluation should strongly prefer `dp_best.pth`.
- Treat post-85 checkpoints as lower-priority candidates unless validation improves again. For real rollout, use epoch-85 `dp_best.pth`, not `dp_latest.pth`.
- A follow-up run should consider lower learning rate, stronger regularization, or fewer effective update steps if the goal is best validation rather than long-run fitting.
- For deployment/evaluation, prefer `dp_best.pth`; `dp_final.pth` should only be used after checking final validation behavior.

Monitoring files:

- `train.log`: raw training log
- `training_status_latest.json`: latest parsed status
- `loss_curve.png` / `loss_curve.csv`: periodically refreshed by the monitor
- `monitor_training.log`: monitor process log

## Recent Arxiv Work Relevant to This Project

Search window: papers submitted or updated in roughly the last two months from 2026-06-20, with older TouchGuide kept as direct background because it is a named reference for tactile guidance. Searches were checked with the arXiv API using tactile robot diffusion, tactile world model, force diffusion policy, classifier/test-time guidance, and related query terms.

The most relevant recent direction is not plain tactile concatenation. Recent work is moving toward inference-time steering, tactile future prediction, and contact-phase-aware use of touch.

### ViTaL: Inference-time Policy Steering via Vision and Touch

Source: https://arxiv.org/abs/2606.14981

Key idea:

- Keep a base generative policy.
- Use a visuo-tactile latent world model to predict future outcomes.
- Use visual verification for global mode selection.
- Use tactile-guided diffusion editing over a shorter horizon for local contact refinement.

Relevance to this project:

- Very close to our intended story: DP proposes actions, foresight predicts tactile consequences, a quality scorer guides denoising.
- Their bi-level split is useful: vision chooses the semantic/global behavior, tactile refines contact execution.
- For board wiping and insertion, this supports separating:
  - long-horizon visual/action mode: where to wipe or where to insert;
  - short-horizon tactile quality: force band, smoothness, contact stability, bounce risk.
- This is the closest recent framing to our current idea, except our score is task-defined physical quality rather than text-conditioned tactile reward.

### Dream-Tac: Tactile World Action Model

Source: https://arxiv.org/abs/2606.08737

Key idea:

- Jointly model actions, future visual observations, and future tactile dynamics.
- Use contact-gated visuo-tactile fusion and contact-aware attention bias.
- Tactile should be emphasized during salient contact changes instead of treated as a uniformly useful dense modality.

Relevance to this project:

- Supports our multi-step foresight direction.
- Our current foresight predicts tactile latent/marker futures separately from DP. A stronger next version could add contact gating or phase-aware weighting so tactile prediction matters most during contact-active windows.
- It also supports making predicted future tactile quality a central part of the policy story, not just an auxiliary visualization.

### ContactWorld: Representation Study for Vision-Tactile World Models

Source: https://arxiv.org/abs/2606.13877

Key idea:

- Studies which representation properties matter for stable long-horizon contact-rich planning.
- The strongest signal is not "more modalities" alone. Spatial structure, temporal continuity, and cross-modal compatibility matter.
- Tactile becomes more important under long-horizon planning, where compounding prediction/contact uncertainty accumulates.

Relevance to this project:

- Supports keeping tactile marker fields/latents spatially structured as long as possible instead of flattening everything too early.
- Supports our concern that a scalar score must be computed over a future window, not only one frame.
- For board wiping, a good score should preserve spatial contact distribution, contact continuity, and force/marker smoothness across the wiping segment.

### FAWAM: Force-Aware World Action Model

Source: https://arxiv.org/abs/2606.08555

Key idea:

- Incorporates force at perception, prediction, and closed-loop execution levels.
- Jointly predicts future actions and end-effector wrench trajectories.
- Uses the predicted wrench trajectory as an execution-time reference for residual correction.

Relevance to this project:

- Very close to the board setting because the good/bad definition is force-band and force-smoothness driven.
- Suggests a practical extension beyond current DP guidance: use predicted future force/marker quality as the target, then apply a small residual correction when observed force deviates from the predicted safe band.
- This does not replace DP guidance; it can become the high-frequency safety/quality correction layer after guided DP chooses the chunk.

### TacForeSight: Force-Guided Tactile World Model

Source: https://arxiv.org/abs/2606.11184

Key idea:

- Models global force and local tactile sensing as asymmetric but complementary contact signals.
- Force gives global interaction intensity, while tactile fields give local spatial contact geometry.

Relevance to this project:

- Supports using both the robot force trace and marker field for board quality labels.
- For model design, this argues against using marker-only labels when force logs are available; force is the cleanest supervision for too-light/too-heavy wiping, while marker field helps contact distribution and smoothness.

### SI-Diff: Force-Domain Diffusion Policy for Insertion

Source: https://arxiv.org/abs/2605.12247

Key idea:

- Learn search and high-precision insertion in one force-domain diffusion policy.
- Use mode conditioning to capture distinct search and insertion behaviors.
- Demonstrates force-domain policy design for peg-in-hole tolerance and zero-shot transfer.

Relevance to this project:

- Strong support for treating insertion as a phase/mode-aware force/tactile problem instead of one flat policy.
- Our insertion scorer should keep a phase-sensitive structure: approach/search, pre-contact, successful insertion, pre-bounce/bounce.
- For DP guidance, this suggests mode-conditioned guidance weights rather than one constant guidance scale across the whole rollout.

### MODIP: Model-Based Optimization for Diffusion Policies

Source: https://arxiv.org/abs/2606.10825

Key idea:

- Use a world model to improve a diffusion policy without directly doing unstable RL through the denoising chain.
- Generate higher-quality trajectories in the world model, then use them as supervised targets for policy adaptation.

Relevance to this project:

- This is not the same as our current gradient guidance, but it gives a strong future extension: use TacQualityEnergy + Foresight to generate improved action chunks offline, then distill them back into DP.
- Practical role: after real rollout proves guidance helps, distill guided actions into a faster policy for deployment.

### Feedback World Model Enables Precise Guidance of Diffusion Policy

Source: https://arxiv.org/abs/2605.15705

Key idea:

- Static open-loop world models become unreliable under distribution shift.
- Use real execution feedback to correct latent predictions online.
- Use action-aware guidance to focus on controllable parts of the prediction.

Relevance to this project:

- Directly relevant to our Foresight risk: predicted tactile futures can drift when the real contact state changes.
- Next Foresight version should keep an online correction state from observed marker/force residuals.
- For board wiping, this means the scorer should compare predicted vs observed contact quality and reduce guidance trust if foresight is inaccurate.

### PACT: Physical Safety Alignment for Diffusion Policies

Source: https://arxiv.org/abs/2606.08414

Key idea:

- Post-train diffusion policies toward constraint-feasible regions using constraint gradients.
- Distill physical constraints across diffusion timesteps, with bounded policy shift.

Relevance to this project:

- Supports using our TacQualityEnergy not only at runtime, but also as a post-training alignment signal.
- For safety-critical force limits, a post-training aligned DP may be more stable than relying only on large runtime guidance.

### Fisher-Preserving Guidance

Source: https://arxiv.org/abs/2605.29937

Key idea:

- Training-free guidance can push diffusion samples off the learned manifold.
- Project guidance updates to preserve the model's local manifold structure.

Relevance to this project:

- Important warning for our classifier guidance: a strong tactile gradient can generate actions that score well under the energy but leave the DP distribution.
- Current trust-region guidance is therefore necessary. A future stronger version can add a Fisher/manifold projection or denoising-sensitivity-based guidance scale.

### Q-Guided Flow / Test-Time Gradient Guidance

Source: https://arxiv.org/abs/2606.11087

Key idea:

- Keep supervised policy training stable.
- Improve policy at test time using value/critic gradients over flow-policy samples.

Relevance to this project:

- Gives a clean conceptual parallel: our TacQualityEnergy is a contact-quality critic, and DP denoising gets a test-time gradient toward higher predicted tactile quality.
- Difference: our critic is physically interpretable and tied to predicted tactile/force consequences, not generic task return.

### IMPACT: Internal-Model Predictive Control for Forceful Manipulation

Source: https://arxiv.org/abs/2606.10818

Key idea:

- Forceful contact tasks such as table wiping benefit from separating high-level task planning from internal predictive control.

Relevance to this project:

- Reinforces the idea that DP should handle task-level wiping motion, while a tactile/force quality layer handles local contact regulation.
- If guidance alone is too slow or unstable, the next practical layer is a lightweight force residual controller for the action chunk.

### TouchGuide

Source: https://arxiv.org/abs/2601.20239

Key idea:

- A pretrained diffusion/flow policy first generates a visually plausible coarse action.
- A task-specific Contact Physical Model provides a tactile feasibility score.
- The score steers the sampling process toward physically feasible actions.

Relevance to this project:

- This is the closest prior to "classifier/score guidance" for tactile manipulation.
- The weakness for our setting is that contrastive feasibility alone may not fully encode task-specific bad outcomes such as board force too small/too large/unstable or insertion pre-bounce risk.
- Our better angle is outcome-labeled or rule-assisted tactile quality energy, not only positive-pair matching.

### Tube Diffusion Policy

Source: https://arxiv.org/abs/2604.23609

Key idea:

- Standard action chunks are weak under contact disturbances because they are not reactive enough.
- Learn an action tube/feedback flow around nominal action chunks.
- Use diffusion correction periodically and streaming feedback control between corrections.

Relevance to this project:

- Our current DP executes chunks and uses action horizon 8. For contact tasks, guidance can improve the proposed chunk, but true robustness may require fast within-chunk correction.
- A practical future extension is to keep DP + guidance at chunk level, then add a lightweight high-frequency tactile correction layer for force deviations.

### DPTG: Diffusion Policy with Tactile Feasibility Guidance

Source: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1851102/full

Key idea:

- Treat tactile feedback as a physical feasibility / phase-awareness constraint rather than just another action-generation input.
- Train a reusable feasibility classifier with rule-assisted labels.
- Plug it into diffusion policy via gradient-based guidance.

Relevance to this project:

- This matches the current TacQualityEnergy direction closely.
- It strengthens the argument that the scoring model should be task-aware and phase-aware, but still decoupled from the base DP.
- For our tasks, the reusable abstraction should be "future tactile outcome quality":
  - board: force in target band + smooth contact + enough contact area/marker response;
  - insertion: low bounce risk + stable insertion tactile state.

## Suggested Architecture Story

The cleanest current story for this project:

1. Base DP learns visuomotor behavior from demonstrations.
2. TactileVAE compresses high-dimensional marker fields into a compact contact latent.
3. Multi-step foresight predicts future tactile consequences of candidate action chunks.
4. TacQualityEnergy scores predicted future tactile consequences using task-specific but physically interpretable heads.
5. During denoising, classifier/energy guidance shifts the action sample toward better predicted tactile outcomes, while a trust region keeps it close to the DP distribution.

This is not reranking. Reranking can remain a diagnostic baseline, but the main method should be gradient guidance inside denoising.

## What Should Improve in This Project

Current architecture:

- Base policy: tactile-concat DP trained on images, proprioception, and frozen TactileVAE latent.
- Tactile encoder: board-trained TactileVAE compresses left marker history.
- Foresight: predicts future tactile consequences over a short horizon.
- Scorer: TacQualityEnergy scores predicted tactile/force quality.
- Runtime: guidance changes the denoising trajectory through scorer gradients; real evidence still requires paired robot rollouts.

Most important improvements, ranked:

1. Add contact-phase gating to both Foresight and guidance.
   - Board: guide strongly only during contact/wiping; reduce or disable during approach and lift.
   - Insertion: guide near contact, pre-bounce, and insertion; reduce during free-space approach.
   - Reason: recent Dream-Tac/SI-Diff/ViTaL all point to mode- or contact-aware tactile use.

2. Make the scorer explicitly multi-horizon.
   - Use `t+1...t+16`, not only one future frame.
   - Board score: force band, contact continuity, marker magnitude, marker temporal smoothness, and spatial contact distribution.
   - Insertion score: margin from pre-bounce/bounce plus sustained insertion likelihood.
   - Reason: ContactWorld and TacForeSight emphasize temporal continuity and short-horizon tactile dynamics.

3. Add guidance trust control.
   - Keep the current trust-region term.
   - Reduce guidance scale when Foresight uncertainty or predicted-vs-observed residual is high.
   - Future option: Fisher/manifold-preserving projected guidance.
   - Reason: strong guidance can leave the DP action manifold.

4. Add online feedback correction as a second layer.
   - DP + TacQualityEnergy handles chunk-level action generation.
   - A small residual correction handles high-frequency force deviations during execution.
   - Reason: Tube DP, FAWAM, Feedback World Model, and IMPACT all suggest contact-rich tasks need within-chunk reactivity.

5. After real evidence, distill guided actions.
   - If real guided rollouts improve force curves/success, collect guided action chunks and train a distilled DP.
   - Reason: runtime guidance is useful for validation and quality improvement, but distillation can reduce deployment latency.

## Recommended Next Improvements

1. Contact-phase gating:
   - Apply stronger tactile quality guidance only in predicted contact-active windows.
   - For board: only during wiping/contact, not approach.
   - For insertion: stronger near pre-contact/insertion alignment, not free-space approach.

2. Multi-horizon tactile score:
   - Score future windows, not a single future frame.
   - Board score should include force band, smoothness, and contact continuity over `t+1...t+16`.
   - Insertion score should include pre-bounce margin and sustained insert likelihood.

3. Separate global and local objectives:
   - Vision/action DP handles global task mode.
   - Tactile quality scorer handles local physical feasibility.
   - This avoids asking one scalar to explain both "where to move" and "how contact feels".

4. Real evidence protocol:
   - Offline validation is necessary but not sufficient.
   - Final claim requires paired real rollouts: same task/initialization family, baseline vs guided, with server-side force traces and explicit pair metadata.

5. Potential novelty framing:
   - "Outcome-conditioned tactile energy guidance for contact-rich diffusion policies."
   - Distinguish from TouchGuide by using future tactile consequence prediction and explicit quality labels/rule-assisted objectives rather than only observation-action feasibility.
   - Distinguish from direct tactile-concat DP by decoupling action generation from tactile quality constraint enforcement.

## Current Bottom Line

The current 260617-only DP training should continue under monitoring. The best validation point is currently epoch 85, while epoch 95 through 160 are worse and indicate sustained plateau/overfit risk. The safest checkpoint for rollout remains `dp_best.pth`.

For the research story, the strongest version is:

`visual/proprio DP action prior + tactile foresight + physically interpretable tactile quality energy + contact-phase/trust-aware classifier guidance`.

This is more defensible than claiming "tactile concat alone" as the main contribution, because recent work is already converging on future tactile prediction, inference-time steering, and contact-aware gating.
