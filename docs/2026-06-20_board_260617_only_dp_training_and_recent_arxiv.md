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

Early training status observed:

- Epoch 1: train `0.815538`, val `0.388930`
- Epoch 5: train `0.077848`, val `0.073704`
- Epoch 10: train `0.041109`, val `0.040868`
- Epoch 15: train `0.027720`, val `0.024991`

Current interpretation:

- Train and validation losses are both decreasing in the early phase.
- No early overfitting signal is visible yet.
- For deployment/evaluation, prefer `dp_best.pth`; `dp_final.pth` should only be used after checking final validation behavior.

Monitoring files:

- `train.log`: raw training log
- `training_status_latest.json`: latest parsed status
- `loss_curve.png` / `loss_curve.csv`: periodically refreshed by the monitor
- `monitor_training.log`: monitor process log

## Recent Work Relevant to This Project

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

### Dream-Tac: Tactile World Action Model

Source: https://arxiv.org/html/2606.08737v1

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

### Force-Guided Tactile World Model

Source: https://arxiv.org/html/2606.11184v1

Key idea:

- Models global force and local tactile sensing as asymmetric but complementary contact signals.
- Force gives global interaction intensity, while tactile fields give local spatial contact geometry.

Relevance to this project:

- Supports using both the robot force trace and marker field for board quality labels.
- For model design, this argues against using marker-only labels when force logs are available; force is the cleanest supervision for too-light/too-heavy wiping, while marker field helps contact distribution and smoothness.

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
