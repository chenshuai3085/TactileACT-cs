# 2026-06-21 Recent Tactile-DP Research Notes

## Current Training Context

- Active run: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260621_tmux`
- Script: `scripts/train/train_dp_tac_concat_board_260617_only_2000_20260621.sh`
- Data: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617`
- Valid data: 79/80 HDF5 episodes; `episode_1.hdf5` is skipped because it lacks `observations/proprio_joint`.
- Policy: DP concat, raw 200x266 `global,wrist` image observations, frozen board TactileVAE latent, proprio.
- Training: 2000 epochs, batch 64, lr `5e-5`, `val_ratio=0.1`, `val_interval=5`, `save_freq=50`, `latest_freq=10`, `topk_k=3`.

## Recent Papers Checked

Source metadata was saved at:
`/home/chenshuai/Project/output/recent_arxiv_tactile_dp_20260621/papers.json`

Additional API-check metadata was saved at:
`/home/chenshuai/Project/output/recent_arxiv_tactile_dp_20260621/additional_api_checks_20260621.json`

| Date | Paper | Relevance |
|---|---|---|
| 2026-06-12 | Inference-time Policy Steering via Vision and Touch | Very close to our direction: pre-trained generative policy plus inference-time visual/tactile steering. The key alignment is tactile-guided diffusion editing, not only candidate reranking. |
| 2026-06-11 | ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation | Supports our focus on spatially structured and temporally continuous tactile/force representations for planning/guidance. |
| 2026-06-07 | Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation | Supports joint modeling of action and future tactile dynamics; similar motivation to Foresight, with contact-gated fusion and acceleration. |
| 2026-06-04 | Multi-Resolution Tactile Imitation Learning | Supports multi-timescale tactile history rather than only a single short window. |
| 2026-06-10 | Ambient Diffusion Policy | Relevant for mixing good and suboptimal data; suggests noise/time-dependent use of lower-quality data instead of naive pooled training. |
| 2026-06-15 | Training and Evaluating Diffusion Policies with Long Context Lengths | Relevant because our current `obs_horizon=2` is short; long context may reduce repeated failed motions in contact-rich tasks. |
| 2026-06-11 | FTP-1: A Generalist Foundation Tactile Policy | Shows the field is moving toward transferable tactile tokens. Useful as background, but too large-scale to directly reproduce here. |

Additional arXiv API checks on 2026-06-21 found several papers that are closer to specific design choices in this project:

| Date | Paper | Link | Project implication |
|---|---|---|---|
| 2026-06-07 | FAWAM: Force-Aware World Action Models for Closed-Loop Contact-Rich Manipulation | http://arxiv.org/abs/2606.08555v2 | Strong support for treating force as a predicted future consequence and feedback signal, not only as a passive observation. This matches the board force-aware Foresight scorer direction. |
| 2026-06-09 | TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation | http://arxiv.org/abs/2606.11184v1 | Very close naming and theme; important to position our contribution around DP gradient guidance with an interpretable task quality energy, not just tactile world-model prediction. |
| 2026-06-07 | Latent Diffusion Policy: Shaping Latent Spaces for Diffusion-Based Robotic Manipulation | http://arxiv.org/abs/2606.08657v1 | Suggests a future policy variant where the action sequence is denoised in a learned latent action space, reducing raw-action denoising burden. Not part of the current 260617-only run. |
| 2026-06-07 | Guided Discovery of New Behaviors using Diffusion Policies | http://arxiv.org/abs/2606.08743v1 | Supports test-time or training-time guided exploration of behavior modes, but our current requirement is not mode discovery; only bounded task-quality gradient correction should be adopted. |
| 2026-06-15 | FlowMPC: Improving Flow Matching policies with World Models | http://arxiv.org/abs/2606.16286v1 | Reinforces the world-model-plus-test-time-control story. This is an adjacent flow-matching policy direction rather than a reason to replace the current DP immediately. |
| 2026-06-11 | QPILOTS: Efficient Test-Time Q-Steering for Flow Policies | http://arxiv.org/abs/2606.14801v1 | Relevant as evidence that test-time objective steering is active in robot generative policies. It should be cited as adjacent steering, not as our exact method. |
| 2026-05-12 | SI-Diff: A Framework for Learning Search and High-Precision Insertion with a Force-Domain Diffusion Policy | http://arxiv.org/abs/2605.12247v1 | Relevant to the insertion side: force-domain diffusion and high-precision insertion support using force/contact signals for policy generation. |
| 2026-04-26 | Tube Diffusion Policy: Reactive Visual-Tactile Policy Learning for Contact-rich Manipulation | http://arxiv.org/abs/2604.23609v1 | Supports reactive visual-tactile contact-rich policy learning, but it is policy learning rather than our Foresight energy guidance mechanism. |

## Current Project Position

The strongest current story is:

1. Train a DP action prior from visual, tactile-latent, and proprio observations.
2. Train Foresight to predict future tactile/force consequences from current observation and candidate actions.
3. Train a task-grounded quality/risk scorer over the predicted consequence.
4. During DP denoising or final clean-action refinement, backpropagate the scorer through Foresight to update the action inside a bounded trust region.

This is explicitly gradient guidance, not reranking. The scorer is not meant to select among sampled candidates after the fact; it provides a differentiable objective that changes the action sequence.

## Design Implications From Recent Work

### 1. Keep Foresight as the central bridge

ViTaL and Dream-Tac both support the idea that a policy needs predicted tactile futures, not only current tactile observations. Our current `action -> Foresight -> tactile/force consequence -> quality score -> gradient` chain is the right backbone.

Concrete next improvement:
- Keep the force-aware multistep Foresight path for board wiping.
- Make the deploy command prefer the force-aware research arm once real robot force traces are collected.
- Avoid presenting marker-only classification accuracy as the final evidence; guidance signal and real force curves matter more.

### 2. Make the board score physically explicit

For board wiping, a good action is not just visually plausible. It should produce:

- enough contact force,
- not too much contact force,
- smooth force change,
- smooth marker/force-field evolution during the wiping contact phase.

The current force-aware candidate is therefore more scientific than the marker-joint scorer because it directly scores force-band consequence. Current offline evidence favors the `margin_only` force-band good-vs-risk score. Smoothness and force-center penalties should remain candidate heads or analysis metrics until paired real force traces prove they improve behavior.

### 3. Add long-context tactile conditioning as a controlled ablation

The current DP uses `obs_horizon=2` and TactileVAE history 8. Recent long-context DP evidence suggests that longer context is often workable if the conditioning method is suitable.

Controlled ablation:
- baseline: current `obs_horizon=2`, `tac_history=8`
- ablation A: `obs_horizon=4`, `tac_history=8`
- ablation B: keep DP context short but give scorer/Foresight multi-scale features, e.g. short window for contact state and longer low-rate window for force smoothness.

This should be evaluated with episode-level validation and real rollout force traces, not only train loss.

### 4. Treat bad/suboptimal data differently across diffusion time

Ambient Diffusion Policy is relevant because board datasets contain positives and several negative modes. Instead of simply pooling all positive and negative data into behavior cloning, one option is:

- Train the action prior mainly on acceptable/positive trajectories.
- Use negative/suboptimal trajectories mainly for the scorer/Foresight/energy model.
- If using negative data in DP, use it in a noise/time-dependent way so it contributes broad support but does not dominate low-noise final action details.

This is a possible future training variant. It is not implemented in the current 260617-only run.

### 5. Consider latent-action DP only after the current baseline is stable

Latent Diffusion Policy suggests that raw-action denoising can be hard because the denoising vector field must handle both scene understanding and precise trajectory generation. For this project the immediate priority is still to finish the raw-action DP concat baseline and validate force/tactile guidance. A later ablation can be:

- encode 16-step joint action chunks into a compact latent action;
- train diffusion in that latent space;
- decode to raw joint actions;
- apply the same Foresight quality gradient either in latent action space or through the decoder to latent space.

This may make guidance smoother, but it adds another learned bottleneck and should not replace the current baseline until the baseline has real rollout evidence.

### 6. Evaluation gate should stay evidence-bound

The useful evaluation levels are:

1. DP training/validation loss and checkpoint stability.
2. Foresight prediction quality on held-out episodes.
3. Scorer accuracy/calibration with episode-level split.
4. Guidance audit: finite gradients, score improvement, bounded action delta.
5. Real paired rollout: baseline vs guided force traces and outcome metadata.

The project should not claim robot improvement before level 5.

## Recommended Near-Term Experiments

1. Finish or monitor the active 260617-only DP training.
   - Use `dp_best.pth` for deployment, not necessarily `dp_final.pth`.
   - Watch for overfitting via `training_status_latest.json`.

2. Run paired real board tests for the force-aware guidance arm.
   - Baseline and guided need server-side `force_trace.csv`.
   - Compare force-band occupancy, excessive force, too-light contact, and force jerk/smoothness.

3. If the new 260617-only DP best is stronger than the existing stable run, update the command sheet to point to it.
   - Only after checking validation and smoke loading.

4. Add a long-context ablation after the current run, not during it.
   - Suggested first ablation: `obs_horizon=4`, same TactileVAE and same data.

5. Do not switch the main method to reranking.
   - If visual mode selection is needed later, keep it as a high-level optional stage.
   - The core PTG contribution should remain differentiable tactile/force consequence guidance.

## Current Claim Boundary

Can say now:
- The training run has started and early loss/validation behavior is healthy.
- The current research direction is aligned with recent visuo-tactile inference-time steering and tactile world-action model work.
- The most promising project-specific novelty is interpretable force/tactile consequence energy for bounded DP gradient guidance.
- As of epoch 55, the 260617-only DP run is still improving on validation loss (`best val=0.015672`), so there is no evidence yet for stopping or switching checkpoints.
- Epoch 60 validation (`val=0.016899`) did not refresh best, but this is currently a small normal fluctuation over one validation point, not enough evidence for overfitting.

Cannot say yet:
- The new 260617-only DP checkpoint is better than the previous stable checkpoint.
- Force-aware guidance improves real board wiping.
- Smoothness heads improve real force behavior.
