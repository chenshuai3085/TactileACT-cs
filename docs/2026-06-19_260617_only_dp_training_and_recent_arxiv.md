# 2026-06-19 260617-only DP Training and Recent ArXiv Review

## Goal

Train a tactile Diffusion Policy only on:

`/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617`

The longer-term purpose is still TacQuality-guided DP: learn a baseline tactile DP on the new 260617 data, then compare baseline vs TacQuality/Foresight guided deployment with server-side force traces.

## Training Run

- Run directory:
  - `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619`
- Launcher:
  - `run_command.sh`
  - tmux session: `dp260617_2000_20260619`
- Dataset:
  - 80 HDF5 episodes found
  - train split: 72 episodes
  - val split: 8 episodes
  - one train episode is skipped by the loader as corrupted/incomplete
- Input:
  - cameras: `global,wrist`
  - raw image shape in HDF5: `200 x 266 x 3`
  - resize/crop: `200,266`
  - proprio: `observations/proprio_joint`
  - action: `actions/joint_abs`
  - tactile side: left
  - tactile history: 8 frames
- Frozen tactile encoder:
  - `/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt`
  - checkpoint norm stats loaded:
    - mean `[-0.33986145, -2.92084837]`
    - std `[1.98048532, 2.76711774]`
- DP config:
  - pred horizon: 16
  - obs horizon: 2
  - action horizon at serving: 8
  - epochs requested: 2000
  - batch size: 64
  - lr: `1e-4`
  - warmup steps: 500
  - diffusion train/inference timesteps: 100/100
  - down dims: `512,1024,2048`
  - train windows: 8192
  - val windows: 1024
  - validation: episode-level split, every epoch
  - checkpointing:
    - `dp_best.pth`: updates by validation loss
    - `dp_latest.pth`: rolling latest every 20 epochs
    - `dp_epoch*.pth`: every 200 epochs
    - `dp_topk_*.pth`: top-3 train loss snapshots

## Historical Interim Training Status

This section is a historical interim snapshot from the live training monitor.
The final state is recorded below in `Early Stop: 2026-06-19 06:03 CST`.

- Snapshot time: 2026-06-19 05:07 CST
- Latest parsed epoch: 95 / 2000
- Latest train loss: `0.009727`
- Latest val loss: `0.014697`
- Current best epoch: 94
- Best val loss: `0.010671`
- Epochs since best: 1
- `latest_val / best_val`: `1.3773`
- GPU memory during training: about 14.7 GB / 24.6 GB
- Output artifacts:
  - `train.log`
  - `loss_curve.csv`
  - `loss_curve.png`
  - `training_snapshot_latest.json`
  - `training_status_latest.json`

Interpretation:

- The run is healthy and still training.
- The early val curve decreased rapidly from `0.176030` at epoch 1 to `0.011638` at epoch 63.
- After a noisy plateau, epoch 94 refreshed the best validation loss to `0.010671`.
- Epoch 95 bounced back to `0.014697`, but because the run just improved at epoch 94, this is still normal short-term validation fluctuation.
- Because validation still reached a new best after epoch 20, continuing the run is currently reasonable.
- Final deployment should use `dp_best.pth`, not necessarily `dp_latest.pth` or `dp_final.pth`.
- `dp_topk_*.pth` files are selected by training loss, not validation loss; they are useful diagnostics, not the primary deployment choice.

## Why Validation Matters Here

The previous 260617-only run requested 2000 epochs but was manually stopped around epoch 830 because:

- best val was at epoch 105 with val loss about `0.011152`
- train loss kept dropping to about `0.001981`
- latest val became about `0.041854`

That pattern means the model was fitting sampled training windows while losing held-out episode performance. For this reason, the scientific checkpoint choice must be validation-based.

## Recent ArXiv Review

Scope: papers from roughly the last two months that are directly relevant to tactile/force/contact-rich diffusion policies or guidance-like mechanisms.

### Tube Diffusion Policy

- Paper: Tube Diffusion Policy: Reactive Visual-Tactile Policy Learning for Contact-rich Manipulation
- arXiv: http://arxiv.org/abs/2604.23609
- Date: 2026-04-26
- Core idea:
  - action chunking is weak for contact-rich tasks because the robot cannot react fast enough to tactile/contact disturbances;
  - TDP learns a feedback flow around nominal action chunks, forming an action tube.
- Relevance:
  - Our current DP is still chunk-based.
  - TacQuality guidance currently refines the final clean action chunk; a stronger next step is to add a residual tactile feedback correction around the chunk during execution.

### SI-Diff

- Paper: SI-Diff: A Framework for Learning Search and High-Precision Insertion with a Force-Domain Diffusion Policy
- arXiv: http://arxiv.org/abs/2605.12247
- Date: 2026-05-12
- Core idea:
  - learn search and precision insertion in one force-domain diffusion policy;
  - use mode conditioning to handle different contact/action modes.
- Relevance:
  - Supports our view that contact phases should not be treated as a single homogeneous behavior.
  - For insertion and board wiping, phase/mode conditioning can be useful:
    - insertion: approach / insert / pre-bounce / bounce / recovery
    - board: approach / stable wiping / too light / too heavy / oscillatory

### Dream-Tac

- Paper: Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation
- arXiv: http://arxiv.org/abs/2606.08737
- Date: 2026-06-07
- Core idea:
  - jointly model actions, future visual observations, and tactile dynamics;
  - use contact-gated visuotactile fusion and contact-aware attention.
- Relevance:
  - Very close to our Foresight idea.
  - Our current story is narrower and more controllable:
    - DP proposes action
    - Foresight predicts tactile consequence
    - TacQuality scores predicted consequence
    - gradient improves the action inside a trust region
  - A future upgrade could add contact-gated fusion to Foresight and DP, so tactile affects the model mainly during actual contact.

### ViTaL / Inference-time Policy Steering via Vision and Touch

- Paper: Inference-time Policy Steering via Vision and Touch
- arXiv: http://arxiv.org/abs/2606.14981
- Date: 2026-06-12
- Core idea:
  - inference-time steering verifies candidate policy actions with future visual and tactile predictions;
  - a learned verifier/reward supports candidate selection and tactile-guided diffusion editing before execution.
- Relevance:
  - This is very close to our broad direction and confirms that tactile future prediction is a timely research line.
  - It also sharpens our distinction:
    - avoid framing our method as simple reranking;
    - emphasize differentiable TacQuality energy, local trust-region gradient refinement, explicit contact-quality criteria tied to force/motion smoothness, and real force-trace evaluation.

### ContactWorld

- Paper: ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation
- arXiv: http://arxiv.org/abs/2606.13877
- Date: 2026-06-11
- Core idea:
  - studies what representation properties make vision-tactile world models useful for stable contact-rich planning.
- Relevance:
  - Supports adding explicit diagnostics for Foresight, not only average reconstruction loss:
    - contact onset/offset timing;
    - force or marker magnitude band;
    - smoothness/oscillation accuracy;
    - ranking preservation between good and bad contact futures.
  - This is directly relevant to whether TacQuality guidance will receive useful gradients.

### TacForeSight

- Paper: TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation
- arXiv: http://arxiv.org/abs/2606.11184
- Date: 2026-06-09
- Core idea:
  - a force-conditioned tactile world model predicts short-horizon tactile latent dynamics for contact-rich manipulation.
- Relevance:
  - Closest name/story overlap with our Foresight component.
  - Our project should avoid vague claims and be precise:
    - what sensor representation is predicted;
    - what quality energy is optimized;
    - how gradients modify DP actions;
    - which real force metrics improve during paired baseline/guided rollouts.

### Latent Diffusion Policy

- Paper: Latent Diffusion Policy: Shaping Latent Spaces for Diffusion-Based Robotic Manipulation
- arXiv: http://arxiv.org/abs/2606.08657
- Date: 2026-06-07
- Core idea:
  - generate in a shaped latent action space instead of raw action space;
  - use a CVAE-style observation-conditioned latent to simplify the diffusion field.
- Relevance:
  - Our DP currently denoises raw joint action chunks.
  - For small 260617 data, raw-action large UNet can overfit quickly.
  - A meaningful research upgrade is a latent-action DP where guidance acts on a low-dimensional action latent, then decodes to smooth joint chunks.

### Long Context Diffusion Policies

- Paper: Training and Evaluating Diffusion Policies with Long Context Lengths
- arXiv: http://arxiv.org/abs/2606.16447
- Date: 2026-06-15
- Core idea:
  - short observation history can fail on tasks needing memory;
  - longer context is not necessarily brittle with proper conditioning and backbone.
- Relevance:
  - Our current obs horizon is 2 and tactile history is 8.
  - For board wiping, force smoothness and oscillation are temporal properties.
  - A reasonable ablation is obs horizon 4/8 and tactile history 16, especially for scoring and Foresight.

### LAGO Policy

- Paper: LAGO Policy: Latency-Aware Asynchronous Diffusion Policies with Goal-Directed Collision-Free Planning for Smooth Manipulation
- arXiv: http://arxiv.org/abs/2606.17982
- Date: 2026-06-16
- Core idea:
  - improve inter-chunk consistency under asynchronous inference;
  - use latency-aware classifier-free guidance and trajectory optimization for smooth execution.
- Relevance:
  - Our deployment executes chunks and can suffer from discontinuity between chunks.
  - TacQuality guidance should include a smoothness/trust-region term across the executed prefix and the new chunk.

### Trust-Region Diffusion Policies

- Paper: Trust-Region Diffusion Policies for Massively Parallel On-Policy RL
- arXiv: http://arxiv.org/abs/2606.15260
- Date: 2026-06-13
- Core idea:
  - enforce a KL trust region over diffusion trajectories for stable policy updates.
- Relevance:
  - Even though this is RL, the trust-region principle matches our guidance implementation.
  - TacQuality guidance should be bounded and accept-only; large score gradients must not move the action far from DP support.

### TouchGuide

- Paper: TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance
- arXiv: http://arxiv.org/abs/2601.20239
- Latest checked version: v6, 2026-05-13
- Core idea:
  - keep a pretrained diffusion/flow policy;
  - use a task-specific Contact Physical Model to produce a tactile feasibility score;
  - steer the action sampling process with that score at inference time.
- Relevance:
  - This is one of the closest references to our current TacQuality route.
  - Key difference for our story:
    - TouchGuide scores current tactile/action feasibility;
    - our method scores predicted future tactile consequences through Foresight, then refines the DP action in a trust region.

### AdaVTF

- Paper: Learning When to See and When to Feel: Adaptive Vision-Torque Fusion for Contact-Aware Manipulation
- arXiv: http://arxiv.org/abs/2604.01414
- Date: 2026-04-01
- Core idea:
  - compare several force/torque-vision fusion strategies inside diffusion policies;
  - adaptively ignore torque before contact and use torque during contact.
- Relevance:
  - Strong support for contact-gated fusion/guidance.
  - For board wiping, TacQuality guidance should not be equally active during approach and contact.

### pi0.7

- Paper: pi0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities
- arXiv: http://arxiv.org/abs/2604.15483
- Date: 2026-04-17
- Core idea:
  - a generalist robot foundation model can be steered by conditioning metadata and classifier-free guidance-like control.
- Relevance:
  - The useful idea is not to copy the large VLA model.
  - The useful idea is to expose quality/style metadata or labels so inference can steer toward desired execution properties.
  - In our setting, those metadata are concrete contact-quality labels:
    - stable wiping
    - too light
    - too heavy
    - oscillatory

### DPTG

- Paper: DPTG: diffusion policy with tactile feasibility guidance
- Source: Frontiers in Robotics and AI, not arXiv
- Link: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1851102/full
- Checked because it is very close to the current project direction.
- Core idea:
  - use tactile feasibility guidance at inference time;
  - report reduced force spikes and force variance.
- Relevance:
  - This is a useful external reference, but should be cited separately from the arXiv-only literature review.
  - It reinforces that final evidence must include real force traces, not only offline prediction or classification metrics.

## Architecture Implications for This Project

1. Keep the current TacQuality-guided route as the main short-term path:
   - DP remains the behavior prior.
   - Foresight predicts tactile consequences.
   - TacQuality gives a differentiable score.
   - Trust-region gradient refinement changes actions only when score improves without leaving DP support.

2. Add phase/contact conditioning:
   - Board wiping should distinguish at least:
     - approach/no-contact
     - stable wiping
     - too light
     - too heavy
     - oscillatory/unstable
   - This should be used in scorer and possibly DP conditioning.

3. Add contact-gated guidance:
   - Guidance should be weak or disabled before contact.
   - Guidance should become active during the wiping contact phase.
   - Gate can be based on marker magnitude/contact area/Fz if available.
   - This is supported by AdaVTF and Dream-Tac.

4. Consider latent-action DP as the next model upgrade:
   - raw joint-action DP is powerful but overfits small datasets easily;
   - latent action can make denoising smoother and guidance more stable.

5. Extend temporal context for scoring/Foresight first, not necessarily DP first:
   - score needs to judge force smoothness;
   - smoothness is not visible from one frame;
   - use 16-frame marker/force windows for scorer/Foresight ablations.

6. Treat TacQuality as future-consequence guidance, not only current feasibility:
   - TouchGuide and DPTG motivate tactile feasibility steering.
   - Our novelty should be: action -> Foresight-predicted tactile future -> quality energy -> trust-region gradient update.
   - This makes the scorer useful before the bad contact has fully happened.

7. Real robot validation remains mandatory:
   - offline val loss only measures behavior cloning fit;
   - TacQuality score/guidance must still be judged by paired baseline/guided rollouts;
   - server-side force traces are required for final evidence.

## Immediate Next Actions

1. Continue the 260617-only DP training.
2. Monitor:
   - latest epoch
   - best val epoch
   - latest val / best val
   - GPU and disk state
3. Use `dp_best.pth` for any deployment test unless later evidence shows `dp_latest.pth` is better.
4. After the run stabilizes or finishes, compare:
   - this 260617-only model
   - previous 260609/260610 full-board model
   - previous plus-peg0617 mixed model
5. For TacQuality guidance paper/story, emphasize:
   - predicted tactile consequence scoring
   - contact-gated quality energy
   - trust-region action refinement
   - force-trace real rollout evaluation

## Live Training Update: 2026-06-19 05:36 CST

The active run is:

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619
```

The training process is still running in tmux session `dp260617_2000_20260619`.
The watcher session `watch260617_20260619` is active and writes status into
`training_watch_status.json`.

Latest parsed state from `train.log`:

| Item | Value |
|---|---:|
| Latest complete epoch | 144 / 2000 |
| Latest train loss | 0.008520 |
| Latest val loss | 0.016110 |
| Current best val loss | 0.010671 |
| Current best val epoch from log messages | 94 |
| Epochs since best | about 50 |
| Tail-20 val min / mean | 0.012857 / 0.015078 |
| Tail-20 train mean | 0.008623 |

Resource state:

- GPU: RTX 4090, about `14.7 GB / 24.6 GB`, high utilization during batches.
- External disk: about `2.2 TB` free.
- Home/root disk: about `45 GB` free; avoid writing large checkpoints to `/home`.

Interpretation:

- The run is healthy: no crash, no OOM, checkpoint writing works.
- Validation has not refreshed the epoch-94 best yet, while training loss keeps
  gradually decreasing.
- This is an early overfitting/plateau warning, but it is not strong enough to
  stop the 2000-epoch run yet because the no-improvement window is still around
  50 epochs and validation briefly recovered near epoch 135.
- Continue monitoring. If validation remains clearly above best for more than
  about 100-150 epochs, treat `dp_best.pth` as the deployable checkpoint and
  consider stopping if GPU is needed for a higher-priority experiment.

Checkpoint interpretation:

- `dp_best.pth`: primary candidate for robot testing and offline comparison.
- `dp_latest.pth`: debugging only unless it later becomes best.
- `dp_topk_*.pth`: selected by training loss, not validation loss; useful for
  diagnosing overfit behavior but not primary deployment checkpoints.

## Prioritized Architecture Improvements After This Survey

### Priority A: keep the main story narrow

The clearest current story is:

```text
RGB + current tactile + qpos
  -> DP behavior prior proposes a clean action chunk
  -> Foresight predicts future tactile consequence of that action
  -> TacQuality energy scores the predicted future contact
  -> trust-region accept-only gradient refinement updates the action
  -> real rollout force traces verify the effect
```

This should remain the main line. It is more defensible than saying "we concat
tactile into DP" because the tactile module has a causal role: it evaluates the
future consequence of the candidate action.

### Priority B: contact-gated guidance

Dream-Tac and AdaVTF both support the same design direction: tactile/force should
matter mainly during contact, not equally during approach. For board wiping:

- approach/no-contact: guidance near zero;
- stable wiping contact: guidance fully active;
- leaving/reset: guidance decays.

Implementation direction:

- use marker magnitude/contact area and, when available, Fz/force magnitude as
  contact gate inputs;
- multiply TacQuality gradient step size by this gate;
- log gate values per server-side rollout so real tests are auditable.

### Priority C: train scorer for guidance, not only classification

TouchGuide and ViTaL indicate that the scorer/verifier should be robust to the
distribution it sees during inference-time editing. Current scorer evidence is
good for clean/predicted-domain samples, but the next version should include:

- clean GT action/marker windows;
- noised action windows sampled with the DP scheduler;
- Foresight-predicted marker windows from noised/refined actions;
- calibration targets for both class accuracy and smooth scalar gradients.

The goal is not only high AUC. The goal is a scalar energy with useful local
gradients under the DP inference distribution.

### Priority D: force-conditioned Foresight / scorer for board wiping

TacForeSight and FAWAM both point to force as a first-class signal for contact
tasks. For board wiping, the quality definition already depends on force band
and force smoothness, so the most valuable next model upgrade is:

- Foresight input includes current force/torque if deployment provides it;
- predicted future includes marker latent plus a force proxy or force-band head;
- TacQuality energy is trained/evaluated against force-in-band ratio and
  derivative smoothness, not just mode labels.

This directly connects the model to the final real metric.

### Priority E: latent-action DP as a later model-capacity fix

The 260617-only dataset has only about 80 episodes. The current DP has a large
UNet and denoises raw joint chunks, so validation overfit is expected. Latent
Diffusion Policy suggests a later route:

- encode action chunks into a compact latent;
- diffuse in latent action space;
- decode to smooth joint chunks;
- apply TacQuality guidance either in latent space or through the decoded action
  with a small trust region.

This is a model change, so it should not interrupt the current run. It is a
second-stage improvement if the 260617-only raw-action DP keeps overfitting.

## Live Training Update: 2026-06-19 05:48 CST

The active run is still healthy and running:

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619
```

Latest parsed state:

| Item | Value |
|---|---:|
| Latest complete epoch | 170 / 2000 |
| Latest train loss | 0.007795 |
| Latest val loss | 0.015911 |
| Best val loss reported by training | 0.010671 |
| Epochs since best | about 76 |
| Latest val / best val | 1.491 |
| Tail-20 val mean | 0.016198 |
| Tail-20 train mean | 0.007868 |

Artifacts:

- `dp_best.pth` exists and remains the deployment candidate.
- `dp_latest.pth` exists but should not be used for robot testing unless it
  later becomes best.
- `loss_curve.png` was refreshed to epoch 164, with log already beyond that.

Current judgment:

- The process is technically healthy: no OOM, GPU busy, checkpoint files exist.
- The validation curve has not refreshed best for about 76 epochs, while train
  loss keeps decreasing.
- This is now a meaningful overfitting/plateau warning.
- Do not stop yet because the current no-improvement window is below the
  planned 100-150 epoch intervention threshold, but if validation remains above
  best through roughly epoch 200-240, the scientific choice is to keep
  `dp_best.pth` and stop the run if GPU is needed for scorer/Foresight work.

## Early Stop: 2026-06-19 06:03 CST

The run was stopped after the validation signal crossed the intervention
threshold.

Final status:

| Item | Value |
|---|---:|
| Last complete epoch | 197 / 2000 |
| Last train loss | 0.007725 |
| Last val loss | 0.019584 |
| Best val loss reported by training | 0.010671 |
| Best epoch reported/inferred | 94 |
| Epochs since best | 103 |
| Latest val / best val | 1.835 |
| Tail-20 val mean | 0.017727 |
| Tail-20 train mean | 0.007256 |

Artifacts:

- recommended checkpoint: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619/dp_best.pth`
- final loss curve: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619/loss_curve.png`
- early-stop summary: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619/early_stop_summary.json`

Decision:

- Training was technically healthy, but validation stayed far above the best
  while train loss continued decreasing.
- Continuing to epoch 2000 would mostly consume GPU time without improving the
  scientifically deployable checkpoint under the current evidence.
- Use `dp_best.pth` for offline/robot testing from this run.
- Do not use `dp_latest.pth` unless intentionally evaluating late-overfit
  behavior.
- This is only an offline validation decision. It does not prove real wiping
  quality; the remaining evidence still requires paired baseline/guided real
  rollouts with server-side force traces.

## Codex Continuation Check: 2026-06-19 06:38 CST

Current machine state was rechecked after the context handoff.

- GPU: no active DP/Foresight/TacQuality training process; only desktop
  processes on the RTX 4090.
- Active training process: none.
- Latest 260617-only DP run:
  - `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619`
- Final run artifacts confirmed:
  - `dp_best.pth`
  - `dp_latest.pth`
  - `dp_topk_ep182_loss0.0070.pth`
  - `dp_topk_ep194_loss0.0066.pth`
  - `dp_topk_ep195_loss0.0067.pth`
  - `loss_curve.csv`
  - `loss_curve.png`
  - `early_stop_summary.json`
- Recommended checkpoint remains:
  - `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619/dp_best.pth`

Scientific interpretation:

- This run should be treated as a useful 260617-only board DP baseline.
- The best checkpoint is validation-selected at epoch `94`.
- The late checkpoint is not better just because train loss is lower.
- The next decision should be based on paired robot tests or stronger offline
  action-quality audits, not on continuing this same training configuration.

## Refined Project Direction From Latest Literature

After rechecking the recent arXiv direction around 2026-06-19, the most
important method signal is consistent:

1. Keep the base DP as a nominal action generator.
2. Predict short-horizon tactile/force consequences before execution.
3. Score those predicted consequences with a task-grounded contact-quality
   energy.
4. Apply bounded, contact-gated, trust-region action refinement.
5. Prove final benefit with paired baseline/guided real rollouts and
   server-side force traces.

This means the strongest story is not simply "classifier guidance" or
"reranking". The stronger framing is:

`DP action chunk -> tactile/force foresight -> differentiable quality energy -> bounded contact-aware gradient refinement`.

Recent papers that support this framing:

- ViTaL / Inference-time Policy Steering via Vision and Touch (`2606.14981`):
  tactile-guided diffusion editing and future tactile verification are directly
  aligned with our goal.
- TacForeSight (`2606.11184`): force-conditioned tactile latent forecasting is
  a close match to the Foresight part of our system and suggests making force a
  first-class prediction/conditioning signal.
- Dream-Tac (`2606.08737`): contact-gated visuotactile world-action modeling
  supports using tactile guidance mainly during contact/wiping, not during the
  whole approach phase.
- Tube Diffusion Policy (`2604.23609`): contact-rich action chunking needs
  reactive correction, which supports adding a fast residual correction or
  execution-time update around DP chunks in later work.

Immediate engineering implications for this project:

- Keep current production tests on final clean-action trust-region guidance.
- Treat DDPM-step guidance as a research ablation until larger sweeps and robot
  evidence prove it is stable.
- For board wiping, make force-band and smoothness evidence primary:
  `force in target band`, `low force derivative/oscillation`, and `stable
  contact during wiping`.
- For insertion, keep the bad class tied to pre-bounce/bounce risk and evaluate
  by episode-level splits and real retry/bounce outcomes.
- The next model improvement with the best scientific value is a force-aware
  Foresight/scorer head, not a larger black-box classifier alone.
