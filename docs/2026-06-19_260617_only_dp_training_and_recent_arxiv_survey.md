# 260617-only Board DP Training and Recent ArXiv Survey

Date: 2026-06-19

## Scope

This note records the 260617-only board DP training run and the June/May 2026 literature scan relevant to the current project story:

`DP action proposal -> Foresight tactile consequence prediction -> TacQuality score/energy -> bounded diffusion-step gradient guidance`

The goal is not reranking. The target is a differentiable tactile/force quality scorer that can guide denoising toward actions with better predicted tactile consequences.

## 1. 260617-only DP Training

### Dataset

- Dataset: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617`
- Note: the requested root `/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban` currently contains this single HDF5 episode directory. It has `80` HDF5 episodes. The subdirectory name is not used as the task semantics by the trainer; the run used the current board/raw-image/tactile DP configuration.
- Run directory: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619`
- Training script: `diffusion/train_dp_tac_concat.py`
- Requested epochs: `2000`
- Actual status: stopped by agent at epoch `197/2000` after validation plateau/overfit evidence.

### Main Configuration

- Cameras: `global,wrist`
- Proprio: `proprio_joint`
- Action: `actions/joint_abs`
- Tactile side: `left`
- Tactile history: `8`
- Frozen tactile VAE:
  `/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt`
- Image input: cached raw image resized/cropped to `200x266`
- Prediction horizon: `16`
- Observation horizon: `2`
- Action horizon: `8`
- Batch size: `64`
- LR: `1e-4`
- Weight decay: `1e-6`
- Diffusion train/inference steps: `100 / 100`
- Down dims: `512,1024,2048`
- Train/val windows: `8192 / 1024`
- Val ratio: `0.1`
- Seed: `1`

Full reproduced command is saved at:

`/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619/run_command.sh`

### Results

- Best checkpoint:
  `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619/dp_best.pth`
- Latest checkpoint:
  `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619/dp_latest.pth`
- Best validation loss: `0.010671`
- Best epoch: `94`
- Last complete epoch: `197`
- Last train/val loss: `0.007725 / 0.019584`
- Tail-20 validation mean: `0.017727`
- Loss curve:
  `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619/loss_curve.png`
- CSV:
  `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619/loss_curve.csv`
- Early stop summary:
  `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619/early_stop_summary.md`

### Interpretation

This run trained normally and reached its best validation point early. After epoch 94, the train loss kept improving while validation loss stayed above the best value and eventually rose to roughly `1.84x` the best validation loss. Continuing to epoch 2000 would likely improve train reconstruction/action noise prediction but hurt validation generalization.

Use `dp_best.pth` for offline or robot tests. Do not use `dp_latest.pth` unless intentionally evaluating late-overfit behavior.

This is an offline loss conclusion only. It does not prove real wiping improvement. Real claims still require paired baseline/guided robot rollouts with server-side force traces.

## 2. Recent ArXiv Scan: Last Two Months

### 2.1 ViTaL: Inference-time Policy Steering via Vision and Touch

- arXiv: `2606.14981`
- Date: 2026-06-12
- Link: https://arxiv.org/html/2606.14981

Core idea: a pretrained generative robot policy is adapted at inference time with two levels:

- visual long-horizon sampling and verification chooses the global behavior mode;
- tactile-guided diffusion editing refines the selected short action chunk for local contact quality.

The most relevant part for this project is that ViTaL scores predicted tactile futures directly in latent space and uses the gradient of the tactile reward to modify diffusion denoising. This is almost the same scientific lane as our current Foresight + TacQuality guidance, but their reward is language-conditioned and their tactile guidance is explicitly short-horizon.

Implication for our project:

- Keep our direction as gradient guidance, not post-hoc reranking.
- Separate global task progress from local tactile quality.
- For board wiping, tactile should guide local pressure/smoothness, not global wipe target selection.
- For insertion, tactile should guide local contact/alignment, while visual/proprio handles hole/pose progress.
- Use bounded diffusion editing/trust region, because direct high-noise gradients can be unstable.

### 2.2 Dream-Tac: Unified Tactile World Action Model

- arXiv: `2606.08737`
- Date: 2026-06
- Link: https://arxiv.org/html/2606.08737v1

Core idea: jointly predict future visual observations, future tactile signals, and robot actions in one world-action model, with contact-aware attention bias that emphasizes tactile signals during salient interaction events.

Implication for our project:

- Our current architecture is a cascaded version: DP proposes actions, Foresight predicts tactile future, TacQuality scores it.
- A stronger future version could jointly train action generation and tactile prediction, but that is a larger redesign.
- Near-term improvement: add contact-aware gating/attention to the tactile branch or quality scorer so tactile features matter most during contact phases.

### 2.3 Tube Diffusion Policy

- arXiv: `2604.23609`
- Date: 2026-04-29
- Link: https://arxiv.org/html/2604.23609v1

Core idea: standard action chunking is too open-loop for contact-rich tasks. Tube Diffusion Policy adds a learned feedback flow around a nominal diffusion action chunk, allowing local reactive corrections during execution.

Implication for our project:

- Our current DP action horizon is `8`; this is still chunk-based and partially open-loop.
- TacQuality guidance should be viewed as local correction around a DP action chunk, not as replacing DP.
- The trust-region limit in our guidance is scientifically important: it keeps refinement near the DP action manifold.
- For deployment, a future stronger version should combine DP chunk proposal with high-frequency tactile correction during contact.

### 2.4 TacForeSight: Force-Guided Tactile World Model

- arXiv: `2606.11184`
- Date: 2026-06-09
- Link: https://arxiv.org/abs/2606.11184

Core idea: a lightweight force-conditioned tactile foresight model predicts short-horizon tactile latent dynamics from tactile observations and high-frequency wrist force/torque. The policy then uses predicted tactile latents as anticipatory contact priors.

This is directly relevant to our current project because our system also uses a tactile foresight module before action execution. Their emphasis on force-conditioned tactile latent prediction suggests that board wiping should not rely only on marker latents; force/torque history should be part of either Foresight conditioning or TacQuality scoring.

Implication for our project:

- Treat Foresight as a central contribution, not just an auxiliary prediction model.
- Add or test force-conditioned Foresight for board wiping when reliable force traces are available.
- Keep the prediction target compact and short-horizon; long horizon should be evaluated carefully because errors compound.
- Use predicted tactile latents as anticipatory priors for guidance, while preserving bounded action updates.

### 2.5 ForceFlow

- arXiv: `2605.11048`
- Date: 2026-05-11
- Link: https://arxiv.org/html/2605.11048v1

Core idea: a force-aware flow matching policy uses force history as a persistent global regulatory signal and jointly predicts actions plus future force. The paper emphasizes that high-dimensional vision can mask low-dimensional force, so force/tactile should be injected in a way that cannot be ignored.

Implication for our project:

- Our board task target is exactly force regulation: proper force magnitude and smooth force change.
- The scorer should not only classify labels; it should preserve differentiable force-band and smoothness terms.
- Consider adding a joint auxiliary head to the policy or Foresight that predicts force/marker quality in addition to action.
- Flow matching is worth considering for a future policy backbone if DP sampling latency becomes a deployment bottleneck.

### 2.6 ContactWorld

- arXiv: `2606.13877`
- Date: 2026-06-11
- Link: https://arxiv.org/html/2606.13877v1

Core idea: contact-rich world models work best when representations are spatially structured and temporally continuous. The paper finds tactile helps most when cross-modal representation compatibility is good, not merely when more modalities are added.

Implication for our project:

- Our tactile marker field and TactileVAE latent are valuable because they preserve contact structure.
- Evaluation should include temporal continuity and long-horizon robustness, not only frame-level accuracy.
- The scorer/guidance evaluation should use episode-level splits and full-episode force curves to avoid leakage and overclaiming.

### 2.7 AT-VLA

- arXiv: `2605.07308`
- Date: 2026-05-08
- Link: https://arxiv.org/html/2605.07308v1

Core idea: tactile information should be injected adaptively, mostly during contact, with a fast tactile stream and slower visual-language stream. Directly adding tactile tokens can hurt pretrained visual reasoning.

Implication for our project:

- For our non-foundation DP, the same principle still applies: tactile guidance should be phase/contact-gated.
- Do not force TacQuality guidance during approach or low-contact segments.
- Use contact strength / marker magnitude / predicted contact probability as a gate for enabling guidance.

### 2.8 World Action Models Survey

- arXiv: `2605.12090`
- Date: 2026-05-12
- Link: https://arxiv.org/abs/2605.12090

Core idea: World Action Models unify future state prediction and action generation, instead of learning reactive observation-to-action mappings.

Implication for our project:

- Our paper story fits the WAM framing if described carefully:
  - baseline DP is reactive action generation;
  - Foresight adds predicted tactile consequences;
  - TacQuality turns predicted consequences into an energy/reward;
  - gradient guidance closes the loop at inference time.
- We should call the current system a cascaded tactile-consequence-guided policy, not a fully joint world-action model.

## 3. Current Project Architecture Assessment

### What is already strong

- The project has a clear contact-rich manipulation motivation: insertion bounce avoidance and board wiping force regulation.
- The tactile representation is structured: marker fields -> TactileVAE latent.
- Foresight provides an action-conditioned predicted tactile consequence.
- TacQuality gives a differentiable quality/energy score.
- Protected DDPM-step guidance already uses bounded updates and accept/fallback logic.

### Main weakness

The weakest link is not DP training itself. It is whether the predicted tactile future and TacQuality gradient remain semantically correct under deployment distribution shift.

Frame/window-level classifier accuracy is not sufficient. A useful guidance scorer must satisfy all of:

1. Good held-out classification/ranking under episode-level split.
2. Correct physical semantics: good force band, stable contact, low bounce risk.
3. Smooth unsaturated gradients with respect to predicted tactile/action.
4. Positive protected DDPM-step improvement under offline rollouts.
5. Real robot paired A/B evidence with force traces.

## 4. Recommended Story Upgrade

The clean story should be:

> Contact-rich robot policies fail because actions are generated before their contact consequences are known. We learn an action-conditioned tactile foresight model and a differentiable tactile quality energy. During DP denoising, candidate actions are edited within a trust region toward predicted tactile futures with better contact quality. This gives diffusion policy a lightweight tactile consequence awareness without retraining the whole policy.

This is novel enough relative to current work because:

- unlike normal tactile-conditioned DP, tactile is used to evaluate predicted outcomes before execution;
- unlike reranking, the score is used as a gradient in the denoising process;
- unlike pure visual world models, the reward is contact-quality-specific;
- unlike direct classifier guidance, the system uses physical safety gates, trust region, and fallback.

## 5. Concrete Next Improvements

### P0: Real A/B force-trace evaluation

Run matched baseline/guided board trials and save server-side force traces per trajectory. This is required before claiming physical improvement.

Metrics:

- mean effective contact force during wiping;
- force-band occupancy ratio;
- jerk/smoothness of force;
- contact dropout ratio;
- task completion / wipe coverage if available;
- safety violations.

### P1: Contact-gated guidance

Enable guidance only when predicted or observed marker/contact proxy exceeds a threshold. This follows the adaptive tactile injection idea and avoids irrelevant gradients during approach.

### P2: Unsaturated score mode

For insertion, avoid saturated `p_good` probability. Use calibrated logit/margin/energy instead so DDPM-step gradients do not vanish.

### P3: Structured temporal scorer

For board, keep the force-band/smoothness objective and extend it to a temporal scorer over predicted tactile windows:

`quality = force_band + contact_stability + low_dforce + low_marker_jerk + no_dropout`

This is more appropriate than pure binary good/bad classification because it provides smoother gradients.

### P4: Future joint training

Longer-term, move from cascaded DP + Foresight toward a joint model that co-predicts action and tactile future, or add auxiliary force/tactile prediction heads inside the policy. This matches recent WAM and ForceFlow trends but should come after current guidance is verified.

## 6. Immediate Recommendation

For the current 260617-only DP:

- Use `dp_best.pth` from epoch 94.
- Do not continue this run to 2000 because validation already plateaued and overfit signs are clear.
- Evaluate it on robot with and without TacQuality guidance.
- Save all force traces on the server side.
- For the paper/story, emphasize tactile consequence-guided denoising, not only tactile-conditioned imitation learning.
