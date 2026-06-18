# 2026-06-19 Recent Tactile Guidance Research and Project Plan

## Scope

This note records the literature check requested while supervising the 260617-only board DP run.

Time window: roughly the last two months before 2026-06-19, with a few older-but-direct steering references kept as context.

Current project target:

```text
DP action proposal
  -> Foresight predicts future tactile consequence
  -> TacQuality scorer / energy evaluates predicted consequence
  -> bounded gradient guidance modifies the action
```

This is gradient guidance / energy guidance, not action reranking as the final mechanism.

## Training Supervision Status

260617-only board DP run:

- dataset: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617`
- run dir: `/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619`
- requested epochs: `2000`
- actual stop: `197/2000`, early-stopped by agent because validation had not improved for more than 100 epochs
- best validation loss: `0.010671` at epoch `94`
- last complete train/val: `0.007725 / 0.019584`
- recommended checkpoint: `dp_best.pth`
- do not use `dp_latest.pth` unless intentionally testing late-overfit behavior

Important caveat: `training_status_latest.json` is stale at epoch 13, but `train.log` and `early_stop_summary.json` show the real final state above.

## Current Local Evidence

Protected DDPM-step guidance evidence now uses step-level accept-only updates and final fallback.

Board wiping protected formal sweep:

- path: `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_20260619_protected_multiep6_start2_seed2_t0_s001/board_ddpm_step_guidance_sweep.json`
- eval points / rows: `12 / 24`
- final score improve rate: `1.0000`
- final score delta mean/min: `0.001762 / 0.000025`
- step/final accept: `1.0000 / 1.0000`
- action delta norm mean: `0.000253`

Insertion protected formal sweep:

- path: `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/insertion_0401_default_protected_multiep8_start2_seed2_t0_s001/insertion_ddpm_step_guidance_sweep.json`
- eval points / rows: `16 / 32`
- final score improve rate: `0.9375`
- final score delta mean/min: `0.000108 / 0.000000`
- step/final accept: `0.9375 / 1.0000`
- action delta norm mean: `0.000110`

Interpretation:

- The scorer is a valid local gradient source in the offline sampler chain.
- The safe region is small-scale, late-step or clean-action trust-region refinement.
- This does not prove real robot improvement. Paired baseline/guided rollouts with server-side force traces are still required.

## Recent Papers

### ViTaL: Inference-time Policy Steering via Vision and Touch

Source: arXiv `2606.14981`, 2026-06-12, <https://arxiv.org/html/2606.14981>

Key idea:

- Uses a visuo-tactile latent world model for outcome prediction.
- Uses visual verification for long-horizon mode selection.
- Uses tactile-guided diffusion editing for local contact refinement.
- Introduces a tactile verifier that scores predicted tactile latents against contact objectives.

Relevance to this project:

- This is the closest match to our current story.
- Their bi-level decomposition gives a clean explanation for our architecture:
  - DP/vision handles global trajectory and task mode;
  - Foresight+TacQuality handles local contact quality;
  - guidance should be short-horizon and bounded.
- Their method also supports our decision not to use aggressive multi-step unprotected DDPM guidance.

Actionable improvement:

- Rename and frame our method as task-specific tactile consequence guidance:
  `local tactile consequence energy guidance`.
- Add a phase-aware board scorer later:
  - approach phase: disable or weaken tactile quality guidance;
  - contact/wiping phase: enable force-band and smoothness guidance;
  - exit phase: disable force-band pressure guidance.
- Keep visual/global mode selection separate from tactile/local refinement.

### Dream-Tac: Unified Tactile World Action Model

Source: arXiv `2606.08737`, 2026-06-07, <https://arxiv.org/abs/2606.08737>

Key idea:

- Jointly models actions, future visual observations, and tactile dynamics.
- Adds contact-gated visuotactile fusion and contact-aware attention bias.
- Uses acceleration/caching for real-time deployment.

Relevance:

- Confirms that predicting tactile future is now a central direction, not a side experiment.
- Our Foresight is a smaller task-specific version of this idea.

Actionable improvement:

- Upgrade Foresight from marker-only latent prediction toward joint prediction:
  - tactile marker latent;
  - force proxy or force-band class;
  - action delta;
  - optional visual latent for global consistency.
- Add contact-gated loss/conditioning so non-contact frames do not dominate training.

### FTP-1: Generalist Foundation Tactile Policy

Source: arXiv `2606.13102`, 2026-06-11, <https://arxiv.org/abs/2606.13102>

Key idea:

- Unifies heterogeneous tactile inputs through morphology-aware tactile tokens.
- Pretrains a shared tactile Transformer expert across many sensors and embodiments.

Relevance:

- Our marker offset, tactile images, force traces, and joint-action proxies are currently task-specific.
- If the project expands to board wiping, writing, card swiping, insertion, and fragile grasping, a unified tactile token representation becomes important.

Actionable improvement:

- Build a small local `TacToken` interface:
  - marker proxy tokens: magnitude, area, center, spread, temporal derivative;
  - raw tactile latent tokens from TactileVAE;
  - force tokens when force traces are available;
  - action/proprio tokens.
- This keeps the scorer extensible without needing FTP-1-scale data.

### Multi-Resolution Tactile Imitation Learning

Source: arXiv `2606.06281`, 2026-06-04, <https://arxiv.org/html/2606.06281v1>

Key idea:

- Combines visual tactile and high-frequency event tactile streams.
- Uses transformer fusion and a flow-matching policy.
- Evaluates contact-rich tasks including board wiping and insertion-like tasks.

Relevance:

- Board wiping quality depends on pressure magnitude and fast pressure variation.
- Our marker-offset history already contains some of this, but the current scorer should explicitly expose temporal features.

Actionable improvement:

- Add short-window temporal features to board quality scoring:
  - marker magnitude mean/std;
  - marker velocity and acceleration;
  - force or force-proxy derivative;
  - in-band ratio over the contact phase.
- For future data collection, store force traces synchronized with marker frames for every real rollout.

### Tube Diffusion Policy

Source: arXiv `2604.23609`, 2026-04-26, <https://arxiv.org/abs/2604.23609>

Key idea:

- Learns a reactive visual-tactile feedback flow around nominal action chunks.
- Addresses the weakness of pure action chunking in contact-rich settings.

Relevance:

- Our DP predicts chunks, and deployment executes a short horizon.
- Contact-rich tasks may need fast correction inside a chunk, especially wiping pressure and insertion bounce avoidance.

Actionable improvement:

- Keep current DP as nominal chunk generator.
- Add a low-level residual/refinement layer later:
  - input: current tactile/force residual;
  - output: small bounded action correction;
  - trained or guided by TacQuality energy.
- This can be framed as a reactive layer rather than replacing the whole DP.

### DPTG: Diffusion Policy with Tactile Feasibility Guidance

Source: Frontiers in Robotics and AI, 2026-06, <https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1851102/full>

Key idea:

- Treats tactile sensing as a physical feasibility constraint.
- Uses a tactile feasibility classifier to guide actions sampled from a vision-driven diffusion policy.
- Uses an adaptive guidance schedule so constraints are active when contact is informative.

Relevance:

- This is close to our TacQuality energy route.
- Their framing supports our scorer as a feasibility/quality constraint instead of a replacement policy.

Actionable improvement:

- Add an explicit `contact_informative_gate`:
  - off before contact;
  - on during wiping / insertion contact;
  - weakened near reset/exit.
- Log when guidance was active in robot trials.

### Older but Directly Relevant

TouchGuide, arXiv `2601.20239`, <https://arxiv.org/abs/2601.20239>

- Inference-time tactile steering with a contact physical model and contrastive feasibility score.
- Useful as a baseline story, but our route is different because we explicitly predict future tactile consequence and use outcome quality labels/energy.

PPGuide, arXiv `2603.10980`, <https://arxiv.org/abs/2603.10980>

- Classifier-based diffusion policy guidance using a performance predictor.
- Useful for explaining why a differentiable predictor/scorer can steer a frozen DP.
- Less tactile-specific than our method.

AdaVTF, arXiv `2604.01414`, <https://arxiv.org/abs/2604.01414>

- Adaptive vision/torque fusion for contact-aware manipulation.
- Supports phase/contact-dependent fusion and gating.

pi0.7 CFG discussion, secondary source: <https://hyper.ai/en/papers/pi07>

- Relevant because classifier-free guidance is used to steer behavior from conditional/unconditional policy differences.
- Our current implementation is classifier/energy guidance, not CFG.

## Project Story After This Survey

The cleanest story is:

```text
Contact-rich DP is strong at generating plausible action chunks,
but it does not explicitly know whether the future contact will be good.

We learn a tactile consequence model that predicts future tactile response
from current observation and candidate action.

We learn a task quality energy on predicted tactile response:
  insertion: avoid pre-bounce/bounce contact;
  board wiping: keep contact pressure in range and smooth.

At inference, we do not rerank only.
We backpropagate the quality energy through Foresight into the action chunk,
using a trust region and accept/fallback protection.
```

Novelty relative to nearby work:

- More task-grounded than TouchGuide contrastive matching because quality labels are tied to bad physical outcomes.
- More lightweight than full Dream-Tac / ViTaL because the first version predicts tactile consequences only.
- More deployable than unbounded classifier guidance because guidance is contact-gated, trust-region bounded, and accept-only.
- More physically interpretable for board wiping because force-band and smoothness metrics are explicit.

## Recommended Next Improvements

Priority 1: Real rollout evidence

- Run matched baseline/guided board trials.
- Save server-side force traces under `baseline/` and `guided/`.
- Report force-in-band ratio, too-low ratio, too-high ratio, force derivative, and coverage/task completion.

Priority 2: Force-aware board scorer

- Current board scorer uses marker/action proxies.
- Add force trace supervision from real robot tests.
- Train a scorer that predicts:
  - binary good/bad;
  - reason: too-small, too-large, oscillatory;
  - continuous quality: in-band + smoothness.

Priority 3: Foresight output upgrade

- Keep precise latent/marker prediction.
- Add optional heads:
  - marker velocity/delta;
  - force proxy or force-band class;
  - contact gate.
- Do not add an extra quality loss unless it is computed from real labels or force traces. Avoid making the Foresight loss depend on a weak teacher.

Priority 4: Safer sampler guidance

- Default: final clean-action trust-region refinement.
- Research ablation: final DDPM step only, `guidance_steps=1`, `guidance_scale=0.001`, `max_delta_norm=0.005`, accept-only and final fallback on.
- Do not default to multi-step DDPM guidance until real rollouts and larger offline sweeps support it.

Priority 5: Reactive correction layer

- Add a small residual controller around DP chunks once force traces are available.
- This addresses the action-chunk reactivity issue highlighted by Tube Diffusion Policy.

## Current Do-Not-Claim Boundary

Do not claim:

- real board force improvement;
- real insertion success improvement;
- general tactile foundation-model capability;
- deployment-proven DDPM-step guidance.

Allowed claim:

- offline evidence shows TacQuality/Foresight can provide finite, bounded, score-improving local gradients under recorded observations;
- 260617-only DP has a usable best checkpoint, but robot performance must be measured separately.
