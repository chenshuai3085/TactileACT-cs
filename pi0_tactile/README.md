# Pi0 Tactile

`pi0_tactile` is the pi0/pi0.5 integration layer for tactile-aware action
generation.  It adds frozen TactileVAE tokens to the pi0 prefix, trains an
action-conditioned tactile foresight auxiliary loss, and supports late flow-step
TacQuality guidance during serving.

The intended pi0.5 deployment contract is:

```text
vision + language + tactile history + state
  -> pi0.5 flow generates a full model action chunk
  -> first robot_action_dim dimensions are scored by tactile foresight
  -> score gradient updates only executable robot dimensions
  -> server executes only the robot action slice
```

This is flow-step gradient guidance, not candidate reranking.

## Files

- `config.py`: `Pi0TactileConfig`, including model action dimensions,
  TactileVAE/foresight paths, and flow guidance knobs.
- `dataset.py`: HDF5 dataset adapter that loads images, qpos, marker offsets,
  prompt placeholders, and normalized robot action chunks.
- `tactile_encoder.py`: frozen `TFAC_V5.tactile_vae.TactileVAE` encoder plus a
  trainable projection to pi0 action-expert token width.
- `foresight_module.py`: wrapper around the existing TFAC foresight transformer.
- `model.py`: pi0/pi0.5 model wrapper with tactile prefix tokens, foresight
  auxiliary loss, action padding/slicing, and guided sampling hooks.
- `guidance.py`: action adapter and late flow-step guidance implementation.
- `score_bridge.py`: differentiable
  `action -> foresight -> TactileVAE decoder -> TacQuality scorer` bridge.
- `serve.py`: WebSocket serving entrypoint with optional flow guidance.
- `test_flow_guidance.py`: local smoke tests for action padding and guidance.

## Action Dimensions

pi0.5 checkpoints commonly use a 32-D model action space.  The current robot,
foresight checkpoints, and TacQuality scorers use 7 executable joint dimensions.
This package separates those two contracts:

```python
config.action_dim = 32        # full pi0/pi0.5 model action dimension
config.robot_action_dim = 7   # executable/scored robot dimensions
config.pi05 = True
```

The model keeps the flow state in `action_dim`, but:

- dataset qpos/actions are loaded as `robot_action_dim`;
- training pads qpos/actions/noise to `action_dim` before pi0 forward;
- foresight and scorer consume only `[..., :robot_action_dim]`;
- guidance gradients are masked to the first `robot_action_dim` dimensions;
- padded dimensions receive zero guidance update and are not executed.

This matches the openpi `PadStatesAndActions` style while keeping our existing
7-D tactile models reusable.

## Flow Guidance

For pi0/pi0.5 flow matching, the clean action estimate at a denoising step is:

```text
x0_est = x_t - t * v_t
```

`Pi0FlowStepGuidance` applies this estimate only in the last configured flow
steps.  By default it treats `v_t` as a detached local clean-action estimator and
optimizes `x_t` with the TacQuality score gradient:

```text
action7_norm
  -> foresight_module
  -> TactileVAE decoder
  -> marker_raw sequence
  -> scorer.score(...)
  -> d score / d x_t
```

Serving safeguards include normalized gradients, a robot-dimension trust region,
optional clamp, second-difference smoothness penalty, finite-gradient checks,
and accept-only-if-improved updates.

## Minimal Training Shape

The dataset returns 7-D robot actions by default.  Padding to pi0.5's full action
space happens inside `Pi0Tactile`.

```python
from pi0_tactile.config import Pi0TactileConfig
from pi0_tactile.model import Pi0Tactile

config = Pi0TactileConfig(
    pi05=True,
    action_dim=32,
    robot_action_dim=7,
    action_horizon=20,
    vae_checkpoint="/path/to/tactile_vae.pt",
    foresight_checkpoint="/path/to/foresight_best.ckpt",
)

model = Pi0Tactile(config)
model.freeze_paligemma()
```

The training forward path returns:

```text
flow_loss
foresight_loss
total_loss = flow_loss + lambda_foresight * foresight_loss
```

## Serving

Minimal unguided serving:

```bash
python -m pi0_tactile.serve \
  --ckpt_dir /path/to/pi0_tactile_ckpt \
  --pi0_weights /path/to/pi0_weights \
  --robot_action_dim 7 \
  --num_flow_steps 10 \
  --port 8766
```

Guided serving:

```bash
python -m pi0_tactile.serve \
  --ckpt_dir /path/to/pi0_tactile_ckpt \
  --pi0_weights /path/to/pi0_weights \
  --robot_action_dim 7 \
  --num_flow_steps 10 \
  --flow_guidance_steps 2 \
  --flow_guidance_scale 0.02 \
  --flow_guidance_max_total_delta 0.08 \
  --flow_guidance_scorer_ckpt /path/to/tac_quality_scorer.pt \
  --flow_guidance_scorer_runtime force_band \
  --flow_guidance_score_mode energy_clipped \
  --send_guidance_report
```

Start with a conservative `flow_guidance_scale` and sweep it on offline logs
before robot rollout.

## Validation

Syntax and guidance smoke tests:

```bash
python -m py_compile \
  pi0_tactile/guidance.py \
  pi0_tactile/score_bridge.py \
  pi0_tactile/config.py \
  pi0_tactile/dataset.py \
  pi0_tactile/model.py \
  pi0_tactile/serve.py \
  pi0_tactile/test_flow_guidance.py

python -m pi0_tactile.test_flow_guidance
```

The smoke test verifies:

- 7-D robot actions pad to 32-D model actions;
- slicing recovers the original 7-D actions;
- guidance improves a differentiable score in a synthetic case;
- only robot dimensions change;
- trust-region bounds are respected.

## Known Limits

- `serve.py` still uses a placeholder ASCII prompt tokenizer.  Real pi0/pi0.5
  deployment should use the tokenizer matching the loaded openpi checkpoint.
- Full pi0.5 import/instantiation requires the openpi environment.  In this
  workspace the plain shell may miss openpi dependencies such as `flax` and
  `jax`.
- `score_bridge.py` expands a decoded single predicted tactile latent over the
  scorer window.  If a multi-step foresight checkpoint is used, add a true
  sequence decode path and run a checkpoint-specific smoke test.
- Guidance should be enabled only after action normalization, foresight
  checkpoint, scorer checkpoint, and robot action dimension are verified to
  match.
