# TacQualityEnergy

`tac_quality_energy` is the reusable scorer module for tactile quality guidance.
It keeps only the core design, not the temporary experiment/gate scripts.

## Purpose

The scorer evaluates predicted or observed tactile consequences and produces a
differentiable quality energy.  In DP inference, the intended chain is:

```text
candidate action
  -> Foresight predicts future tactile marker field
  -> TacQualityEnergy scores tactile consequence
  -> d score / d action guides a small trust-region refinement
```

This is classifier/scorer guidance, not reranking.

## Current Deployed Scorers

The current rollout defaults and research priority are task-specific runtimes
selected by the latest audit:

- socket insertion: `InsertionRiskScorerRuntime`, `score_mode=good_margin`.
  This uses the unsaturated good-vs-risk binary logit margin rather than the
  saturated `p_good` probability.
- board wiping: `ForceBandTacQualityEnergyRuntime`,
  `score_mode=quality`, with the `marker_joint_action` feature variant from the
  s12 force-band scorer.

The older distilled/manual-board scorer is kept as an ablation, not the current
default.

## Distilled Ablation Model

`DistilledTacQualityEnergy` uses a shared MLP encoder with five parallel heads:

```text
binary head   : good/bad logits
reason head   : quality reason logits
quality head  : continuous quality logit
teacher head  : RF-teacher soft P(good) distillation logit
energy head   : residual free energy
```

Its fused energy is:

```text
energy =
  0.45 * quality_logit
+ 0.30 * teacher_logit
+ 0.15 * good_margin
+ 0.10 * reason_margin
+ 0.10 * free_energy

energy_clipped = tanh(energy / 4.0) * 4.0
```

## Files

- `model.py`: multi-head `DistilledTacQualityEnergy` ablation architecture.
- `proxy_features.py`: differentiable marker/action proxy features.
- `runtime.py`: checkpoint-backed runtime with differentiable preprocessing.
- `ptg_proxy_runtime.py`: older board PTGProxyScorerV2 ablation runtime.
- `insertion_runtime.py`: insertion default risk scorer runtime.
- `force_band_runtime.py`: board ForceBand runtime; current board candidate uses
  `marker_joint_action` deploy features with `score_mode=quality`.
- `foresight_bridge.py`: differentiable `raw action -> Foresight -> marker` bridge.
- `serving_guidance.py`: serving-time adapter for clean-action trust-region guidance.
- `trust_region.py`: accepted gradient-ascent update for DP action tensors.

## Serving Contract

The formal guided server uses this package through:

```text
DP denoising produces clean action chunk
  -> action is denormalized to raw joint units
  -> Foresight predicts future tactile marker consequence
  -> scorer energy is differentiated w.r.t. action
  -> trust-region refinement accepts only improving bounded updates
```

The package currently supports:

- `InsertionRiskScorerRuntime` for current insertion guidance
  (`score_mode=good_margin`, the unsaturated binary logit margin selected by
  the matched DDPM/Foresight score-mode ablation).  The older `profile` mode
  remains available as a conservative ablation.
- `ForceBandTacQualityEnergyRuntime` for the current board guidance candidate
  (`marker_joint_action`, `score_mode=quality`).
- `PTGProxyScorerV2Runtime` for the older board proxy-scorer ablation.
- `DistilledTacQualityEnergyRuntime` as the cross-task distilled ablation.

## Minimal usage

Current board scorer:

```python
import torch
from TFAC_V5.tac_quality_energy import ForceBandTacQualityEnergyRuntime

runtime = ForceBandTacQualityEnergyRuntime(
    "/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/force_band_tac_quality_energy_best.pt",
    device="cuda:0",
)

left_marker = torch.randn(8, 8, 9, 9, 2, device=runtime.device, requires_grad=True)
joint_action = torch.randn(8, 8, 7, device=runtime.device, requires_grad=True)
score = runtime.score(left_marker, joint_action_seq=joint_action, mode="quality")
grad = torch.autograd.grad(score.sum(), joint_action)[0]
```

Current insertion scorer:

```python
import torch
from TFAC_V5.tac_quality_energy import InsertionRiskScorerRuntime

runtime = InsertionRiskScorerRuntime(
    "/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt",
    device="cuda:0",
)

left_marker = torch.randn(8, 8, 9, 9, 2, device=runtime.device, requires_grad=True)
joint_action = torch.randn(8, 8, 7, device=runtime.device, requires_grad=True)
score = runtime.score(left_marker, joint_action_seq=joint_action, mode="good_margin")
grad = torch.autograd.grad(score.sum(), joint_action)[0]
```

Distilled ablation scorer:

```python
import torch
from TFAC_V5.tac_quality_energy import DistilledTacQualityEnergyRuntime

runtime = DistilledTacQualityEnergyRuntime(
    "/home/chenshuai/Project/output/manual_board_tac_quality_energy/distilled_tac_quality_energy_final.pt",
    device="cuda:0",
)

left_marker = torch.randn(8, 8, 9, 9, 2, device=runtime.device, requires_grad=True)
joint_action = torch.randn(8, 8, 7, device=runtime.device, requires_grad=True)
task_id = torch.zeros(8, dtype=torch.long, device=runtime.device)

score = runtime.score(
    left_marker,
    joint_action_seq=joint_action,
    task_id=task_id,
    mode="energy_clipped",
)
grad = torch.autograd.grad(score.sum(), joint_action)[0]
```
