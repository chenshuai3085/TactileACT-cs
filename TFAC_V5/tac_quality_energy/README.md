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

## Model

The model uses a shared MLP encoder with five parallel heads:

```text
binary head   : good/bad logits
reason head   : quality reason logits
quality head  : continuous quality logit
teacher head  : RF-teacher soft P(good) distillation logit
energy head   : residual free energy
```

The deployed energy is:

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

- `model.py`: multi-head `DistilledTacQualityEnergy` architecture.
- `proxy_features.py`: differentiable marker/action proxy features.
- `runtime.py`: checkpoint-backed runtime with differentiable preprocessing.
- `trust_region.py`: accepted gradient-ascent update for DP action tensors.

## Current reference results

Manual-board mixed task GroupKFold:

```text
binary balanced acc     0.9070 +/- 0.0094
binary AUC              0.9757 +/- 0.0045
energy binary AUC       0.9745 +/- 0.0055
reason macro-F1         0.7881 +/- 0.0141
quality corr            0.7842 +/- 0.0115
teacher_pred_corr       0.9411 +/- 0.0044
```

Real-window action-gradient smoke:

```text
score mode       energy_clipped
step scale       0.5
insertion improve rate  0.9844
board improve rate      1.0000
```

## Minimal usage

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
