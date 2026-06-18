# 2026-06-19 Insertion Matched-Foresight Guidance Audit

## Purpose

This audit refreshes the insertion TacQuality guidance evidence using Foresight checkpoints that match the current model architecture. The previous insertion guidance evidence used `latent_foresight_full`, which loaded with `100` missing keys from the visual backbone. That result is kept as historical caveat evidence, not the main insertion readiness proof.

## Insertion Scorer

- runtime: `InsertionRiskScorerRuntime`
- checkpoint: `/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt`
- score mode: `profile`
- offline data samples: `5600`
- binary AUC: `0.9877 +/- 0.0050`
- binary bACC: `0.9437 +/- 0.0163`
- reason macro F1: `0.7894 +/- 0.0154`
- quality corr: `0.7656 +/- 0.0385`

## Foresight Checkpoint Compatibility

| Foresight | dataset | state load result |
|---|---|---|
| `latent_foresight_full` | unknown / legacy | `100 missing / 0 unexpected` |
| `latent_foresight_0209` | `/home/chenshuai/data/dataset/0209-0210` | `0 missing / 0 unexpected` |
| `latent_foresight_0401` | `/home/chenshuai/data/dataset/260401_k14_truncated` | `0 missing / 0 unexpected` |

## Clean-Action Gradient Audit

| audit | Foresight dir | missing / unexpected | pass | improved | accept | score delta mean | action delta norm mean |
|---|---|---:|---|---:|---:|---:|---:|
| legacy full caveat | /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full | 100 / 0 | True | 1.0000 | 0.9688 | 0.266473 | 0.074925 |
| matched 0209 | /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_0209 | 0 / 0 | True | 0.9167 | 0.8438 | 0.106714 | 0.067100 |
| matched 0401 | /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_0401 | 0 / 0 | True | 1.0000 | 1.0000 | 0.140125 | 0.079864 |

Interpretation:

- Both matched Foresight audits pass without missing keys.
- `0401` is slightly cleaner on this audit: improved `1.0000`, accept `1.0000`.
- `0209` still passes, but some samples are not accepted because accept-only-improved rejects zero/non-improving refinements.

## Noisy-Action Audit

Each cell reports `final_minus_noisy`, and `clean` is `final_minus_clean`.

| audit | overall pass | Foresight missing / unexpected | per-noise behavior |
|---|---|---:|---|
| legacy full caveat | True | 100 / 0 | 0.0: +0.3883, clean 0.3883; 0.05: +0.0825, clean -0.0419; 0.1: +0.1037, clean -0.3165; 0.2: +0.0618, clean -0.8885; 0.4: +0.1043, clean -0.6241 |
| matched 0209 | True | 0 / 0 | 0.0: +0.1673, clean 0.1673; 0.05: +0.0704, clean 0.3946; 0.1: +0.0764, clean 0.3718; 0.2: +0.0726, clean -0.1435; 0.4: +0.0336, clean -0.1604 |
| matched 0401 | True | 0 / 0 | 0.0: +0.0702, clean 0.0702; 0.05: +0.0446, clean 0.2856; 0.1: +0.0514, clean 0.0794; 0.2: +0.0461, clean -0.2416; 0.4: +0.0356, clean -0.6690 |

The matched checkpoints pass all tested perturbation levels. As with board, this proves local score-gradient robustness around perturbed chunks, not full DDPM-step guidance or real robot improvement.

## Recommendation

Use matched insertion Foresight checkpoints for future insertion TacQuality guidance evidence:

- `latent_foresight_0401` is the preferred current audit checkpoint because it is fully compatible and has the cleanest gradient audit.
- `latent_foresight_0209` remains valid and compatible for older 0209 data distribution checks.
- Do not use `latent_foresight_full` as primary insertion evidence unless its missing visual-backbone keys are fixed or intentionally documented as a marker-only ablation.

## Evidence Boundaries

- Offline scorer metrics prove insertion risk/good-contact separability only.
- Matched Foresight gradient/noisy audits prove differentiable guidance readiness only.
- Final insertion claim still requires real baseline/guided rollouts with success, bounce/retry, and contact-quality outcomes.
