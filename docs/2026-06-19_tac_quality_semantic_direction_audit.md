# 2026-06-19 TacQuality Semantic Direction Audit

## Goal

The goal is to check whether the current tactile quality scorers are suitable
as DP classifier/energy guidance potentials, not just whether they classify
good/bad samples correctly.

For guidance, a good scorer should satisfy two levels:

1. Global semantic ordering:
   bad-contact samples should get lower scores than good-contact samples.
2. Local semantic gradient:
   near a bad-contact sample, the score gradient should point toward a
   physically better contact mode.

Existing local score-landscape checks proved that moving along the raw gradient
usually increases the score.  This audit adds the missing question: does that
score increase correspond to the intended physical semantics?

## Method

Script:

- `TFAC_V5/tac_quality_energy/eval_semantic_guidance_direction.py`

Command:

```bash
python TFAC_V5/tac_quality_energy/eval_semantic_guidance_direction.py \
  --device cuda:0 \
  --max_samples_per_class 192
```

Outputs:

- `/home/chenshuai/Project/output/tac_quality_semantic_direction_audit/tac_quality_semantic_direction_audit.json`
- `/home/chenshuai/Project/output/tac_quality_semantic_direction_audit/tac_quality_semantic_direction_audit.md`
- `/home/chenshuai/Project/output/tac_quality_semantic_direction_audit/tac_quality_semantic_direction_audit_rows.csv`
- `/home/chenshuai/Project/output/tac_quality_semantic_direction_audit/insertion_p_good_bad_to_good_curves.png`
- `/home/chenshuai/Project/output/tac_quality_semantic_direction_audit/insertion_profile_bad_to_good_curves.png`
- `/home/chenshuai/Project/output/tac_quality_semantic_direction_audit/board_default_quality_bad_to_good_curves.png`
- `/home/chenshuai/Project/output/tac_quality_semantic_direction_audit/board_s12_quality_bad_to_good_curves.png`

Protocol:

- Insertion:
  - good: `good_insert`
  - bad: `pre_bounce_risk`, `impact_or_recovery`
  - excluded: `weak_approach`, because it is not the user's bounce/failure definition.
- Board:
  - good: `positive`
  - bad: `too_small`, `too_large`, `oscillate`

For each bad mode:

- `bad_to_good`: interpolate real bad samples toward the good centroid.
  The score should increase.
- `good_to_bad`: interpolate real good samples toward the bad centroid.
  The score should decrease.
- Gradient projection: compute whether the local scorer gradient has the
  correct dot-product sign with the semantic direction.

## Results

### Summary

| task | deployed mode | semantic recommended mode | correction pass | strict pass |
|---|---|---|---:|---:|
| insertion | `profile` | `p_good` | true | false |
| board default 20260618 | `quality` | `quality` | false | false |
| board s12 20260619 | `quality` | `quality` | true | false |

`correction pass` focuses on bad-to-good repair gradients.  This is the most
important condition for DP guidance, because deployment uses small trust-region
updates plus accept/fallback protection.

`strict pass` additionally requires good-to-bad protection gradients to pass in
all modes.  None of the current scorers fully passes strict mode, so real
deployment should keep accept-only/fallback protection.

### Score Mode Comparison

| task | best mode | curve pass | bad-to-good gradient pass | strict gradient pass | bad-to-good projection mean |
|---|---|---:|---:|---:|---:|
| insertion | `p_good` | 1.0000 | 1.0000 | 0.7500 | 0.8255 |
| board default 20260618 | `quality` | 1.0000 | 0.6667 | 0.5000 | 0.6354 |
| board s12 20260619 | `quality` | 1.0000 | 1.0000 | 0.6667 | 0.7969 |

Important details:

- Insertion `p_good` is better than the deployed `profile` mode for semantic
  direction:
  - `p_good` bad-to-good projection mean: `0.8255`
  - `profile` bad-to-good projection mean: `0.6797`
  - `p_good` also has better protection gradient rate.
- Board default 20260618 scorer has correct global score curves, but fails
  bad-to-good correction for at least one bad mode.  It is not ideal as the
  long-term board guidance potential.
- Board s12 20260619 scorer has the best board semantic guidance geometry:
  - curve pass: `1.0000`
  - bad-to-good correction gradient pass: `1.0000`
  - bad-to-good projection mean: `0.7969`

## Interpretation

The key finding is not that the current scorers are unusable.  It is more
specific:

- The global classification/ordering semantics are strong.
- The current deployed guidance modes are not always the best local gradient
  potentials.
- For insertion, the first semantic audit made `p_good` look promising, but
  follow-up DDPM-step and cross-score tests show bounded `p_good` saturates;
  the next insertion A/B candidate should use unsaturated `good_margin`
  instead.
- For board, the s12 scorer with `quality` mode is better aligned with
  bad-to-good semantic correction than the older 20260618 default scorer.

This directly affects DP classifier guidance:

```text
DP action -> Foresight predicted tactile consequence -> score -> gradient
```

If the score mode has good global accuracy but weak semantic gradient
projection, it can still be risky for gradient guidance.  Therefore score mode
selection should use semantic direction audits, not only AUC/F1/Spearman.

## Recommendation

Current conservative recommendation:

1. Keep accept-only and final fallback enabled for all guidance.
2. Insertion:
   - `p_good` looked better in the semantic direction audit, but the follow-up
     DDPM-step sweep shows it saturates at score 1.0 and gives zero sampler
     improvement;
   - keep `profile` as the current conservative protected DDPM-step score mode
     until the next real A/B test is explicitly switched;
   - the follow-up score-mode ablation below shows that `good_margin` is a
     better next insertion A/B candidate than bounded `p_good`, because it is
     an unsaturated logit margin and does not degrade the other scorer heads in
     the protected offline sweep.
3. Board:
   - prefer the s12 scorer for the next offline/robot ablation candidate;
   - the follow-up protected DDPM-step sweep with s12 passed and produced a
     larger mean score gain than the old/default board scorer.

Do not claim real robot improvement from this audit.  It proves only offline
semantic score geometry under saved data distributions.

## Follow-Up DDPM-Step Sweeps

After the semantic audit, two protected DDPM-step sweeps were run with the
recommended score/scorer candidates.

### Insertion p_good

Path:

- `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/insertion_0401_p_good_protected_multiep8_start2_seed2_t0_s001/insertion_ddpm_step_guidance_sweep.json`

Result:

| mode | eval points | rows | improve | mean delta | min delta | step accept | final accept | action norm mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `p_good` | 16 | 32 | 0.0000 | 0.000000 | 0.000000 | 1.0000 | 1.0000 | 0.000112 |

Interpretation:

- `p_good` is semantically well ordered, but in the matched DDPM/Foresight
  chain it is saturated at or near 1.0.
- The gradient is finite but the scored objective has no measurable room to
  improve.
- Therefore `p_good` should not replace insertion `profile` for current
  DDPM-step guidance.

### Insertion score-mode cross-score ablation

Path:

- `/home/chenshuai/Project/output/tac_quality_score_mode_ablation/insertion_0401_profile_pgood_energy_goodmargin_cross_score_20260619/insertion_score_mode_ablation.json`

Protocol:

- guidance modes: `profile`, `p_good`, `energy`, `good_margin`
- eval modes: `profile`, `p_good`, `log_p_good`, `energy`,
  `good_margin`, `quality_logit`, `neg_risk`
- dataset: `/home/chenshuai/data/dataset/260401_k14_truncated`
- DP: `/home/chenshuai/Project/output/ckpt/dp_tac_concat_02090210/dp_final.pth`
- Foresight: `/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_0401/foresight_best.ckpt`
- evidence boundary: offline sampler/cross-score evidence only

Result:

| guidance mode | rows | final accept | action norm | own delta | own improve | profile delta | energy delta | good_margin delta | quality_logit delta | min quality delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `profile` | 32 | 1.0000 | 0.000110 | 0.000108 | 0.9375 | 0.000108 | 0.000500 | 0.001160 | 0.000768 | -0.000054 |
| `p_good` | 32 | 1.0000 | 0.000112 | 0.000000 | 0.0000 | 0.000094 | 0.000379 | 0.001245 | 0.000508 | -0.000584 |
| `energy` | 32 | 1.0000 | 0.000110 | 0.000500 | 0.9375 | 0.000108 | 0.000500 | 0.001160 | 0.000768 | -0.000054 |
| `good_margin` | 32 | 1.0000 | 0.000102 | 0.001154 | 0.9375 | 0.000093 | 0.000394 | 0.001154 | 0.000557 | 0.000000 |

Interpretation:

- `p_good` is saturated and gives zero own-score improvement.
- `energy` avoids saturation and matches the conservative `profile` behavior.
- `good_margin` gives the largest own-score gain, keeps action updates slightly
  smaller than `profile`, and has non-negative `profile`, `energy`,
  `good_margin`, `quality_logit`, and `neg_risk` deltas in this protected
  sweep.
- Therefore the next insertion A/B candidate should be the unsaturated
  `good_margin` score mode, not the bounded `p_good` probability.
- This is still not a real robot success/bounce claim.

### Board s12

Path:

- `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_s12_260617_20260619_protected_multiep6_start2_seed2_t0_s001/board_ddpm_step_guidance_sweep.json`

Result:

| scorer | eval points | rows | improve | mean delta | min delta | step accept | final accept | action norm mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `board_s12 quality` | 12 | 24 | 1.0000 | 0.003919 | 0.000013 | 1.0000 | 1.0000 | 0.000227 |

Comparison with old/default protected board sweep:

| scorer | mean delta | min delta | action norm mean |
|---|---:|---:|---:|
| old/default marker_joint | 0.001762 | 0.000025 | 0.000253 |
| s12 marker_joint | 0.003919 | 0.000013 | 0.000227 |

Interpretation:

- s12 has stronger semantic bad-to-good geometry and stronger protected
  sampler score gain.
- It is now the better board A/B candidate.
- This is still offline sampler evidence, not real board force improvement.
