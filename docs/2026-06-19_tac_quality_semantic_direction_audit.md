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
- For insertion, `p_good` should be tested as the default guidance score mode.
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
   - test `p_good` as the guidance score mode in offline DDPM-step and real
     rollout smoke before changing deployment default;
   - keep `profile` as the conservative current setting until that smoke passes.
3. Board:
   - prefer the s12 scorer for the next offline/robot ablation candidate;
   - before deployment, rerun Foresight alignment and DDPM-step protected sweep
     with the s12 checkpoint as the active scorer.

Do not claim real robot improvement from this audit.  It proves only offline
semantic score geometry under saved data distributions.
