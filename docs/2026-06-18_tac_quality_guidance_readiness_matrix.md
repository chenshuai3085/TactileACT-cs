# 2026-06-18 TacQuality Guidance Readiness Matrix

## Scope

This note aligns the insertion and board-wiping TacQuality scorers under the same evidence standard:

```text
DP clean action
  -> task Foresight predicts future tactile consequence
  -> TacQuality scorer gives differentiable score
  -> trust-region gradient update on action
```

This is classifier/energy guidance on the clean action chunk. It is not reranking.

## Current Best Candidates

| task | candidate scorer | checkpoint | score mode | status |
|---|---|---|---|---|
| insertion | `InsertionRiskScorerRuntime` | `/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt` | `profile` config, runtime fallback to `energy_clipped` | current default candidate |
| board | `ForceBandTacQualityEnergyRuntime` with 260617 positive | `/home/chenshuai/Project/output/board_force_band_tac_quality_energy_with_260617_positive_20260618/force_band_tac_quality_energy_best.pt` | `quality` | best 260617-regime candidate; not yet promoted to final default without real force curves |

## Scorer Quality Evidence

| task | eval protocol | main metrics | evidence file |
|---|---|---|---|
| insertion | grouped CV over insertion windows | binary bACC `0.9437`, macro F1 `0.9364`, AUC `0.9877`, reason macro F1 `0.7894`, quality corr `0.7656` | `/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_eval.json` |
| board | grouped train/val with old positive + 260617 positive + too-small/too-large/oscillate negatives | held-out AUC `0.999929`, bACC `0.997172`, reason macro F1 `0.997304`, quality Spearman `0.971293` | `/home/chenshuai/Project/output/board_force_band_tac_quality_energy_with_260617_positive_20260618/train_result.json` |

Interpretation:

- Insertion has harder reason separation, but strong binary/risk signal and meaningful continuous quality correlation.
- Board has very strong regime separation after adding 260617 as positive, but continuous physical-force validation still requires real rollout curves.

## Foresight Gradient Guidance Evidence

| task | Foresight | samples | pass | finite grad | positive grad | improved | trust-region | score delta mean | action delta norm mean |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| insertion | `/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full/foresight_best.ckpt` | 24 | true | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.266473 | 0.074925 |
| board | `/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload/foresight_best.ckpt` | 24 | true | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000054 | 0.000799 |

Evidence files:

- insertion: `/home/chenshuai/Project/output/insertion_guidance_gradient_audit_real_foresight_profile_20260618/guidance_gradient_audit.json`
- board: `/home/chenshuai/Project/output/tac_quality_force_band_with260617_guidance_gradient_audit_quality/guidance_gradient_audit.json`

Interpretation:

- Both tasks have valid differentiable chains from action to predicted tactile score through Foresight.
- Board uses a much smaller action trust-region step, so score delta/action delta are much smaller by design.
- These audits prove gradient availability and bounded refinement, not real robot improvement.

## Server Entrypoint Smoke Evidence

| task | entrypoint | pass | scorer runtime | score mode | not reranking | evidence file |
|---|---|---|---|---|---|---|
| insertion | `for_show_xiaomi.serve_dp_tac_quality_guided` | true | `InsertionRiskScorerRuntime` | `profile` | true | `/home/chenshuai/Project/output/tac_quality_guided_server_packet/auto_discovered/insertion_guided_server_real_foresight_smoke.json` |
| board | `for_show_xiaomi.serve_dp_tac_quality_guided` | true | `ForceBandTacQualityEnergyRuntime` | `quality` | true | `/home/chenshuai/Project/output/tac_quality_guided_server_packet/with260617_scorer_real_foresight_smoke_20260618.json` |

Interpretation:

- Both task entrypoints can load DP, Foresight, scorer, and run final clean-action trust-region guidance.
- Board smoke used a temporary rollout config that points to the new with-260617-positive scorer:
  - `/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_with260617_scorer_tmp.json`
- The permanent rollout config should not be promoted until real force-curve testing confirms benefit.

## Current DP Training Context

The 260617-only board DP is still training:

- run dir: `/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000`
- current observation: around epoch `575/2000`
- best remains epoch `105`, `val=0.011152`
- latest validation remains much worse than best

Deployment/testing should therefore use:

```text
/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000/dp_best.pth
```

not `dp_latest.pth`.

## Remaining Evidence Gap

The goal is not fully complete because the final claim is real tactile consequence improvement during robot execution. The missing evidence is:

### Insertion

- baseline vs guided real rollouts;
- success rate, bounce count, retry count;
- safety/smoothness of the executed action trajectory.

### Board

- baseline vs guided real rollouts using the same DP checkpoint and same initial conditions;
- server-side per-trajectory force curves;
- contact-phase force-in-band ratio;
- force derivative/smoothness;
- marker smoothness;
- wipe coverage or task-completion proxy.

## Recommendation

For the next real test round:

1. Use insertion `default_guided` as the insertion guidance candidate.
2. Use board with the temporary with-260617-positive ForceBand config as the board guidance candidate.
3. Always collect matched baseline and guided trials.
4. Treat all current offline metrics as readiness evidence, not final performance evidence.

## Board Real-Rollout Command Packet

The copy-paste command sheet has been updated for the current board candidate:

- file: `for_show_xiaomi/guide_forshow.sh`
- baseline port: `8765`
- guided port: `8766`
- DP checkpoint: `/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000/dp_best.pth`
- guided scorer config: `/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_with260617_scorer_tmp.json`
- server-side rollout root: `/home/chenshuai/Project/output/board_force_rollouts/260617_only_with260617_scorer`

Expected rollout layout:

```text
/home/chenshuai/Project/output/board_force_rollouts/260617_only_with260617_scorer/
  baseline/<trial>/force_trace.csv
  baseline/<trial>/force_trace.npz
  baseline/<trial>/force_curve.png
  baseline/<trial>/metadata.json
  guided/<trial>/force_trace.csv
  guided/<trial>/force_trace.npz
  guided/<trial>/force_curve.png
  guided/<trial>/metadata.json
```

After real robot trials, run:

```bash
conda run --no-capture-output -n TactileACT python for_show_xiaomi/eval_board_force_rollouts.py \
  --root /home/chenshuai/Project/output/board_force_rollouts/260617_only_with260617_scorer \
  --tag board_260617_forceband_with260617
```

This will generate contact-phase force summaries under:

```text
/home/chenshuai/Project/output/board_force_rollout_eval/board_260617_forceband_with260617/
```

## Board Force Evaluation Metric Smoke

The real-rollout evaluator was extended to report explicit contact-phase force-quality metrics:

- force-in-band ratio;
- acceptable-force ratio;
- too-low / too-high ratio;
- absolute force error to target band center;
- force delta and jerk;
- smoothness score.

Validation command:

```bash
conda run --no-capture-output -n TactileACT python for_show_xiaomi/eval_board_force_rollouts.py \
  --root /tmp/board_force_eval_smoke \
  --output_dir /tmp/board_force_eval_smoke_out \
  --tag smoke_rerun \
  --force_band_center 8.5 \
  --force_band_sigma 1.0 \
  --force_accept_low 5.0 \
  --force_accept_high 12.0 \
  --force_smooth_delta_target 0.5
```

Smoke output:

- `/tmp/board_force_eval_smoke_out/smoke_rerun/board_force_rollout_summary.md`
- `/tmp/board_force_eval_smoke_out/smoke_rerun/board_force_rollout_group_summary.csv`
- `/tmp/board_force_eval_smoke_out/smoke_rerun/board_force_overview.png`
- `/tmp/board_force_eval_smoke_out/smoke_rerun/board_force_group_curves.png`

The synthetic smoke separates a bad low/unstable-force trace from a good stable-force trace:

| group | in-band | acceptable | too low | abs error | dF mean | smooth score |
|---|---:|---:|---:|---:|---:|---:|
| baseline smoke | 0.1429 | 0.4000 | 0.6000 | 4.5588 | 0.9895 | 0.1382 |
| guided smoke | 1.0000 | 1.0000 | 0.0000 | 0.1651 | 0.0255 | 0.9503 |

Interpretation:

- The evaluator can quantify exactly the board-wiping quality target: contact-phase force should be inside the desired band and smooth.
- This is only an evaluator smoke test, not a real robot guidance result.
- The real claim still requires matched baseline/guided robot trials saved under `/home/chenshuai/Project/output/board_force_rollouts/260617_only_with260617_scorer`.
