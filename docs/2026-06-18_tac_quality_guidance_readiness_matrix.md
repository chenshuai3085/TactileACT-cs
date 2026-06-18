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
| board | `ForceBandTacQualityEnergyRuntime`, feature `marker_joint_action` | `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/force_band_tac_quality_energy_best.pt` | `quality` | current board guidance candidate; deploy-aligned feature, Foresight-chain score no longer saturated |

## Scorer Quality Evidence

| task | eval protocol | main metrics | evidence file |
|---|---|---|---|
| insertion | grouped CV over insertion windows | binary bACC `0.9437`, macro F1 `0.9364`, AUC `0.9877`, reason macro F1 `0.7894`, quality corr `0.7656` | `/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_eval.json` |
| board | grouped train/val over predicted-domain deploy features: predicted marker proxy + candidate joint-action proxy | held-out AUC `0.9997`, bACC `0.9828`, reason macro F1 `0.9703`, quality Spearman `0.9239` | `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/train_result.json` |

Interpretation:

- Insertion has harder reason separation, but strong binary/risk signal and meaningful continuous quality correlation.
- Board marker-joint scorer is slightly weaker than the old feature-leaking/eef-assisted classifier on held-out metrics, but it matches the real serving contract: the scorer sees Foresight-predicted marker plus candidate joint action, not future ground-truth `eef_abs`.

## Foresight Gradient Guidance Evidence

| task | Foresight | samples | pass | finite grad | positive grad | improved | trust-region | score delta mean | action delta norm mean |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| insertion | `/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full/foresight_best.ckpt` | 24 | true | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.266473 | 0.074925 |
| board | `/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload/foresight_best.ckpt` | 24 | true | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.001320 | 0.000791 |

Evidence files:

- insertion: `/home/chenshuai/Project/output/insertion_guidance_gradient_audit_real_foresight_profile_20260618/guidance_gradient_audit.json`
- board: `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/guidance_gradient_audit_quality/guidance_gradient_audit.json`

Interpretation:

- Both tasks have valid differentiable chains from action to predicted tactile score through Foresight.
- Board uses a much smaller action trust-region step, so score delta/action delta are much smaller by design.
- The older with-260617 scorer passed finite-gradient checks but had saturated Foresight-chain scores. The current marker-joint scorer fixes that deployment mismatch and has a larger, better-ordered `quality` score in the real Foresight chain.
- These audits prove gradient availability and bounded refinement, not real robot improvement.

## Server Entrypoint Smoke Evidence

| task | entrypoint | pass | scorer runtime | score mode | not reranking | evidence file |
|---|---|---|---|---|---|---|
| insertion | `for_show_xiaomi.serve_dp_tac_quality_guided` | true | `InsertionRiskScorerRuntime` | `profile` | true | `/home/chenshuai/Project/output/tac_quality_guided_server_packet/auto_discovered/insertion_guided_server_real_foresight_smoke.json` |
| board | `for_show_xiaomi.serve_dp_tac_quality_guided` | config/build smoke true | `ForceBandTacQualityEnergyRuntime` | `quality` | true | `/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_marker_joint_20260618.json` |

Interpretation:

- Insertion entrypoint has a real-Foresight server smoke file.
- Board command sheet now points to the marker-joint rollout config and `marker_joint_guided` arm. The config can build `ForceBandTacQualityEnergyRuntime` with feature dim `64`; a full server dry-run can be rerun when GPU memory is free.
- This still does not prove real force-curve improvement. It only proves the current command/config no longer points at the old saturated scorer.

## Board Score-Mode Saturation Finding And Fix

The current with-260617 ForceBand scorer was re-evaluated through the real board Foresight chain with old positive, 260617 positive, too-small, too-large, and oscillate samples.

Evidence file:

- `docs/2026-06-18_with260617_forceband_foresight_alignment.md`

Output root:

- `/home/chenshuai/Project/output/tac_quality_force_band_with260617_score_mode_sweep_20260618`

Key metrics:

| mode | pred AUC(good) | pred/GT Spearman | score vs force-quality Spearman | label mean range |
|---|---:|---:|---:|---:|
| `p_good` | 0.6514 | 0.8019 | -0.4942 | 0.00000041 |
| `energy_clipped` | 0.6431 | 0.5331 | -0.3978 | 0.00074194 |
| `profile` | 0.6431 | 0.4133 | -0.3665 | 0.00275540 |
| `quality` | 0.5604 | 0.4385 | -0.0011 | 0.00029819 |
| `reason_good` | 0.5253 | 0.6455 | -0.1848 | 0.00000058 |

Interpretation:

- Offline classifier metrics remain strong, but the deployable score has very small dynamic range after Foresight/GT future marker scoring.
- The old with-260617 scorer should be treated as a rejected/intermediate candidate, not the current board default.
- The marker-joint scorer fixes the main deployment mismatch by removing future `eef_abs` from the feature vector.

Marker-joint evidence:

| metric | value |
|---|---:|
| held-out AUC | 0.9997 |
| held-out bACC | 0.9828 |
| held-out reason macro F1 | 0.9703 |
| held-out quality Spearman | 0.9239 |
| Foresight pred AUC(good), quality mode | 0.9991 |
| Foresight GT AUC(good), quality mode | 0.8986 |
| pred/GT Spearman, quality mode | 0.6130 |
| score vs force-band quality Spearman | 0.4733 |
| gradient audit pass | true |

Evidence files:

- `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/train_result.json`
- `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/foresight_alignment_quality/foresight_score_alignment.json`
- `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/guidance_gradient_audit_quality/guidance_gradient_audit.json`

## Current DP Training Context

The 260617-only board DP is still training:

- run dir: `/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000`
- current observation: around epoch `779/2000`
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
2. Use board `marker_joint_guided` as the board guidance candidate.
3. Always collect matched baseline and guided trials.
4. Treat all current offline metrics as readiness evidence, not final performance evidence.

## Board Real-Rollout Command Packet

The copy-paste command sheet has been updated for the current board candidate:

- file: `for_show_xiaomi/guide_forshow.sh`
- baseline port: `8765`
- guided port: `8766`
- DP checkpoint: `/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000/dp_best.pth`
- guided scorer config: `/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_marker_joint_20260618.json`
- guided arm: `marker_joint_guided`
- server-side rollout root: `/home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_scorer`

Expected rollout layout:

```text
/home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_scorer/
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
  --root /home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_scorer \
  --tag board_260617_marker_joint_scorer
```

This will generate contact-phase force summaries under:

```text
/home/chenshuai/Project/output/board_force_rollout_eval/board_260617_marker_joint_scorer/
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
- The real claim still requires matched baseline/guided robot trials saved under `/home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_scorer`.
