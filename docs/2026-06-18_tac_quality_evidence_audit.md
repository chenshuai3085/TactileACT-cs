# 2026-06-18 TacQuality Evidence Audit

## Scope

This audit checks the current TacQuality scorer/guidance evidence for both insertion and board wiping.

The target chain is:

```text
DP clean action
  -> Foresight predicts future tactile consequence
  -> TacQuality scorer evaluates quality/risk
  -> trust-region gradient update refines the action chunk
```

This is classifier/energy guidance. It is not reranking.

Full machine-readable outputs:

- `/home/chenshuai/Project/output/tac_quality_evidence_audit_20260618/tac_quality_evidence_audit.json`
- `/home/chenshuai/Project/output/tac_quality_evidence_audit_20260618/tac_quality_evidence_audit.md`

## Insertion

Current candidate:

- runtime: `InsertionRiskScorerRuntime`
- checkpoint: `/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt`
- score mode: `profile` config; current runtime falls back to `energy_clipped` when weighted profile is unavailable

Evidence:

| item | value |
|---|---:|
| grouped CV AUC | 0.9877 |
| grouped CV balanced accuracy | 0.9437 |
| grouped CV reason macro F1 | 0.7894 |
| grouped CV quality corr | 0.7656 |
| Foresight gradient audit pass | true |
| finite grad rate | 1.0000 |
| positive grad rate | 1.0000 |
| improved rate | 1.0000 |
| trust-region pass rate | 1.0000 |
| server dry-run pass | true |
| not reranking | true |

Evidence files:

- `/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_eval.json`
- `/home/chenshuai/Project/output/insertion_guidance_gradient_audit_real_foresight_profile_20260618/guidance_gradient_audit.json`
- `/home/chenshuai/Project/output/tac_quality_guided_server_packet/insertion_default_real_foresight_smoke_20260618/override_vae_guided_server_dry_run_smoke.json`

## Board Wiping

Current candidate:

- runtime: `ForceBandTacQualityEnergyRuntime`
- feature variant: `marker_joint_action`
- checkpoint: `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/force_band_tac_quality_energy_best.pt`
- score mode: `quality`

Evidence:

| item | value |
|---|---:|
| grouped held-out AUC | 0.9997 |
| grouped held-out balanced accuracy | 0.9828 |
| grouped held-out reason macro F1 | 0.9703 |
| grouped held-out quality Spearman | 0.9239 |
| Foresight pred AUC(good) | 0.9991 |
| Foresight GT AUC(good) | 0.8986 |
| pred/GT score Spearman | 0.6130 |
| pred score vs force-band quality Spearman | 0.4733 |
| Foresight gradient audit pass | true |
| finite grad rate | 1.0000 |
| positive grad rate | 1.0000 |
| improved rate | 1.0000 |
| trust-region pass rate | 1.0000 |
| server dry-run pass | true |
| not reranking | true |

Evidence files:

- `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/train_result.json`
- `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/foresight_alignment_quality/foresight_score_alignment.json`
- `/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/guidance_gradient_audit_quality/guidance_gradient_audit.json`
- `/home/chenshuai/Project/output/tac_quality_guided_server_packet/marker_joint_board_real_foresight_smoke_20260618/guided_server_dry_run_smoke.json`

## Gate Result

| gate | result |
|---|---|
| insertion offline scorer ready | true |
| insertion gradient guidance ready | true |
| board offline scorer ready | true |
| board Foresight-score alignment ready | true |
| board gradient guidance ready | true |
| real rollout improvement proven | false |

## Conclusion

Current evidence supports using:

- insertion: `InsertionRiskScorerRuntime`
- board: `ForceBandTacQualityEnergyRuntime(marker_joint_action, score_mode=quality)`

as the current TacQuality classifier/energy guidance candidates.

However, the full goal is not complete yet. The missing evidence is matched real robot rollouts:

- insertion: baseline vs guided success, bounce, retry, and safety metrics;
- board: baseline vs guided server-side force curves, contact-phase force-in-band ratio, force smoothness, marker smoothness, and wipe-quality proxy.

Offline scorer metrics, Foresight alignment, gradient audits, and server dry-runs are readiness evidence only. They must not be written as real robot improvement.
