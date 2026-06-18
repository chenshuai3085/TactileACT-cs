# TacQuality Real-Rollout Command Preflight

Date: 2026-06-19

## Purpose

Prepare the current baseline/guided real-robot command sheet for the two TacQuality guidance tasks:

- board wiping with the 260617-only DP checkpoint;
- socket insertion with the matched 0401 Foresight checkpoint.

This document only records command/preflight evidence. It does not claim real-robot improvement. Real improvement still requires paired baseline/guided robot rollouts with saved force/outcome logs.

## Command Sheet Updated

File:

```text
for_show_xiaomi/guide_forshow.sh
```

The file remains a copy-paste command sheet, not a launcher. Running it directly exits after printing instructions.

Current board commands:

- baseline server: port `8765`
- guided server: port `8766`
- DP checkpoint directory:

```text
/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260618_ext
```

- DP checkpoint: `dp_best.pth`
- board Foresight:

```text
/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload/foresight_best.ckpt
```

- board guided arm: `marker_joint_guided`
- board scorer runtime: `ForceBandTacQualityEnergyRuntime`
- board guidance location: after DP has produced the final clean action chunk

Current insertion commands:

- baseline server: port `8785`
- guided server: port `8786`
- DP checkpoint directory:

```text
/home/chenshuai/Project/output/ckpt/dp_tac_concat_02090210
```

- DP checkpoint: `dp_final.pth`
- TactileVAE override:

```text
/home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt
```

- insertion Foresight is now the matched 0401 checkpoint:

```text
/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_0401/foresight_best.ckpt
```

The older insertion `latent_foresight_full` checkpoint is not used as the default real-test chain because previous audits showed a missing-key caveat. It remains historical/ablation evidence only.

## Server-Side Rollout Logging

The deployment server uses:

```text
for_show_xiaomi/server_rollout_logger.py
```

Every real rollout gets one separate directory. The grouping is automatic:

```text
/home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_scorer/baseline/
/home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_scorer/guided/
/home/chenshuai/Project/output/insertion_rollouts/default_insertion_risk_scorer/baseline/
/home/chenshuai/Project/output/insertion_rollouts/default_insertion_risk_scorer/guided/
```

Each rollout directory saves:

- `force_trace.csv`
- `force_trace.npz`
- `force_curve.png`
- `metadata.json`

The CSV includes force, qpos, eef, returned action, action_norm, marker proxy, and guidance report scalar fields when available.

The client commands in `guide_forshow.sh` intentionally pass `--disable_force_log`; this avoids duplicate client-side force logs. The force/trajectory evidence for this comparison is saved on the GPU server side.

## Preflight Result

Command:

```bash
conda run --no-capture-output -n TactileACT python for_show_xiaomi/preflight_tac_quality_deploy.py
```

Result:

```json
{
  "preflight_pass": true,
  "path_ok": true,
  "config_ok": true,
  "busy_ports": []
}
```

Generated files:

```text
/home/chenshuai/Project/output/tac_quality_deploy_preflight/tac_quality_deploy_preflight.json
/home/chenshuai/Project/output/tac_quality_deploy_preflight/tac_quality_deploy_preflight.md
```

## Dry-Run Smoke Evidence

All smoke runs were done without starting a robot server. They check loading, model wiring, and clean-action guidance/refinement interfaces.

### Board Baseline

Output:

```text
/home/chenshuai/Project/output/tac_quality_guided_server_packet/board_260617_baseline_smoke_20260619/guided_server_dry_run_smoke.json
```

Result:

- `dry_run_guidance_smoke_pass=true`
- task: `board`
- arm: `baseline`
- guidance disabled
- Foresight load: `missing=0`, `unexpected=0`

### Board Marker-Joint Guided

Output:

```text
/home/chenshuai/Project/output/tac_quality_guided_server_packet/board_260617_marker_joint_guided_smoke_20260619/guided_server_dry_run_smoke.json
```

Result:

- `dry_run_guidance_smoke_pass=true`
- task: `board`
- arm: `marker_joint_guided`
- scorer runtime: `ForceBandTacQualityEnergyRuntime`
- score mode: `quality`
- improved rate: `1.0`
- accept rate: `1.0`
- finite grad rate: `1.0`
- positive grad rate: `1.0`
- Foresight load: `missing=0`, `unexpected=0`

### Insertion Default Guided

Output:

```text
/home/chenshuai/Project/output/tac_quality_guided_server_packet/insertion_0401_default_guided_smoke_20260619/guided_server_dry_run_smoke.json
```

Result:

- `dry_run_guidance_smoke_pass=true`
- task: `insertion`
- arm: `default_guided`
- scorer runtime: `InsertionRiskScorerRuntime`
- score mode: `profile`
- improved rate: `1.0`
- accept rate: `1.0`
- finite grad rate: `1.0`
- positive grad rate: `1.0`
- Foresight load: `missing=0`, `unexpected=0`

## Evaluation After Robot Tests

Board force-curve evaluation:

```bash
conda run --no-capture-output -n TactileACT python for_show_xiaomi/assign_rollout_pair_ids.py \
  --root /home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_scorer \
  --prefix board

conda run --no-capture-output -n TactileACT python for_show_xiaomi/eval_board_force_rollouts.py \
  --root /home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_scorer \
  --tag board_260617_marker_joint_scorer
```

Insertion rollout evaluation:

```bash
conda run --no-capture-output -n TactileACT python for_show_xiaomi/assign_rollout_pair_ids.py \
  --root /home/chenshuai/Project/output/insertion_rollouts/default_insertion_risk_scorer \
  --prefix insertion

conda run --no-capture-output -n TactileACT python for_show_xiaomi/eval_insertion_rollouts.py \
  --root /home/chenshuai/Project/output/insertion_rollouts/default_insertion_risk_scorer \
  --output_dir /home/chenshuai/Project/output/insertion_rollout_eval \
  --tag insertion_default_risk_scorer
```

Unified summary after both tasks:

```bash
conda run --no-capture-output -n TactileACT python for_show_xiaomi/eval_tac_quality_real_rollouts.py \
  --board_root /home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_scorer \
  --insertion_root /home/chenshuai/Project/output/insertion_rollouts/default_insertion_risk_scorer \
  --output_dir /home/chenshuai/Project/output/tac_quality_real_rollout_eval \
  --tag current_tac_quality
```

## Current Claim Boundary

Supported now:

- Command paths exist and preflight passes.
- Board baseline/guided and insertion guided dry-runs load and pass.
- Server-side force/trajectory logging is wired and grouped by `baseline`/`guided`.
- Guidance remains final clean-action trust-region refinement, not reranking and not every-step DDPM guidance.

Not supported yet:

- No real-robot improvement claim.
- No claim that guided is better than baseline until matched rollout force/outcome logs are collected and evaluated.
- No claim that `latent_foresight_full` is the default insertion chain.
