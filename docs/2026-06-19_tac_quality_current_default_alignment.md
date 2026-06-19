# 2026-06-19 TacQuality Current Default Alignment

## Purpose

This note records a small but important alignment fix for the current DP
classifier/scorer-guidance stack.

The project currently has multiple historical rollout config files.  The active
recommendation is:

| task | current arm | scorer | score mode |
|---|---|---|---|
| board | `marker_joint_s12_guided` | `ForceBandTacQualityEnergyRuntime` with marker-joint s12 checkpoint | `quality` |
| insertion | `good_margin_guided` | `InsertionRiskScorerRuntime` | `good_margin` |

Older defaults still pointed to `default_guided`/older config files in several
server and offline-audit entry points.  That made it easy to accidentally run
the stale board scorer or insertion `profile` score when a command forgot to
pass `--rollout_arm_config`.

## Change

Defaults were aligned to:

```text
/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_20260619.json
```

Updated entry points:

- `for_show_xiaomi/serve_dp_tac_quality_guided.py`
- `for_show_xiaomi/preflight_tac_quality_deploy.py`
- `TFAC_V5/tac_quality_energy/serving_guidance.py`
- `TFAC_V5/tac_quality_energy/eval_ddpm_step_guidance_audit.py`
- `TFAC_V5/tac_quality_energy/eval_noisy_action_guidance_audit.py`
- `TFAC_V5/tac_quality_energy/eval_guidance_gradient_audit.py`
- `TFAC_V5/tac_quality_energy/sweep_board_ddpm_step_guidance.py`
- `TFAC_V5/tac_quality_energy/sweep_insertion_ddpm_step_guidance.py`
- `TFAC_V5/tac_quality_energy/eval_insertion_score_mode_ablation.py`

The change does not retrain models and does not change scorer math.  It only
changes default paths/arms so new runs use the current recommended scorer unless
an older scorer is explicitly requested.

## Verification

Commands run:

```bash
python -m py_compile \
  TFAC_V5/tac_quality_energy/serving_guidance.py \
  TFAC_V5/tac_quality_energy/eval_ddpm_step_guidance_audit.py \
  TFAC_V5/tac_quality_energy/eval_noisy_action_guidance_audit.py \
  TFAC_V5/tac_quality_energy/sweep_board_ddpm_step_guidance.py \
  TFAC_V5/tac_quality_energy/sweep_insertion_ddpm_step_guidance.py \
  TFAC_V5/tac_quality_energy/eval_insertion_score_mode_ablation.py \
  TFAC_V5/tac_quality_energy/eval_guidance_gradient_audit.py \
  for_show_xiaomi/serve_dp_tac_quality_guided.py \
  for_show_xiaomi/preflight_tac_quality_deploy.py
```

```bash
python for_show_xiaomi/preflight_tac_quality_deploy.py
```

Preflight result:

```text
preflight_pass=true
path_ok=true
config_ok=true
busy_ports=[]
```

Current CPU dry-run serving smoke was also refreshed while the 260617-only DP
training was active on GPU.  The smoke used `--gpu -1` and synthetic Foresight
to avoid interfering with the training process.

Board s12 smoke:

```text
output: /home/chenshuai/Project/output/tac_quality_guided_server_packet/current_20260619_board_s12_cpu_smoke/guided_server_dry_run_smoke.json
task: board
arm: marker_joint_s12_guided
runtime: ForceBandTacQualityEnergyRuntime
score_mode: quality
pass: true
finite_grad_rate: 1.0
positive_grad_rate: 1.0
accept_rate: 1.0
score_delta_mean: 0.00006324
raw_action_delta_mean: 0.00057086
contact_gate_value: 1.0
not_reranking: true
```

Insertion good-margin smoke:

```text
output: /home/chenshuai/Project/output/tac_quality_guided_server_packet/current_20260619_insertion_good_margin_cpu_smoke/guided_server_dry_run_smoke.json
task: insertion
arm: good_margin_guided
runtime: InsertionRiskScorerRuntime
score_mode: good_margin
pass: true
finite_grad_rate: 1.0
positive_grad_rate: 1.0
accept_rate: 1.0
score_delta_mean: 0.185566
raw_action_delta_mean: 0.069957
not_reranking: true
```

Default parser/runtime check in the `TactileACT` environment:

```text
ddpm default arm: marker_joint_s12_guided
board sweep default arm: marker_joint_s12_guided
insertion sweep default arm: good_margin_guided
score-mode ablation default arm: good_margin_guided
```

## Current Evidence Boundary

Offline evidence supports:

- board s12 scorer has strong held-out classification and valid bounded
  Foresight-gradient behavior;
- insertion good-margin scorer avoids saturated `p_good` and has the strongest
  current offline score-mode evidence;
- final clean-action trust-region guidance and final-step protected DDPM
  guidance are the safest current guidance modes.

Not proven yet:

- real board wiping improvement;
- real insertion success/bounce/retry improvement;
- final best scorer after paired robot tests.

The next required evidence is paired real rollouts with server-side logs:

- board: baseline vs `marker_joint_s12_guided`, force-band occupancy and
  smoothness metrics;
- insertion: baseline vs `good_margin_guided`, success/bounce/retry metadata
  plus contact-quality traces.
