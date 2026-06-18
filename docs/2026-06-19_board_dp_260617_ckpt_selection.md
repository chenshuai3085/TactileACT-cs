# 2026-06-19 Board 260617 DP Checkpoint Selection

## Decision

Use the newer 260617-only DP run for current board baseline/guided real-test commands:

```text
/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619/dp_best.pth
```

The older `20260618_ext` run remains useful historical evidence, but it is no
longer the default command-sheet checkpoint.

## Evidence

### Validation

| run | best val | best epoch | stopped epoch | latest val / best | recommended ckpt |
|---|---:|---:|---:|---:|---|
| `20260619` | 0.010671 | 94 | 197 | 1.835 | `dp_best.pth` |
| `20260618_ext` | 0.011387 | 105 | 523 | 3.009 | `dp_best.pth` |

The `20260619` run has the lower validation loss and a less severe late-overfit
tail. Both runs should use `dp_best.pth`, not `dp_latest.pth`.

### Deploy Load Check

New `20260619` board stack dry-run:

```text
/home/chenshuai/Project/output/tac_quality_guided_server_packet/board_260617_20260619_marker_joint_guided_smoke_20260619/guided_server_dry_run_smoke.json
```

Result:

- pass: `true`
- DP variant: `tactile_vae_frozen`
- Foresight: multistep, horizon 16, `0 missing / 0 unexpected`
- scorer runtime: `ForceBandTacQualityEnergyRuntime`
- score delta mean: `+0.001191`
- accept rate: `1.0000`
- finite grad rate: `1.0000`
- max delta within trust region: `true`

### Matched DDPM-Step Smoke

Same real 260617 episode/frame, same scorer/Foresight, same DDIM setting:

- dataset: `/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617`
- episode: `2`
- start: `80`
- seeds: `1,2,3,4`
- inference steps: `4`
- guidance steps: `1`
- guidance scale: `0.001`

| run | final improve | final score delta mean | finite grad | action delta norm mean | evidence |
|---|---:|---:|---:|---:|---|
| `20260619` | 1.0000 | +0.000052 | 1.0000 | 0.000107 | `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_20260619_ep2_s80_t0_s001_seed1_4/ddpm_step_guidance_audit.json` |
| `20260618_ext` | 1.0000 | +0.000050 | 1.0000 | 0.000116 | `/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_20260618ext_ep2_s80_t0_s001_seed1_4/ddpm_step_guidance_audit.json` |

This is not real robot evidence. It only says the new checkpoint loads correctly
and gives at least the same local TacQuality guidance behavior as the previous
default under this offline smoke.

## Updated Local Artifacts

- Command sheet: `for_show_xiaomi/guide_forshow.sh`
- Preflight checker: `for_show_xiaomi/preflight_tac_quality_deploy.py`
- Readiness builder: `TFAC_V5/tac_quality_energy/build_current_readiness_matrix.py`
- Readiness matrix: `docs/2026-06-18_tac_quality_guidance_readiness_matrix.md`

Preflight after the switch:

```text
preflight_pass=true
path_ok=true
config_ok=true
busy_ports=[]
```

## Boundary

The current evidence supports using `20260619/dp_best.pth` for controlled real
rollout tests. It does not prove better wiping quality. That still requires
matched baseline/guided real rollouts with server-side `force_trace.csv` and
force-band/smoothness metrics.
