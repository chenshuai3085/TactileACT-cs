# Board Force Test Runbook

This file is for real robot board-wiping tests.  It records the standard flow
for comparing one DP checkpoint with and without tactile quality guidance while
saving force traces for every trial.

## 1. Start Policy Servers

Current recommended comparison for the new `260609+260610+0617` DP:

```bash
cd /home/chenshuai/Project/TactileACT-cs
sed -n '1,260p' for_show_xiaomi/guide_forshow.sh
```

`guide_forshow.sh` is now a copy-paste command sheet.  Running the file itself
only prints help and does not start services.  Copy and run these two blocks
from the file:

- `1. Current recommended baseline`: same DP, no guidance baseline, port `8765`
- `2. Current recommended guided`: same DP, PTGProxy final clean-action guidance, port `8766`
- `3. Status check`: confirm the ports/processes after launch

## 2. Run Robot Client And Record Force

Use one force-log tag for one comparison batch.  Example:

```bash
TAG=board_newdp_ptgproxy_$(date +%Y%m%d_%H%M)
GPU_SERVER_IP=127.0.0.1
```

Baseline trials:

```bash
python for_show_xiaomi/ws_client.py \
  --host ${GPU_SERVER_IP} \
  --port 8765 \
  --force_log_dir /home/chenshuai/Project/output/board_force_rollouts/${TAG}/baseline
```

Guided trials:

```bash
python for_show_xiaomi/ws_client.py \
  --host ${GPU_SERVER_IP} \
  --port 8766 \
  --force_log_dir /home/chenshuai/Project/output/board_force_rollouts/${TAG}/guided
```

Each trial saves:

- `force_trace.csv`
- `force_trace.npz`
- `force_curve.png`
- `metadata.json`

## 3. Evaluate Force Curves

Evaluate all trials in this comparison batch:

```bash
conda run --no-capture-output -n TactileACT python for_show_xiaomi/eval_board_force_rollouts.py \
  --root /home/chenshuai/Project/output/board_force_rollouts/${TAG} \
  --tag ${TAG}
```

Outputs:

- `board_force_rollout_summary.csv`: per-trial statistics
- `board_force_rollout_group_summary.csv`: grouped by arm/port
- `board_force_rollout_summary.json`
- `board_force_rollout_summary.md`
- `board_force_overview.png`

Key columns:

- `ft_fz_mean`: average Z force
- `ft_fz_p95`: high Z force tail
- `ft_f_mag_mean`: average force magnitude
- `ft_fz_delta_abs_mean`: mean absolute Z-force change, proxy for smoothness
- `ft_f_mag_delta_abs_mean`: mean absolute force-magnitude change
