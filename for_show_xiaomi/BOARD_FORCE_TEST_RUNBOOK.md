# Board Force Test Runbook

This file is for real robot board-wiping tests.  It records the standard flow
for comparing one DP checkpoint with and without tactile quality guidance.
The policy server saves one force/trajectory trace directory for every real
robot wipe.

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

Server-side rollout logs are saved under two fixed groups:

```bash
/home/chenshuai/Project/output/board_force_rollouts/server/baseline/
/home/chenshuai/Project/output/board_force_rollouts/server/guided/
```

Each wipe trajectory gets one separate timestamped directory.  Each directory
saves:

- `force_trace.csv`
- `force_trace.npz`
- `force_curve.png`
- `metadata.json`

The CSV includes robot/tactile force columns, qpos/eef trajectory columns,
server action columns, marker proxy columns, and guidance report scalar columns.

## 2. Run Robot Client

Example client setup:

```bash
GPU_SERVER_IP=127.0.0.1
```

Baseline trials:

```bash
python for_show_xiaomi/ws_client.py \
  --host ${GPU_SERVER_IP} \
  --port 8765 \
  --disable_force_log
```

Guided trials:

```bash
python for_show_xiaomi/ws_client.py \
  --host ${GPU_SERVER_IP} \
  --port 8766 \
  --disable_force_log
```

`--disable_force_log` disables duplicate client-side logging.  The server still
saves the full rollout trace because it receives the real observation and knows
the returned action.

## 3. Evaluate Force Curves

Evaluate and visualize all server-side baseline/guided rollout traces:

```bash
conda run --no-capture-output -n TactileACT python for_show_xiaomi/eval_board_force_rollouts.py \
  --root /home/chenshuai/Project/output/board_force_rollouts/server \
  --tag board_server_all
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
