#!/usr/bin/env bash
set -euo pipefail

# For-show launcher for board DP / tactile guidance services.
#
# Default behavior is safe: running this script without a mode only prints help.
# Services are launched in the background with nohup. Logs go to /tmp by default.

REPO_DIR="${REPO_DIR:-/home/chenshuai/Project/TactileACT-cs}"
LOG_DIR="${LOG_DIR:-/tmp/guide_forshow}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-TactileACT}"
HOST="${HOST:-0.0.0.0}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"

CONDA_RUN=("$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" python -u)

BOARD_FORESIGHT_DIR="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload"
BOARD_FORESIGHT_CKPT="${BOARD_FORESIGHT_DIR}/foresight_best.ckpt"

OLD_FULL_DIR="/home/chenshuai/Project/output/dp_tac_concat_board_260609_260610_left_boardvae_rawimg200x266_ph16_oh2_e1000"
OLD_FULL_CKPT_NAME="dp_topk_ep504_loss0.0020.pth"
OLD_FULL_CKPT="${OLD_FULL_DIR}/${OLD_FULL_CKPT_NAME}"

OLD_POS_DIR="/home/chenshuai/Project/output/dp_tac_concat_board_positive_only_260609_left_boardvae_rawimg200x266_ph16_oh2_e1000"
OLD_POS_BEST_CKPT_NAME="dp_best.pth"
OLD_POS_LATEST_CKPT_NAME="dp_latest.pth"
OLD_POS_BEST_CKPT="${OLD_POS_DIR}/${OLD_POS_BEST_CKPT_NAME}"
OLD_POS_LATEST_CKPT="${OLD_POS_DIR}/${OLD_POS_LATEST_CKPT_NAME}"

NEW_PLUS_PEG_DIR="/home/chenshuai/Project/output/dp_tac_concat_board_260609_260610_plus_peg0617_left_boardvae_rawimg200x266_ph16_oh2_e1000"
NEW_PLUS_PEG_CKPT_NAME="dp_best.pth"
NEW_PLUS_PEG_CKPT="${NEW_PLUS_PEG_DIR}/${NEW_PLUS_PEG_CKPT_NAME}"

BOARD_LATENT_SCORER="/home/chenshuai/Project/output/board_latent_energy/ce_margin_e10/board_latent_energy_best.pt"
PTG_ROLLOUT_CONFIG="/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs.json"
PTG_PROXY_SCORER="/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_final.pt"

usage() {
  cat <<'EOF'
Usage:
  ./for_show_xiaomi/guide_forshow.sh <mode>

Environment overrides:
  GPU=0 HOST=0.0.0.0 LOG_DIR=/tmp/guide_forshow DRY_RUN=1

Current recommended new-DP comparison, using the new 260609+260610+0617 DP:
  new_ptg_baseline_8765     New DP dp_best, no guidance, port 8765
  new_ptg_guided_8766       New DP dp_best + PTGProxy final clean-action guidance, port 8766
  new_ptg_pair              Launch both commands above

Historical board runs from 2026-06-16:
  old_full_baseline20_8766      Old full-data DP baseline, 20 denoise steps, port 8766
  old_full_guided20_8765        Old full-data DP + BoardLatentEnergy guidance, port 8765
  old_full_pair20               Launch the two old full-data 20-step services
  old_pos_baseline50_8776       Positive-only DP baseline, 50 denoise steps, port 8776
  old_pos_guided50_8775         Positive-only DP + BoardLatentEnergy guidance, port 8775
  old_pos_pair50                Launch the two positive-only 50-step services
  old_four_services100          Historical full+positive baseline/guided 100-step setup

Historical board guided sweep from 2026-06-17:
  old_full_guided_matrix        Four full-data guided ports:
                                  8765 steps=10 scale=0.005
                                  8766 steps=20 scale=0.005
                                  8775 steps=10 scale=0.010
                                  8776 steps=20 scale=0.010
  old_full_guided_fast8785      Full-data guided, 30 denoise steps, port 8785

Utilities:
  status                        Show known ports, matching processes, and GPU memory

Notes:
  - This script does not kill old services automatically.
  - If a port is occupied, stop the old service first or choose another port by editing this file.
  - Use DRY_RUN=1 to print commands without launching services.
EOF
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[error] Missing file: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "[error] Missing directory: $path" >&2
    exit 1
  fi
}

port_in_use() {
  local port="$1"
  ss -ltnp 2>/dev/null | grep -E ":${port}[[:space:]]" >/dev/null 2>&1
}

ensure_free_ports() {
  local port
  for port in "$@"; do
    if port_in_use "$port"; then
      echo "[error] Port ${port} is already in use:" >&2
      ss -ltnp 2>/dev/null | grep -E ":${port}[[:space:]]" >&2 || true
      exit 1
    fi
  done
}

print_cmd() {
  printf 'cd %q\n' "$REPO_DIR"
  printf '%q ' "$@"
  printf '\n'
}

run_bg() {
  local name="$1"
  local port="$2"
  local log_file="$3"
  shift 3

  if [[ "${SKIP_PORT_CHECK:-0}" != "1" ]]; then
    ensure_free_ports "$port"
  fi

  mkdir -p "$LOG_DIR"

  echo "[launch] ${name}"
  echo "[port] ${port}"
  echo "[log] ${log_file}"
  echo "[cmd]"
  print_cmd "$@"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] not launching"
    return 0
  fi

  (
    cd "$REPO_DIR"
    nohup "$@" >"$log_file" 2>&1 &
    echo $! >"${log_file%.log}.pid"
  )

  sleep 2
  if [[ -s "${log_file%.log}.pid" ]]; then
    echo "[pid] $(cat "${log_file%.log}.pid")"
  fi
  tail -20 "$log_file" || true
}

status() {
  echo "=== known ports ==="
  ss -ltnp 2>/dev/null | grep -E ':8765|:8766|:8775|:8776|:8785' || true
  echo
  echo "=== matching processes ==="
  pgrep -af 'serve_dp_tac_quality_guided|serve_board_dp_foresight_guided|serve_dp_policy' || true
  echo
  echo "=== gpu ==="
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
}

check_new_ptg_paths() {
  require_dir "$NEW_PLUS_PEG_DIR"
  require_file "${NEW_PLUS_PEG_DIR}/config.json"
  require_file "$NEW_PLUS_PEG_CKPT"
  require_dir "$BOARD_FORESIGHT_DIR"
  require_file "$BOARD_FORESIGHT_CKPT"
  require_file "$PTG_ROLLOUT_CONFIG"
  require_file "$PTG_PROXY_SCORER"
}

check_old_full_paths() {
  require_dir "$OLD_FULL_DIR"
  require_file "${OLD_FULL_DIR}/config.json"
  require_file "$OLD_FULL_CKPT"
  require_dir "$BOARD_FORESIGHT_DIR"
  require_file "$BOARD_FORESIGHT_CKPT"
  require_file "$BOARD_LATENT_SCORER"
}

check_old_pos_paths() {
  require_dir "$OLD_POS_DIR"
  require_file "${OLD_POS_DIR}/config.json"
  require_file "$OLD_POS_BEST_CKPT"
  require_file "$OLD_POS_LATEST_CKPT"
  require_dir "$BOARD_FORESIGHT_DIR"
  require_file "$BOARD_FORESIGHT_CKPT"
  require_file "$BOARD_LATENT_SCORER"
}

start_new_ptg_baseline_8765() {
  check_new_ptg_paths
  run_bg "new_plus_peg_ptg_baseline_no_guidance" 8765 "${LOG_DIR}/new_plus_peg_ptg_baseline_8765.log" \
    "${CONDA_RUN[@]}" -m for_show_xiaomi.serve_dp_tac_quality_guided \
    --task board \
    --arm baseline \
    --disable_guidance \
    --ckpt_dir "$NEW_PLUS_PEG_DIR" \
    --ckpt_name "$NEW_PLUS_PEG_CKPT_NAME" \
    --foresight_dir "$BOARD_FORESIGHT_DIR" \
    --foresight_ckpt "$BOARD_FORESIGHT_CKPT" \
    --rollout_arm_config "$PTG_ROLLOUT_CONFIG" \
    --host "$HOST" \
    --port 8765 \
    --gpu "$GPU" \
    --num_inference_steps 100 \
    --action_skip 0 \
    --action_horizon 8
}

start_new_ptg_guided_8766() {
  check_new_ptg_paths
  run_bg "new_plus_peg_ptg_guided" 8766 "${LOG_DIR}/new_plus_peg_ptg_guided_8766.log" \
    "${CONDA_RUN[@]}" -m for_show_xiaomi.serve_dp_tac_quality_guided \
    --task board \
    --arm default_guided \
    --ckpt_dir "$NEW_PLUS_PEG_DIR" \
    --ckpt_name "$NEW_PLUS_PEG_CKPT_NAME" \
    --foresight_dir "$BOARD_FORESIGHT_DIR" \
    --foresight_ckpt "$BOARD_FORESIGHT_CKPT" \
    --rollout_arm_config "$PTG_ROLLOUT_CONFIG" \
    --host "$HOST" \
    --port 8766 \
    --gpu "$GPU" \
    --num_inference_steps 100 \
    --action_skip 0 \
    --action_horizon 8 \
    --send_guidance_report
}

start_new_ptg_pair() {
  ensure_free_ports 8765 8766
  SKIP_PORT_CHECK=1 start_new_ptg_baseline_8765
  SKIP_PORT_CHECK=1 start_new_ptg_guided_8766
}

start_old_policy_baseline() {
  local name="$1"
  local port="$2"
  local ckpt_dir="$3"
  local ckpt_name="$4"
  local steps="$5"
  local action_skip="$6"
  local action_horizon="$7"
  local dump_dir="$8"
  local log_file="${LOG_DIR}/${name}.log"

  require_dir "$ckpt_dir"
  require_file "${ckpt_dir}/config.json"
  require_file "${ckpt_dir}/${ckpt_name}"

  run_bg "$name" "$port" "$log_file" \
    "${CONDA_RUN[@]}" for_show_xiaomi/serve_dp_policy.py \
    --ckpt_dir "$ckpt_dir" \
    --ckpt_name "$ckpt_name" \
    --host "$HOST" \
    --port "$port" \
    --gpu "$GPU" \
    --num_inference_steps "$steps" \
    --action_skip "$action_skip" \
    --action_horizon "$action_horizon" \
    --debug_dump_first_obs \
    --debug_dump_dir "$dump_dir"
}

start_old_boardlatent_guided() {
  local name="$1"
  local port="$2"
  local dp_ckpt="$3"
  local steps="$4"
  local guidance_steps="$5"
  local guidance_scale="$6"
  local action_skip="$7"
  local action_horizon="$8"
  local log_file="${LOG_DIR}/${name}.log"

  require_file "$dp_ckpt"
  require_file "$BOARD_FORESIGHT_CKPT"
  require_file "$BOARD_LATENT_SCORER"

  run_bg "$name" "$port" "$log_file" \
    "${CONDA_RUN[@]}" for_show_xiaomi/serve_board_dp_foresight_guided.py \
    --dp_ckpt "$dp_ckpt" \
    --foresight_dir "$BOARD_FORESIGHT_DIR" \
    --foresight_ckpt "$BOARD_FORESIGHT_CKPT" \
    --scorer_ckpt "$BOARD_LATENT_SCORER" \
    --host "$HOST" \
    --port "$port" \
    --gpu "$GPU" \
    --num_inference_steps "$steps" \
    --guidance_steps "$guidance_steps" \
    --guidance_scale "$guidance_scale" \
    --guidance_path latent_only \
    --score_mode expert_margin \
    --alignment shift1 \
    --action_skip "$action_skip" \
    --action_horizon "$action_horizon" \
    --send_guidance_report
}

start_old_full_baseline20_8766() {
  check_old_full_paths
  start_old_policy_baseline \
    old_full_baseline20_8766 8766 "$OLD_FULL_DIR" "$OLD_FULL_CKPT_NAME" \
    20 8 8 \
    /home/chenshuai/Project/output/tactileact_dp_first_obs_full_ep504_skip8_steps20
}

start_old_full_guided20_8765() {
  check_old_full_paths
  start_old_boardlatent_guided \
    old_full_guided20_8765 8765 "$OLD_FULL_CKPT" \
    20 2 0.001 8 8
}

start_old_full_pair20() {
  ensure_free_ports 8765 8766
  SKIP_PORT_CHECK=1 start_old_full_baseline20_8766
  SKIP_PORT_CHECK=1 start_old_full_guided20_8765
}

start_old_pos_baseline50_8776() {
  check_old_pos_paths
  start_old_policy_baseline \
    old_pos_baseline50_8776 8776 "$OLD_POS_DIR" "$OLD_POS_BEST_CKPT_NAME" \
    50 8 8 \
    /home/chenshuai/Project/output/tactileact_dp_first_obs_positive_best_skip8_steps50
}

start_old_pos_guided50_8775() {
  check_old_pos_paths
  start_old_boardlatent_guided \
    old_pos_guided50_8775 8775 "$OLD_POS_BEST_CKPT" \
    50 2 0.001 8 8
}

start_old_pos_pair50() {
  ensure_free_ports 8775 8776
  SKIP_PORT_CHECK=1 start_old_pos_baseline50_8776
  SKIP_PORT_CHECK=1 start_old_pos_guided50_8775
}

start_old_full_baseline100_8766() {
  check_old_full_paths
  start_old_policy_baseline \
    old_full_baseline100_8766 8766 "$OLD_FULL_DIR" "$OLD_FULL_CKPT_NAME" \
    100 6 10 \
    /home/chenshuai/Project/output/tactileact_dp_first_obs_full_ep504_skip6_steps100
}

start_old_full_guided100_8765() {
  check_old_full_paths
  start_old_boardlatent_guided \
    old_full_guided100_8765 8765 "$OLD_FULL_CKPT" \
    100 10 0.005 6 10
}

start_old_pos_baseline100_8776() {
  check_old_pos_paths
  start_old_policy_baseline \
    old_pos_baseline100_8776 8776 "$OLD_POS_DIR" "$OLD_POS_LATEST_CKPT_NAME" \
    100 6 10 \
    /home/chenshuai/Project/output/tactileact_dp_first_obs_positive_latest_skip6_steps100
}

start_old_pos_guided100_8775() {
  check_old_pos_paths
  start_old_boardlatent_guided \
    old_pos_guided100_8775 8775 "$OLD_POS_LATEST_CKPT" \
    100 10 0.005 6 10
}

start_old_four_services100() {
  ensure_free_ports 8765 8766 8775 8776
  SKIP_PORT_CHECK=1 start_old_full_baseline100_8766
  SKIP_PORT_CHECK=1 start_old_full_guided100_8765
  SKIP_PORT_CHECK=1 start_old_pos_baseline100_8776
  SKIP_PORT_CHECK=1 start_old_pos_guided100_8775
}

start_old_full_guided_matrix() {
  check_old_full_paths
  ensure_free_ports 8765 8766 8775 8776
  SKIP_PORT_CHECK=1 start_old_boardlatent_guided \
    old_full_guided_matrix_8765_s10_g0005 8765 "$OLD_FULL_CKPT" \
    100 10 0.005 6 10
  SKIP_PORT_CHECK=1 start_old_boardlatent_guided \
    old_full_guided_matrix_8766_s20_g0005 8766 "$OLD_FULL_CKPT" \
    100 20 0.005 6 10
  SKIP_PORT_CHECK=1 start_old_boardlatent_guided \
    old_full_guided_matrix_8775_s10_g001 8775 "$OLD_FULL_CKPT" \
    100 10 0.01 6 10
  SKIP_PORT_CHECK=1 start_old_boardlatent_guided \
    old_full_guided_matrix_8776_s20_g001 8776 "$OLD_FULL_CKPT" \
    100 20 0.01 6 10
}

start_old_full_guided_fast8785() {
  check_old_full_paths
  start_old_boardlatent_guided \
    old_full_guided_fast8785_steps30_s10_g001 8785 "$OLD_FULL_CKPT" \
    30 10 0.01 6 10
}

mode="${1:-help}"
case "$mode" in
  help|-h|--help)
    usage
    ;;
  status)
    status
    ;;
  new_ptg_baseline_8765)
    start_new_ptg_baseline_8765
    ;;
  new_ptg_guided_8766)
    start_new_ptg_guided_8766
    ;;
  new_ptg_pair)
    start_new_ptg_pair
    ;;
  old_full_baseline20_8766)
    start_old_full_baseline20_8766
    ;;
  old_full_guided20_8765)
    start_old_full_guided20_8765
    ;;
  old_full_pair20)
    start_old_full_pair20
    ;;
  old_pos_baseline50_8776)
    start_old_pos_baseline50_8776
    ;;
  old_pos_guided50_8775)
    start_old_pos_guided50_8775
    ;;
  old_pos_pair50)
    start_old_pos_pair50
    ;;
  old_four_services100)
    start_old_four_services100
    ;;
  old_full_guided_matrix)
    start_old_full_guided_matrix
    ;;
  old_full_guided_fast8785)
    start_old_full_guided_fast8785
    ;;
  *)
    echo "[error] Unknown mode: $mode" >&2
    echo >&2
    usage >&2
    exit 2
    ;;
esac
