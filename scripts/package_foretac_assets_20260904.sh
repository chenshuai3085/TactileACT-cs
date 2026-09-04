#!/usr/bin/env bash
set -Eeuo pipefail

OUT=/home/chenshuai/foretac_asset_bundles_20260904
mkdir -p "$OUT"
PROJECT_ARCHIVE="$OUT/foretac_projects_code_models_20260904.tar.gz"
WORLD_ARCHIVE="$OUT/worldarena_and_track3_deployment_20260904.tar.gz"

COMMON_EXCLUDES=(
  --exclude='*/.git' --exclude='*/.git/*' --exclude='*/__pycache__' --exclude='*/__pycache__/*'
  --exclude='*/.cache' --exclude='*/.cache/*' --exclude='*/wandb' --exclude='*/wandb/*'
  --exclude='*/tensorboard' --exclude='*/tensorboard/*' --exclude='*/runs' --exclude='*/runs/*'
  --exclude='*.mp4' --exclude='*.MP4' --exclude='*.mov' --exclude='*.MOV' --exclude='*.avi' --exclude='*.AVI'
  --exclude='*.mkv' --exclude='*.MKV' --exclude='*.webm' --exclude='*.WEBM' --exclude='*.qt' --exclude='*.gif'
  --exclude='*.hdf5' --exclude='*.h5' --exclude='*.npy' --exclude='*.npz' --exclude='*.parquet'
  --exclude='*.arrow' --exclude='*.pkl' --exclude='*.msgpack' --exclude='*.log' --exclude='*.log.*'
  --exclude='*.jsonl' --exclude='*.jsonl.*' --exclude='*/logs' --exclude='*/logs/*'
)
PROJECT_ROOTS=(
  'home/chenshuai/Project/miACT' 'home/chenshuai/Project/miACT (1)' 'home/chenshuai/Project/mi_pih'
  'home/chenshuai/Project/omnisf' 'home/chenshuai/Project/omnisf-master' 'home/chenshuai/Project/openpi'
  'home/chenshuai/Project/scripts' 'home/chenshuai/Project/TactileACT-cs' 'home/chenshuai/Project/vtm-manuscript'
)
WORLD_ROOTS=('home/chenshuai/Project/WorldArena2.0' 'home/chenshuai/track3_deploy_share_portable_20260821_213226')

echo '[1/5] ForeTac projects'
tar -czf "$PROJECT_ARCHIVE" -C / "${COMMON_EXCLUDES[@]}" \
  --exclude='home/chenshuai/Project/TactileACT-cs/outputs/logs' \
  --exclude='home/chenshuai/Project/TactileACT-cs/outputs/logs/*' \
  --exclude='home/chenshuai/Project/TactileACT-cs/outputs/multitask_replay_records' \
  --exclude='home/chenshuai/Project/TactileACT-cs/outputs/multitask_replay_records/*' \
  --exclude='home/chenshuai/Project/TactileACT-cs/assets/PACE_videos' \
  --exclude='home/chenshuai/Project/TactileACT-cs/assets/PACE_videos/*' \
  --exclude='home/chenshuai/Project/TactileACT-cs/assets/bounce_truncated_videos' \
  --exclude='home/chenshuai/Project/TactileACT-cs/assets/bounce_truncated_videos/*' \
  --exclude='home/chenshuai/Project/TactileACT-cs/tmp_vtm_ProjectPage*' \
  --exclude='home/chenshuai/Project/TactileACT-cs/web_media_source_bundle*' \
  --exclude='home/chenshuai/Project/TactileACT-cs/promo_build' \
  --exclude='home/chenshuai/Project/TactileACT-cs/promo_build/*' \
  --exclude='home/chenshuai/Project/TactileACT-cs/web_promo_build' \
  --exclude='home/chenshuai/Project/TactileACT-cs/web_promo_build/*' \
  --exclude='*/scripts/upload_modelscope.py' \
  --exclude='home/chenshuai/Project/TactileACT-cs/outputs/*/visualizations/*/episode_*' \
  --exclude='home/chenshuai/Project/TactileACT-cs/outputs/*/visualizations/*/episode_*/*' \
  --exclude='*.zip' --exclude='*.tar' --exclude='*.tar.gz' "${PROJECT_ROOTS[@]}"

echo '[2/5] WorldArena and Track3 deployment'
tar -czf "$WORLD_ARCHIVE" -C / "${COMMON_EXCLUDES[@]}" \
  --exclude='home/chenshuai/Project/WorldArena2.0/*/episode_*' \
  --exclude='home/chenshuai/Project/WorldArena2.0/*/episode_*/*' \
  --exclude='home/chenshuai/Project/WorldArena2.0/*/260*' \
  --exclude='home/chenshuai/Project/WorldArena2.0/.cache' \
  --exclude='home/chenshuai/Project/WorldArena2.0/.cache/*' \
  --exclude='home/chenshuai/Project/WorldArena2.0/*.tar.gz' \
  --exclude='*/_deps' --exclude='*/_deps/*' \
  --exclude='*/.competition_runtime' --exclude='*/.competition_runtime/*' \
  --exclude='*/.mplcache' --exclude='*/.mplcache/*' \
  --exclude='*/obs_logs*' --exclude='*/obs_logs*/*' \
  --exclude='*.png' --exclude='*.PNG' --exclude='*.jpg' --exclude='*.JPG' --exclude='*.jpeg' --exclude='*.JPEG' \
  --exclude='*.zip' --exclude='*.jsonl' --exclude='*.jsonl.*' "${WORLD_ROOTS[@]}"

echo '[3/5] Documentation'
cat > "$OUT/DATASET_AND_MEDIA_EXCLUSIONS.md" <<'EOF'
# ForeTac 本机资产归档说明（2026-09-04）

归档保留源码、配置、论文/方案、部署脚本、训练脚本、checkpoint（`.pt/.pth/.ckpt/.safetensors`）以及可复现所需的文本和 JSON 配置。

## 明确排除

- 数据集/轨迹/数组：`.hdf5`、`.h5`、`.npy`、`.npz`、`.parquet`、`.arrow`、`.pkl`、`.msgpack`。
- 视频/媒体：`.mp4`、`.mov`、`.avi`、`.mkv`、`.webm`、`.qt`；Track3 的 PNG/JPG 渲染图也排除。
- 运行日志和缓存：`.log`、`.jsonl`、`.git`、`__pycache__`、`.cache`、`wandb`、`runs`、`tensorboard`。
- ForeTac 回放记录、宣传媒体构建目录、临时网页压缩包。
- WorldArena 的 `episode_*`、日期批次目录和比赛运行产物。
- Track3 的 `obs_logs*`、`_deps`、`.competition_runtime`、`.mplcache` 和图像/日志产物。

## 数据集使用说明（仅记录，不随包上传）

- ForeTac/TactileACT：触觉 marker、HDF5 轨迹、RGB/触觉历史、Board/Vase/Card/Chip 等任务数据。
- miACT/mi_pih：PIH、擦拭、插孔等机器人轨迹与实机数据；只保留采集/读取代码和配置。
- OmniSF/OmniSF-master/OpenPI：只保留训练和环境适配代码；下载数据、仿真 episode 和缓存不打包。
- WorldArena/Track3：比赛任务 episode、观测和评测数据；只保留部署方案、训练/服务脚本、配置与说明文档。
- vtm-manuscript：论文正文、附录、参考文献、图表源码和构建说明；视频和大体量媒体不打包。

完整数据路径、原始数据/权重位置和不可恢复项见 `recovery_inventory_20260831.md` 及 `MANIFEST.md`。
EOF

echo '[4/5] Manifest and checksums'
{
  echo '# ForeTac 本机资产归档清单'; echo; date -Is; echo
  for archive in "$PROJECT_ARCHIVE" "$WORLD_ARCHIVE"; do
    echo "## $(basename "$archive")"; stat -c 'size_bytes=%s' "$archive"
    tar -tzf "$archive" | tee "$archive.list" | wc -l | awk '{print "entries=" $1}'
    sha256sum "$archive"; echo
  done
  echo '## Source roots'; printf '%s\n' "${PROJECT_ROOTS[@]}" "${WORLD_ROOTS[@]}"; echo
  echo '## Existing preserved artifacts (not duplicated)'
  sha256sum /home/chenshuai/internal_key_projects_code_20260829.tar.gz /home/chenshuai/foretac_remote_audit_preserved_20260831.tar.gz /home/chenshuai/recovery_snapshot_20260831.tar.gz /home/chenshuai/recovery_inventory_20260831.md
} > "$OUT/MANIFEST.md"
sha256sum "$PROJECT_ARCHIVE" "$WORLD_ARCHIVE" "$OUT/DATASET_AND_MEDIA_EXCLUSIONS.md" "$OUT/MANIFEST.md" > "$OUT/SHA256SUMS"
echo '[5/5] Done'; du -sh "$OUT"/*
