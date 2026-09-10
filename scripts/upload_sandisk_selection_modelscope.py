#!/usr/bin/env python3
"""Upload selected SANDISK paths directly, preserving their directory tree."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from modelscope.hub.api import HubApi


ROOT = Path("/media/chenshuai/SANDISK ELE")
REPO = "chenshuai3085/pih-tactile-episodes-20260904"
ITEMS = [
    "111", "0708_video", "20251229_teleop_joint_next_state_as_action",
    "参赛注册-2986", "参赛注册-3012", "基于颗粒物衰减法的室内换气率无线测量装置",
    "数媒", "无忧搜寻-基于Hadoop和AI的智能救援系统", "云智教育APP", "证明材料",
    "证书", "智慧助老轮椅", "桌面整理", "A11-2891", "Cursor_workspace",
    "pi05_lerobot_cowa", "rm_models-main", "tactileact_output", "zhuomian",
    "证明材料.zip", "clean_dataset.py", "for_show.py",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=ROOT)
    ap.add_argument("--repo-id", default=REPO)
    ap.add_argument("--state", type=Path, default=Path("tmp/modelscope_sandisk_upload_state.jsonl"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-items", type=int, default=0)
    args = ap.parse_args()
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        raise SystemExit("MODELSCOPE_API_TOKEN is required")
    for item in ITEMS:
        if not (args.source / item).exists():
            raise SystemExit(f"Missing: {args.source / item}")

    state = {}
    if args.state.exists():
        for line in args.state.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line); state[row["item"]] = row
    args.state.parent.mkdir(parents=True, exist_ok=True)
    api = HubApi(); api.login(token)
    info = api.repo_info(args.repo_id, repo_type="dataset")
    if getattr(info, "visibility", None) != 1:
        raise SystemExit(f"Refusing upload: visibility={getattr(info, 'visibility', None)}, expected private")
    selected = ITEMS[:args.max_items] if args.max_items else ITEMS
    for item in selected:
        src = args.source / item
        if state.get(item, {}).get("status") == "success":
            print(f"[DONE_BEFORE] {item}", flush=True); continue
        dest = f"SANDISK_ELE/{item}"
        started = time.time()
        try:
            for attempt in range(1, 6):
                try:
                    if src.is_dir():
                        result = api.upload_folder(
                            repo_id=args.repo_id, folder_path=src, path_in_repo=dest,
                            repo_type="dataset", token=token, max_workers=args.workers,
                            commit_message=f"Upload SANDISK selection: {item}",
                            commit_description="Selected SANDISK path; source remains unchanged.",
                        )
                    else:
                        result = api.upload_file(
                            path_or_fileobj=src, path_in_repo=dest, repo_id=args.repo_id,
                            repo_type="dataset", token=token,
                            commit_message=f"Upload SANDISK selection: {item}",
                            commit_description="Selected SANDISK file; source remains unchanged.",
                        )
                    break
                except Exception as exc:
                    if attempt == 5: raise
                    delay = min(120, 15 * attempt)
                    print(f"[RETRY {attempt}/5] {item}: {exc}; sleep {delay}s", flush=True)
                    time.sleep(delay)
            files = [p for p in src.rglob("*") if p.is_file()] if src.is_dir() else [src]
            row = {"item": item, "status": "success", "files": len(files), "bytes": sum(p.stat().st_size for p in files), "elapsed_s": round(time.time()-started,1), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            with args.state.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
            state[item] = row
            print(f"[DONE] {item} files={row['files']} bytes={row['bytes']}", flush=True)
        except Exception as exc:
            row = {"item": item, "status": "failed", "error": repr(exc), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            with args.state.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
            print(f"[FAIL] {item}: {exc}", flush=True)
            return 1
    print(f"finished state={args.state}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
