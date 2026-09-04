#!/usr/bin/env python3
"""Upload external-drive PIH episode files to a private ModelScope dataset.

The source tree is read-only. Uploads are committed one source directory at a
time so an interrupted run can be resumed using the JSONL state file.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from modelscope.hub.api import HubApi


DEFAULT_SOURCE = "/media/chenshuai/EXTERNAL_USB/pih_dataset"
DEFAULT_REPO = "chenshuai3085/pih-tactile-episodes-20260904"
DEFAULT_STATE = "/home/chenshuai/Project/TactileACT-cs/tmp/modelscope_pih_upload_state.jsonl"
PATTERNS = ("*.hdf5", "*.h5", "*.mp4")


def file_inventory(source: Path) -> tuple[list[tuple[Path, list[Path]]], list[Path]]:
    by_dir: dict[Path, list[Path]] = {}
    skipped: list[Path] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".hdf5", ".h5", ".mp4"}:
            continue
        if path.stat().st_size == 0:
            skipped.append(path)
            continue
        by_dir.setdefault(path.parent, []).append(path)
    return sorted((directory, files) for directory, files in by_dir.items()), skipped


def load_state(path: Path) -> dict[str, dict]:
    state: dict[str, dict] = {}
    if not path.exists():
        return state
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            state[item["directory"]] = item
    return state


def append_state(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start-at", type=int, default=0)
    parser.add_argument("--max-dirs", type=int, default=0)
    args = parser.parse_args()

    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        raise SystemExit("MODELSCOPE_API_TOKEN is required")
    source = Path(args.source).resolve()
    state_path = Path(args.state)
    directories, skipped = file_inventory(source)
    state = load_state(state_path)
    print(f"source={source}")
    print(f"repo={args.repo_id} (private dataset expected)")
    print(f"directories={len(directories)} files={sum(len(x[1]) for x in directories)}")
    print(f"zero_byte_skipped={len(skipped)}")
    for path in skipped:
        print(f"[SKIP_ZERO] {path.relative_to(source)}")

    api = HubApi()
    # Explicit login keeps the run independent of cached cookies.
    api.login(token)
    info = api.repo_info(args.repo_id, repo_type="dataset")
    if getattr(info, "visibility", None) != 1:
        raise SystemExit(f"Refusing upload: repository visibility is {getattr(info, 'visibility', None)}, expected 1/private")

    selected = directories[args.start_at:]
    if args.max_dirs:
        selected = selected[: args.max_dirs]
    failures = 0
    for index, (directory, files) in enumerate(selected, start=args.start_at):
        relative = directory.relative_to(source).as_posix()
        if state.get(relative, {}).get("status") == "success":
            print(f"[{index + 1}/{len(directories)}] [DONE_BEFORE] {relative}", flush=True)
            continue
        started = time.time()
        try:
            # Network interruptions are common during multi-terabyte runs;
            # retry the whole directory while the SDK reuses already-uploaded blobs.
            last_error = None
            for attempt in range(1, 6):
                try:
                    result = api.upload_folder(
                        repo_id=args.repo_id,
                        folder_path=directory,
                        path_in_repo=relative,
                        allow_patterns=list(PATTERNS),
                        repo_type="dataset",
                        token=token,
                        max_workers=args.workers,
                        commit_message=f"Upload PIH episodes: {relative}",
                        commit_description="Private external-drive PIH episode data; source paths preserved.",
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt == 5:
                        raise
                    delay = min(60, 5 * attempt)
                    print(f"[RETRY {attempt}/5] {relative}: {exc}; sleeping {delay}s", flush=True)
                    time.sleep(delay)
            item = {
                "directory": relative,
                "status": "success",
                "files": len(files),
                "bytes": sum(p.stat().st_size for p in files),
                "elapsed_s": round(time.time() - started, 1),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            append_state(state_path, item)
            state[relative] = item
            print(f"[{index + 1}/{len(directories)}] [DONE] {relative} files={len(files)}", flush=True)
        except Exception as exc:  # keep later directories resumable
            failures += 1
            item = {"directory": relative, "status": "failed", "files": len(files), "error": repr(exc), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            append_state(state_path, item)
            state[relative] = item
            print(f"[{index + 1}/{len(directories)}] [FAIL] {relative}: {exc}", flush=True)
    print(f"finished failures={failures} state={state_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
