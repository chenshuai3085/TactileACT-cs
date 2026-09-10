#!/usr/bin/env python3
"""Stream a large SANDISK selection as a tar archive directly to ModelScope."""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import time
from pathlib import Path

from modelscope.hub.api import HubApi


SOURCE_ROOT = Path("/media/chenshuai/SANDISK ELE")
REPO_ID = "chenshuai3085/pih-tactile-episodes-20260904"
REPO_PATH = "project_archives/SANDISK_ELE_selected_20260910.tar"
ITEMS = [
    "111", "0708_video", "20251229_teleop_joint_next_state_as_action",
    "参赛注册-2986", "参赛注册-3012", "基于颗粒物衰减法的室内换气率无线测量装置",
    "数媒", "无忧搜寻-基于Hadoop和AI的智能救援系统", "云智教育APP", "证明材料",
    "证书", "智慧助老轮椅", "桌面整理", "A11-2891", "Cursor_workspace",
    "pi05_lerobot_cowa", "rm_models-main", "tactileact_output", "zhuomian",
    "证明材料.zip", "clean_dataset.py", "for_show.py",
]


def tar_command(root: Path) -> list[str]:
    return [
        "tar", "--create", "--file=-", "--format=posix", "--numeric-owner",
        "--directory", str(root), "--transform", "s,^,SANDISK_ELE/,", "--",
        *ITEMS,
    ]


class TarStream(io.BufferedIOBase):
    """Restartable read-only stream backed by a tar subprocess."""

    def __init__(self, command: list[str], size: int):
        self.command = command
        self.size = size
        self.position = 0
        self.process: subprocess.Popen | None = None
        self._start()

    def _start(self) -> None:
        if self.process is not None:
            self.process.stdout.close()
            self.process.wait()
        self.process = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_END:
            if offset != 0:
                raise OSError("only seek(0, SEEK_END) is supported")
            return self.size
        if whence == io.SEEK_SET and offset == 0:
            self._start()
            return 0
        raise OSError("stream only supports seek(0) and seek(0, SEEK_END)")

    def read(self, size: int = -1) -> bytes:
        if self.process is None or self.process.stdout is None:
            return b""
        data = self.process.stdout.read(size)
        self.position += len(data)
        return data

    def close(self) -> None:
        if self.process is not None:
            self.process.stdout.close()
            self.process.wait()
            self.process = None
        super().close()


def stream_size(command: list[str]) -> int:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    total = 0
    while True:
        chunk = process.stdout.read(16 * 1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    code = process.wait()
    if code:
        raise RuntimeError(f"tar size pass failed ({code}): {stderr[-2000:]}")
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--repo-path", default=REPO_PATH)
    args = parser.parse_args()
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        raise SystemExit("MODELSCOPE_API_TOKEN is required")
    for item in ITEMS:
        if not (args.source / item).exists():
            raise SystemExit(f"Missing source item: {args.source / item}")

    command = tar_command(args.source)
    print("Scanning source and calculating streamed tar size...", flush=True)
    size = stream_size(command)
    print(f"Archive stream size: {size / 1073741824:.3f} GiB", flush=True)
    api = HubApi()
    api.login(token)
    info = api.repo_info(args.repo_id, repo_type="dataset")
    if getattr(info, "visibility", None) != 1:
        raise SystemExit(f"Refusing upload: repository visibility={getattr(info, 'visibility', None)}, expected private")
    stream = TarStream(command, size)
    try:
        for attempt in range(1, 6):
            try:
                result = api.upload_file(
                    path_or_fileobj=stream,
                    path_in_repo=args.repo_path,
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    token=token,
                    commit_message="Upload complete SANDISK project selection as streamed tar archive",
                    commit_description="All requested SANDISK paths included; local source remained read-only and unchanged.",
                    disable_tqdm=False,
                )
                print(f"Upload succeeded: {result}", flush=True)
                return 0
            except Exception as exc:
                if attempt == 5:
                    raise
                delay = min(120, 15 * attempt)
                print(f"Upload attempt {attempt}/5 failed: {exc}; retrying in {delay}s", flush=True)
                time.sleep(delay)
                stream.seek(0)
    finally:
        stream.close()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
