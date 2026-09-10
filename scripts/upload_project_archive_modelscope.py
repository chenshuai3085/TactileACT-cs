#!/usr/bin/env python3
"""Upload one large project archive to the existing private ModelScope dataset."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from modelscope.hub.api import HubApi


DEFAULT_FILE = "/home/chenshuai/Project/ForeTac_projects_merged_no_ckpt_20260910.zip"
DEFAULT_REPO = "chenshuai3085/pih-tactile-episodes-20260904"
DEFAULT_REPO_PATH = "project_archives/ForeTac_projects_merged_no_ckpt_20260910.zip"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=DEFAULT_FILE)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--repo-path", default=DEFAULT_REPO_PATH)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    source = Path(args.file).resolve()
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if not source.is_file():
        raise SystemExit(f"Archive not found: {source}")
    if not token:
        raise SystemExit("MODELSCOPE_API_TOKEN is required")

    api = HubApi()
    api.login(token)
    info = api.repo_info(args.repo_id, repo_type="dataset")
    visibility = getattr(info, "visibility", None)
    if visibility != 1:
        raise SystemExit(f"Refusing upload: repository visibility={visibility}, expected private (1)")

    size_gib = source.stat().st_size / 1073741824
    print(f"Uploading {source} ({size_gib:.3f} GiB)")
    print(f"Destination: {args.repo_id}:{args.repo_path} (private dataset)", flush=True)
    for attempt in range(1, args.retries + 1):
        try:
            result = api.upload_file(
                path_or_fileobj=source,
                path_in_repo=args.repo_path,
                repo_id=args.repo_id,
                repo_type="dataset",
                token=token,
                commit_message="Upload merged project archive without checkpoint files",
                commit_description="Original local archive retained; includes project data/results as packaged.",
                disable_tqdm=False,
            )
            print(f"Upload succeeded: {result}")
            return 0
        except Exception as exc:
            if attempt >= args.retries:
                raise
            delay = min(120, 15 * attempt)
            print(f"Upload attempt {attempt}/{args.retries} failed: {exc}; retrying in {delay}s", flush=True)
            time.sleep(delay)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
