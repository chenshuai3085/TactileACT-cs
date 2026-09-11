#!/usr/bin/env python3
"""Merge the requested project archives while excluding model checkpoints."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import zipfile
from pathlib import Path


DEFAULT_PROJECT_DIR = Path("/home/chenshuai/Project")
DEFAULT_NAMES = [
    "HomePage-ForeTac.zip",
    "miACT.zip",
    "miACT (1).zip",
    "mi_pih.zip",
    "omnisf-master.zip",
    "TactileACT-cs.zip",
    "vtm-manuscript.zip",
]
CHECKPOINT_PARTS = {"ckpt", "checkpoint", "checkpoints"}
WEIGHT_SUFFIXES = {".pth", ".pt", ".ckpt", ".safetensors", ".onnx", ".bin"}
STORED_SUFFIXES = {
    ".7z", ".avi", ".deb", ".gif", ".jpg", ".jpeg", ".mp3", ".mp4", ".mov",
    ".pdf", ".png", ".pptx", ".qt", ".tar", ".tgz", ".gz", ".webm", ".webp",
    ".zip",
}
LARGE_DATA_SUFFIXES = {
    ".h5", ".hdf5", ".npy", ".npz", ".parquet", ".arrow", ".pkl", ".pickle",
    ".msgpack", ".mp4", ".mov", ".avi", ".mkv", ".webm", ".qt", ".gif",
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".wav", ".mp3",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
}
LARGE_PATH_PARTS = {
    "ckpt", "checkpoint", "checkpoints", "dataset", "datasets", "data", "episodes",
    "episode", "outputs", "output", "results", "runs", "wandb", "tensorboard",
    "videos", "video", "media", "assets",
}


def exclusion_reason(name: str, code_only: bool = False) -> str | None:
    normalized = name.replace("\\", "/")
    parts = {part.lower() for part in normalized.split("/")}
    suffix = Path(normalized).suffix.lower()
    if parts & CHECKPOINT_PARTS:
        return "checkpoint_path"
    if suffix in WEIGHT_SUFFIXES:
        return f"weight_suffix:{suffix}"
    if code_only:
        lower_parts = {part.lower() for part in normalized.split("/")}
        if lower_parts & LARGE_PATH_PARTS:
            return "code_only_large_path"
        if suffix in LARGE_DATA_SUFFIXES:
            return f"code_only_large_suffix:{suffix}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--code-only", action="store_true", help="Also exclude datasets, outputs, media and archives")
    args = parser.parse_args()

    inputs = [args.project_dir / name for name in DEFAULT_NAMES]
    missing = [str(p) for p in inputs if not p.is_file()]
    if missing:
        raise SystemExit("Missing input archives:\n" + "\n".join(missing))
    if args.output.resolve() in {p.resolve() for p in inputs}:
        raise SystemExit("Output must not overwrite an input archive")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    included = []
    excluded = []
    names_seen: set[str] = set()
    with zipfile.ZipFile(args.output, "w", allowZip64=True) as out:
        for source in inputs:
            with zipfile.ZipFile(source) as zf:
                for source_info in zf.infolist():
                    name = source_info.filename.replace("\\", "/")
                    if source_info.is_dir():
                        continue
                    reason = exclusion_reason(name, args.code_only)
                    if reason:
                        excluded.append({"archive": source.name, "path": name, "size": source_info.file_size, "reason": reason})
                        continue
                    if name in names_seen:
                        # Keep both projects losslessly if they happen to share a path.
                        name = f"__duplicate_from_{source.stem}/{name}"
                    names_seen.add(name)
                    info = copy.copy(source_info)
                    info.filename = name
                    suffix = Path(name).suffix.lower()
                    info.compress_type = zipfile.ZIP_STORED if suffix in STORED_SUFFIXES else zipfile.ZIP_DEFLATED
                    with zf.open(source_info, "r") as src, out.open(info, "w", force_zip64=True) as dst:
                        shutil.copyfileobj(src, dst, length=1024 * 1024)
                    included.append({"archive": source.name, "path": name, "size": source_info.file_size})

        manifest = {
            "description": "Merged project archives with checkpoints/model weight files excluded.",
            "inputs": DEFAULT_NAMES,
            "included_files": len(included),
            "included_bytes": sum(x["size"] for x in included),
            "excluded_files": len(excluded),
            "excluded_bytes": sum(x["size"] for x in excluded),
            "excluded_rules": [
                "path component ckpt/checkpoint/checkpoints",
                "suffix .pth/.pt/.ckpt/.safetensors/.onnx/.bin",
            ],
            "excluded": excluded,
        }
        manifest_data = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        out.writestr("ARCHIVE_MANIFEST_NO_CKPT.json", manifest_data, compress_type=zipfile.ZIP_DEFLATED)
        readme = (
            "This archive merges the seven requested project packages.\n"
            "Checkpoint directories and common model weight suffixes were excluded.\n"
            "Results/outputs and other non-checkpoint project files were retained.\n"
            "See ARCHIVE_MANIFEST_NO_CKPT.json for the exact excluded file list.\n"
        )
        out.writestr("ARCHIVE_README_NO_CKPT.txt", readme, compress_type=zipfile.ZIP_DEFLATED)

    print(json.dumps({
        "output": str(args.output),
        "included_files": len(included),
        "included_gib": sum(x["size"] for x in included) / 1073741824,
        "excluded_files": len(excluded),
        "excluded_gib": sum(x["size"] for x in excluded) / 1073741824,
        "output_gib": args.output.stat().st_size / 1073741824,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
