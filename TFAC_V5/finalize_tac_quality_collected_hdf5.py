"""Finalize one collected TacQuality rollout HDF5 into the scheduled path.

The guided DP server does not save rollout HDF5 files.  In practice, the robot
client or a separate collection script writes a file such as ``episode_0.hdf5``.
This helper copies or moves that collected file to the current schedule row's
``recommended_path`` and verifies the minimal schema required by the formal
TacQuality gates.

It is an execution utility only; it does not evaluate policy quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_finalize_collected_hdf5")
DEFAULT_NEXT_STEP = Path(
    "/home/chenshuai/Project/output/tac_quality_next_collection_step/"
    "formal_paired12/tac_quality_next_collection_step.json"
)
DEFAULT_SCORER_FREEZE_MANIFEST = Path(
    "/home/chenshuai/Project/output/tac_quality_scorer_freeze_manifest/"
    "tac_quality_scorer_freeze_manifest.json"
)
FORCE_KEYS = ("ft", "observations/tac/left/force6d", "observations/tac/right/force6d")
MARKER_KEYS = ("observations/tac/left/marker_offset", "observations/tac/right/marker_offset")
ACTION_KEYS = ("actions/joint_abs", "actions/eef_abs")


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_optional_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}")


def discover_hdf5(root: Path) -> List[Path]:
    if root.is_file() and root.suffix in {".hdf5", ".h5"}:
        return [root]
    if not root.exists():
        return []
    return sorted([*root.rglob("*.hdf5"), *root.rglob("*.h5")], key=lambda p: (p.stat().st_mtime, str(p)))


def newest_hdf5(root: Path) -> Optional[Path]:
    files = discover_hdf5(root)
    return files[-1] if files else None


def first_existing(f: h5py.File, keys: tuple[str, ...]) -> Optional[Dict[str, Any]]:
    for key in keys:
        if key in f:
            d = f[key]
            return {"key": key, "shape": [int(x) for x in d.shape], "dtype": str(d.dtype)}
    return None


def valid_len(info: Optional[Dict[str, Any]], min_steps: int) -> bool:
    return bool(info and info.get("shape") and info["shape"][0] >= min_steps)


def audit_hdf5(path: Path, min_steps: int) -> Dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "schema_ok": False,
            "reason": "file does not exist",
        }
    try:
        with h5py.File(path, "r") as f:
            force = first_existing(f, FORCE_KEYS)
            left_marker = first_existing(f, ("observations/tac/left/marker_offset",))
            right_marker = first_existing(f, ("observations/tac/right/marker_offset",))
            action = first_existing(f, ACTION_KEYS)
            attrs = {
                "success": "success" in f.attrs,
                "stopped_early": "stopped_early" in f.attrs,
                "tac_quality_scorer_freeze_manifest_sha256": "tac_quality_scorer_freeze_manifest_sha256" in f.attrs,
                "tac_quality_scorer_freeze_git_commit": "tac_quality_scorer_freeze_git_commit" in f.attrs,
            }
    except Exception as exc:
        return {
            "path": str(path),
            "exists": True,
            "schema_ok": False,
            "reason": f"failed to open/read HDF5: {exc}",
        }
    missing = []
    if not valid_len(force, min_steps):
        missing.append("force_source")
    if not valid_len(left_marker, min_steps):
        missing.append("observations/tac/left/marker_offset")
    if not valid_len(right_marker, min_steps):
        missing.append("observations/tac/right/marker_offset")
    if not valid_len(action, min_steps):
        missing.append("action_source")
    return {
        "path": str(path),
        "exists": True,
        "schema_ok": not missing,
        "missing_or_short_required_fields": missing,
        "force_source": force,
        "left_marker": left_marker,
        "right_marker": right_marker,
        "action_source": action,
        "attrs": attrs,
        "missing_optional_attrs": [name for name, present in attrs.items() if not present],
    }


def write_explicit_attrs(path: Path, *, success: Optional[bool], stopped_early: Optional[bool]) -> Dict[str, Any]:
    requested = {
        "success": success,
        "stopped_early": stopped_early,
    }
    written: Dict[str, bool] = {}
    if all(value is None for value in requested.values()):
        return {
            "requested": False,
            "written": written,
            "path": str(path),
        }
    with h5py.File(path, "a") as f:
        for name, value in requested.items():
            if value is None:
                continue
            f.attrs[name] = bool(value)
            written[name] = bool(value)
    return {
        "requested": True,
        "written": written,
        "path": str(path),
    }


def write_freeze_attrs(path: Path, manifest_path: Optional[str]) -> Dict[str, Any]:
    if not manifest_path:
        return {
            "requested": False,
            "written": {},
            "path": str(path),
        }
    manifest = Path(manifest_path)
    if not manifest.exists():
        return {
            "requested": True,
            "written": {},
            "path": str(path),
            "error": f"scorer freeze manifest not found: {manifest}",
        }
    try:
        data = load_json(manifest)
    except Exception as exc:
        return {
            "requested": True,
            "written": {},
            "path": str(path),
            "error": f"failed to read scorer freeze manifest: {exc}",
        }
    digest = sha256_file(manifest)
    written = {
        "tac_quality_scorer_freeze_manifest": str(manifest),
        "tac_quality_scorer_freeze_manifest_sha256": digest or "",
        "tac_quality_scorer_freeze_git_commit": str(data.get("git_commit", "")),
        "tac_quality_scorer_freeze_pass": bool(data.get("scorer_freeze_manifest_pass", False)),
    }
    with h5py.File(path, "a") as f:
        for name, value in written.items():
            f.attrs[name] = value
    return {
        "requested": True,
        "written": written,
        "path": str(path),
        "manifest": str(manifest),
        "manifest_sha256": digest,
    }


def choose_source(args: argparse.Namespace) -> Path:
    if args.source:
        return Path(args.source)
    if args.source_dir:
        found = newest_hdf5(Path(args.source_dir))
        if found is None:
            raise FileNotFoundError(f"No HDF5 found under --source_dir {args.source_dir}")
        return found
    raise ValueError("Provide --source or --source_dir")


def build(args: argparse.Namespace) -> Dict[str, Any]:
    next_step = load_json(Path(args.next_step))
    if not next_step.get("has_next_step"):
        raise RuntimeError("No pending next collection step; run post-collection instead.")
    row = next_step["next_row"]
    target = Path(next_step["recommended_path"])
    source = choose_source(args)
    source_audit = audit_hdf5(source, args.min_steps)
    explicit_success = parse_optional_bool(args.success)
    explicit_stopped_early = parse_optional_bool(args.stopped_early)

    operation = "none"
    copied_or_moved = False
    explicit_attrs = {
        "requested": explicit_success is not None or explicit_stopped_early is not None,
        "written": {},
        "path": str(target),
    }
    freeze_attrs: Dict[str, Any] = {
        "requested": bool(args.scorer_freeze_manifest),
        "written": {},
        "path": str(target),
    }
    refusal_reason = None
    if not source.exists():
        refusal_reason = "source file does not exist"
    elif source.resolve() == target.resolve():
        operation = "already_at_target"
        copied_or_moved = True
    elif target.exists() and not args.overwrite:
        refusal_reason = "target exists; pass --overwrite to replace it"
    elif not source_audit.get("schema_ok", False) and not args.allow_bad_schema:
        refusal_reason = "source schema check failed; pass --allow_bad_schema only for manual debugging"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            operation = "dry_run_move" if args.move else "dry_run_copy"
        elif args.move:
            shutil.move(str(source), str(target))
            operation = "move"
            copied_or_moved = True
        else:
            shutil.copy2(source, target)
            operation = "copy"
            copied_or_moved = True
        if copied_or_moved and not args.dry_run:
            explicit_attrs = write_explicit_attrs(
                target,
                success=explicit_success,
                stopped_early=explicit_stopped_early,
            )
            freeze_attrs = write_freeze_attrs(target, args.scorer_freeze_manifest)
            if freeze_attrs.get("error"):
                refusal_reason = freeze_attrs["error"]

    target_audit = audit_hdf5(target, args.min_steps)
    finalize_pass = bool(
        refusal_reason is None
        and source_audit.get("schema_ok", False)
        and (args.dry_run or target_audit.get("schema_ok", False))
        and str(target).endswith(str(row.get("recommended_filename")))
    )
    result = {
        "purpose": "Finalize one collected TacQuality rollout HDF5 into the schedule recommended_path.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "dry_run": bool(args.dry_run),
        "move": bool(args.move),
        "overwrite": bool(args.overwrite),
        "operation": operation,
        "copied_or_moved": copied_or_moved,
        "explicit_outcome_attrs": explicit_attrs,
        "scorer_freeze_attrs": freeze_attrs,
        "refusal_reason": refusal_reason,
        "finalize_pass": finalize_pass,
        "next_step": str(args.next_step),
        "next_row": row,
        "source": str(source),
        "target": str(target),
        "source_audit": source_audit,
        "target_audit": target_audit,
        "post_finalize_commands": [
            "python TFAC_V5/build_tac_quality_collection_progress.py",
            "python TFAC_V5/audit_tac_quality_rollout_hdf5_schema.py",
            "python TFAC_V5/build_tac_quality_rollout_pairing.py --tag formal_paired12",
            "python TFAC_V5/build_tac_quality_next_collection_step.py",
        ],
        "guardrails": [
            "Default mode copies the file and keeps the original; use --move only after manual confirmation.",
            "The target is never overwritten unless --overwrite is provided.",
            "success/stopped_early attrs are written only when explicitly provided by --success/--stopped_early.",
            "TacQuality scorer-freeze provenance attrs are written from --scorer_freeze_manifest by default.",
            "This utility only finalizes file placement and schema; it is not rollout quality evidence.",
        ],
    }
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    row = result["next_row"]
    lines = [
        "# TacQuality Finalize Collected HDF5",
        "",
        f"- finalize_pass: `{result['finalize_pass']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- dry_run: `{result['dry_run']}`",
        f"- operation: `{result['operation']}`",
        f"- explicit_outcome_attrs: `{result['explicit_outcome_attrs']}`",
        f"- scorer_freeze_attrs: `{result['scorer_freeze_attrs']}`",
        f"- refusal_reason: `{result['refusal_reason']}`",
        "",
        "## Schedule Row",
        "",
        f"- task: `{row['task']}`",
        f"- pair_id: `{row['pair_id']}`",
        f"- arm: `{row['arm']}`",
        f"- recommended_filename: `{row['recommended_filename']}`",
        "",
        "## Paths",
        "",
        f"- source: `{result['source']}`",
        f"- target: `{result['target']}`",
        "",
        "## Schema",
        "",
        f"- source_schema_ok: `{result['source_audit'].get('schema_ok')}`",
        f"- target_schema_ok: `{result['target_audit'].get('schema_ok')}`",
        f"- source_missing: `{result['source_audit'].get('missing_or_short_required_fields')}`",
        f"- target_missing: `{result['target_audit'].get('missing_or_short_required_fields')}`",
        "",
        "## After Finalize",
        "",
    ]
    for command in result["post_finalize_commands"]:
        lines.append(f"- `{command}`")
    lines.extend(["", "## Guardrails", ""])
    for item in result["guardrails"]:
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next_step", default=str(DEFAULT_NEXT_STEP))
    parser.add_argument("--source", default=None, help="Collected HDF5 file to finalize.")
    parser.add_argument("--source_dir", default=None, help="Use the newest HDF5 under this directory.")
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="formal_paired12_latest")
    parser.add_argument("--move", action="store_true", help="Move instead of copying the source HDF5.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow_bad_schema", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--min_steps", type=int, default=3)
    parser.add_argument("--success", default=None, help="Optional explicit rollout success attr: true/false.")
    parser.add_argument("--stopped_early", default=None, help="Optional explicit stopped_early attr: true/false.")
    parser.add_argument("--scorer_freeze_manifest", default=str(DEFAULT_SCORER_FREEZE_MANIFEST))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(args)
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "tac_quality_finalize_collected_hdf5.json"
    md_path = out_dir / "tac_quality_finalize_collected_hdf5.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "finalize_pass": result["finalize_pass"],
                "scientific_evidence": result["scientific_evidence"],
                "dry_run": result["dry_run"],
                "operation": result["operation"],
                "source": result["source"],
                "target": result["target"],
                "refusal_reason": result["refusal_reason"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
