"""Prepare formal baseline-vs-guided rollout validation inputs.

This script does not evaluate whether TacQuality guidance is successful.  It
checks whether two rollout directories are ready for
eval_real_rollout_quality_gate.py and writes:

  - a readiness JSON/Markdown report;
  - a pairing CSV template, matched by sorted HDF5 order;
  - a metadata CSV template for task success / early-stop labels;
  - the exact command to run the formal rollout quality gate.

Use it after collecting baseline DP and TacQuality-guided DP HDF5 rollouts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import h5py


DEFAULT_OUT = Path("/home/chenshuai/Project/output/real_rollout_validation_ready")
REQUIRED_KEYS = [
    "ft",
    "observations/tac/left/force6d",
    "observations/tac/right/force6d",
    "observations/tac/left/marker_offset",
    "observations/tac/right/marker_offset",
]
ACTION_ALTERNATIVES = ["actions/eef_abs", "actions/joint_abs"]


def discover_hdf5(root: Path) -> List[Path]:
    if root.is_file() and root.suffix in {".hdf5", ".h5"}:
        return [root]
    return sorted([*root.rglob("*.hdf5"), *root.rglob("*.h5")])


def inspect_file(path: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "path": str(path),
        "stem": path.stem,
        "exists": path.exists(),
        "readable": False,
        "n_steps": None,
        "missing_required_keys": [],
        "has_action": False,
        "success_attr": None,
        "stopped_early_attr": None,
        "error": None,
    }
    try:
        with h5py.File(path, "r") as f:
            row["readable"] = True
            missing = [key for key in REQUIRED_KEYS if key not in f]
            row["missing_required_keys"] = missing
            row["has_action"] = any(key in f for key in ACTION_ALTERNATIVES)
            lengths = []
            for key in REQUIRED_KEYS + ACTION_ALTERNATIVES:
                if key in f and hasattr(f[key], "shape") and len(f[key].shape) > 0:
                    lengths.append(int(f[key].shape[0]))
            row["n_steps"] = min(lengths) if lengths else None
            row["success_attr"] = _attr_to_jsonable(f.attrs.get("success", None))
            row["stopped_early_attr"] = _attr_to_jsonable(f.attrs.get("stopped_early", None))
    except Exception as exc:  # pragma: no cover - report path, do not hide it.
        row["error"] = repr(exc)
    row["ready"] = bool(
        row["readable"]
        and not row["missing_required_keys"]
        and row["has_action"]
        and row["n_steps"] is not None
        and row["n_steps"] >= 8
    )
    return row


def _attr_to_jsonable(value: Any) -> Optional[Any]:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_pairing_template(baseline: List[Path], guided: List[Path], path: Path) -> None:
    n = min(len(baseline), len(guided))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "baseline", "guided"])
        for i in range(n):
            writer.writerow([f"trial_{i + 1:03d}", baseline[i].name, guided[i].name])


def write_metadata_template(paths: Iterable[Path], path: Path) -> None:
    seen = set()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "success", "stopped_early"])
        for p in paths:
            if p.stem in seen:
                continue
            seen.add(p.stem)
            writer.writerow([p.stem, "", ""])


def resolve_rollout_path(value: str, root: Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    direct = root / p
    if direct.exists():
        return direct
    matches = list(root.rglob(value))
    if len(matches) == 1:
        return matches[0]
    stem_matches = [m for m in root.rglob("*.hdf5") if m.stem == value]
    stem_matches += [m for m in root.rglob("*.h5") if m.stem == value]
    if len(stem_matches) == 1:
        return stem_matches[0]
    raise FileNotFoundError(f"Cannot resolve rollout path {value!r} under {root}")


def parse_boolish(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return None
    if text in {"1", "true", "yes", "y", "success", "succeeded", "pass"}:
        return 1.0
    if text in {"0", "false", "no", "n", "fail", "failed"}:
        return 0.0
    return float(value)


def inspect_pairing_csv(path: Optional[str], baseline_root: Path, guided_root: Path) -> Dict[str, Any]:
    if not path:
        return {"provided": False, "ready": None, "n_pairs": None, "issues": []}
    issues: List[str] = []
    rows = []
    csv_path = Path(path)
    if not csv_path.exists():
        return {"provided": True, "ready": False, "path": str(csv_path), "n_pairs": 0, "issues": ["pairing_csv does not exist"]}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        missing = {"baseline", "guided"} - fields
        if missing:
            issues.append(f"pairing_csv missing columns: {sorted(missing)}")
        for i, row in enumerate(reader):
            pair_id = row.get("pair_id") or f"pair_{i:04d}"
            try:
                baseline = resolve_rollout_path(row.get("baseline", ""), baseline_root)
            except Exception as exc:
                issues.append(f"{pair_id}: cannot resolve baseline {row.get('baseline')!r}: {exc}")
                baseline = None
            try:
                guided = resolve_rollout_path(row.get("guided", ""), guided_root)
            except Exception as exc:
                issues.append(f"{pair_id}: cannot resolve guided {row.get('guided')!r}: {exc}")
                guided = None
            rows.append(
                {
                    "pair_id": pair_id,
                    "baseline": str(baseline) if baseline else None,
                    "guided": str(guided) if guided else None,
                }
            )
    if not rows:
        issues.append("pairing_csv has no rows")
    return {
        "provided": True,
        "ready": not issues,
        "path": str(csv_path),
        "n_pairs": len(rows),
        "issues": issues,
        "pairs_preview": rows[:10],
    }


def metadata_keys_for_path(path: Path) -> List[str]:
    return [str(path), path.name, path.stem]


def inspect_metadata_csv(path: Optional[str], rollout_paths: List[Path]) -> Dict[str, Any]:
    if not path:
        return {"provided": False, "ready": None, "covered_files": 0, "issues": []}
    issues: List[str] = []
    csv_path = Path(path)
    if not csv_path.exists():
        return {"provided": True, "ready": False, "path": str(csv_path), "covered_files": 0, "issues": ["metadata_csv does not exist"]}
    metadata: Dict[str, Dict[str, float]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if not ({"file", "stem", "path"} & fields):
            issues.append("metadata_csv must contain at least one of columns: file, stem, path")
        for line_no, row in enumerate(reader, start=2):
            keys = []
            for col in ("file", "path"):
                if row.get(col):
                    p = Path(row[col])
                    keys.extend([str(p), p.name, p.stem])
            if row.get("stem"):
                keys.append(row["stem"])
            try:
                success = parse_boolish(row.get("success", row.get("task_success")))
                stopped = parse_boolish(row.get("stopped_early", row.get("early_stop")))
            except Exception as exc:
                issues.append(f"line {line_no}: invalid boolish value: {exc}")
                continue
            if success is None:
                issues.append(f"line {line_no}: missing success/task_success value")
            if stopped is None:
                issues.append(f"line {line_no}: missing stopped_early/early_stop value")
            if not keys:
                issues.append(f"line {line_no}: missing file/path/stem key")
                continue
            if success is not None and stopped is not None:
                for key in keys:
                    metadata[key] = {"success": success, "stopped_early": stopped}

    uncovered = []
    for p in rollout_paths:
        if not any(key in metadata for key in metadata_keys_for_path(p)):
            uncovered.append(str(p))
    if uncovered:
        issues.append(f"metadata_csv does not cover {len(uncovered)} rollout files")
    return {
        "provided": True,
        "ready": not issues,
        "path": str(csv_path),
        "covered_files": len(rollout_paths) - len(uncovered),
        "total_files": len(rollout_paths),
        "issues": issues,
        "uncovered_preview": uncovered[:10],
    }


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Real Rollout Validation Readiness",
        "",
        f"- task: `{result['task']}`",
        f"- ready_for_quality_gate: `{result['ready_for_quality_gate']}`",
        f"- baseline_n: `{result['baseline']['n']}`",
        f"- guided_n: `{result['guided']['n']}`",
        f"- min_episodes: `{result['min_episodes']}`",
        f"- pairing_csv_template: `{result['outputs']['pairing_csv_template']}`",
        f"- metadata_csv_template: `{result['outputs']['metadata_csv_template']}`",
        f"- pairing_csv_provided: `{result['pairing_csv_check']['provided']}`",
        f"- metadata_csv_provided: `{result['metadata_csv_check']['provided']}`",
        "",
        "## Gate Command",
        "",
        "```bash",
        result["gate_command"],
        "```",
        "",
        "## Blocking Issues",
        "",
    ]
    if result["blocking_issues"]:
        for issue in result["blocking_issues"]:
            lines.append(f"- {issue}")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Provided CSV Checks",
            "",
            "```json",
            json.dumps(
                {
                    "pairing_csv_check": result["pairing_csv_check"],
                    "metadata_csv_check": result["metadata_csv_check"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            "```",
        ]
    )
    lines.extend(["", "## Notes", ""])
    lines.extend(
        [
            "- Fill `success` and `stopped_early` in the metadata CSV if the HDF5 attrs are absent.",
            "- Replace the pairing CSV ordering if baseline/guided trials are not sorted in matching order.",
            "- A readiness pass only means the gate can run; it does not mean guidance is validated.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n": len(rows),
        "ready_n": sum(bool(r["ready"]) for r in rows),
        "has_success_attr_n": sum(r["success_attr"] is not None for r in rows),
        "has_stopped_early_attr_n": sum(r["stopped_early_attr"] is not None for r in rows),
        "min_steps": min([r["n_steps"] for r in rows if r["n_steps"] is not None], default=None),
        "files": rows,
    }


def build(args: argparse.Namespace) -> Dict[str, Any]:
    baseline_root = Path(args.baseline_dir)
    guided_root = Path(args.guided_dir)
    baseline_paths = discover_hdf5(baseline_root)
    guided_paths = discover_hdf5(guided_root)
    baseline = [inspect_file(p) for p in baseline_paths]
    guided = [inspect_file(p) for p in guided_paths]

    out_dir = Path(args.output_dir) / (args.tag or f"{args.task}_validation_ready")
    out_dir.mkdir(parents=True, exist_ok=True)
    pairing_csv = out_dir / "pairing_template.csv"
    metadata_csv = out_dir / "metadata_template.csv"
    write_pairing_template(baseline_paths, guided_paths, pairing_csv)
    write_metadata_template([*baseline_paths, *guided_paths], metadata_csv)
    pairing_check = inspect_pairing_csv(args.pairing_csv, baseline_root, guided_root)
    metadata_check = inspect_metadata_csv(args.metadata_csv, [*baseline_paths, *guided_paths])

    blocking: List[str] = []
    if len(baseline_paths) < args.min_episodes:
        blocking.append(f"baseline has {len(baseline_paths)} files, needs >= {args.min_episodes}")
    if len(guided_paths) < args.min_episodes:
        blocking.append(f"guided has {len(guided_paths)} files, needs >= {args.min_episodes}")
    bad_baseline = [r["path"] for r in baseline if not r["ready"]]
    bad_guided = [r["path"] for r in guided if not r["ready"]]
    if bad_baseline:
        blocking.append(f"{len(bad_baseline)} baseline files are not gate-ready")
    if bad_guided:
        blocking.append(f"{len(bad_guided)} guided files are not gate-ready")
    metadata_ready = bool(metadata_check["provided"] and metadata_check["ready"])
    if not all(r["success_attr"] is not None for r in baseline + guided) and not metadata_ready:
        blocking.append("some files lack success attr; fill metadata_template.csv or provide --metadata_csv")
    if not all(r["stopped_early_attr"] is not None for r in baseline + guided) and not metadata_ready:
        blocking.append("some files lack stopped_early attr; fill metadata_template.csv or provide --metadata_csv")
    if pairing_check["provided"] and not pairing_check["ready"]:
        blocking.extend([f"pairing_csv: {issue}" for issue in pairing_check["issues"]])
    if metadata_check["provided"] and not metadata_check["ready"]:
        blocking.extend([f"metadata_csv: {issue}" for issue in metadata_check["issues"]])

    pairing_for_command = Path(args.pairing_csv) if args.pairing_csv else pairing_csv
    metadata_for_command = Path(args.metadata_csv) if args.metadata_csv else metadata_csv
    gate_command = (
        f"python TFAC_V5/eval_real_rollout_quality_gate.py "
        f"--task {args.task} "
        f"--baseline_dir {baseline_root} "
        f"--guided_dir {guided_root} "
        f"--pairing_csv {pairing_for_command} "
        f"--metadata_csv {metadata_for_command} "
        f"--output_dir /home/chenshuai/Project/output/real_rollout_quality_gate "
        f"--tag {args.task}_baseline_vs_guided"
    )
    result = {
        "task": args.task,
        "baseline_dir": str(baseline_root),
        "guided_dir": str(guided_root),
        "min_episodes": int(args.min_episodes),
        "ready_for_quality_gate": not blocking,
        "blocking_issues": blocking,
        "baseline": summarize_group(baseline),
        "guided": summarize_group(guided),
        "pairing_csv_check": pairing_check,
        "metadata_csv_check": metadata_check,
        "outputs": {
            "pairing_csv_template": str(pairing_csv),
            "metadata_csv_template": str(metadata_csv),
        },
        "gate_command": gate_command,
    }
    json_path = out_dir / "real_rollout_validation_readiness.json"
    md_path = out_dir / "real_rollout_validation_readiness.md"
    result["outputs"]["json"] = str(json_path)
    result["outputs"]["markdown"] = str(md_path)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["insertion", "board"], required=True)
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--guided_dir", required=True)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--tag", default=None)
    parser.add_argument("--min_episodes", type=int, default=10)
    parser.add_argument("--pairing_csv", default=None)
    parser.add_argument("--metadata_csv", default=None)
    return parser.parse_args()


def main() -> None:
    result = build(parse_args())
    print(
        json.dumps(
            {
                "ready_for_quality_gate": result["ready_for_quality_gate"],
                "baseline_n": result["baseline"]["n"],
                "guided_n": result["guided"]["n"],
                "blocking_issues": result["blocking_issues"],
                "json": result["outputs"]["json"],
                "markdown": result["outputs"]["markdown"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
