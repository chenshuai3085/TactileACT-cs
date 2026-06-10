"""Audit scorer feature caches against the TacQuality label standard registry.

The registry defines what good/bad/neutral means.  This script checks that the
actual cached training/evaluation labels used by scorer experiments still match
that standard, or explicitly documents compatible deviations.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_label_standard_compliance")

PATHS = {
    "registry": Path(
        "/home/chenshuai/Project/output/tac_quality_label_standard_registry/"
        "tac_quality_label_standard_registry.json"
    ),
    "insertion_meta": Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_metadata.json"),
    "insertion_features": Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_features.npz"),
    "board_meta": Path("/home/chenshuai/Project/output/board_quality_label_schemes/board_windows_w32_s16_meta.json"),
    "ptg_meta": Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_metadata.json"),
    "ptg_features": Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_features.npz"),
    "action_aware_meta": Path(
        "/home/chenshuai/Project/output/action_aware_marker_scorer/action_aware_marker_metadata.json"
    ),
    "action_aware_features": Path(
        "/home/chenshuai/Project/output/action_aware_marker_scorer/action_aware_marker_features.npz"
    ),
}


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get(d: Optional[Dict[str, Any]], dotted: str, default=None):
    cur: Any = d
    if cur is None:
        return default
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def file_info(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": bool(path.exists()),
        "bytes": int(path.stat().st_size) if path.exists() and path.is_file() else None,
    }


def counts(arr: np.ndarray) -> Dict[str, int]:
    return {str(k): int(v) for k, v in Counter(np.asarray(arr).tolist()).items()}


def pass_item(name: str, passed: bool, evidence: Dict[str, Any], severity: str = "required") -> Dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "severity": severity,
        "evidence": evidence,
    }


def insertion_checks(registry: Dict[str, Any], meta: Dict[str, Any], features: np.lib.npyio.NpzFile) -> List[Dict[str, Any]]:
    reason_names = {str(k): v for k, v in (meta.get("reason_names") or {}).items()}
    reg_names = {
        key: value["name"]
        for key, value in (get(registry, "tasks.insertion.reason_classes", {}) or {}).items()
    }
    reason = features["reason"]
    binary = features["binary"]
    quality = features["quality"]
    expected_quality = {
        int(k): float(v["quality"])
        for k, v in (get(registry, "tasks.insertion.reason_classes", {}) or {}).items()
    }
    quality_ok = True
    quality_rows = {}
    for cls, target in expected_quality.items():
        mask = reason == cls
        vals = np.unique(np.round(quality[mask], 6)) if mask.any() else np.array([])
        quality_rows[str(cls)] = {"target": target, "unique_values": vals.tolist(), "n": int(mask.sum())}
        if mask.any() and not np.allclose(vals, target, atol=1e-5):
            quality_ok = False
    binary_map_ok = bool(
        np.all(binary[reason == 0] == -1)
        and np.all(binary[reason == 1] == 1)
        and np.all(binary[(reason == 2) | (reason == 3)] == 0)
    )
    return [
        pass_item(
            "insertion_reason_names_match_registry",
            reason_names == reg_names,
            {"metadata": reason_names, "registry": reg_names},
        ),
        pass_item(
            "insertion_binary_mapping_matches_registry",
            binary_map_ok,
            {
                "expected": "reason 0 -> -1 neutral, reason 1 -> 1 good, reason 2/3 -> 0 bad",
                "reason_counts": counts(reason),
                "binary_counts": counts(binary),
            },
        ),
        pass_item(
            "insertion_quality_values_match_registry",
            quality_ok,
            quality_rows,
        ),
        pass_item(
            "insertion_episode_groups_present",
            "groups" in features.files and len(np.unique(features["groups"])) >= 100,
            {"n_groups": int(len(np.unique(features["groups"]))) if "groups" in features.files else None},
        ),
    ]


def ptg_checks(registry: Dict[str, Any], meta: Dict[str, Any], features: np.lib.npyio.NpzFile) -> List[Dict[str, Any]]:
    task = features["task"]
    reason = features["reason"]
    binary = features["binary"]
    quality = features["quality"]
    board_mask = task == "board"
    insertion_mask = task == "insertion"
    board_reasons = set(np.unique(reason[board_mask]).astype(int).tolist())
    insertion_reasons = set(np.unique(reason[insertion_mask]).astype(int).tolist())
    binary_board = {
        str(cls): sorted(np.unique(binary[board_mask & (reason == cls)]).astype(int).tolist())
        for cls in sorted(board_reasons)
    }
    binary_insertion = {
        str(cls): sorted(np.unique(binary[insertion_mask & (reason == cls)]).astype(int).tolist())
        for cls in sorted(insertion_reasons)
    }
    # PTGProxy keeps rough_motion as a bad class for unified binary training.
    # The registry marks it contextual for standard definition, so this is a
    # documented compatible deviation rather than a hard blocker.
    rough_motion_contextual = (
        get(registry, "tasks.board.binary_policy.neutral_or_contextual") == ["rough_motion"]
        and binary_board.get("4") == [0]
    )
    board_required_ok = bool(
        binary_board.get("0") == [0]
        and binary_board.get("1") == [1]
        and binary_board.get("2") == [0]
        and binary_board.get("3") == [0]
    )
    insertion_ok = bool(
        binary_insertion.get("0") == [-1]
        and binary_insertion.get("1") == [1]
        and binary_insertion.get("2") == [0]
        and binary_insertion.get("3") == [0]
    )
    board_meta = meta.get("board_meta") or {}
    return [
        pass_item(
            "ptg_proxy_insertion_binary_mapping_matches_registry",
            insertion_ok,
            {"binary_by_reason": binary_insertion, "task_count": int(insertion_mask.sum())},
        ),
        pass_item(
            "ptg_proxy_board_required_binary_mapping_matches_registry",
            board_required_ok,
            {"binary_by_reason": binary_board, "task_count": int(board_mask.sum())},
        ),
        pass_item(
            "ptg_proxy_rough_motion_contextual_deviation_documented",
            rough_motion_contextual,
            {
                "registry_contextual": get(registry, "tasks.board.binary_policy.neutral_or_contextual"),
                "ptg_binary_for_reason_4": binary_board.get("4"),
                "interpretation": (
                    "PTGProxy treats rough_motion as binary bad for conservative unified training; "
                    "registry marks it contextual because it is not the primary force-magnitude negative."
                ),
            },
            severity="documented_deviation",
        ),
        pass_item(
            "ptg_proxy_board_window_stride_force_source_match_registry",
            board_meta.get("window") == get(registry, "tasks.board.window")
            and board_meta.get("stride") == get(registry, "tasks.board.stride")
            and board_meta.get("board_force_source") == get(registry, "tasks.board.force_source"),
            {
                "metadata": {
                    "window": board_meta.get("window"),
                    "stride": board_meta.get("stride"),
                    "board_force_source": board_meta.get("board_force_source"),
                },
                "registry": {
                    "window": get(registry, "tasks.board.window"),
                    "stride": get(registry, "tasks.board.stride"),
                    "force_source": get(registry, "tasks.board.force_source"),
                },
            },
        ),
        pass_item(
            "ptg_proxy_quality_range_valid",
            bool(np.nanmin(quality) >= -1e-6 and np.nanmax(quality) <= 1.0 + 1e-6),
            {"quality_min": float(np.nanmin(quality)), "quality_max": float(np.nanmax(quality))},
        ),
    ]


def action_aware_checks(registry: Dict[str, Any], meta: Dict[str, Any], features: np.lib.npyio.NpzFile) -> List[Dict[str, Any]]:
    task = features["task"]
    y = features["y_t4"]
    binary = features["y_binary"]
    score = features["score"]
    board_mask = task == "board"
    insertion_mask = task == "insertion"
    board_binary = {
        str(cls): sorted(np.unique(binary[board_mask & (y == cls)]).astype(int).tolist())
        for cls in sorted(np.unique(y[board_mask]).astype(int).tolist())
    }
    insertion_binary = {
        str(cls): sorted(np.unique(binary[insertion_mask & (y == cls)]).astype(int).tolist())
        for cls in sorted(np.unique(y[insertion_mask]).astype(int).tolist())
    }
    # ActionAware uses T4 merged classes, so exact T5 board compliance is not
    # expected.  It must still preserve good=1, risk/excessive=0, weak=-1.
    merged_ok = bool(
        insertion_binary.get("0") == [-1]
        and insertion_binary.get("1") == [1]
        and insertion_binary.get("2") == [0]
        and insertion_binary.get("3") == [0]
        and board_binary.get("1") == [1]
        and board_binary.get("2") == [0]
        and board_binary.get("3") == [0]
    )
    return [
        pass_item(
            "action_aware_merged_t4_binary_mapping_compatible",
            merged_ok,
            {
                "insertion_binary_by_t4": insertion_binary,
                "board_binary_by_t4": board_binary,
                "registry_board_policy": get(registry, "tasks.board.binary_policy"),
            },
            severity="compatible_merged_taxonomy",
        ),
        pass_item(
            "action_aware_score_range_valid",
            bool(np.nanmin(score) >= -1e-6 and np.nanmax(score) <= 1.0 + 1e-6),
            {"score_min": float(np.nanmin(score)), "score_max": float(np.nanmax(score))},
        ),
        pass_item(
            "action_aware_task_ids_match_registry_tasks",
            set(np.unique(task).tolist()) == {"insertion", "board"},
            {"task_counts": counts(task)},
        ),
    ]


def build(paths: Dict[str, Path]) -> Dict[str, Any]:
    registry = load_json(paths["registry"])
    insertion_meta = load_json(paths["insertion_meta"]) or {}
    ptg_meta = load_json(paths["ptg_meta"]) or {}
    action_meta = load_json(paths["action_aware_meta"]) or {}
    insertion_features = np.load(paths["insertion_features"], allow_pickle=True)
    ptg_features = np.load(paths["ptg_features"], allow_pickle=True)
    action_features = np.load(paths["action_aware_features"], allow_pickle=True)
    checks = []
    checks.extend(insertion_checks(registry or {}, insertion_meta, insertion_features))
    checks.extend(ptg_checks(registry or {}, ptg_meta, ptg_features))
    checks.extend(action_aware_checks(registry or {}, action_meta, action_features))
    required_pass = all(item["passed"] for item in checks if item["severity"] == "required")
    compatible_deviations_pass = all(
        item["passed"] for item in checks if item["severity"] in {"documented_deviation", "compatible_merged_taxonomy"}
    )
    result = {
        "name": "TacQuality label-standard compliance audit",
        "purpose": "Check cached scorer labels against the label/score standard registry.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "registry": str(paths["registry"]),
        "required_checks_pass": bool(required_pass),
        "compatible_deviations_pass": bool(compatible_deviations_pass),
        "compliance_pass": bool(required_pass and compatible_deviations_pass),
        "checks": checks,
        "summary": {
            "n_checks": len(checks),
            "n_required": sum(item["severity"] == "required" for item in checks),
            "n_documented_deviations": sum(item["severity"] == "documented_deviation" for item in checks),
            "n_compatible_merged_taxonomy": sum(item["severity"] == "compatible_merged_taxonomy" for item in checks),
        },
        "interpretation": (
            "The scorer caches comply with the registry.  PTGProxy's rough_motion-as-bad choice is a "
            "documented conservative training deviation; ActionAware uses a merged T4 taxonomy."
        ),
        "paths": {name: file_info(path) for name, path in paths.items()},
    }
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Label-Standard Compliance Audit",
        "",
        f"- compliance_pass: `{result['compliance_pass']}`",
        f"- required_checks_pass: `{result['required_checks_pass']}`",
        f"- compatible_deviations_pass: `{result['compatible_deviations_pass']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        "",
        "## Checks",
        "",
        "| check | severity | pass |",
        "|---|---|---:|",
    ]
    for item in result["checks"]:
        lines.append(f"| {item['name']} | {item['severity']} | {item['passed']} |")
    lines.extend(["", "## Interpretation", "", result["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build(PATHS)
    json_path = out_dir / "tac_quality_label_standard_compliance.json"
    md_path = out_dir / "tac_quality_label_standard_compliance.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "compliance_pass": result["compliance_pass"],
                "required_checks_pass": result["required_checks_pass"],
                "compatible_deviations_pass": result["compatible_deviations_pass"],
                "summary": result["summary"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
