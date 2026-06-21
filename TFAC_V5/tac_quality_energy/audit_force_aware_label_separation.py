#!/usr/bin/env python3
"""Audit force-aware board scorer label separation.

This is a lightweight evidence extractor, not a training job.  It reads an
existing held-out force-aware Foresight gradient audit and checks whether the
selected board score separates the actual board quality labels:

    good  vs  too_small / too_large / oscillate

The result is useful for answering a narrow but important question before real
rollouts: does the score encode the intended good/bad standard, or does it only
produce gradients?
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


DEFAULT_AUDIT = Path(
    "/home/chenshuai/Project/output/force_aware_score_weight_sweep/"
    "20260621_104503/margin_only/20260621_104505/audit_results.json"
)
DEFAULT_OUT_ROOT = Path("/home/chenshuai/Project/output/force_aware_label_separation")
EXPECTED_BAD_LABELS = ("too_small", "too_large", "oscillate")


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} JSON root is not an object")
    return data


def save_json(data: Mapping[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get(data: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def fnum(value: Any, ndigits: int = 4) -> str:
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return "NA"


def label_metrics(label_data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n": int(label_data.get("n", 0) or 0),
        "base_score_mean": as_float(get(label_data, "base_score.mean")),
        "base_score_median": as_float(get(label_data, "base_score.median")),
        "base_good_prob_mean": as_float(get(label_data, "base_good_prob.mean")),
        "score_delta_mean": as_float(get(label_data, "score_delta.mean")),
        "good_prob_delta_mean": as_float(get(label_data, "good_prob_delta.mean")),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    audit = load_json(args.audit_json)
    by_label = audit.get("by_label", {})
    if not isinstance(by_label, Mapping) or "good" not in by_label:
        raise ValueError(f"{args.audit_json} does not contain by_label.good")

    labels = {label: label_metrics(data) for label, data in by_label.items() if isinstance(data, Mapping)}
    good = labels["good"]
    bad_labels = {label: labels[label] for label in EXPECTED_BAD_LABELS if label in labels}
    if set(bad_labels) != set(EXPECTED_BAD_LABELS):
        missing = sorted(set(EXPECTED_BAD_LABELS) - set(bad_labels))
        raise ValueError(f"Missing expected bad labels in audit: {missing}")

    worst_bad_score = max(v["base_score_mean"] for v in bad_labels.values())
    worst_bad_good_prob = max(v["base_good_prob_mean"] for v in bad_labels.values())
    good_vs_worst_bad_margin = good["base_score_mean"] - worst_bad_score

    checks = [
        {
            "name": "score_good_bad_auc",
            "pass": as_float(get(audit, "scorer_metrics.score_good_bad_auc")) >= args.min_auc,
            "value": as_float(get(audit, "scorer_metrics.score_good_bad_auc")),
            "threshold": args.min_auc,
        },
        {
            "name": "force_band_balanced_acc",
            "pass": as_float(get(audit, "scorer_metrics.band_balanced_acc")) >= args.min_band_bacc,
            "value": as_float(get(audit, "scorer_metrics.band_balanced_acc")),
            "threshold": args.min_band_bacc,
        },
        {
            "name": "good_score_positive",
            "pass": good["base_score_mean"] > args.min_good_score,
            "value": good["base_score_mean"],
            "threshold": args.min_good_score,
        },
        {
            "name": "all_bad_scores_negative",
            "pass": all(v["base_score_mean"] < args.max_bad_score for v in bad_labels.values()),
            "value": {k: v["base_score_mean"] for k, v in bad_labels.items()},
            "threshold": args.max_bad_score,
        },
        {
            "name": "good_vs_bad_margin",
            "pass": good_vs_worst_bad_margin >= args.min_good_bad_margin,
            "value": good_vs_worst_bad_margin,
            "threshold": args.min_good_bad_margin,
        },
        {
            "name": "good_probability_high",
            "pass": good["base_good_prob_mean"] >= args.min_good_prob,
            "value": good["base_good_prob_mean"],
            "threshold": args.min_good_prob,
        },
        {
            "name": "bad_probability_low",
            "pass": worst_bad_good_prob <= args.max_bad_good_prob,
            "value": worst_bad_good_prob,
            "threshold": args.max_bad_good_prob,
        },
    ]

    out_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pass": all(item["pass"] for item in checks),
        "source_audit": str(args.audit_json),
        "foresight_ckpt": get(audit, "setup.ckpt"),
        "split": get(audit, "setup.split"),
        "num_samples": get(audit, "setup.num_samples"),
        "score_weights": get(audit, "setup.score_weights"),
        "scorer_metrics": audit.get("scorer_metrics", {}),
        "guidance_metrics": audit.get("guidance_metrics", {}),
        "labels": labels,
        "separation": {
            "good_score_mean": good["base_score_mean"],
            "bad_score_means": {k: v["base_score_mean"] for k, v in bad_labels.items()},
            "worst_bad_score_mean": worst_bad_score,
            "good_vs_worst_bad_margin": good_vs_worst_bad_margin,
            "good_prob_mean": good["base_good_prob_mean"],
            "worst_bad_good_prob_mean": worst_bad_good_prob,
        },
        "checks": checks,
        "evidence_boundary": (
            "Held-out offline label-separation evidence for the force-aware board score. "
            "This proves score/label alignment on the audited split, not real robot improvement."
        ),
        "paths": {
            "json": str(out_dir / "force_aware_label_separation.json"),
            "markdown": str(out_dir / "force_aware_label_separation.md"),
        },
    }
    save_json(result, out_dir / "force_aware_label_separation.json")
    write_markdown(result, out_dir / "force_aware_label_separation.md")
    return result


def write_markdown(result: Mapping[str, Any], path: Path) -> None:
    labels = result["labels"]
    sep = result["separation"]
    lines = [
        "# Force-Aware Board Label Separation Audit",
        "",
        f"Created: `{result['created_at']}`",
        "",
        "## Summary",
        "",
        f"- pass: `{result['pass']}`",
        f"- source audit: `{result['source_audit']}`",
        f"- split / samples: `{result['split']}` / `{result['num_samples']}`",
        f"- score weights: `{json.dumps(result['score_weights'], ensure_ascii=False)}`",
        f"- force-band bACC: `{fnum(get(result, 'scorer_metrics.band_balanced_acc'))}`",
        f"- good/bad AUC: `{fnum(get(result, 'scorer_metrics.score_good_bad_auc'))}`",
        f"- good score mean: `{fnum(sep['good_score_mean'])}`",
        f"- worst bad score mean: `{fnum(sep['worst_bad_score_mean'])}`",
        f"- good-vs-worst-bad margin: `{fnum(sep['good_vs_worst_bad_margin'])}`",
        f"- good prob mean / worst bad good prob mean: "
        f"`{fnum(sep['good_prob_mean'])}` / `{fnum(sep['worst_bad_good_prob_mean'], 8)}`",
        "",
        "## By Label",
        "",
        "| label | n | score mean | score median | good prob mean | score delta mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in ["good", "too_small", "too_large", "oscillate"]:
        item = labels[label]
        lines.append(
            f"| `{label}` | {item['n']} | {item['base_score_mean']:.4f} | "
            f"{item['base_score_median']:.4f} | {item['base_good_prob_mean']:.8f} | "
            f"{item['score_delta_mean']:.4f} |"
        )
    lines.extend([
        "",
        "## Checks",
        "",
        "| check | pass | value | threshold |",
        "|---|---:|---|---:|",
    ])
    for item in result["checks"]:
        value = json.dumps(item["value"], ensure_ascii=False)
        lines.append(f"| `{item['name']}` | `{item['pass']}` | `{value}` | `{item['threshold']}` |")
    lines.extend([
        "",
        "## Evidence Boundary",
        "",
        str(result["evidence_boundary"]),
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit_json", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--min_auc", type=float, default=0.99)
    parser.add_argument("--min_band_bacc", type=float, default=0.95)
    parser.add_argument("--min_good_score", type=float, default=0.0)
    parser.add_argument("--max_bad_score", type=float, default=0.0)
    parser.add_argument("--min_good_bad_margin", type=float, default=10.0)
    parser.add_argument("--min_good_prob", type=float, default=0.90)
    parser.add_argument("--max_bad_good_prob", type=float, default=0.01)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "json": result["paths"]["json"],
        "markdown": result["paths"]["markdown"],
        "pass": result["pass"],
        "separation": result["separation"],
        "checks": result["checks"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
