#!/usr/bin/env python3
"""Audit insertion scorer label separation.

This is a lightweight evidence extractor, not a training job.  It loads the
already trained insertion risk scorer and the saved insertion risk feature
windows, then checks whether the deployed guidance score

    good_margin = logit(good_insert) - logit(bad)

is aligned with the intended labels:

    good_insert  vs  pre_bounce_risk / impact_or_recovery

`weak_approach` is reported as neutral and is not used in the binary pass/fail
checks.  The audit complements the existing gradient/config-consistency checks:
it answers whether the score itself encodes the intended good/bad standard.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.tac_quality_energy.insertion_runtime import (  # noqa: E402
    DEFAULT_CKPT,
    InsertionRiskScorerRuntime,
)


DEFAULT_FEATURES = Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_features.npz")
DEFAULT_METADATA = Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_metadata.json")
DEFAULT_OUT_ROOT = Path("/home/chenshuai/Project/output/insertion_label_separation")
REASON_NAMES = {
    0: "weak_approach",
    1: "good_insert",
    2: "pre_bounce_risk",
    3: "impact_or_recovery",
}
GOOD_REASON = 1
BAD_REASONS = (2, 3)
NEUTRAL_REASON = 0


def save_json(data: Mapping[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} JSON root is not an object")
    return data


def fnum(value: Any, ndigits: int = 4) -> str:
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return "NA"


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average tied ranks.
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            avg = (start + 1 + end) / 2.0
            ranks[order[start:end]] = avg
        start = end
    rank_sum_pos = ranks[labels].sum()
    n_pos = float(len(pos))
    n_neg = float(len(neg))
    return float((rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores)
    hits = labels[order].astype(np.float64)
    precision = np.cumsum(hits) / (np.arange(len(hits), dtype=np.float64) + 1.0)
    return float((precision * hits).sum() / n_pos)


def balanced_accuracy(labels: np.ndarray, pred: np.ndarray) -> float:
    labels = labels.astype(bool)
    pred = pred.astype(bool)
    pos = labels
    neg = ~labels
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    tpr = float((pred[pos] == 1).mean())
    tnr = float((pred[neg] == 0).mean())
    return 0.5 * (tpr + tnr)


def summarize(values: np.ndarray) -> dict[str, Any]:
    if len(values) == 0:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def grouped_mean(values: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(groups)
    means = np.array([float(np.mean(values[groups == g])) for g in unique], dtype=np.float64)
    return unique, means


def infer_scores(
    runtime: InsertionRiskScorerRuntime,
    marker: np.ndarray,
    action: np.ndarray,
    batch_size: int,
) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {
        "good_margin": [],
        "p_good": [],
        "quality_logit": [],
        "quality_score": [],
        "risk_prob": [],
        "reason_pred": [],
        "reason_prob": [],
    }
    for start in range(0, len(marker), batch_size):
        end = min(start + batch_size, len(marker))
        marker_t = torch.from_numpy(marker[start:end]).to(runtime.device)
        action_t = torch.from_numpy(action[start:end]).to(runtime.device)
        with torch.no_grad():
            out = runtime(left_marker_seq=marker_t, joint_action_seq=action_t)
        chunks["good_margin"].append(out["good_margin"].detach().cpu().numpy())
        chunks["p_good"].append(out["p_good"].detach().cpu().numpy())
        chunks["quality_logit"].append(out["quality_logit"].detach().cpu().numpy())
        chunks["quality_score"].append(out["quality_score"].detach().cpu().numpy())
        chunks["risk_prob"].append(out["risk_prob"].detach().cpu().numpy())
        reason_prob = out["reason_prob"].detach().cpu().numpy()
        chunks["reason_prob"].append(reason_prob)
        chunks["reason_pred"].append(reason_prob.argmax(axis=1))
    return {k: np.concatenate(v, axis=0) for k, v in chunks.items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    metadata = load_json(args.metadata_json) if args.metadata_json.exists() else {}
    reason_names = {
        int(k): str(v)
        for k, v in metadata.get("reason_names", REASON_NAMES).items()
    }
    data = np.load(args.features_npz, allow_pickle=True)
    marker = data["marker"].astype(np.float32)
    action = data["action"].astype(np.float32)
    reason = data["reason"].astype(np.int64)
    binary = data["binary"].astype(np.int64)
    groups = data["groups"].astype(str)
    sample_ids = data["sample_ids"].astype(str)
    ep_type = data["ep_type"].astype(str)

    runtime = InsertionRiskScorerRuntime(str(args.checkpoint), device=args.device)
    scores = infer_scores(runtime, marker, action, args.batch_size)
    good_margin = scores["good_margin"]
    p_good = scores["p_good"]
    quality_logit = scores["quality_logit"]
    risk_prob = scores["risk_prob"]
    reason_pred = scores["reason_pred"]
    reason_prob = scores["reason_prob"]

    good_mask = reason == GOOD_REASON
    bad_mask = np.isin(reason, BAD_REASONS)
    neutral_mask = reason == NEUTRAL_REASON
    eval_mask = good_mask | bad_mask
    eval_good = good_mask[eval_mask]
    eval_scores = good_margin[eval_mask]
    eval_p_good = p_good[eval_mask]

    pred_binary = eval_scores > args.good_margin_threshold
    reason_pred_eval = reason_pred[eval_mask]
    reason_true_eval = reason[eval_mask]

    labels: dict[str, Any] = {}
    for rid in sorted(np.unique(reason).tolist()):
        mask = reason == rid
        name = reason_names.get(int(rid), str(rid))
        labels[name] = {
            "reason_id": int(rid),
            "n": int(mask.sum()),
            "groups": int(len(np.unique(groups[mask]))),
            "ep_types": sorted(np.unique(ep_type[mask]).tolist()),
            "good_margin": summarize(good_margin[mask]),
            "p_good": summarize(p_good[mask]),
            "quality_logit": summarize(quality_logit[mask]),
            "risk_prob": summarize(risk_prob[mask]),
            "pred_reason_counts": {
                reason_names.get(int(k), str(int(k))): int(v)
                for k, v in zip(*np.unique(reason_pred[mask], return_counts=True))
            },
            "reason_prob_mean": {
                reason_names.get(i, str(i)): float(np.mean(reason_prob[mask, i]))
                for i in range(reason_prob.shape[1])
            },
        }

    good_score_mean = float(np.mean(good_margin[good_mask]))
    bad_score_means = {
        reason_names.get(rid, str(rid)): float(np.mean(good_margin[reason == rid]))
        for rid in BAD_REASONS
    }
    worst_bad_score = max(bad_score_means.values())
    neutral_score_mean = float(np.mean(good_margin[neutral_mask])) if neutral_mask.any() else float("nan")
    good_group_names, good_group_scores = grouped_mean(good_margin[good_mask], groups[good_mask])
    bad_group_names, bad_group_scores = grouped_mean(good_margin[bad_mask], groups[bad_mask])

    reason_acc = float(np.mean(reason_pred_eval == reason_true_eval)) if eval_mask.any() else float("nan")
    bad_reason_mask = reason_true_eval != GOOD_REASON
    bad_reason_acc = (
        float(np.mean(reason_pred_eval[bad_reason_mask] == reason_true_eval[bad_reason_mask]))
        if bad_reason_mask.any()
        else float("nan")
    )
    confusion: dict[str, dict[str, int]] = {}
    for true_id in sorted(np.unique(reason_true_eval).tolist()):
        row: dict[str, int] = {}
        true_name = reason_names.get(int(true_id), str(int(true_id)))
        for pred_id in sorted(np.unique(reason_pred_eval).tolist()):
            pred_name = reason_names.get(int(pred_id), str(int(pred_id)))
            row[pred_name] = int(np.sum((reason_true_eval == true_id) & (reason_pred_eval == pred_id)))
        confusion[true_name] = row

    metrics = {
        "sample_level": {
            "n_eval": int(eval_mask.sum()),
            "n_good": int(good_mask.sum()),
            "n_bad": int(bad_mask.sum()),
            "n_neutral": int(neutral_mask.sum()),
            "good_bad_auc_good_margin": binary_auc(eval_good, eval_scores),
            "good_bad_ap_good_margin": average_precision(eval_good, eval_scores),
            "good_bad_auc_p_good": binary_auc(eval_good, eval_p_good),
            "good_bad_balanced_acc_margin0": balanced_accuracy(eval_good, pred_binary),
            "reason_acc_excluding_neutral": reason_acc,
            "bad_reason_acc": bad_reason_acc,
        },
        "episode_group_level": {
            "good_groups": int(len(good_group_names)),
            "bad_groups": int(len(bad_group_names)),
            "good_bad_auc_group_mean_good_margin": binary_auc(
                np.concatenate([
                    np.ones(len(good_group_scores), dtype=bool),
                    np.zeros(len(bad_group_scores), dtype=bool),
                ]),
                np.concatenate([good_group_scores, bad_group_scores]),
            ),
            "good_group_score": summarize(good_group_scores),
            "bad_group_score": summarize(bad_group_scores),
        },
        "confusion_excluding_neutral": confusion,
    }

    separation = {
        "good_score_mean": good_score_mean,
        "bad_score_means": bad_score_means,
        "worst_bad_score_mean": worst_bad_score,
        "good_vs_worst_bad_margin": good_score_mean - worst_bad_score,
        "neutral_score_mean": neutral_score_mean,
        "good_p_good_mean": float(np.mean(p_good[good_mask])),
        "worst_bad_p_good_mean": max(
            float(np.mean(p_good[reason == rid]))
            for rid in BAD_REASONS
        ),
        "bad_risk_prob_mean": float(np.mean(risk_prob[bad_mask])),
        "good_risk_prob_mean": float(np.mean(risk_prob[good_mask])),
    }

    checks = [
        {
            "name": "sample_auc_good_margin",
            "pass": metrics["sample_level"]["good_bad_auc_good_margin"] >= args.min_auc,
            "value": metrics["sample_level"]["good_bad_auc_good_margin"],
            "threshold": args.min_auc,
        },
        {
            "name": "group_auc_good_margin",
            "pass": metrics["episode_group_level"]["good_bad_auc_group_mean_good_margin"] >= args.min_group_auc,
            "value": metrics["episode_group_level"]["good_bad_auc_group_mean_good_margin"],
            "threshold": args.min_group_auc,
        },
        {
            "name": "balanced_acc_margin0",
            "pass": metrics["sample_level"]["good_bad_balanced_acc_margin0"] >= args.min_balanced_acc,
            "value": metrics["sample_level"]["good_bad_balanced_acc_margin0"],
            "threshold": args.min_balanced_acc,
        },
        {
            "name": "good_margin_positive",
            "pass": separation["good_score_mean"] > args.min_good_score,
            "value": separation["good_score_mean"],
            "threshold": args.min_good_score,
        },
        {
            "name": "bad_margins_negative",
            "pass": all(v < args.max_bad_score for v in bad_score_means.values()),
            "value": bad_score_means,
            "threshold": args.max_bad_score,
        },
        {
            "name": "good_vs_worst_bad_margin",
            "pass": separation["good_vs_worst_bad_margin"] >= args.min_good_bad_margin,
            "value": separation["good_vs_worst_bad_margin"],
            "threshold": args.min_good_bad_margin,
        },
        {
            "name": "reason_acc_excluding_neutral",
            "pass": reason_acc >= args.min_reason_acc,
            "value": reason_acc,
            "threshold": args.min_reason_acc,
        },
    ]

    out_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pass": all(item["pass"] for item in checks),
        "features_npz": str(args.features_npz),
        "metadata_json": str(args.metadata_json),
        "checkpoint": str(args.checkpoint),
        "score_definition": "good_margin = binary_logits[:, good] - binary_logits[:, bad]",
        "label_definition": {
            "good": reason_names.get(GOOD_REASON, "good_insert"),
            "bad": [reason_names.get(r, str(r)) for r in BAD_REASONS],
            "neutral_report_only": reason_names.get(NEUTRAL_REASON, "weak_approach"),
        },
        "setup": {
            "device": args.device,
            "batch_size": args.batch_size,
            "num_samples": int(len(reason)),
            "num_groups": int(len(np.unique(groups))),
            "num_sample_ids": int(len(np.unique(sample_ids))),
            "good_margin_threshold": args.good_margin_threshold,
        },
        "metrics": metrics,
        "labels": labels,
        "separation": separation,
        "checks": checks,
        "evidence_boundary": (
            "Offline label-separation audit for the trained insertion scorer on saved "
            "feature windows. It proves score/label alignment on this dataset and "
            "episode-group summaries, not real robot insertion improvement."
        ),
        "paths": {
            "json": str(out_dir / "insertion_label_separation.json"),
            "markdown": str(out_dir / "insertion_label_separation.md"),
        },
    }
    save_json(result, out_dir / "insertion_label_separation.json")
    write_markdown(result, out_dir / "insertion_label_separation.md")
    return result


def write_markdown(result: Mapping[str, Any], path: Path) -> None:
    metrics = result["metrics"]
    sep = result["separation"]
    labels = result["labels"]
    lines = [
        "# Insertion Label Separation Audit",
        "",
        f"Created: `{result['created_at']}`",
        "",
        "## Summary",
        "",
        f"- pass: `{result['pass']}`",
        f"- features: `{result['features_npz']}`",
        f"- checkpoint: `{result['checkpoint']}`",
        f"- score: `{result['score_definition']}`",
        f"- labels: `{json.dumps(result['label_definition'], ensure_ascii=False)}`",
        f"- samples/groups: `{result['setup']['num_samples']}` / `{result['setup']['num_groups']}`",
        f"- sample AUC good_margin: `{fnum(metrics['sample_level']['good_bad_auc_good_margin'])}`",
        f"- group-mean AUC good_margin: `{fnum(metrics['episode_group_level']['good_bad_auc_group_mean_good_margin'])}`",
        f"- balanced accuracy at margin>0: `{fnum(metrics['sample_level']['good_bad_balanced_acc_margin0'])}`",
        f"- reason accuracy excluding neutral: `{fnum(metrics['sample_level']['reason_acc_excluding_neutral'])}`",
        f"- good score mean: `{fnum(sep['good_score_mean'])}`",
        f"- worst bad score mean: `{fnum(sep['worst_bad_score_mean'])}`",
        f"- good-vs-worst-bad margin: `{fnum(sep['good_vs_worst_bad_margin'])}`",
        f"- neutral score mean: `{fnum(sep['neutral_score_mean'])}`",
        "",
        "## By Label",
        "",
        "| label | n | groups | good_margin mean | p_good mean | risk_prob mean | quality_logit mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ["weak_approach", "good_insert", "pre_bounce_risk", "impact_or_recovery"]:
        item = labels.get(label)
        if not item:
            continue
        lines.append(
            f"| `{label}` | {item['n']} | {item['groups']} | "
            f"{fnum(item['good_margin'].get('mean'))} | "
            f"{fnum(item['p_good'].get('mean'), 6)} | "
            f"{fnum(item['risk_prob'].get('mean'), 6)} | "
            f"{fnum(item['quality_logit'].get('mean'))} |"
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
        "## Confusion Excluding Neutral",
        "",
        "```json",
        json.dumps(metrics["confusion_excluding_neutral"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Evidence Boundary",
        "",
        str(result["evidence_boundary"]),
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features_npz", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--metadata_json", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CKPT))
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--good_margin_threshold", type=float, default=0.0)
    parser.add_argument("--min_auc", type=float, default=0.98)
    parser.add_argument("--min_group_auc", type=float, default=0.95)
    parser.add_argument("--min_balanced_acc", type=float, default=0.90)
    parser.add_argument("--min_good_score", type=float, default=0.0)
    parser.add_argument("--max_bad_score", type=float, default=0.0)
    parser.add_argument("--min_good_bad_margin", type=float, default=8.0)
    parser.add_argument("--min_reason_acc", type=float, default=0.70)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "json": result["paths"]["json"],
        "markdown": result["paths"]["markdown"],
        "pass": result["pass"],
        "metrics": result["metrics"],
        "separation": result["separation"],
        "checks": result["checks"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
