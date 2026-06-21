#!/usr/bin/env python3
"""Sweep force-aware board TacQuality score weights.

This script does not train a new model.  It reuses the force-aware Foresight
gradient audit and compares several energy definitions on the same split.  The
goal is to avoid choosing a DP guidance scorer only because it classifies well;
the selected score should also produce a meaningful bounded action update.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.tac_quality_energy import eval_force_aware_foresight_guidance as eval_force


DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/force_aware_score_weight_sweep")


WEIGHT_PRESETS: list[dict[str, Any]] = [
    {
        "name": "default_margin_contact_center_smooth",
        "description": "Current force-aware score: band margin + contact - force-center - force-smooth.",
        "weights": {"band_margin": 1.0, "contact_logprob": 0.20, "force_center": 0.25, "force_smooth": 0.10, "action_smooth": 0.0},
    },
    {
        "name": "margin_only",
        "description": "Only good-vs-risk force-band margin; tests whether penalties are actually needed.",
        "weights": {"band_margin": 1.0, "contact_logprob": 0.0, "force_center": 0.0, "force_smooth": 0.0, "action_smooth": 0.0},
    },
    {
        "name": "margin_contact",
        "description": "Band margin plus contact confidence.",
        "weights": {"band_margin": 1.0, "contact_logprob": 0.20, "force_center": 0.0, "force_smooth": 0.0, "action_smooth": 0.0},
    },
    {
        "name": "margin_center",
        "description": "Band margin plus force-center penalty.",
        "weights": {"band_margin": 1.0, "contact_logprob": 0.0, "force_center": 0.25, "force_smooth": 0.0, "action_smooth": 0.0},
    },
    {
        "name": "margin_smooth",
        "description": "Band margin plus force-smoothness penalty.",
        "weights": {"band_margin": 1.0, "contact_logprob": 0.0, "force_center": 0.0, "force_smooth": 0.10, "action_smooth": 0.0},
    },
    {
        "name": "margin_action_smooth",
        "description": "Band margin plus action-acceleration smoothness penalty.",
        "weights": {"band_margin": 1.0, "contact_logprob": 0.0, "force_center": 0.0, "force_smooth": 0.0, "action_smooth": 0.05},
    },
    {
        "name": "margin_force_action_smooth",
        "description": "Band margin plus both predicted-force smoothness and action smoothness penalties.",
        "weights": {"band_margin": 1.0, "contact_logprob": 0.0, "force_center": 0.0, "force_smooth": 0.10, "action_smooth": 0.05},
    },
    {
        "name": "strong_smooth",
        "description": "More conservative force smoothness emphasis.",
        "weights": {"band_margin": 1.0, "contact_logprob": 0.20, "force_center": 0.20, "force_smooth": 0.30, "action_smooth": 0.0},
    },
    {
        "name": "strong_center",
        "description": "More conservative force magnitude centering emphasis.",
        "weights": {"band_margin": 1.0, "contact_logprob": 0.20, "force_center": 0.50, "force_smooth": 0.10, "action_smooth": 0.0},
    },
    {
        "name": "weak_margin_balanced_penalty",
        "description": "Lower band margin weight, stronger physical regularization.",
        "weights": {"band_margin": 0.6, "contact_logprob": 0.25, "force_center": 0.30, "force_smooth": 0.20, "action_smooth": 0.0},
    },
]


def metric(result: Dict[str, Any], dotted: str, default: float = 0.0) -> float:
    cur: Any = result
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    try:
        return float(cur)
    except Exception:
        return default


def guidance_score(row: Dict[str, Any]) -> float:
    """Rank useful guidance without rewarding unbounded action changes."""

    score_delta = row["score_delta_mean"]
    improve = row["improved_rate"]
    finite = row["finite_grad_rate"]
    positive = row["positive_grad_rate"]
    trust = row["trust_region_pass_rate"]
    auc = row["score_good_bad_auc"]
    bacc = row["band_balanced_acc"]
    action_delta = row["action_delta_norm_mean"]
    raw_delta = row["raw_action_delta_norm_mean"]

    # Saturate score delta so huge energy scales do not dominate.  Prefer
    # nontrivial action deltas, but penalize very large raw deltas.
    score_delta_term = min(score_delta / 3.0, 1.0)
    action_term = min(action_delta / 0.05, 1.0)
    raw_penalty = max(0.0, raw_delta - 0.75) * 0.25
    return (
        0.20 * auc
        + 0.15 * bacc
        + 0.20 * improve
        + 0.15 * finite
        + 0.10 * positive
        + 0.10 * trust
        + 0.06 * score_delta_term
        + 0.04 * action_term
        - raw_penalty
    )


def run_one(args: argparse.Namespace, preset: Dict[str, Any], out_root: Path) -> Dict[str, Any]:
    weights = preset["weights"]
    eval_args = argparse.Namespace(
        foresight_dir=args.foresight_dir,
        ckpt=args.ckpt,
        out_dir=out_root / preset["name"],
        split=args.split,
        samples_per_episode=args.samples_per_episode,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        gpu=args.gpu,
        refine_steps=args.refine_steps,
        action_step=args.action_step,
        max_total_delta=args.max_total_delta,
        w_band_margin=weights["band_margin"],
        w_contact=weights["contact_logprob"],
        w_force_center=weights["force_center"],
        w_force_smooth=weights["force_smooth"],
        w_action_smooth=weights["action_smooth"],
    )
    result = eval_force.run(eval_args)
    row = {
        "name": preset["name"],
        "description": preset["description"],
        "weights": weights,
        "audit_json": result["paths"]["json"],
        "num_samples": result["setup"]["num_samples"],
        "band_balanced_acc": metric(result, "scorer_metrics.band_balanced_acc"),
        "contact_acc": metric(result, "scorer_metrics.contact_acc"),
        "score_good_bad_auc": metric(result, "scorer_metrics.score_good_bad_auc"),
        "finite_grad_rate": metric(result, "guidance_metrics.finite_grad_rate"),
        "positive_grad_rate": metric(result, "guidance_metrics.positive_grad_rate"),
        "improved_rate": metric(result, "guidance_metrics.improved_rate"),
        "trust_region_pass_rate": metric(result, "guidance_metrics.trust_region_pass_rate"),
        "score_delta_mean": metric(result, "summaries.score_delta.mean"),
        "score_delta_p05": metric(result, "summaries.score_delta.p05"),
        "action_delta_norm_mean": metric(result, "summaries.action_delta_norm.mean"),
        "raw_action_delta_norm_mean": metric(result, "summaries.raw_action_delta_norm.mean"),
    }
    row["ranking_score"] = guidance_score(row)
    return row


def write_markdown(summary: Dict[str, Any], path: Path) -> None:
    rows = summary["rows_sorted"]
    lines = [
        "# Force-Aware Board Score Weight Sweep",
        "",
        "目的：比较擦黑板 force-aware TacQuality 能量函数的不同权重，判断哪个更适合 DP classifier/energy guidance。",
        "",
        "This is offline Foresight-gradient evidence only; it does not claim real robot improvement.",
        "",
        "## Setup",
        "",
        f"- split: `{summary['setup']['split']}`",
        f"- samples_per_episode: `{summary['setup']['samples_per_episode']}`",
        f"- max_samples: `{summary['setup']['max_samples']}`",
        f"- refine: steps `{summary['setup']['refine_steps']}`, action_step `{summary['setup']['action_step']}`, max_total_delta `{summary['setup']['max_total_delta']}`",
        f"- foresight ckpt: `{summary['setup']['ckpt']}`",
        "",
        "## Ranking",
        "",
        "| rank | preset | ranking | AUC | bACC | improve | score delta | action delta | raw delta |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(rows, start=1):
        lines.append(
            f"| {i} | `{row['name']}` | {row['ranking_score']:.4f} | "
            f"{row['score_good_bad_auc']:.4f} | {row['band_balanced_acc']:.4f} | "
            f"{row['improved_rate']:.4f} | {row['score_delta_mean']:.4f} | "
            f"{row['action_delta_norm_mean']:.4f} | {row['raw_action_delta_norm_mean']:.4f} |"
        )
    best = rows[0]
    lines.extend(
        [
            "",
            "## Best Preset",
            "",
            f"- name: `{best['name']}`",
            f"- weights: `{best['weights']}`",
            f"- why: ranking combines scorer separation, finite positive gradients, improvement rate, trust-region pass, and nontrivial bounded action update.",
            "",
            "## Interpretation",
            "",
            "- `margin_only` tests whether the learned force-band classifier alone is enough.",
            "- Center/smooth penalties are preferred only if they preserve AUC/improve while keeping action changes bounded and physically meaningful.",
            "- The final choice still needs real paired force_trace rollouts before claiming robot improvement.",
            "",
            "## All Presets",
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"### `{row['name']}`",
                "",
                f"- description: {row['description']}",
                f"- weights: `{row['weights']}`",
                f"- audit json: `{row['audit_json']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foresight_dir", type=Path, default=eval_force.DEFAULT_FORESIGHT_DIR)
    parser.add_argument("--ckpt", type=Path, default=eval_force.DEFAULT_CKPT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--samples_per_episode", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--refine_steps", type=int, default=4)
    parser.add_argument("--action_step", type=float, default=0.02)
    parser.add_argument("--max_total_delta", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_dir / stamp
    out_root.mkdir(parents=True, exist_ok=False)
    rows: List[Dict[str, Any]] = []
    for preset in WEIGHT_PRESETS:
        print(f"[sweep] {preset['name']} {preset['weights']}", flush=True)
        rows.append(run_one(args, preset, out_root))
    rows_sorted = sorted(rows, key=lambda r: r["ranking_score"], reverse=True)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "setup": {
            "foresight_dir": str(args.foresight_dir),
            "ckpt": str(args.ckpt),
            "split": args.split,
            "samples_per_episode": args.samples_per_episode,
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "refine_steps": args.refine_steps,
            "action_step": args.action_step,
            "max_total_delta": args.max_total_delta,
        },
        "rows": rows,
        "rows_sorted": rows_sorted,
        "best": rows_sorted[0],
        "evidence_boundary": (
            "Offline force-aware Foresight-gradient weight sweep only. "
            "No real robot improvement is claimed."
        ),
    }
    json_path = out_root / "force_aware_score_weight_sweep.json"
    md_path = out_root / "force_aware_score_weight_sweep.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "best": summary["best"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
