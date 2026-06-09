"""Plan sample size for formal real rollout quality validation.

The real rollout gate requires a positive quality delta with a positive lower
confidence bound, plus no degradation in task success / early stop / bad-rate
constraints.  This planner estimates how many baseline-vs-guided rollouts are
needed before collecting data.

It is a planning tool only.  The actual pass/fail decision remains
eval_real_rollout_quality_gate.py on recorded HDF5 rollouts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List


OUT_DIR = Path("/home/chenshuai/Project/output/real_rollout_sample_size_plan")
Z95_ONE_SIDED = 1.645


def normal_required_n(delta: float, std: float, *, paired: bool) -> int:
    if delta <= 0:
        return math.inf
    if std <= 1e-12:
        return 2 if paired else 4
    if paired:
        # One-sample CI on paired deltas: mean - z * sd/sqrt(n) > 0.
        return max(2, int(math.ceil((Z95_ONE_SIDED * std / delta) ** 2)))
    # Difference of two independent means with equal n per group.
    return max(2, int(math.ceil(2.0 * (Z95_ONE_SIDED * std / delta) ** 2)))


def success_drop_guard_n(max_success_rate_drop: float, baseline_success_rate: float, guided_success_rate: float) -> Dict[str, object]:
    margin = guided_success_rate - (baseline_success_rate - max_success_rate_drop)
    return {
        "baseline_success_rate": baseline_success_rate,
        "guided_success_rate": guided_success_rate,
        "max_success_rate_drop": max_success_rate_drop,
        "nominal_margin": margin,
        "nominal_constraint_ok": bool(margin >= 0.0),
        "note": (
            "The rollout gate checks observed success/stopped rates directly.  "
            "Use metadata_csv so these constraints are explicit."
        ),
    }


def build(args: argparse.Namespace) -> Dict[str, object]:
    planned_delta = max(args.expected_quality_delta, args.min_quality_delta)
    paired_n = normal_required_n(planned_delta, args.expected_paired_delta_std, paired=True)
    unpaired_n = normal_required_n(planned_delta, args.expected_group_quality_std, paired=False)
    recommended_paired = max(args.min_episodes, paired_n, args.min_recommended)
    recommended_unpaired = max(args.min_episodes, unpaired_n, args.min_recommended)

    rows: List[Dict[str, object]] = []
    for delta in args.delta_grid:
        rows.append(
            {
                "quality_delta": delta,
                "paired_n_pairs": normal_required_n(delta, args.expected_paired_delta_std, paired=True),
                "unpaired_n_per_group": normal_required_n(delta, args.expected_group_quality_std, paired=False),
            }
        )

    result = {
        "task": args.task,
        "scope": "Planning only; run eval_real_rollout_quality_gate.py for the formal decision.",
        "gate_aligned_constraints": {
            "min_episodes": args.min_episodes,
            "min_quality_delta": args.min_quality_delta,
            "bootstrap_ci_lower_must_be_positive": True,
            "max_bad_rate_increase": args.max_bad_rate_increase,
            "max_success_rate_drop": args.max_success_rate_drop,
        },
        "assumptions": {
            "expected_quality_delta": args.expected_quality_delta,
            "planned_delta_for_ci": planned_delta,
            "expected_paired_delta_std": args.expected_paired_delta_std,
            "expected_group_quality_std": args.expected_group_quality_std,
            "normal_approximation_z_one_sided_95": Z95_ONE_SIDED,
        },
        "recommendation": {
            "paired_design_recommended": True,
            "paired_n_pairs": int(recommended_paired),
            "unpaired_n_per_group": int(recommended_unpaired),
            "minimum_collection_plan": (
                f"Collect at least {int(recommended_paired)} paired baseline/guided trials for {args.task}; "
                f"if unpaired, collect at least {int(recommended_unpaired)} baseline and "
                f"{int(recommended_unpaired)} guided rollouts."
            ),
            "why_paired": (
                "Paired trials reduce variance from scene/initial-condition differences and directly match "
                "the paired bootstrap CI used by the rollout gate when --pairing_csv is provided."
            ),
        },
        "success_rate_guard": success_drop_guard_n(
            args.max_success_rate_drop,
            args.expected_baseline_success_rate,
            args.expected_guided_success_rate,
        ),
        "delta_grid": rows,
        "commands": {
            "prepare": (
                "python TFAC_V5/prepare_real_rollout_validation.py "
                f"--task {args.task} --baseline_dir <baseline_dir> --guided_dir <guided_dir> "
                "--pairing_csv <pairs.csv> --metadata_csv <metadata.csv>"
            ),
            "gate": (
                "python TFAC_V5/eval_real_rollout_quality_gate.py "
                f"--task {args.task} --baseline_dir <baseline_dir> --guided_dir <guided_dir> "
                "--pairing_csv <pairs.csv> --metadata_csv <metadata.csv>"
            ),
        },
    }
    return result


def write_markdown(result: Dict[str, object], path: Path) -> None:
    rec = result["recommendation"]
    lines = [
        "# Real Rollout Sample Size Plan",
        "",
        f"- task: `{result['task']}`",
        f"- paired_n_pairs: `{rec['paired_n_pairs']}`",
        f"- unpaired_n_per_group: `{rec['unpaired_n_per_group']}`",
        f"- minimum_collection_plan: {rec['minimum_collection_plan']}",
        "",
        "## Assumptions",
        "",
        "```json",
        json.dumps(result["assumptions"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Delta Grid",
        "",
        "| quality_delta | paired_n_pairs | unpaired_n_per_group |",
        "|---:|---:|---:|",
    ]
    for row in result["delta_grid"]:
        lines.append(f"| {row['quality_delta']} | {row['paired_n_pairs']} | {row['unpaired_n_per_group']} |")
    lines.extend(
        [
            "",
            "## Commands",
            "",
            "```bash",
            result["commands"]["prepare"],
            result["commands"]["gate"],
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_delta_grid(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["insertion", "board"], required=True)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default=None)
    parser.add_argument("--min_episodes", type=int, default=10)
    parser.add_argument("--min_recommended", type=int, default=12)
    parser.add_argument("--min_quality_delta", type=float, default=0.03)
    parser.add_argument("--max_bad_rate_increase", type=float, default=0.05)
    parser.add_argument("--max_success_rate_drop", type=float, default=0.0)
    parser.add_argument("--expected_quality_delta", type=float, default=0.08)
    parser.add_argument("--expected_paired_delta_std", type=float, default=0.10)
    parser.add_argument("--expected_group_quality_std", type=float, default=0.18)
    parser.add_argument("--expected_baseline_success_rate", type=float, default=1.0)
    parser.add_argument("--expected_guided_success_rate", type=float, default=1.0)
    parser.add_argument("--delta_grid", type=parse_delta_grid, default=parse_delta_grid("0.03,0.05,0.08,0.10,0.15,0.20"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) / (args.tag or f"{args.task}_sample_size_plan")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build(args)
    json_path = out_dir / "real_rollout_sample_size_plan.json"
    md_path = out_dir / "real_rollout_sample_size_plan.md"
    result["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "task": result["task"],
                "paired_n_pairs": result["recommendation"]["paired_n_pairs"],
                "unpaired_n_per_group": result["recommendation"]["unpaired_n_per_group"],
                "json": str(json_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
