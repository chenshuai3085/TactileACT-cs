"""Build the TacQuality label/score standard registry.

A classifier/scorer for DP guidance is only meaningful if the good/bad and
quality-score targets are explicit.  This registry records the current
authoritative weak/human-derived standards for socket insertion and board
wiping, plus the evidence artifacts that validate those standards.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_label_standard_registry")

PATHS = {
    "insertion_latent_eval": Path("/home/chenshuai/Project/output/ptg_quality_eval/tactile_quality_model_eval.json"),
    "insertion_risk_eval": Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_eval.json"),
    "board_force_calibration": Path(
        "/home/chenshuai/Project/output/board_target_force_calibration/board_target_force_calibration.json"
    ),
    "board_label_scheme": Path(
        "/home/chenshuai/Project/output/board_quality_label_schemes/w32_s16/board_quality_scheme_eval.json"
    ),
    "ptg_proxy_eval": Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_eval.json"),
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


def build_registry(paths: Dict[str, Path]) -> Dict[str, Any]:
    data = {name: load_json(path) for name, path in paths.items()}
    board_cal = data["board_force_calibration"]
    board_scheme = data["board_label_scheme"]
    insertion_eval = data["insertion_latent_eval"]
    insertion_risk = data["insertion_risk_eval"]
    ptg = data["ptg_proxy_eval"]

    board_target = get(board_cal, "recommended.board_target_force")
    board_sigma = get(board_cal, "recommended.board_force_sigma")
    board_range = get(board_cal, "recommended.acceptable_mean_force_range")
    board_p95 = get(board_cal, "recommended.acceptable_force_p95_upper")

    registry = {
        "name": "TacQuality label/score standard registry",
        "purpose": (
            "Authoritative current definitions of good/bad labels and continuous quality targets "
            "for DP classifier-guidance scorer training/evaluation."
        ),
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "version": "2026-06-10",
        "tasks": {
            "insertion": {
                "data_source": "/home/chenshuai/data/dataset/0414 annotations.pkl + HDF5 marker/action streams",
                "authoritative_split": "episode-level GroupKFold; frame-level random split is diagnostic only",
                "reason_classes": {
                    "0": {
                        "name": "weak_approach",
                        "binary": "neutral",
                        "quality": 0.30,
                        "definition": "approach or weak contact frames not yet reliable insertion contact",
                        "use": "neutral for binary loss; retained for reason/quality calibration",
                    },
                    "1": {
                        "name": "good_insert",
                        "binary": "good",
                        "quality": 1.00,
                        "definition": "successful insertion/contact frames labeled insert",
                        "use": "positive target for insertion guidance",
                    },
                    "2": {
                        "name": "pre_bounce_risk",
                        "binary": "bad",
                        "quality": 0.05,
                        "definition": "frames immediately before lift/bounce in bounce episodes",
                        "use": "early negative target; primary proactive risk class",
                    },
                    "3": {
                        "name": "impact_or_recovery",
                        "binary": "bad",
                        "quality": 0.00,
                        "definition": "bounce/impact/recovery contact after collision risk has materialized",
                        "use": "hard negative target",
                    },
                },
                "binary_policy": {
                    "good": ["good_insert"],
                    "bad": ["pre_bounce_risk", "impact_or_recovery"],
                    "neutral": ["weak_approach"],
                    "why": (
                        "The user-defined bad insertion data are in bounce episodes.  The useful proactive "
                        "boundary is good insert versus pre-bounce/bounce-risk, not approach versus insert."
                    ),
                },
                "primary_metrics": {
                    "latent_insert_vs_prebounce_group_balanced_accuracy": get(
                        insertion_eval, "group_cv.LDA.balanced_accuracy.mean"
                    ),
                    "latent_insert_vs_prebounce_group_auc": get(insertion_eval, "group_cv.LDA.roc_auc.mean"),
                    "risk_scorer_binary_auc": get(insertion_risk, "mixed_group_cv.binary_auc.mean"),
                    "risk_scorer_balanced_accuracy": get(
                        insertion_risk, "mixed_group_cv.binary_balanced_accuracy.mean"
                    ),
                    "risk_scorer_quality_corr": get(insertion_risk, "mixed_group_cv.quality_corr.mean"),
                },
                "deployment_target": (
                    "Minimize pre-bounce/bounce-risk probability or maximize insertion quality score "
                    "through bounded final-clean-action guidance."
                ),
            },
            "board": {
                "data_source": "/home/chenshuai/data/dataset/260522_v8l_caheiban success HDF5 windows",
                "authoritative_split": "episode-level GroupKFold over board windows",
                "force_source": get(board_scheme, "best_classification.force_source", "left_force"),
                "window": get(board_scheme, "args.window"),
                "stride": get(board_scheme, "args.stride"),
                "target_force": board_target,
                "force_sigma": board_sigma,
                "acceptable_mean_force_range": board_range,
                "acceptable_force_p95_upper": board_p95,
                "reason_classes": {
                    "0": {
                        "name": "too_light",
                        "binary": "bad",
                        "quality": "low",
                        "definition": "mean force is below the suitable force band",
                        "use": "negative target: insufficient contact/cleaning force",
                    },
                    "1": {
                        "name": "good_smooth",
                        "binary": "good",
                        "quality": "high",
                        "definition": "force magnitude is in the suitable band and force/action variation is smooth",
                        "use": "positive target for board wiping guidance",
                    },
                    "2": {
                        "name": "too_heavy",
                        "binary": "bad",
                        "quality": "low",
                        "definition": "mean or peak force is above the suitable force band",
                        "use": "negative target: excessive force",
                    },
                    "3": {
                        "name": "rough_force",
                        "binary": "bad",
                        "quality": "low_mid",
                        "definition": "force changes are rough or jerky even if mean force is not too large",
                        "use": "negative/smoothness target",
                    },
                    "4": {
                        "name": "rough_motion",
                        "binary": "neutral_or_bad",
                        "quality": "mid",
                        "definition": "motion/action proxy is rough relative to smooth demonstrations",
                        "use": "reason class for smooth action guidance; not the primary force-magnitude negative",
                    },
                },
                "binary_policy": {
                    "good": ["good_smooth"],
                    "bad": ["too_light", "too_heavy", "rough_force"],
                    "neutral_or_contextual": ["rough_motion"],
                    "why": (
                        "The board task has no human labels, so weak labels are derived from suitable force "
                        "magnitude and smooth force/action change.  Good action requires both enough force and "
                        "smooth variation."
                    ),
                },
                "primary_metrics": {
                    "best_scheme": get(board_scheme, "best_classification.scheme"),
                    "best_feature_set": get(board_scheme, "best_classification.feature_set"),
                    "best_classifier": get(board_scheme, "best_classification.model"),
                    "balanced_accuracy": get(board_scheme, "best_classification.balanced_accuracy"),
                    "macro_f1": get(board_scheme, "best_classification.macro_f1"),
                    "good_auc": get(board_scheme, "best_classification.good_auc"),
                    "score_corr": get(board_scheme, "best_classification.score_corr"),
                    "regression_quality_corr": get(board_scheme, "best_regression.quality_corr"),
                    "ptg_proxy_binary_auc": get(ptg, "mixed_group_cv.binary_auc.mean"),
                    "ptg_proxy_quality_corr": get(ptg, "mixed_group_cv.quality_corr.mean"),
                },
                "deployment_target": (
                    "Maximize board quality score while keeping mean force near target_force, avoiding "
                    "too-light/too-heavy force, reducing force/action roughness, and respecting action range."
                ),
            },
        },
        "shared_guidance_policy": {
            "recommended_mode": "final clean-action bounded accept-only refinement",
            "not_allowed_as_final_claim": [
                "frame-level random split as primary accuracy",
                "score-only improvement without smoothness/range proxy checks",
                "fixed-step ActionAware guidance without line search",
                "synthetic smoke as real rollout evidence",
            ],
            "required_for_final_completion": [
                "formal insertion baseline-vs-guided real rollout gate",
                "formal board baseline-vs-guided real rollout gate",
                "formal insertion scorer ablation gate",
                "formal board scorer ablation gate",
            ],
        },
        "artifacts": {name: file_info(path) for name, path in paths.items()},
    }

    registry["registry_pass"] = bool(
        board_target is not None
        and board_sigma is not None
        and get(board_scheme, "best_classification.balanced_accuracy", 0.0) >= 0.88
        and get(board_scheme, "best_regression.quality_corr", 0.0) >= 0.95
        and get(insertion_risk, "mixed_group_cv.binary_auc.mean", 0.0) >= 0.95
        and get(insertion_eval, "data.n_groups", 0) >= 100
    )
    return registry


def write_markdown(registry: Dict[str, Any], path: Path) -> None:
    insertion = registry["tasks"]["insertion"]
    board = registry["tasks"]["board"]
    lines = [
        "# TacQuality Label/Score Standard Registry",
        "",
        f"- registry_pass: `{registry['registry_pass']}`",
        f"- scientific_evidence: `{registry['scientific_evidence']}`",
        f"- version: `{registry['version']}`",
        "",
        "## Insertion Standard",
        "",
        f"- split: {insertion['authoritative_split']}",
        f"- deployment_target: {insertion['deployment_target']}",
        "",
        "| id | name | binary | quality | definition |",
        "|---:|---|---|---:|---|",
    ]
    for key, row in insertion["reason_classes"].items():
        lines.append(f"| {key} | {row['name']} | {row['binary']} | {row['quality']} | {row['definition']} |")
    lines.extend(
        [
            "",
            "## Board Standard",
            "",
            f"- split: {board['authoritative_split']}",
            f"- target_force: `{board['target_force']}`",
            f"- force_sigma: `{board['force_sigma']}`",
            f"- acceptable_mean_force_range: `{board['acceptable_mean_force_range']}`",
            f"- acceptable_force_p95_upper: `{board['acceptable_force_p95_upper']}`",
            f"- deployment_target: {board['deployment_target']}",
            "",
            "| id | name | binary | quality | definition |",
            "|---:|---|---|---|---|",
        ]
    )
    for key, row in board["reason_classes"].items():
        lines.append(f"| {key} | {row['name']} | {row['binary']} | {row['quality']} | {row['definition']} |")
    lines.extend(
        [
            "",
            "## Shared Guidance Policy",
            "",
            f"- recommended_mode: `{registry['shared_guidance_policy']['recommended_mode']}`",
            "",
            "### Not Allowed As Final Claim",
            "",
        ]
    )
    for item in registry["shared_guidance_policy"]["not_allowed_as_final_claim"]:
        lines.append(f"- {item}")
    lines.extend(["", "### Required For Final Completion", ""])
    for item in registry["shared_guidance_policy"]["required_for_final_completion"]:
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = build_registry(PATHS)
    json_path = out_dir / "tac_quality_label_standard_registry.json"
    md_path = out_dir / "tac_quality_label_standard_registry.md"
    json_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(registry, md_path)
    print(
        json.dumps(
            {
                "registry_pass": registry["registry_pass"],
                "insertion_binary_policy": registry["tasks"]["insertion"]["binary_policy"],
                "board_target_force": registry["tasks"]["board"]["target_force"],
                "board_force_sigma": registry["tasks"]["board"]["force_sigma"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
