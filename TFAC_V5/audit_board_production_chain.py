"""Audit whether board-wiping PTG has production Foresight/DP evidence.

The current board evidence is strong but not final:

  scorer-level gradient readiness: passed
  learned surrogate action -> tactile -> score chain: passed
  production board DP/Foresight full chain: only valid if real checkpoints are
  trained on / configured for the board-wiping dataset.

This script makes that boundary machine-readable.  It scans known config files,
checks whether any production DP or Foresight config references the board
dataset, and summarizes existing board evidence.  It deliberately does not
count the surrogate checkpoint as production Foresight/DP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


BOARD_DATASET = Path("/home/chenshuai/data/dataset/260522_v8l_caheiban")
OUT_DIR = Path("/home/chenshuai/Project/output/board_production_chain_audit")

DEFAULT_ROOTS = [
    Path("/home/chenshuai/Project/output"),
    Path("/home/chenshuai/Project/TactileACT-cs"),
]

DEFAULT_EVIDENCE = {
    "board_readiness": Path(
        "/home/chenshuai/Project/output/board_guidance_readiness/board_ptg_v2_energy_readiness_N240_safe_step.json"
    ),
    "board_surrogate": Path("/home/chenshuai/Project/output/board_tactile_surrogate/board_tactile_surrogate_eval.json"),
    "ptg_proxy_v2_eval": Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_eval.json"),
    "evidence_summary": Path("/home/chenshuai/Project/output/ptg_guidance_evidence/ptg_guidance_evidence_summary.json"),
}

CONFIG_NAMES = {
    "args.json",
    "config.json",
    "dp_config.json",
    "cpm_config.json",
    "tfs_config.json",
    "tfm_config.json",
}

CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {"_value": data}
    except Exception as exc:
        return {"_load_error": str(exc)}


def get(d: Optional[Dict[str, Any]], dotted: str, default=None):
    if d is None:
        return default
    cur: Any = d
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def iter_config_files(roots: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.name in CONFIG_NAMES and path not in seen:
                seen.add(path)
                out.append(path)
    return sorted(out)


def normalize_path_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(normalize_path_text(v) for v in value)
    if isinstance(value, dict):
        return " ".join(normalize_path_text(v) for v in value.values())
    return str(value)


def dataset_covered(config: Dict[str, Any], dataset: Path) -> bool:
    text = normalize_path_text(config)
    dataset_str = str(dataset)
    dataset_name = dataset.name
    return dataset_str in text or dataset_name in text


def classify_config(path: Path, config: Dict[str, Any]) -> str:
    text = (str(path) + " " + normalize_path_text(config)).lower()
    if "surrogate" in text:
        return "surrogate"
    if "foresight" in text or "latent_foresight" in text:
        return "foresight"
    if "diffusion" in text or "/dp_" in text or "dp_config" in path.name or "num_train_timesteps" in text:
        return "dp"
    if "scorer" in text or "quality" in text:
        return "scorer"
    return "other"


def checkpoint_siblings(config_path: Path) -> List[str]:
    parent = config_path.parent
    if not parent.exists():
        return []
    ckpts = [
        str(p)
        for p in sorted(parent.iterdir())
        if p.is_file() and p.suffix.lower() in CHECKPOINT_SUFFIXES and "surrogate" not in p.name.lower()
    ]
    return ckpts[:20]


def summarize_config(path: Path, config: Dict[str, Any], dataset: Path) -> Dict[str, Any]:
    kind = classify_config(path, config)
    covered = dataset_covered(config, dataset)
    key_subset = {}
    for key in [
        "dataset_dir",
        "dataset_dirs",
        "hdf5_dir",
        "data_dir",
        "save_dir",
        "name",
        "model_name",
        "camera_names",
        "action_key",
        "proprio_key",
        "num_episodes",
        "n_train",
        "n_val",
        "variant",
    ]:
        if key in config:
            key_subset[key] = config[key]
    return {
        "path": str(path),
        "kind": kind,
        "covers_board_dataset": bool(covered),
        "has_checkpoint_sibling": bool(checkpoint_siblings(path)),
        "checkpoint_siblings": checkpoint_siblings(path),
        "keys": key_subset,
    }


def evidence_summary(paths: Dict[str, Path]) -> Dict[str, Any]:
    board_readiness = load_json(paths["board_readiness"])
    board_surrogate = load_json(paths["board_surrogate"])
    ptg_proxy = load_json(paths["ptg_proxy_v2_eval"])
    summary = load_json(paths["evidence_summary"])
    return {
        "board_scorer_quality": {
            "path": str(paths["ptg_proxy_v2_eval"]),
            "exists": ptg_proxy is not None,
            "mixed_binary_auc": get(ptg_proxy, "mixed_group_cv.binary_auc.mean"),
            "mixed_quality_corr": get(ptg_proxy, "mixed_group_cv.quality_corr.mean"),
        },
        "board_scorer_readiness": {
            "path": str(paths["board_readiness"]),
            "exists": board_readiness is not None,
            "passes": get(board_readiness, "interpretation.passes_board_guidance_readiness"),
            "score_improved_rate": get(board_readiness, "summary.score_improved_rate"),
            "score_delta_mean": get(board_readiness, "summary.score_delta.mean"),
        },
        "board_surrogate_full_chain": {
            "path": str(paths["board_surrogate"]),
            "exists": board_surrogate is not None,
            "passes": get(board_surrogate, "interpretation.passes_board_surrogate_full_chain"),
            "marker_mae_mean": get(board_surrogate, "eval.marker_mae.mean"),
            "score_improved_rate": get(board_surrogate, "guidance_probe.score_improved_rate"),
        },
        "global_evidence_summary": {
            "path": str(paths["evidence_summary"]),
            "exists": summary is not None,
            "objective_complete": get(summary, "completion_assessment.objective_complete"),
            "reason": get(summary, "completion_assessment.reason"),
        },
    }


def build_audit(args) -> Dict[str, Any]:
    roots = [Path(x) for x in args.search_roots]
    configs = []
    for path in iter_config_files(roots):
        data = load_json(path)
        if not data or "_load_error" in data:
            continue
        row = summarize_config(path, data, Path(args.board_dataset))
        if args.include_all_configs or row["covers_board_dataset"] or row["kind"] in {"dp", "foresight"}:
            configs.append(row)

    production_board = [
        row
        for row in configs
        if row["covers_board_dataset"] and row["kind"] in {"dp", "foresight"} and row["has_checkpoint_sibling"]
    ]
    board_dp = [row for row in production_board if row["kind"] == "dp"]
    board_foresight = [row for row in production_board if row["kind"] == "foresight"]

    evidence = evidence_summary(DEFAULT_EVIDENCE)
    production_pass = bool(board_dp and board_foresight and args.require_both_dp_and_foresight)
    if not args.require_both_dp_and_foresight:
        production_pass = bool(production_board)

    return {
        "board_dataset": str(Path(args.board_dataset)),
        "search_roots": [str(x) for x in roots],
        "require_both_dp_and_foresight": bool(args.require_both_dp_and_foresight),
        "config_counts": {
            "scanned_or_reported": len(configs),
            "production_board_candidates": len(production_board),
            "board_dp_candidates": len(board_dp),
            "board_foresight_candidates": len(board_foresight),
        },
        "production_board_candidates": production_board,
        "reported_configs": configs,
        "existing_evidence": evidence,
        "verdict": {
            "passes_board_production_full_chain_prereq": production_pass,
            "reason": (
                "Found board-covered production DP and Foresight checkpoints."
                if production_pass
                else "No production DP+Foresight checkpoint pair was found whose config covers the board dataset. "
                "Board surrogate evidence remains useful but is not production full-chain evidence."
            ),
            "next_required_step": (
                "Train or locate board-specific production DP and Foresight checkpoints, then run "
                "action -> production Foresight -> PTG board energy -> dscore/daction refinement."
            ),
        },
    }


def write_markdown(audit: Dict[str, Any], path: Path) -> None:
    ev = audit["existing_evidence"]
    lines = [
        "# Board Production PTG Chain Audit",
        "",
        "## Verdict",
        "",
        f"- passes_board_production_full_chain_prereq: `{audit['verdict']['passes_board_production_full_chain_prereq']}`",
        f"- reason: {audit['verdict']['reason']}",
        f"- next_required_step: {audit['verdict']['next_required_step']}",
        "",
        "## Existing Evidence",
        "",
        "| item | pass/metric | value |",
        "|---|---|---:|",
        f"| board scorer AUC | mixed_binary_auc | {ev['board_scorer_quality']['mixed_binary_auc']} |",
        f"| board scorer quality corr | mixed_quality_corr | {ev['board_scorer_quality']['mixed_quality_corr']} |",
        f"| scorer readiness | passes | {ev['board_scorer_readiness']['passes']} |",
        f"| scorer readiness | improved rate | {ev['board_scorer_readiness']['score_improved_rate']} |",
        f"| surrogate full-chain | passes | {ev['board_surrogate_full_chain']['passes']} |",
        f"| surrogate full-chain | improved rate | {ev['board_surrogate_full_chain']['score_improved_rate']} |",
        "",
        "## Production Candidates",
        "",
    ]
    if audit["production_board_candidates"]:
        for row in audit["production_board_candidates"]:
            lines.extend(
                [
                    f"### {row['kind']}: `{row['path']}`",
                    "",
                    f"- has_checkpoint_sibling: `{row['has_checkpoint_sibling']}`",
                    f"- checkpoint_siblings: `{row['checkpoint_siblings']}`",
                    f"- keys: `{row['keys']}`",
                    "",
                ]
            )
    else:
        lines.append("No board-covered production DP/Foresight checkpoint config was found.")
        lines.append("")
    lines.extend(
        [
            "## Boundary",
            "",
            "The board surrogate proves the mechanism `action -> predicted tactile -> PTG energy -> action gradient`, "
            "but it is not the production DP/Foresight chain.  The objective should remain incomplete until the "
            "production board chain is verified.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board_dataset", default=str(BOARD_DATASET))
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--search_roots", nargs="*", default=[str(x) for x in DEFAULT_ROOTS])
    parser.add_argument("--include_all_configs", action="store_true")
    parser.add_argument("--require_both_dp_and_foresight", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = build_audit(args)
    json_path = out_dir / "board_production_chain_audit.json"
    md_path = out_dir / "board_production_chain_audit.md"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(audit, md_path)
    print(json.dumps(audit["verdict"], ensure_ascii=False, indent=2))
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
