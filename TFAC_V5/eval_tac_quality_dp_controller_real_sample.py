"""Real-sample audit for TacQualityDPGuidanceController.

This verifies the final DP-facing controller on real insertion and board
windows.  The controller is stricter than the lower-level refiner: it encodes
that gradients must be recomputed from the current score_fn every call and that
stale gradient reuse is not an allowed API path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.tac_quality_dp_guidance_controller import from_guidance_profile  # noqa: E402
from TFAC_V5.tac_quality_guidance_runtime import TacQualityGuidanceRuntime  # noqa: E402


INSERTION_FEATURES = Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_features.npz")
BOARD_DIR = Path("/home/chenshuai/data/dataset/260522_v8l_caheiban")
OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_dp_guidance_controller")


def summarize(x) -> Dict[str, float]:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def pad_last(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) >= window:
        return x[-window:]
    pad = np.repeat(x[:1], window - len(x), axis=0)
    return np.concatenate([pad, x], axis=0)


def sample_insertion(args) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    data = np.load(args.insertion_features, allow_pickle=True)
    marker = data["marker"]
    action = data["action"]
    reason = data["reason"]
    quality = data["quality"]
    rng = np.random.default_rng(args.seed)
    selected: List[int] = []
    per_reason = max(1, args.n_insertion // max(1, len(np.unique(reason))))
    for cls in sorted(np.unique(reason).tolist()):
        idx = np.flatnonzero(reason == cls)
        if len(idx):
            selected.extend(rng.choice(idx, min(per_reason, len(idx)), replace=False).tolist())
    if len(selected) < args.n_insertion:
        rest = np.setdiff1d(np.arange(len(marker)), np.asarray(selected, dtype=np.int64), assume_unique=False)
        selected.extend(rng.choice(rest, min(args.n_insertion - len(selected), len(rest)), replace=False).tolist())
    idx = np.asarray(selected[: args.n_insertion], dtype=np.int64)
    return (
        marker[idx].astype(np.float32),
        action[idx].astype(np.float32),
        {
            "source": str(args.insertion_features),
            "n_samples": int(len(idx)),
            "reason_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(reason[idx], return_counts=True))},
            "quality": summarize(quality[idx]),
        },
    )


def sample_board(args):
    rows = []
    for path in sorted((Path(args.board_dir) / "success").glob("*.hdf5")):
        with h5py.File(path, "r") as f:
            n = min(
                len(f["observations/tac/left/marker_offset"]),
                len(f["observations/tac/right/marker_offset"]),
                len(f["actions/eef_abs"]),
                len(f["actions/joint_abs"]),
            )
        for start in range(0, max(1, n - args.board_window + 1), args.board_stride):
            end = min(n, start + args.board_window)
            if end - start >= max(8, args.board_window // 2):
                rows.append((path, start, end))
    rng = np.random.default_rng(args.seed + 23)
    if len(rows) > args.n_board:
        rows = [rows[i] for i in sorted(rng.choice(len(rows), args.n_board, replace=False).tolist())]
    if not rows:
        raise RuntimeError(f"No board windows found under {args.board_dir}")
    left, right, eef, joint = [], [], [], []
    for path, start, end in rows:
        with h5py.File(path, "r") as f:
            left.append(pad_last(f["observations/tac/left/marker_offset"][start:end], args.board_window))
            right.append(pad_last(f["observations/tac/right/marker_offset"][start:end], args.board_window))
            eef.append(pad_last(f["actions/eef_abs"][start:end], args.board_window))
            joint.append(pad_last(f["actions/joint_abs"][start:end], args.board_window))
    return (
        np.stack(left).astype(np.float32),
        np.stack(right).astype(np.float32),
        np.stack(eef).astype(np.float32),
        np.stack(joint).astype(np.float32),
        {"source": str(args.board_dir), "n_samples": int(len(rows))},
    )


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    runtime = TacQualityGuidanceRuntime(device=args.device)
    device = runtime.device

    ins_marker_np, ins_action_np, ins_meta = sample_insertion(args)
    ins_marker = torch.tensor(ins_marker_np, dtype=torch.float32, device=device)
    ins_action = torch.tensor(ins_action_np, dtype=torch.float32, device=device)
    ins_controller = from_guidance_profile("insertion", scale=args.insertion_scale, clamp_norm_action=False)

    def ins_score(action):
        return runtime.score("insertion", ins_marker, action, mode="profile")

    ins_guided, ins_report = ins_controller.guide(ins_action, ins_score)

    left_np, right_np, eef_np, joint_np, board_meta = sample_board(args)
    left = torch.tensor(left_np, dtype=torch.float32, device=device)
    right = torch.tensor(right_np, dtype=torch.float32, device=device)
    eef = torch.tensor(eef_np, dtype=torch.float32, device=device)
    joint = torch.tensor(joint_np, dtype=torch.float32, device=device)
    board_controller = from_guidance_profile("board", scale=args.board_scale, clamp_norm_action=False)

    def board_score(action):
        return runtime.score("board", left, action, right_marker_seq=right, eef_action_seq=eef, mode="profile")

    board_guided, board_report = board_controller.guide(joint, board_score)

    result = {
        "purpose": "Real-sample audit for DP-facing TacQuality guidance controller.",
        "scope": "Controller-level API/guardrail check on real data tensors; not robot validation.",
        "device": str(device),
        "seed": int(args.seed),
        "insertion": {
            "meta": ins_meta,
            "report": ins_report,
            "guided_shape": list(ins_guided.shape),
            "passes_controller_real_sample": bool(
                ins_report["improved_rate"] >= args.pass_improve_rate
                and ins_report["finite_grad_rate"] >= 0.999
                and ins_report["positive_grad_rate"] >= 0.999
                and ins_report["max_delta_within_trust_region"]
                and not ins_report["guardrails"]["stale_gradient_reuse_allowed"]
            ),
        },
        "board": {
            "meta": board_meta,
            "report": board_report,
            "guided_shape": list(board_guided.shape),
            "passes_controller_real_sample": bool(
                board_report["improved_rate"] >= args.pass_improve_rate
                and board_report["finite_grad_rate"] >= 0.999
                and board_report["positive_grad_rate"] >= 0.999
                and board_report["max_delta_within_trust_region"]
                and not board_report["guardrails"]["stale_gradient_reuse_allowed"]
            ),
        },
    }
    result["overall_pass"] = bool(
        result["insertion"]["passes_controller_real_sample"]
        and result["board"]["passes_controller_real_sample"]
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "controller_real_sample_audit.json"
    md_path = out_dir / "controller_real_sample_audit.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# TacQuality DP Guidance Controller Real-Sample Audit",
        "",
        f"- overall_pass: `{result['overall_pass']}`",
        f"- insertion_pass: `{result['insertion']['passes_controller_real_sample']}`",
        f"- board_pass: `{result['board']['passes_controller_real_sample']}`",
        f"- insertion_improved_rate: `{ins_report['improved_rate']}`",
        f"- insertion_score_delta_mean: `{ins_report['score_delta']['mean']}`",
        f"- board_improved_rate: `{board_report['improved_rate']}`",
        f"- board_score_delta_mean: `{board_report['score_delta']['mean']}`",
        "",
        "Guardrail: stale gradient reuse is not allowed by the controller API.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "overall_pass": result["overall_pass"],
                "insertion_improved_rate": ins_report["improved_rate"],
                "insertion_score_delta_mean": ins_report["score_delta"]["mean"],
                "board_improved_rate": board_report["improved_rate"],
                "board_score_delta_mean": board_report["score_delta"]["mean"],
                "json": str(json_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--n_insertion", type=int, default=256)
    parser.add_argument("--n_board", type=int, default=256)
    parser.add_argument("--board_window", type=int, default=32)
    parser.add_argument("--board_stride", type=int, default=32)
    parser.add_argument("--insertion_features", type=Path, default=INSERTION_FEATURES)
    parser.add_argument("--board_dir", type=Path, default=BOARD_DIR)
    parser.add_argument("--insertion_scale", type=float, default=0.04)
    parser.add_argument("--board_scale", type=float, default=0.0008)
    parser.add_argument("--pass_improve_rate", type=float, default=0.95)
    parser.add_argument("--output_dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
