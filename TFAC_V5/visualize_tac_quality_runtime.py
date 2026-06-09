"""Visualize the current TacQuality runtime scorer on insertion and board data.

This is a diagnostic artifact for the DP classifier-guidance scorer.  It
visualizes the actual runtime score/gradient behavior rather than a separate
offline classifier:

  - PCA projection of proxy features colored by score and task/label;
  - score and gradient-norm distributions;
  - insertion score vs annotated quality/reason;
  - board score vs force/smoothness proxies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.tac_quality_guidance_runtime import TacQualityGuidanceRuntime  # noqa: E402


INSERTION_FEATURES = Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_features.npz")
BOARD_DIR = Path("/home/chenshuai/data/dataset/260522_v8l_caheiban")
OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_runtime_visualization")


REASON_NAMES = {
    0: "good",
    1: "low_quality",
    2: "pre_bounce",
    3: "bounce",
}


def summarize(x) -> Dict[str, float]:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"n": 0}
    return {
        "n": int(len(arr)),
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


def action_smoothness(actions: np.ndarray) -> np.ndarray:
    if actions.shape[1] < 3:
        return np.zeros(actions.shape[0], dtype=np.float32)
    accel = actions[:, 2:] - 2 * actions[:, 1:-1] + actions[:, :-2]
    return np.linalg.norm(accel, axis=-1).mean(axis=1).astype(np.float32)


def marker_mag(marker: np.ndarray) -> np.ndarray:
    return np.linalg.norm(marker.reshape(marker.shape[0], marker.shape[1], -1), axis=-1).mean(axis=1)


def sample_insertion(args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    data = np.load(args.insertion_features, allow_pickle=True)
    marker = data["marker"].astype(np.float32)
    action = data["action"].astype(np.float32)
    reason = data["reason"].astype(np.int64)
    quality = data["quality"].astype(np.float32)
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
    meta = {
        "source": str(args.insertion_features),
        "n_samples": int(len(idx)),
        "reason_counts": {REASON_NAMES.get(int(k), str(k)): int(v) for k, v in zip(*np.unique(reason[idx], return_counts=True))},
        "quality": summarize(quality[idx]),
    }
    return marker[idx], action[idx], reason[idx], quality[idx], meta


def board_rows(args):
    rows = []
    for path in sorted((Path(args.board_dir) / "success").glob("*.hdf5")):
        with h5py.File(path, "r") as f:
            n = min(
                len(f["observations/tac/left/marker_offset"]),
                len(f["observations/tac/right/marker_offset"]),
                len(f["actions/eef_abs"]),
                len(f["actions/joint_abs"]),
                len(f["observations/tac/left/force6d"]),
            )
        for start in range(0, max(1, n - args.board_window + 1), args.board_stride):
            end = min(n, start + args.board_window)
            if end - start >= max(8, args.board_window // 2):
                rows.append((path, start, end))
    rng = np.random.default_rng(args.seed + 23)
    if len(rows) > args.n_board:
        rows = [rows[i] for i in sorted(rng.choice(len(rows), args.n_board, replace=False).tolist())]
    return rows


def force_mag(force6: np.ndarray) -> np.ndarray:
    arr = np.asarray(force6, dtype=np.float32)
    if arr.shape[-1] >= 3:
        return np.linalg.norm(arr[..., :3], axis=-1)
    return np.abs(arr.reshape(len(arr), -1))


def sample_board(args):
    rows = board_rows(args)
    if not rows:
        raise RuntimeError(f"No board windows found under {args.board_dir}")
    left, right, eef, joint, force_mean, force_std = [], [], [], [], [], []
    for path, start, end in rows:
        with h5py.File(path, "r") as f:
            left.append(pad_last(f["observations/tac/left/marker_offset"][start:end], args.board_window))
            right.append(pad_last(f["observations/tac/right/marker_offset"][start:end], args.board_window))
            eef.append(pad_last(f["actions/eef_abs"][start:end], args.board_window))
            joint.append(pad_last(f["actions/joint_abs"][start:end], args.board_window))
            fmag = force_mag(f["observations/tac/left/force6d"][start:end])
            force_mean.append(float(np.mean(fmag)))
            force_std.append(float(np.std(fmag)))
    left = np.stack(left).astype(np.float32)
    right = np.stack(right).astype(np.float32)
    eef = np.stack(eef).astype(np.float32)
    joint = np.stack(joint).astype(np.float32)
    force_mean = np.asarray(force_mean, dtype=np.float32)
    force_std = np.asarray(force_std, dtype=np.float32)
    meta = {
        "source": str(args.board_dir),
        "n_samples": int(len(rows)),
        "window": int(args.board_window),
        "stride": int(args.board_stride),
        "force_mean": summarize(force_mean),
        "force_std": summarize(force_std),
    }
    return left, right, eef, joint, force_mean, force_std, meta


def score_grad(runtime: TacQualityGuidanceRuntime, task: str, marker, action, **kwargs):
    action_req = torch.tensor(action, dtype=torch.float32, device=runtime.device, requires_grad=True)
    marker_t = torch.tensor(marker, dtype=torch.float32, device=runtime.device)
    if task == "insertion":
        score = runtime.score("insertion", marker_t, action_req, mode="profile")
        diag = runtime.diagnostics("insertion", marker_t, action_req)
    else:
        right_t = torch.tensor(kwargs["right"], dtype=torch.float32, device=runtime.device)
        eef_t = torch.tensor(kwargs["eef"], dtype=torch.float32, device=runtime.device)
        score = runtime.score(
            "board",
            marker_t,
            action_req,
            right_marker_seq=right_t,
            eef_action_seq=eef_t,
            mode="profile",
        )
        diag = runtime.diagnostics(
            "board",
            marker_t,
            action_req,
            right_marker_seq=right_t,
            eef_action_seq=eef_t,
        )
    grad = torch.autograd.grad(score.sum(), action_req, retain_graph=False)[0]
    out = {k: v.detach().cpu().numpy() for k, v in diag.items() if torch.is_tensor(v)}
    out["score"] = score.detach().cpu().numpy()
    out["grad_norm"] = grad.detach().flatten(1).norm(dim=1).cpu().numpy()
    return out


def feature_matrix(marker: np.ndarray, action: np.ndarray) -> np.ndarray:
    marker_flat = marker.reshape(marker.shape[0], -1)
    action_flat = action.reshape(action.shape[0], -1)
    stats = np.column_stack(
        [
            marker_mag(marker),
            marker_flat.std(axis=1),
            action_flat.mean(axis=1),
            action_flat.std(axis=1),
            action_smoothness(action),
        ]
    )
    return np.concatenate([stats, marker_flat[:, :: max(1, marker_flat.shape[1] // 64)], action_flat], axis=1)


def shared_feature_matrix(marker: np.ndarray, action: np.ndarray) -> np.ndarray:
    marker_flat = marker.reshape(marker.shape[0], -1)
    action_flat = action.reshape(action.shape[0], -1)
    return np.column_stack(
        [
            marker_mag(marker),
            marker_flat.std(axis=1),
            np.percentile(np.abs(marker_flat), 50, axis=1),
            np.percentile(np.abs(marker_flat), 95, axis=1),
            action_flat.mean(axis=1),
            action_flat.std(axis=1),
            np.percentile(np.abs(action_flat), 50, axis=1),
            np.percentile(np.abs(action_flat), 95, axis=1),
            action_smoothness(action),
        ]
    ).astype(np.float32)


def scatter(ax, z, c, title, cmap="viridis", label=None):
    im = ax.scatter(z[:, 0], z[:, 1], c=c, s=16, alpha=0.75, cmap=cmap, edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if label is not None:
        ax.text(0.02, 0.98, label, transform=ax.transAxes, va="top", ha="left", fontsize=9)
    return im


def plot_runtime_space(ins, board, out_dir: Path) -> Dict[str, str]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures: Dict[str, str] = {}

    X = np.vstack([ins["shared_features"], board["shared_features"]])
    task = np.concatenate([np.zeros(len(ins["score"])), np.ones(len(board["score"]))])
    score = np.concatenate([ins["score"], board["score"]])
    grad = np.concatenate([ins["grad_norm"], board["grad_norm"]])
    z = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(X))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im0 = scatter(axes[0], z, task, "PCA by task", cmap="coolwarm")
    im1 = scatter(axes[1], z, score, "PCA by TacQuality score", cmap="viridis")
    im2 = scatter(axes[2], z, np.log10(grad + 1e-8), "PCA by log10(action grad norm)", cmap="magma")
    for ax, im in zip(axes, [im0, im1, im2]):
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = fig_dir / "runtime_pca_task_score_grad.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures["runtime_pca_task_score_grad"] = str(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(ins["score"], bins=45, alpha=0.65, density=True, label="insertion")
    axes[0].hist(board["score"], bins=45, alpha=0.65, density=True, label="board")
    axes[0].set_title("TacQuality score distribution")
    axes[0].set_xlabel("profile energy")
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False)
    axes[1].hist(np.log10(ins["grad_norm"] + 1e-8), bins=45, alpha=0.65, density=True, label="insertion")
    axes[1].hist(np.log10(board["grad_norm"] + 1e-8), bins=45, alpha=0.65, density=True, label="board")
    axes[1].set_title("Action-gradient norm distribution")
    axes[1].set_xlabel("log10 grad norm")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    path = fig_dir / "runtime_score_grad_distributions.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures["runtime_score_grad_distributions"] = str(path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    reason = ins["reason"]
    im = scatter(axes[0], ins["z"], reason, "Insertion PCA by reason", cmap="tab10")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    axes[1].scatter(ins["quality"], ins["score"], s=14, alpha=0.65)
    axes[1].set_xlabel("annotated quality")
    axes[1].set_ylabel("TacQuality score")
    axes[1].set_title("Insertion score vs quality")
    axes[2].scatter(ins["quality"], ins["grad_norm"], s=14, alpha=0.65)
    axes[2].set_xlabel("annotated quality")
    axes[2].set_ylabel("action grad norm")
    axes[2].set_title("Insertion grad norm vs quality")
    fig.tight_layout()
    path = fig_dir / "insertion_runtime_quality_reason.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures["insertion_runtime_quality_reason"] = str(path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im = scatter(axes[0], board["z"], board["force_mean"], "Board PCA by force mean", cmap="viridis")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    axes[1].scatter(board["force_mean"], board["score"], s=14, alpha=0.65)
    axes[1].set_xlabel("force mean")
    axes[1].set_ylabel("TacQuality score")
    axes[1].set_title("Board score vs force mean")
    axes[2].scatter(board["smoothness"], board["score"], s=14, alpha=0.65)
    axes[2].set_xlabel("joint action smoothness")
    axes[2].set_ylabel("TacQuality score")
    axes[2].set_title("Board score vs action smoothness")
    fig.tight_layout()
    path = fig_dir / "board_runtime_force_smoothness.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures["board_runtime_force_smoothness"] = str(path)

    return figures


def corr_safe(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def write_markdown(summary: Dict[str, object], path: Path) -> None:
    lines = [
        "# TacQuality Runtime Visualization",
        "",
        f"- visualization_pass: `{summary['visualization_pass']}`",
        f"- out_dir: `{summary['out_dir']}`",
        "",
        "## Diagnostics",
        "",
        f"- insertion score-quality corr: `{summary['diagnostics']['insertion_score_quality_corr']}`",
        f"- insertion score-grad corr: `{summary['diagnostics']['insertion_score_grad_corr']}`",
        f"- board score-force corr: `{summary['diagnostics']['board_score_force_corr']}`",
        f"- board score-smoothness corr: `{summary['diagnostics']['board_score_smoothness_corr']}`",
        "",
        "## Figures",
        "",
    ]
    for name, path_value in summary["figures"].items():
        lines.append(f"- {name}: `{path_value}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--n_insertion", type=int, default=512)
    parser.add_argument("--n_board", type=int, default=512)
    parser.add_argument("--board_window", type=int, default=32)
    parser.add_argument("--board_stride", type=int, default=32)
    parser.add_argument("--insertion_features", type=Path, default=INSERTION_FEATURES)
    parser.add_argument("--board_dir", type=Path, default=BOARD_DIR)
    parser.add_argument("--output_dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    runtime = TacQualityGuidanceRuntime(device=args.device)

    ins_marker, ins_action, ins_reason, ins_quality, ins_meta = sample_insertion(args)
    ins_diag = score_grad(runtime, "insertion", ins_marker, ins_action)
    ins_feat = feature_matrix(ins_marker, ins_action)
    ins_shared = shared_feature_matrix(ins_marker, ins_action)
    ins_z = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(ins_feat))
    ins = {
        **ins_diag,
        "features": ins_feat,
        "shared_features": ins_shared,
        "z": ins_z,
        "reason": ins_reason,
        "quality": ins_quality,
        "meta": ins_meta,
    }

    left, right, eef, joint, force_mean, force_std, board_meta = sample_board(args)
    board_diag = score_grad(runtime, "board", left, joint, right=right, eef=eef)
    board_feat = feature_matrix(left, joint)
    board_shared = shared_feature_matrix(left, joint)
    board_z = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(board_feat))
    board = {
        **board_diag,
        "features": board_feat,
        "shared_features": board_shared,
        "z": board_z,
        "force_mean": force_mean,
        "force_std": force_std,
        "smoothness": action_smoothness(joint),
        "meta": board_meta,
    }

    figures = plot_runtime_space(ins, board, args.output_dir)
    diagnostics = {
        "insertion_score": summarize(ins["score"]),
        "board_score": summarize(board["score"]),
        "insertion_grad_norm": summarize(ins["grad_norm"]),
        "board_grad_norm": summarize(board["grad_norm"]),
        "insertion_score_quality_corr": corr_safe(ins["score"], ins_quality),
        "insertion_score_grad_corr": corr_safe(ins["score"], ins["grad_norm"]),
        "board_score_force_corr": corr_safe(board["score"], force_mean),
        "board_score_smoothness_corr": corr_safe(board["score"], board["smoothness"]),
    }
    summary = {
        "purpose": "Visualization of the deployed TacQualityGuidanceRuntime score/gradient behavior.",
        "scope": "Real-sample visualization; not a robot rollout gate.",
        "device": str(runtime.device),
        "seed": int(args.seed),
        "out_dir": str(args.output_dir),
        "insertion_meta": ins_meta,
        "board_meta": board_meta,
        "diagnostics": diagnostics,
        "figures": figures,
    }
    summary["visualization_pass"] = bool(
        all(Path(p).exists() and Path(p).stat().st_size > 0 for p in figures.values())
        and diagnostics["insertion_grad_norm"]["p05"] > 1e-8
        and diagnostics["board_grad_norm"]["p05"] > 1e-8
    )
    json_path = args.output_dir / "tac_quality_runtime_visualization.json"
    md_path = args.output_dir / "tac_quality_runtime_visualization.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, md_path)
    print(
        json.dumps(
            {
                "visualization_pass": summary["visualization_pass"],
                "figures": figures,
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
