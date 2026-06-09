"""Train a lightweight board tactile consequence surrogate.

This is a pragmatic bridge before a full board Foresight/DP checkpoint exists.
It learns:

  current left/right marker window + future eef/joint action chunk
      -> future left/right marker window

Then it verifies the guidance chain:

  action -> surrogate predicted tactile -> PTG v2 board energy -> dscore/daction

The split is episode-level, not random windows, to avoid leakage.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.eval_tac_energy_guided_denoising import summarize
from TFAC_V5.ptg_proxy_scorer_v2_runtime import PTGProxyScorerV2Runtime, TASK_TO_ID
from TFAC_V5.tac_quality_guidance_config import get_guidance_profile


BOARD_DIR = Path("/home/chenshuai/data/dataset/260522_v8l_caheiban")
OUT_DIR = Path("/home/chenshuai/Project/output/board_tactile_surrogate")


@dataclass
class NormStats:
    marker_mean: np.ndarray
    marker_std: np.ndarray
    eef_mean: np.ndarray
    eef_std: np.ndarray
    joint_mean: np.ndarray
    joint_std: np.ndarray

    def to_jsonable(self) -> Dict[str, object]:
        return {k: v.tolist() for k, v in asdict(self).items()}


class BoardTactileSurrogate(nn.Module):
    def __init__(self, window: int = 8, hidden: int = 512, dropout: float = 0.05):
        super().__init__()
        self.window = window
        marker_dim = window * 9 * 9 * 2
        action_dim = window * (6 + 7)
        in_dim = marker_dim * 2 + action_dim
        out_dim = marker_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, left_cur, right_cur, eef_action, joint_action):
        bsz = left_cur.shape[0]
        x = torch.cat(
            [
                left_cur.reshape(bsz, -1),
                right_cur.reshape(bsz, -1),
                eef_action.reshape(bsz, -1),
                joint_action.reshape(bsz, -1),
            ],
            dim=-1,
        )
        y = self.net(x)
        marker_dim = self.window * 9 * 9 * 2
        left = y[:, :marker_dim].reshape(bsz, self.window, 9, 9, 2)
        right = y[:, marker_dim:].reshape(bsz, self.window, 9, 9, 2)
        return left, right


def collect_samples(args):
    rows = []
    paths = sorted((Path(args.data_dir) / "success").glob("*.hdf5"))
    for path in paths:
        with h5py.File(path, "r") as f:
            left = f["observations/tac/left/marker_offset"][:]
            right = f["observations/tac/right/marker_offset"][:]
            eef = f["actions/eef_abs"][:]
            joint = f["actions/joint_abs"][:]
        n = min(len(left), len(right), len(eef), len(joint))
        max_t = n - args.window - 1
        for t in range(args.window - 1, max_t, args.stride):
            cur_slice = slice(t - args.window + 1, t + 1)
            fut_slice = slice(t + 1, t + 1 + args.window)
            if fut_slice.stop > n:
                continue
            rows.append(
                {
                    "episode": path.stem,
                    "left_cur": left[cur_slice],
                    "right_cur": right[cur_slice],
                    "eef": eef[fut_slice],
                    "joint": joint[fut_slice],
                    "left_target": left[fut_slice],
                    "right_target": right[fut_slice],
                }
            )
    return rows


def split_by_episode(rows, val_frac: float, seed: int):
    episodes = np.array(sorted({r["episode"] for r in rows}))
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    n_val = max(1, int(round(len(episodes) * val_frac)))
    val_eps = set(episodes[:n_val].tolist())
    train_idx = np.array([i for i, r in enumerate(rows) if r["episode"] not in val_eps], dtype=np.int64)
    val_idx = np.array([i for i, r in enumerate(rows) if r["episode"] in val_eps], dtype=np.int64)
    return train_idx, val_idx, sorted(val_eps)


def stack(rows, key):
    return np.stack([r[key] for r in rows]).astype(np.float32)


def build_arrays(rows):
    return {
        "left_cur": stack(rows, "left_cur"),
        "right_cur": stack(rows, "right_cur"),
        "eef": stack(rows, "eef"),
        "joint": stack(rows, "joint"),
        "left_target": stack(rows, "left_target"),
        "right_target": stack(rows, "right_target"),
        "episode": np.array([r["episode"] for r in rows]),
    }


def fit_norm(arrays, idx) -> NormStats:
    marker_train = np.concatenate(
        [
            arrays["left_cur"][idx].reshape(-1, 2),
            arrays["right_cur"][idx].reshape(-1, 2),
            arrays["left_target"][idx].reshape(-1, 2),
            arrays["right_target"][idx].reshape(-1, 2),
        ],
        axis=0,
    )
    return NormStats(
        marker_mean=marker_train.mean(axis=0).astype(np.float32),
        marker_std=(marker_train.std(axis=0) + 1e-6).astype(np.float32),
        eef_mean=arrays["eef"][idx].reshape(-1, 6).mean(axis=0).astype(np.float32),
        eef_std=(arrays["eef"][idx].reshape(-1, 6).std(axis=0) + 1e-6).astype(np.float32),
        joint_mean=arrays["joint"][idx].reshape(-1, 7).mean(axis=0).astype(np.float32),
        joint_std=(arrays["joint"][idx].reshape(-1, 7).std(axis=0) + 1e-6).astype(np.float32),
    )


def normalize_arrays(arrays, stats: NormStats):
    marker_mean = stats.marker_mean.reshape(1, 1, 1, 1, 2)
    marker_std = stats.marker_std.reshape(1, 1, 1, 1, 2)
    return {
        "left_cur": (arrays["left_cur"] - marker_mean) / marker_std,
        "right_cur": (arrays["right_cur"] - marker_mean) / marker_std,
        "left_target": (arrays["left_target"] - marker_mean) / marker_std,
        "right_target": (arrays["right_target"] - marker_mean) / marker_std,
        "eef": (arrays["eef"] - stats.eef_mean.reshape(1, 1, 6)) / stats.eef_std.reshape(1, 1, 6),
        "joint": (arrays["joint"] - stats.joint_mean.reshape(1, 1, 7)) / stats.joint_std.reshape(1, 1, 7),
        "episode": arrays["episode"],
    }


def make_loader(normed, idx, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(normed["left_cur"][idx]).float(),
        torch.from_numpy(normed["right_cur"][idx]).float(),
        torch.from_numpy(normed["eef"][idx]).float(),
        torch.from_numpy(normed["joint"][idx]).float(),
        torch.from_numpy(normed["left_target"][idx]).float(),
        torch.from_numpy(normed["right_target"][idx]).float(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    n = 0
    for left, right, eef, joint, left_t, right_t in loader:
        left = left.to(device)
        right = right.to(device)
        eef = eef.to(device)
        joint = joint.to(device)
        left_t = left_t.to(device)
        right_t = right_t.to(device)
        pred_l, pred_r = model(left, right, eef, joint)
        loss = F.smooth_l1_loss(pred_l, left_t) + F.smooth_l1_loss(pred_r, right_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        total += float(loss.item()) * len(left)
        n += len(left)
    return total / max(n, 1)


@torch.no_grad()
def eval_model(model, loader, device, stats: NormStats):
    model.eval()
    mse, mae = [], []
    marker_std = torch.tensor(stats.marker_std, dtype=torch.float32, device=device).view(1, 1, 1, 1, 2)
    for left, right, eef, joint, left_t, right_t in loader:
        left = left.to(device)
        right = right.to(device)
        eef = eef.to(device)
        joint = joint.to(device)
        left_t = left_t.to(device)
        right_t = right_t.to(device)
        pred_l, pred_r = model(left, right, eef, joint)
        pred_raw = torch.cat([pred_l * marker_std, pred_r * marker_std], dim=0)
        tgt_raw = torch.cat([left_t * marker_std, right_t * marker_std], dim=0)
        err = pred_raw - tgt_raw
        mse.append(err.square().flatten(1).mean(dim=1).cpu().numpy())
        mae.append(err.abs().flatten(1).mean(dim=1).cpu().numpy())
    return {"marker_mse": summarize(np.concatenate(mse)), "marker_mae": summarize(np.concatenate(mae))}


def unnorm_marker(x, stats: NormStats, device):
    mean = torch.tensor(stats.marker_mean, dtype=torch.float32, device=device).view(1, 1, 1, 1, 2)
    std = torch.tensor(stats.marker_std, dtype=torch.float32, device=device).view(1, 1, 1, 1, 2)
    return x * std + mean


def norm_action(raw, mean, std, device):
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 1, -1)
    std_t = torch.tensor(std, dtype=torch.float32, device=device).view(1, 1, -1)
    return (raw - mean_t) / std_t


def full_chain_guidance_probe(args, model, scorer, arrays, normed, val_idx, stats: NormStats, device):
    profile = get_guidance_profile("board")
    rng = np.random.default_rng(args.seed)
    idx = val_idx
    if len(idx) > args.n_grad_eval:
        idx = rng.choice(idx, args.n_grad_eval, replace=False)

    left_cur = torch.tensor(normed["left_cur"][idx], dtype=torch.float32, device=device)
    right_cur = torch.tensor(normed["right_cur"][idx], dtype=torch.float32, device=device)
    eef_raw = torch.tensor(arrays["eef"][idx], dtype=torch.float32, device=device, requires_grad=True)
    joint_raw = torch.tensor(arrays["joint"][idx], dtype=torch.float32, device=device, requires_grad=True)
    eef_norm = norm_action(eef_raw, stats.eef_mean, stats.eef_std, device)
    joint_norm = norm_action(joint_raw, stats.joint_mean, stats.joint_std, device)
    pred_l_norm, pred_r_norm = model(left_cur, right_cur, eef_norm, joint_norm)
    pred_l = unnorm_marker(pred_l_norm, stats, device)
    pred_r = unnorm_marker(pred_r_norm, stats, device)
    task_id = torch.full((len(idx),), TASK_TO_ID["board"], dtype=torch.long, device=device)
    score = scorer.weighted_energy_score(
        pred_l,
        right_marker_seq=pred_r,
        eef_action_seq=eef_raw,
        joint_action_seq=joint_raw,
        task_id=task_id,
        quality_weight=profile.energy.quality,
        binary_weight=profile.energy.binary_margin,
        reason_weight=profile.energy.reason_margin,
        clip=True,
    )
    grad_eef, grad_joint = torch.autograd.grad(score.sum(), [eef_raw, joint_raw], retain_graph=False)
    eef_norm_g = grad_eef.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-8)
    joint_norm_g = grad_joint.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-8)
    with torch.no_grad():
        eef_step = eef_raw + profile.refinement.action_step * grad_eef / eef_norm_g
        joint_step = joint_raw + profile.refinement.action_step * grad_joint / joint_norm_g
    eef_step_norm = norm_action(eef_step, stats.eef_mean, stats.eef_std, device)
    joint_step_norm = norm_action(joint_step, stats.joint_mean, stats.joint_std, device)
    pred_l2_norm, pred_r2_norm = model(left_cur, right_cur, eef_step_norm, joint_step_norm)
    pred_l2 = unnorm_marker(pred_l2_norm, stats, device)
    pred_r2 = unnorm_marker(pred_r2_norm, stats, device)
    score2 = scorer.weighted_energy_score(
        pred_l2,
        right_marker_seq=pred_r2,
        eef_action_seq=eef_step,
        joint_action_seq=joint_step,
        task_id=task_id,
        quality_weight=profile.energy.quality,
        binary_weight=profile.energy.binary_margin,
        reason_weight=profile.energy.reason_margin,
        clip=True,
    )
    delta = (score2 - score.detach()).detach().cpu().numpy()
    finite = (
        torch.isfinite(grad_eef).flatten(1).all(dim=1)
        & torch.isfinite(grad_joint).flatten(1).all(dim=1)
    ).detach().cpu().numpy()
    positive = (
        (grad_eef.flatten(1).norm(dim=1) > 1e-8)
        & (grad_joint.flatten(1).norm(dim=1) > 1e-8)
    ).detach().cpu().numpy()
    return {
        "n": int(len(idx)),
        "score": summarize(score.detach().cpu().numpy()),
        "score_after_action_step": summarize(score2.detach().cpu().numpy()),
        "score_delta": summarize(delta),
        "score_improved_rate": float(np.mean(delta > 0)),
        "finite_action_grad_rate": float(np.mean(finite)),
        "positive_action_grad_rate": float(np.mean(positive)),
        "eef_grad_norm": summarize(grad_eef.flatten(1).norm(dim=1).detach().cpu().numpy()),
        "joint_grad_norm": summarize(grad_joint.flatten(1).norm(dim=1).detach().cpu().numpy()),
        "passes_surrogate_full_chain": bool(np.mean(finite) == 1.0 and np.mean(positive) == 1.0 and np.mean(delta > 0) >= 0.95),
        "scope": "Board surrogate full-chain only: action -> learned board tactile surrogate -> PTG v2 scorer.",
    }


def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    rows = collect_samples(args)
    if not rows:
        raise RuntimeError("No board samples found")
    arrays = build_arrays(rows)
    train_idx, val_idx, val_eps = split_by_episode(rows, args.val_frac, args.seed)
    stats = fit_norm(arrays, train_idx)
    normed = normalize_arrays(arrays, stats)

    train_loader = make_loader(normed, train_idx, args.batch_size, shuffle=True)
    val_loader = make_loader(normed, val_idx, args.batch_size, shuffle=False)
    model = BoardTactileSurrogate(window=args.window, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_mae = float("inf")
    history = []
    for epoch in tqdm(range(args.epochs), desc="Train board tactile surrogate"):
        train_loss = train_epoch(model, train_loader, opt, device)
        metrics = eval_model(model, val_loader, device, stats)
        val_mae = metrics["marker_mae"]["mean"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_marker_mae": val_mae})
        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = eval_model(model, val_loader, device, stats)
    scorer = PTGProxyScorerV2Runtime(args.scorer_ckpt, device=str(device))
    probe = full_chain_guidance_probe(args, model, scorer, arrays, normed, val_idx, stats, device)
    result = {
        "config": vars(args),
        "data": {
            "n_samples": int(len(rows)),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_episodes": int(len(set(arrays["episode"].tolist()))),
            "val_episodes": val_eps,
        },
        "norm_stats": stats.to_jsonable(),
        "history": history,
        "eval": final_metrics,
        "guidance_probe": probe,
        "interpretation": {
            "passes_board_surrogate_full_chain": bool(
                final_metrics["marker_mae"]["mean"] < args.max_marker_mae
                and probe["passes_surrogate_full_chain"]
            ),
            "scope": "Surrogate full-chain evidence. It reduces risk before training full board Foresight/DP, but does not replace it.",
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": vars(args),
        "norm_stats": stats.to_jsonable(),
        "eval": final_metrics,
        "guidance_probe": probe,
    }
    torch.save(ckpt, OUT_DIR / "board_tactile_surrogate_final.pt")
    print(json.dumps({k: v for k, v in result.items() if k not in {"history", "norm_stats"}}, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default=str(BOARD_DIR))
    parser.add_argument("--scorer_ckpt", default="/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_final.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--n_grad_eval", type=int, default=256)
    parser.add_argument("--max_marker_mae", type=float, default=2.5)
    parser.add_argument("--output", default=str(OUT_DIR / "board_tactile_surrogate_eval.json"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
