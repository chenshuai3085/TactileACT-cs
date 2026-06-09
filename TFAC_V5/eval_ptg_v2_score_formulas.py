"""Search PTG-v2 score formulas on the same DP candidate pool.

eval_ptg_v2_reranking.py runs one score mode per process, so DP candidates are
resampled for each mode.  This script samples candidates once, computes all
PTG-v2 heads and action smoothness features, then evaluates simple formulas with
per-frame z-score normalization.

The target is still L1-to-expert for insertion, which is only an offline proxy.
The purpose is diagnostic: does any deployable combination of PTG-v2 heads give
a stable ranking signal on real DP sampled candidates?
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "TFAC_V5"))

from TFAC_V5.eval_action_aware_reranking import ForesightOnlyReranker, load_tactile_vae_norm  # noqa: E402
from TFAC_V5.dp_reranking import _collect_insertion_frames, _find_hdf5_files, _load_frame_data  # noqa: E402
from TFAC_V5.eval_ptg_v2_reranking import predict_marker_for_candidates  # noqa: E402
from TFAC_V5.ptg_proxy_scorer_v2_runtime import PTGProxyScorerV2Runtime  # noqa: E402


OUT_DIR = Path("/home/chenshuai/Project/output/ptg_v2_reranking")
WINDOW = 8


def action_features(actions: np.ndarray) -> dict[str, np.ndarray]:
    chunk = actions[:, :WINDOW]
    delta = np.diff(chunk, axis=1) if chunk.shape[1] > 1 else np.zeros_like(chunk[:, :1])
    speed = np.linalg.norm(delta, axis=-1)
    accel = np.diff(delta, axis=1) if delta.shape[1] > 1 else np.zeros_like(delta[:, :1])
    accel_norm = np.linalg.norm(accel, axis=-1)
    return {
        "action_speed_mean": speed.mean(axis=1),
        "action_speed_p90": np.percentile(speed, 90, axis=1),
        "action_accel_mean": accel_norm.mean(axis=1),
        "action_accel_p90": np.percentile(accel_norm, 90, axis=1),
        "action_abs_delta_max": np.abs(delta).max(axis=(1, 2)),
        "action_first_last_l2": np.linalg.norm(chunk[:, -1] - chunk[:, 0], axis=-1),
    }


def group_z(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    for g in np.unique(groups):
        idx = groups == g
        vals = x[idx].astype(np.float64)
        out[idx] = (vals - vals.mean()) / (vals.std() + 1e-8)
    return out


def select_metrics(score: np.ndarray, l1: np.ndarray, groups: np.ndarray) -> dict:
    selected, random_sel, oracle, corr = [], [], [], []
    wins = 0
    rng = np.random.default_rng(123)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        best = idx[np.argmax(score[idx])]
        rnd = idx[int(rng.integers(len(idx)))]
        selected.append(l1[best])
        random_sel.append(l1[rnd])
        oracle.append(l1[idx].min())
        wins += int(l1[best] < l1[rnd])
        if np.std(score[idx]) > 1e-8 and np.std(l1[idx]) > 1e-8:
            corr.append(float(np.corrcoef(score[idx], -l1[idx])[0, 1]))
        else:
            corr.append(0.0)
    selected = np.asarray(selected)
    random_sel = np.asarray(random_sel)
    oracle = np.asarray(oracle)
    return {
        "selected_l1_mean": float(selected.mean()),
        "random_l1_mean": float(random_sel.mean()),
        "oracle_l1_mean": float(oracle.mean()),
        "beats_random": float(wins / len(selected)),
        "score_l1_corr_mean": float(np.mean(corr)),
        "oracle_gap_ratio": float((selected.mean() - oracle.mean()) / max(oracle.mean(), 1e-8)),
    }


@torch.no_grad()
def build_cache(args) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUT_DIR / f"ptg_v2_candidates_K{args.K}_N{args.n_eval}_seed{args.seed}.npz"
    if cache_path.exists() and not args.force_rebuild:
        return cache_path

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    reranker = ForesightOnlyReranker(
        dp_config_path=args.dp_config,
        dp_ckpt_path=args.dp_ckpt,
        foresight_ckpt_path=args.foresight_ckpt,
        foresight_dir=args.foresight_dir,
        device=args.device,
        K=args.K,
    )
    scorer = PTGProxyScorerV2Runtime(args.scorer_ckpt, device=args.device)
    vae_ckpt = reranker.foresight_config.get(
        "tactile_vae_ckpt", "/home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt"
    )
    vae_mean, vae_std = load_tactile_vae_norm(vae_ckpt)

    with open(os.path.join(args.data_dir, "annotations.pkl"), "rb") as f:
        ann = pickle.load(f)
    frames = _collect_insertion_frames(
        _find_hdf5_files(args.data_dir),
        ann,
        cs=reranker.pred_horizon,
        obs_horizon=reranker.obs_horizon,
    )
    rng = np.random.default_rng(args.seed)
    if len(frames) > args.n_eval:
        frames = [frames[i] for i in rng.choice(len(frames), args.n_eval, replace=False)]

    rows, groups, frame_ids = [], [], []
    for frame_id, (hdf5_path, ep_name, t) in enumerate(tqdm(frames, desc="Build PTG-v2 candidate cache")):
        try:
            data = _load_frame_data(reranker, hdf5_path, t, reranker.obs_horizon)
            obs_cond = reranker.build_obs_cond(data["images_obs"], data["qpos_obs"], data["marker_hists"])
            actions = reranker.generate_candidates(obs_cond, K=args.K)
            marker_raw = predict_marker_for_candidates(
                reranker,
                actions,
                data["qpos_raw"],
                data["marker_window"],
                data["foresight_images"],
                vae_mean,
                vae_std,
            )
            marker_seq = marker_raw.unsqueeze(1).expand(-1, WINDOW, -1, -1, -1)
            action_seq = actions[:, :WINDOW, :]
            task_id = torch.zeros(args.K, dtype=torch.long, device=reranker.device)
            out = scorer.forward(marker_seq, marker_seq, joint_action_seq=action_seq, task_id=task_id)
            actions_np = actions.detach().cpu().numpy()
            expert_flat = data["action_expert"].reshape(-1)
            l1 = np.array([np.abs(actions_np[i].reshape(-1) - expert_flat).mean() for i in range(args.K)])
            afeats = action_features(actions_np)
            reason_prob = out["reason_prob"].detach().cpu().numpy()
            row = {
                "p_good": out["p_good"].detach().cpu().numpy(),
                "log_p_good": out["log_p_good"].detach().cpu().numpy(),
                "quality": out["quality_score"].detach().cpu().numpy(),
                "reason_good": reason_prob[:, 1],
                "reason_risk": reason_prob[:, 2],
                "reason_impact": reason_prob[:, 3],
                "l1": l1.astype(np.float32),
                **{k: v.astype(np.float32) for k, v in afeats.items()},
            }
            rows.append(row)
            groups.extend([frame_id] * args.K)
            frame_ids.extend([f"{ep_name}:{t}"] * args.K)
        except Exception as exc:
            print(f"Skip {ep_name} t={t}: {exc}")

    if not rows:
        raise RuntimeError("No rows built")
    keys = rows[0].keys()
    arrays = {k: np.concatenate([r[k] for r in rows]).astype(np.float32) for k in keys}
    arrays["groups"] = np.asarray(groups, dtype=np.int64)
    arrays["frame_ids"] = np.asarray(frame_ids)
    np.savez_compressed(cache_path, **arrays)
    print(f"Saved {cache_path}")
    return cache_path


def formula_score(Z: dict[str, np.ndarray], params: dict) -> np.ndarray:
    score = Z[params["base"]].copy()
    for name, weight in params.get("bonuses", {}).items():
        score += weight * Z[name]
    for name, weight in params.get("penalties", {}).items():
        score -= weight * Z[name]
    return score


def evaluate_formulas(cache_path: Path, args):
    data = np.load(cache_path, allow_pickle=True)
    groups = data["groups"].astype(np.int64)
    l1 = data["l1"].astype(np.float64)
    feature_names = [
        "quality",
        "p_good",
        "log_p_good",
        "reason_good",
        "reason_risk",
        "reason_impact",
        "action_speed_p90",
        "action_accel_p90",
        "action_abs_delta_max",
        "action_first_last_l2",
    ]
    Z = {name: group_z(data[name].astype(np.float64), groups) for name in feature_names}

    rows = []
    for base in ["quality", "p_good", "log_p_good", "reason_good"]:
        rows.append({"name": base, "params": {"base": base}, "metrics": select_metrics(Z[base], l1, groups)})

    weights = [0.1, 0.25, 0.5, 1.0]
    penalty_sets = [
        {},
        {"action_accel_p90": None},
        {"action_speed_p90": None},
        {"action_abs_delta_max": None},
        {"action_accel_p90": None, "action_abs_delta_max": None},
    ]
    bonus_sets = [
        {},
        {"reason_good": 0.25},
        {"p_good": 0.25},
        {"reason_good": 0.25, "p_good": 0.25},
    ]
    for base in ["quality", "p_good", "log_p_good"]:
        for penalties in penalty_sets:
            names = list(penalties)
            grids = [()] if not names else product(weights, repeat=len(names))
            for ws in grids:
                for bonuses in bonus_sets:
                    params = {"base": base, "penalties": dict(zip(names, ws)), "bonuses": bonuses}
                    score = formula_score(Z, params)
                    rows.append(
                        {
                            "name": "formula",
                            "params": params,
                            "metrics": select_metrics(score, l1, groups),
                        }
                    )
    rows = sorted(rows, key=lambda r: (r["metrics"]["selected_l1_mean"], -r["metrics"]["beats_random"]))
    result = {
        "cache": str(cache_path),
        "n": int(len(l1)),
        "n_groups": int(len(np.unique(groups))),
        "top": rows[:25],
        "baselines": [r for r in rows if r["name"] != "formula"][:10],
    }
    out_path = OUT_DIR / f"ptg_v2_formula_eval_{cache_path.stem}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": rows[0], "baselines": result["baselines"]}, ensure_ascii=False, indent=2))
    print(f"Saved {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp_ckpt", default="/home/chenshuai/Project/output/dp_tac_vae_shift4_0414/dp_best.pth")
    parser.add_argument("--dp_config", default="/home/chenshuai/Project/output/dp_tac_vae_shift4_0414/config.json")
    parser.add_argument("--foresight_ckpt", default="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full/foresight_best.ckpt")
    parser.add_argument("--foresight_dir", default="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full")
    parser.add_argument("--scorer_ckpt", default="/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_final.pt")
    parser.add_argument("--data_dir", default="/home/chenshuai/data/dataset/0414")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--n_eval", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--cache", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    cache = Path(parsed.cache) if parsed.cache else build_cache(parsed)
    evaluate_formulas(cache, parsed)
