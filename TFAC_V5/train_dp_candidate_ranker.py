"""Train a lightweight calibration ranker for real DP sampled candidates.

The action-aware scorer is good at classifying tactile quality, but real DP
candidate actions are clustered and its raw score is only weakly aligned with
candidate ranking.  This script builds a DP-candidate dataset and trains a small
ranking calibration head:

    DP candidates -> Foresight marker -> action-aware scorer/proxy features
                  -> ranker predicts -L1(candidate, expert)

This is an offline calibration experiment.  The target is L1-to-expert as a
proxy ranking signal; later it should be replaced or mixed with real tactile
quality / human labels.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "TFAC_V5"))

from TFAC_V5.action_aware_scorer_runtime import (  # noqa: E402
    WINDOW,
    action_proxy_features_torch,
    marker_proxy_features_torch,
)
from TFAC_V5.eval_action_aware_reranking import (  # noqa: E402
    ForesightOnlyReranker,
    load_tactile_vae_norm,
)
from TFAC_V5.dp_reranking import _collect_insertion_frames, _find_hdf5_files, _load_frame_data  # noqa: E402
from TFAC_V5.action_aware_scorer_runtime import ActionAwareScorerRuntime  # noqa: E402


OUT_DIR = Path("/home/chenshuai/Project/output/dp_candidate_ranker")


@torch.no_grad()
def predict_marker_for_candidates(reranker, actions_raw, qpos_raw, marker_window, foresight_images, vae_mean, vae_std):
    k = actions_raw.shape[0]
    fs_images = [img.expand(k, *img.shape[1:]) for img in foresight_images]
    fs_chunk = reranker.foresight_chunk
    action_fs = actions_raw[:, :fs_chunk, :]
    action_fs_norm = reranker._fs_norm_action(action_fs)

    qpos_raw_t = torch.tensor(qpos_raw, dtype=torch.float32, device=reranker.device)
    qpos_fs_norm = reranker._fs_norm_qpos(qpos_raw_t.unsqueeze(0)).expand(k, -1)
    z_pred, _, _, _, _, _ = reranker.foresight(fs_images, action_fs_norm, qpos=qpos_fs_norm)
    if z_pred.dim() == 3:
        z_pred = z_pred[:, -1]
    if reranker.foresight_config.get("residual_prediction", False):
        marker_win = marker_window.unsqueeze(0).to(reranker.device)
        z_cur_raw, _ = reranker.foresight.tactile_vae.encode_single_frame(marker_win)
        z_pred = z_cur_raw.reshape(1, -1).expand_as(z_pred) + z_pred

    c = reranker.foresight_config.get("tactile_vae_latent_dim", 16)
    marker_norm = reranker.foresight.tactile_vae.decoder(z_pred.reshape(k, c, 3, 3))
    vae_mean = vae_mean.to(reranker.device).view(1, 1, 1, 2)
    vae_std = vae_std.to(reranker.device).view(1, 1, 1, 2)
    return marker_norm * vae_std + vae_mean


@torch.no_grad()
def candidate_features(scorer, marker_raw, actions_raw):
    k = actions_raw.shape[0]
    marker_seq = marker_raw.unsqueeze(1).expand(-1, WINDOW, -1, -1, -1)
    action_seq = actions_raw[:, :WINDOW, :]
    task_id = torch.zeros(k, dtype=torch.long, device=actions_raw.device)
    out = scorer.forward(marker_seq, action_seq, task_id)
    p_good = torch.softmax(out["binary_logits"], dim=-1)[:, 1:2]
    log_p_good = F.log_softmax(out["binary_logits"], dim=-1)[:, 1:2]
    quality = torch.sigmoid(out["score"]).unsqueeze(-1)
    t4 = torch.softmax(out["t4_logits"], dim=-1)
    mproxy = marker_proxy_features_torch(marker_seq)
    aproxy = action_proxy_features_torch(action_seq, scorer.action_dim)
    hybrid = log_p_good + 0.5 * quality
    return torch.cat([p_good, log_p_good, quality, hybrid, t4, mproxy, aproxy], dim=-1)


def build_dataset(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUT_DIR / f"dp_candidates_K{args.K}_N{args.n_frames}_seed{args.seed}.npz"
    meta_path = OUT_DIR / f"dp_candidates_K{args.K}_N{args.n_frames}_seed{args.seed}_meta.json"
    if cache_path.exists() and not args.force_rebuild:
        return cache_path

    reranker = ForesightOnlyReranker(
        dp_config_path=args.dp_config,
        dp_ckpt_path=args.dp_ckpt,
        foresight_ckpt_path=args.foresight_ckpt,
        foresight_dir=args.foresight_dir,
        device=args.device,
        K=args.K,
    )
    scorer = ActionAwareScorerRuntime(args.scorer_ckpt, device=args.device)
    vae_mean, vae_std = load_tactile_vae_norm(
        reranker.foresight_config.get(
            "tactile_vae_ckpt", "/home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt"
        )
    )
    with open(os.path.join(args.data_dir, "annotations.pkl"), "rb") as f:
        ann = pickle.load(f)
    frames = _collect_insertion_frames(
        _find_hdf5_files(args.data_dir),
        ann,
        cs=reranker.pred_horizon,
        obs_horizon=reranker.obs_horizon,
    )
    rng = np.random.default_rng(args.seed)
    if len(frames) > args.n_frames:
        frames = [frames[i] for i in rng.choice(len(frames), args.n_frames, replace=False)]

    all_x, all_l1, all_groups, all_frame_ids, all_scores = [], [], [], [], []
    for frame_id, (hdf5_path, ep_name, t) in enumerate(tqdm(frames, desc="Build DP candidate dataset")):
        try:
            data = _load_frame_data(reranker, hdf5_path, t, reranker.obs_horizon)
            obs_cond = reranker.build_obs_cond(data["images_obs"], data["qpos_obs"], data["marker_hists"])
            actions = reranker.generate_candidates(obs_cond, K=args.K)
            marker_raw = predict_marker_for_candidates(
                reranker, actions, data["qpos_raw"], data["marker_window"], data["foresight_images"], vae_mean, vae_std
            )
            feats = candidate_features(scorer, marker_raw, actions).detach().cpu().numpy()
            actions_np = actions.detach().cpu().numpy()
            expert_flat = data["action_expert"].reshape(-1)
            l1 = np.array([np.abs(actions_np[i].reshape(-1) - expert_flat).mean() for i in range(args.K)], dtype=np.float32)
            all_x.append(feats.astype(np.float32))
            all_l1.append(l1)
            all_groups.extend([frame_id] * args.K)
            all_frame_ids.extend([f"{ep_name}:{t}"] * args.K)
            all_scores.append(feats[:, :4].astype(np.float32))
        except Exception as exc:
            print(f"Skip {ep_name} t={t}: {exc}")

    x = np.concatenate(all_x, axis=0)
    l1 = np.concatenate(all_l1, axis=0)
    groups = np.asarray(all_groups, dtype=np.int64)
    frame_ids = np.asarray(all_frame_ids)
    base_scores = np.concatenate(all_scores, axis=0)
    np.savez_compressed(cache_path, X=x, l1=l1, y=-l1, groups=groups, frame_ids=frame_ids, base_scores=base_scores)
    meta = {
        "K": args.K,
        "n_requested": args.n_frames,
        "n_frames": int(len(np.unique(groups))),
        "n_candidates": int(len(x)),
        "feature_dim": int(x.shape[1]),
        "feature_layout": "p_good, log_p_good, quality, hybrid, t4_probs(4), marker_proxy(18), action_proxy(10)",
        "target": "-L1(candidate, expert)",
        "note": "Offline ranker calibration target; replace/mix with real tactile quality later.",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {cache_path}")
    return cache_path


def select_metrics(pred, l1, groups):
    by_group = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[int(g)].append(i)
    selected, random_sel, oracle, corr = [], [], [], []
    rng = np.random.default_rng(123)
    wins = 0
    for idxs in by_group.values():
        idxs = np.asarray(idxs, dtype=np.int64)
        best = idxs[np.argmax(pred[idxs])]
        rnd = idxs[int(rng.integers(len(idxs)))]
        selected.append(l1[best])
        random_sel.append(l1[rnd])
        oracle.append(l1[idxs].min())
        wins += int(l1[best] < l1[rnd])
        if np.std(pred[idxs]) > 1e-8 and np.std(l1[idxs]) > 1e-8:
            corr.append(float(np.corrcoef(pred[idxs], -l1[idxs])[0, 1]))
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


def model_suite():
    return {
        "base_quality": None,
        "base_hybrid": None,
        "ridge": Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))]),
        "gbr": GradientBoostingRegressor(n_estimators=180, max_depth=3, learning_rate=0.04, random_state=42),
        "rf": RandomForestRegressor(n_estimators=240, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", MLPRegressor(hidden_layer_sizes=(96, 48), max_iter=500, early_stopping=True, random_state=42)),
        ]),
    }


def evaluate_rankers(cache_path, args):
    data = np.load(cache_path, allow_pickle=True)
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    l1 = data["l1"].astype(np.float32)
    groups = data["groups"].astype(np.int64)

    results = {"cache": str(cache_path), "data": {"n": int(len(x)), "n_groups": int(len(np.unique(groups))), "dim": int(x.shape[1])}, "models": {}}
    cv = GroupKFold(n_splits=min(args.folds, len(np.unique(groups))))
    for name, model in model_suite().items():
        fold_rows = []
        for tr, te in cv.split(x, y, groups):
            if name == "base_quality":
                pred = x[te, 2]
            elif name == "base_hybrid":
                pred = x[te, 3]
            else:
                model.fit(x[tr], y[tr])
                pred = model.predict(x[te])
            row = select_metrics(pred, l1[te], groups[te])
            row["rmse"] = float(np.sqrt(mean_squared_error(y[te], pred)))
            fold_rows.append(row)
        agg = {}
        for key in fold_rows[0]:
            vals = [r[key] for r in fold_rows]
            agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        results["models"][name] = {"folds": fold_rows, "aggregate": agg}
        print(name, agg, flush=True)

    out_path = OUT_DIR / f"ranker_eval_{Path(cache_path).stem}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp_ckpt", default="/home/chenshuai/Project/output/dp_tac_vae_shift4_0414/dp_best.pth")
    parser.add_argument("--dp_config", default="/home/chenshuai/Project/output/dp_tac_vae_shift4_0414/config.json")
    parser.add_argument("--foresight_ckpt", default="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full/foresight_best.ckpt")
    parser.add_argument("--foresight_dir", default="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full")
    parser.add_argument("--scorer_ckpt", default="/home/chenshuai/Project/output/action_aware_marker_scorer_joint_abs/action_aware_marker_scorer_final.pt")
    parser.add_argument("--data_dir", default="/home/chenshuai/data/dataset/0414")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--n_frames", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--cache", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cache = Path(args.cache) if args.cache else build_dataset(args)
    evaluate_rankers(cache, args)
