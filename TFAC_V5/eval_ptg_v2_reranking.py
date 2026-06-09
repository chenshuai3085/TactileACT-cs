"""Evaluate PTG proxy scorer v2 on insertion DP candidates.

This mirrors eval_action_aware_reranking.py but scores candidates with the new
unified PTGProxyScorerV2Runtime.  Foresight currently predicts one marker field
from the insertion tactile stream, so the first integration test feeds that
predicted marker to both left/right inputs of v2.  This is not the final
two-hand tactile setup, but it tells whether the v2 quality/risk heads carry a
useful ranking signal on real DP sampled candidates.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "TFAC_V5"))

from TFAC_V5.eval_action_aware_reranking import (  # noqa: E402
    ForesightOnlyReranker,
    load_tactile_vae_norm,
)
from TFAC_V5.dp_reranking import _collect_insertion_frames, _find_hdf5_files, _load_frame_data  # noqa: E402
from TFAC_V5.ptg_proxy_scorer_v2_runtime import PTGProxyScorerV2Runtime  # noqa: E402


OUT_DIR = Path("/home/chenshuai/Project/output/ptg_v2_reranking")
WINDOW = 8


@torch.no_grad()
def predict_marker_for_candidates(reranker, actions_raw, qpos_raw, marker_window, foresight_images, vae_mean, vae_std):
    k = actions_raw.shape[0]
    fs_images = [img.expand(k, *img.shape[1:]) for img in foresight_images]
    fs_chunk = reranker.foresight_chunk
    action_fs = actions_raw[:, :fs_chunk, :]
    action_fs_norm = reranker._fs_norm_action(action_fs)

    if isinstance(qpos_raw, np.ndarray):
        qpos_raw_t = torch.tensor(qpos_raw, dtype=torch.float32, device=reranker.device)
    else:
        qpos_raw_t = qpos_raw.to(reranker.device)
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
def score_candidates(reranker, scorer, actions, data, vae_mean, vae_std, args):
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
    task_id = torch.zeros(actions.shape[0], dtype=torch.long, device=reranker.device)
    out = scorer.forward(
        marker_seq,
        marker_seq,
        joint_action_seq=action_seq,
        task_id=task_id,
    )
    if args.score_mode == "quality":
        score = out["quality_score"]
    elif args.score_mode == "p_good":
        score = out["p_good"]
    elif args.score_mode == "log_p_good":
        score = out["log_p_good"]
    elif args.score_mode == "reason_good":
        score = out["reason_prob"][:, 1]
    elif args.score_mode == "guidance":
        score = scorer.guidance_score(
            marker_seq,
            marker_seq,
            joint_action_seq=action_seq,
            task_id=task_id,
            quality_weight=args.quality_weight,
            good_weight=args.good_weight,
            reason_weight=args.reason_weight,
            action_smooth_weight=args.action_smooth_weight,
        )
    else:
        raise ValueError(args.score_mode)
    return score, out, marker_raw


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def run_eval(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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

    rows = []
    wins = 0
    noise_scales = [float(x) for x in args.noise_scales.split(",")]
    for hdf5_path, ep_name, t in tqdm(frames, desc=f"PTG-v2 rerank {args.score_mode}"):
        try:
            data = _load_frame_data(reranker, hdf5_path, t, reranker.obs_horizon)
            expert = torch.tensor(data["action_expert"], dtype=torch.float32, device=reranker.device)
            if args.candidate_mode == "dp_sampling":
                obs_cond = reranker.build_obs_cond(data["images_obs"], data["qpos_obs"], data["marker_hists"])
                actions = reranker.generate_candidates(obs_cond, K=args.K)
            elif args.candidate_mode == "simulated":
                action_std = expert.std().clamp(min=1e-4)
                candidates = []
                if args.include_expert:
                    candidates.append(expert)
                for i in range(args.K - len(candidates)):
                    scale = noise_scales[i % len(noise_scales)]
                    candidates.append(expert + torch.randn_like(expert) * action_std * scale)
                actions = torch.stack(candidates)
            else:
                raise ValueError(args.candidate_mode)

            scores, out, _ = score_candidates(reranker, scorer, actions, data, vae_mean, vae_std, args)
            s = scores.detach().cpu().numpy()
            actions_np = actions.detach().cpu().numpy()
            expert_flat = data["action_expert"].reshape(-1)
            l1 = np.array([np.abs(actions_np[i].reshape(-1) - expert_flat).mean() for i in range(args.K)])
            rank = np.argsort(-s)
            best_idx = int(rank[0])
            random_idx = int(rng.integers(args.K))
            wins += int(l1[best_idx] < l1[random_idx])
            corr = np.corrcoef(s, -l1)[0, 1] if np.std(s) > 1e-8 and np.std(l1) > 1e-8 else 0.0
            reason_prob = out["reason_prob"].detach().cpu().numpy()
            rows.append(
                {
                    "episode": ep_name,
                    "t": int(t),
                    "best_idx": best_idx,
                    "random_idx": random_idx,
                    "score_best": float(s[best_idx]),
                    "score_random": float(s[random_idx]),
                    "l1_best": float(l1[best_idx]),
                    "l1_random": float(l1[random_idx]),
                    "l1_oracle": float(l1.min()),
                    "l1_worst": float(l1.max()),
                    "score_l1_corr": float(corr),
                    "score_range": float(s.max() - s.min()),
                    "p_good_best": float(out["p_good"].detach().cpu().numpy()[best_idx]),
                    "quality_best": float(out["quality_score"].detach().cpu().numpy()[best_idx]),
                    "reason_good_best": float(reason_prob[best_idx, 1]),
                }
            )
        except Exception as exc:
            print(f"Skip {ep_name} t={t}: {exc}")

    if not rows:
        raise RuntimeError("No valid rows")
    result = {
        "config": vars(args),
        "n_frames": len(rows),
        "beats_random_l1": wins / len(rows),
        "l1_best": summarize([r["l1_best"] for r in rows]),
        "l1_random": summarize([r["l1_random"] for r in rows]),
        "l1_oracle": summarize([r["l1_oracle"] for r in rows]),
        "score_l1_corr": summarize([r["score_l1_corr"] for r in rows]),
        "score_range": summarize([r["score_range"] for r in rows]),
        "rows": rows,
    }
    out_path = OUT_DIR / f"{args.candidate_mode}_{args.score_mode}_K{args.K}_N{len(rows)}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))
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
    parser.add_argument("--n_eval", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate_mode", default="dp_sampling", choices=["dp_sampling", "simulated"])
    parser.add_argument("--include_expert", action="store_true")
    parser.add_argument("--noise_scales", default="0.05,0.1,0.2,0.4,0.8")
    parser.add_argument(
        "--score_mode",
        default="guidance",
        choices=["quality", "p_good", "log_p_good", "reason_good", "guidance"],
    )
    parser.add_argument("--quality_weight", type=float, default=1.0)
    parser.add_argument("--good_weight", type=float, default=0.15)
    parser.add_argument("--reason_weight", type=float, default=0.25)
    parser.add_argument("--action_smooth_weight", type=float, default=0.02)
    return parser.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
