"""Evaluate insertion-specific risk scorer on DP sampled candidates."""

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

from TFAC_V5.eval_action_aware_reranking import ForesightOnlyReranker, load_tactile_vae_norm  # noqa: E402
from TFAC_V5.dp_reranking import _collect_insertion_frames, _find_hdf5_files, _load_frame_data  # noqa: E402
from TFAC_V5.eval_ptg_v2_reranking import predict_marker_for_candidates  # noqa: E402
from TFAC_V5.insertion_risk_scorer_runtime import InsertionRiskScorerRuntime, WINDOW  # noqa: E402


OUT_DIR = Path("/home/chenshuai/Project/output/insertion_risk_reranking")


@torch.no_grad()
def score_candidates(reranker, scorer, actions, data, vae_mean, vae_std, mode):
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
    score = scorer.score(marker_seq, action_seq, mode=mode)
    out = scorer.forward(marker_seq, action_seq)
    return score, out


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
    scorer = InsertionRiskScorerRuntime(args.scorer_ckpt, device=args.device)
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
    for hdf5_path, ep_name, t in tqdm(frames, desc=f"Insertion-risk rerank {args.score_mode}"):
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

            score, out = score_candidates(reranker, scorer, actions, data, vae_mean, vae_std, args.score_mode)
            s = score.detach().cpu().numpy()
            actions_np = actions.detach().cpu().numpy()
            expert_flat = data["action_expert"].reshape(-1)
            l1 = np.array([np.abs(actions_np[i].reshape(-1) - expert_flat).mean() for i in range(args.K)])
            rank = np.argsort(-s)
            best_idx = int(rank[0])
            random_idx = int(rng.integers(args.K))
            wins += int(l1[best_idx] < l1[random_idx])
            corr = np.corrcoef(s, -l1)[0, 1] if np.std(s) > 1e-8 and np.std(l1) > 1e-8 else 0.0
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
                    "risk_best": float(out["risk_prob"].detach().cpu().numpy()[best_idx]),
                    "quality_best": float(out["quality_score"].detach().cpu().numpy()[best_idx]),
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
    parser.add_argument("--scorer_ckpt", default="/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt")
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
        default="risk_guidance",
        choices=["quality", "p_good", "log_p_good", "neg_risk", "risk_guidance"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
