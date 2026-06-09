"""Evaluate full action -> Foresight -> scorer gradient flow.

This is the integration test that matters for classifier guidance.  It does not
optimize candidate reranking.  It checks whether TacQualityEnergy can provide a
usable gradient all the way back to the action trajectory:

  action_seq -> LatentForesight -> decoded marker -> energy scorer -> dS/daction

The test uses insertion because the current trained Foresight/DP stack is for
the socket insertion dataset.  Board wiping already has a strong scorer, but it
needs a board-specific Foresight/DP stack before this same full-chain test can
be run there.
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

from TFAC_V5.dp_reranking import _collect_insertion_frames, _find_hdf5_files, _load_frame_data  # noqa: E402
from TFAC_V5.eval_action_aware_reranking import ForesightOnlyReranker, load_tactile_vae_norm  # noqa: E402
from TFAC_V5.insertion_risk_scorer_runtime import InsertionRiskScorerRuntime, WINDOW  # noqa: E402


OUT_DIR = Path("/home/chenshuai/Project/output/full_chain_guidance_gradient")


def freeze(module):
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def predict_marker_differentiable(reranker, actions_raw, qpos_raw, marker_window, foresight_images, vae_mean, vae_std):
    """Differentiable counterpart of predict_marker_for_candidates."""
    k = actions_raw.shape[0]
    fs_images = [img.expand(k, *img.shape[1:]) for img in foresight_images]
    action_fs = actions_raw[:, : reranker.foresight_chunk, :]
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
    return marker_norm * vae_std + vae_mean, z_pred


def action_smoothness(actions):
    if actions.shape[1] < 3:
        return torch.zeros(actions.shape[0], device=actions.device)
    accel = actions[:, 2:] - 2 * actions[:, 1:-1] + actions[:, :-2]
    return torch.linalg.norm(accel, dim=-1).mean(dim=1)


def score_actions(reranker, scorer, actions, data, vae_mean, vae_std, mode, smooth_weight):
    marker_raw, z_pred = predict_marker_differentiable(
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
    if smooth_weight:
        score = score - smooth_weight * action_smoothness(action_seq)
    return score, marker_raw, z_pred


def summarize(x):
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def build_action_batch(args, expert, rng):
    expert = expert.to(dtype=torch.float32)
    action_std = expert.std().clamp(min=1e-4)
    rows = [expert]
    for i in range(args.K - 1):
        scale = args.noise_scales[i % len(args.noise_scales)]
        rows.append(expert + torch.randn_like(expert) * action_std * scale)
    return torch.stack(rows, dim=0)


def evaluate_frame(args, reranker, scorer, data, vae_mean, vae_std, rng):
    expert = torch.tensor(data["action_expert"], dtype=torch.float32, device=reranker.device)
    base_actions = build_action_batch(args, expert, rng).detach().clone().requires_grad_(True)
    score0, marker0, z0 = score_actions(reranker, scorer, base_actions, data, vae_mean, vae_std, args.score_mode, args.smooth_weight)
    grad = torch.autograd.grad(score0.sum(), base_actions, retain_graph=False)[0]

    grad_flat_norm = grad.flatten(1).norm(dim=1)
    action_flat_norm = base_actions.detach().flatten(1).norm(dim=1).clamp_min(1e-8)
    grad_rel = grad_flat_norm / action_flat_norm
    grad_unit = grad / grad_flat_norm.view(-1, 1, 1).clamp_min(1e-8)

    with torch.no_grad():
        stepped = base_actions + args.step_size * grad_unit
    stepped = stepped.detach().clone().requires_grad_(True)
    score1, marker1, z1 = score_actions(reranker, scorer, stepped, data, vae_mean, vae_std, args.score_mode, args.smooth_weight)

    delta = (score1 - score0.detach()).detach()
    marker_delta = (marker1 - marker0.detach()).flatten(1).norm(dim=1)
    z_delta = (z1 - z0.detach()).flatten(1).norm(dim=1)
    return {
        "score_before": score0.detach().cpu().numpy(),
        "score_after": score1.detach().cpu().numpy(),
        "score_delta": delta.cpu().numpy(),
        "grad_norm": grad_flat_norm.detach().cpu().numpy(),
        "grad_relative_norm": grad_rel.detach().cpu().numpy(),
        "marker_delta_norm": marker_delta.detach().cpu().numpy(),
        "z_delta_norm": z_delta.detach().cpu().numpy(),
        "finite_grad": torch.isfinite(grad).flatten(1).all(dim=1).detach().cpu().numpy(),
    }


def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    reranker = ForesightOnlyReranker(
        dp_config_path=args.dp_config,
        dp_ckpt_path=args.dp_ckpt,
        foresight_ckpt_path=args.foresight_ckpt,
        foresight_dir=args.foresight_dir,
        device=args.device,
        K=args.K,
    )
    freeze(reranker.foresight)
    scorer = InsertionRiskScorerRuntime(args.scorer_ckpt, device=args.device)
    freeze(scorer)

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
    if len(frames) > args.n_eval:
        frames = [frames[i] for i in rng.choice(len(frames), args.n_eval, replace=False)]

    rows = []
    accum = {k: [] for k in ["score_before", "score_after", "score_delta", "grad_norm", "grad_relative_norm", "marker_delta_norm", "z_delta_norm", "finite_grad"]}
    for hdf5_path, ep_name, t in tqdm(frames, desc="Full-chain guidance gradient"):
        try:
            data = _load_frame_data(reranker, hdf5_path, t, reranker.obs_horizon)
            out = evaluate_frame(args, reranker, scorer, data, vae_mean, vae_std, rng)
            for key in accum:
                accum[key].append(out[key])
            rows.append(
                {
                    "episode": ep_name,
                    "t": int(t),
                    "score_delta_mean": float(np.mean(out["score_delta"])),
                    "score_improved_rate": float(np.mean(out["score_delta"] > 0)),
                    "grad_norm_mean": float(np.mean(out["grad_norm"])),
                    "finite_grad_rate": float(np.mean(out["finite_grad"])),
                }
            )
        except Exception as exc:
            print(f"Skip {ep_name} t={t}: {exc}")

    if not rows:
        raise RuntimeError("No valid full-chain rows")

    merged = {k: np.concatenate(v, axis=0) for k, v in accum.items()}
    result = {
        "config": vars(args),
        "n_frames": len(rows),
        "n_action_samples": int(len(merged["score_delta"])),
        "summary": {
            "score_before": summarize(merged["score_before"]),
            "score_after": summarize(merged["score_after"]),
            "score_delta": summarize(merged["score_delta"]),
            "score_improved_rate": float(np.mean(merged["score_delta"] > 0)),
            "grad_norm": summarize(merged["grad_norm"]),
            "grad_relative_norm": summarize(merged["grad_relative_norm"]),
            "finite_grad_rate": float(np.mean(merged["finite_grad"])),
            "marker_delta_norm": summarize(merged["marker_delta_norm"]),
            "z_delta_norm": summarize(merged["z_delta_norm"]),
        },
        "interpretation": {
            "passes_full_chain_gradient": bool(
                np.mean(merged["finite_grad"]) > 0.99
                and np.mean(merged["grad_norm"] > 1e-8) > 0.99
                and np.mean(merged["score_delta"] > 0) > 0.95
            ),
            "meaning": "The scorer can provide gradients through action -> Foresight -> decoded tactile -> energy score.",
            "board_note": "Board wiping needs a board Foresight/DP stack before this full-chain test can be repeated there.",
        },
        "rows": rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp_ckpt", default="/home/chenshuai/Project/output/dp_tac_vae_shift4_0414/dp_best.pth")
    parser.add_argument("--dp_config", default="/home/chenshuai/Project/output/dp_tac_vae_shift4_0414/config.json")
    parser.add_argument("--foresight_ckpt", default="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full/foresight_best.ckpt")
    parser.add_argument("--foresight_dir", default="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full")
    parser.add_argument("--scorer_ckpt", default="/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt")
    parser.add_argument("--data_dir", default="/home/chenshuai/data/dataset/0414")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--n_eval", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score_mode", default="energy_clipped", choices=["quality", "log_p_good", "risk_guidance", "energy", "energy_clipped"])
    parser.add_argument("--step_size", type=float, default=0.02)
    parser.add_argument("--smooth_weight", type=float, default=0.0)
    parser.add_argument("--noise_scales", type=lambda s: [float(x) for x in s.split(",")], default="0.0,0.05,0.1,0.2")
    parser.add_argument("--output", default=str(OUT_DIR / "insertion_full_chain_energy_clipped_K8_N16.json"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
