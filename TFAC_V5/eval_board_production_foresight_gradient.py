"""Evaluate board production Foresight -> PTG scorer action gradients.

This is a production-Foresight gradient probe, not a DP rollout.  It checks the
core requirement for classifier guidance on board wiping:

  state trajectory -> board Foresight -> decoded tactile marker
      -> PTG board energy -> denergy/d(state trajectory)

The current board DP checkpoint is still missing, so the action variable here is
the same state trajectory conditioning used by the board Foresight pretraining
config.  A full DP denoising evaluation should replace this input with DP
candidate actions once the board DP checkpoint exists.
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
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.dataset import ForesightEpisodicDataset
from TFAC_V5.eval_tac_energy_guided_denoising import summarize
from TFAC_V5.pretrain_latent_foresight import LatentForesightPretrainModel, _scan_episode_paths
from TFAC_V5.ptg_proxy_scorer_v2_runtime import PTGProxyScorerV2Runtime, TASK_TO_ID
from TFAC_V5.tac_quality_guidance_config import get_guidance_profile


OUT_DIR = Path("/home/chenshuai/Project/output/board_production_foresight_gradient")


def freeze(module: torch.nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_model(config: dict, device: torch.device) -> LatentForesightPretrainModel:
    cam_names = config["camera_names"]
    model = LatentForesightPretrainModel(
        camera_names=cam_names,
        cam_backbone_mapping={cam_name: 0 for cam_name in cam_names},
        hidden_dim=int(config["hidden_dim"]),
        state_dim=int(config.get("state_dim", 7)),
        foresight_layers=int(config.get("foresight_layers", 3)),
        foresight_nheads=int(config.get("foresight_nheads", 8)),
        foresight_dim_feedforward=int(config.get("foresight_dim_feedforward", 2048)),
        dropout=float(config.get("dropout", 0.1)),
        tactile_mode=config.get("tactile_mode", "marker"),
        max_history=int(config.get("max_history", 8)),
        foresight_change_weight=bool(config.get("foresight_change_weight", False)),
        predict_horizon=int(config.get("predict_horizon", 1)),
        tactile_vae_ckpt=config.get("tactile_vae_ckpt"),
        tactile_vae_latent_dim=int(config.get("tactile_vae_latent_dim", 16)),
        use_gtcl=bool(config.get("use_gtcl", False)),
        gtcl_epsilon=float(config.get("gtcl_epsilon", 2.5)),
        gtcl_proj_dim=int(config.get("gtcl_proj_dim", 64)),
        gtcl_temperature=float(config.get("gtcl_temperature", 0.07)),
        use_delta_pred=bool(config.get("use_delta_pred", False)),
        residual_prediction=bool(config.get("residual_prediction", False)),
    ).to(device)
    return model


def load_state(model: LatentForesightPretrainModel, ckpt_path: str | Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"load_state strict=False: missing={len(missing)}, unexpected={len(unexpected)}")


def make_dataset(config: dict, norm_stats: dict, n_episodes: int, seed: int) -> ForesightEpisodicDataset:
    paths = _scan_episode_paths(config["dataset_dir"])
    rng = np.random.default_rng(seed)
    if n_episodes and len(paths) > n_episodes:
        paths = [paths[i] for i in sorted(rng.choice(len(paths), n_episodes, replace=False))]
    return ForesightEpisodicDataset(
        paths,
        config["dataset_dir"],
        config["camera_names"],
        norm_stats,
        chunk_size=int(config["chunk_size"]),
        foresight_horizon=int(config.get("foresight_horizon", 10)),
        proprio_key=config.get("proprio_key", "proprio_joint"),
        action_key=config.get("action_key", "actions/joint_abs"),
        tac_side=config.get("tac_side", "left"),
        tac_img_key=config.get("tac_img_key", "img"),
        tactile_mode=config.get("tactile_mode", "marker"),
        history_len=int(config.get("history_len", 1)),
        tactile_vae_window=int(config.get("tactile_vae_window", 8)),
        preload=False,
        use_state_trajectory=bool(config.get("use_state_trajectory", True)),
    )


def to_device_list(items, device: torch.device):
    return [x.to(device, non_blocking=True).float() for x in items]


def decode_predicted_marker(model: LatentForesightPretrainModel, z_pred: torch.Tensor, config: dict):
    if z_pred.dim() == 3:
        z_pred = z_pred[:, -1]
    c = int(config.get("tactile_vae_latent_dim", 16))
    marker_norm = model.tactile_vae.decoder(z_pred.reshape(z_pred.shape[0], c, 3, 3))
    return marker_norm


def action_smoothness(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] < 3:
        return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    accel = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    return torch.linalg.norm(accel, dim=-1).mean(dim=1)


def score_state_trajectory(args, model, scorer, config, images, qpos, action_traj):
    z_pred, _, _, _, _, _ = model(images, action_traj, future_images=None, qpos=qpos)
    marker_norm = decode_predicted_marker(model, z_pred, config)
    marker_seq = marker_norm.unsqueeze(1).expand(-1, args.score_window, -1, -1, -1)
    action_seq = action_traj[:, : args.score_window, :]
    task_id = torch.full(
        (action_traj.shape[0],),
        TASK_TO_ID["board"],
        dtype=torch.long,
        device=action_traj.device,
    )
    score = scorer.weighted_energy_score(
        marker_seq,
        right_marker_seq=marker_seq,
        joint_action_seq=action_seq,
        task_id=task_id,
        quality_weight=args.quality_weight,
        binary_weight=args.binary_weight,
        reason_weight=args.reason_weight,
        clip=not args.no_clip,
    )
    if args.smooth_weight:
        score = score - args.smooth_weight * action_smoothness(action_seq)
    return score, marker_norm, z_pred


def eval_batch(args, model, scorer, config, batch, device):
    images, qpos, action, _is_pad, _future_images, _history_images = batch
    images = to_device_list(images, device)
    qpos = qpos.to(device, non_blocking=True).float()
    base_action = action.to(device, non_blocking=True).float()
    x = base_action.detach().clone().requires_grad_(True)
    score0, marker0, z0 = score_state_trajectory(args, model, scorer, config, images, qpos, x)
    grad = torch.autograd.grad(score0.sum(), x, retain_graph=False)[0]
    grad_norm = grad.flatten(1).norm(dim=1)
    grad_unit = grad / grad_norm.view(-1, 1, 1).clamp_min(1e-8)
    with torch.no_grad():
        proposal = x + args.step_size * grad_unit
        delta = proposal - base_action
        delta_norm = delta.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-8)
        scale = torch.clamp(args.max_total_delta / delta_norm, max=1.0)
        proposal = base_action + delta * scale
    score1, marker1, z1 = score_state_trajectory(
        args,
        model,
        scorer,
        config,
        images,
        qpos,
        proposal.detach().clone().requires_grad_(True),
    )
    return {
        "score_before": score0.detach().cpu().numpy(),
        "score_after": score1.detach().cpu().numpy(),
        "score_delta": (score1 - score0.detach()).detach().cpu().numpy(),
        "grad_norm": grad_norm.detach().cpu().numpy(),
        "finite_grad": torch.isfinite(grad).flatten(1).all(dim=1).detach().cpu().numpy(),
        "action_delta_norm": (proposal - base_action).detach().flatten(1).norm(dim=1).cpu().numpy(),
        "marker_delta_norm": (marker1 - marker0.detach()).detach().flatten(1).norm(dim=1).cpu().numpy(),
        "z_delta_norm": (z1 - z0.detach()).detach().flatten(1).norm(dim=1).cpu().numpy(),
    }


def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    config = load_json(Path(args.foresight_dir) / "args.json")
    norm_stats = load_pickle(Path(args.foresight_dir) / "dataset_stats.pkl")
    model = build_model(config, device)
    load_state(model, args.foresight_ckpt)
    freeze(model)
    scorer = PTGProxyScorerV2Runtime(args.scorer_ckpt, device=str(device))
    freeze(scorer)

    profile = get_guidance_profile("board")
    if args.quality_weight is None:
        args.quality_weight = profile.energy.quality
    if args.binary_weight is None:
        args.binary_weight = profile.energy.binary_margin
    if args.reason_weight is None:
        args.reason_weight = profile.energy.reason_margin
    if args.step_size is None:
        args.step_size = profile.refinement.action_step
    if args.max_total_delta is None:
        args.max_total_delta = profile.refinement.max_total_delta

    dataset = make_dataset(config, norm_stats, args.n_episodes, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    accum = {
        "score_before": [],
        "score_after": [],
        "score_delta": [],
        "grad_norm": [],
        "finite_grad": [],
        "action_delta_norm": [],
        "marker_delta_norm": [],
        "z_delta_norm": [],
    }
    rows = []
    n_seen = 0
    for batch_idx, batch in enumerate(tqdm(loader, desc="Board production Foresight gradient")):
        out = eval_batch(args, model, scorer, config, batch, device)
        for key, value in out.items():
            accum[key].append(value)
        rows.append(
            {
                "batch": int(batch_idx),
                "score_delta_mean": float(np.mean(out["score_delta"])),
                "score_improved_rate": float(np.mean(out["score_delta"] > 0)),
                "grad_norm_mean": float(np.mean(out["grad_norm"])),
                "finite_grad_rate": float(np.mean(out["finite_grad"])),
            }
        )
        n_seen += len(out["score_delta"])
        if args.n_eval and n_seen >= args.n_eval:
            break

    merged = {k: np.concatenate(v, axis=0)[: args.n_eval] for k, v in accum.items()}
    score_improved_rate = float(np.mean(merged["score_delta"] > 0))
    finite_grad_rate = float(np.mean(merged["finite_grad"]))
    nonzero_grad_rate = float(np.mean(merged["grad_norm"] > 1e-8))
    result = {
        "config": vars(args),
        "foresight_config": {
            "dir": str(args.foresight_dir),
            "ckpt": str(args.foresight_ckpt),
            "hidden_dim": config.get("hidden_dim"),
            "foresight_layers": config.get("foresight_layers"),
            "num_epochs": config.get("num_epochs"),
            "use_state_trajectory": config.get("use_state_trajectory"),
        },
        "n_samples": int(len(merged["score_delta"])),
        "summary": {
            "score_before": summarize(merged["score_before"]),
            "score_after": summarize(merged["score_after"]),
            "score_delta": summarize(merged["score_delta"]),
            "score_improved_rate": score_improved_rate,
            "grad_norm": summarize(merged["grad_norm"]),
            "finite_grad_rate": finite_grad_rate,
            "nonzero_grad_rate": nonzero_grad_rate,
            "action_delta_norm": summarize(merged["action_delta_norm"]),
            "marker_delta_norm": summarize(merged["marker_delta_norm"]),
            "z_delta_norm": summarize(merged["z_delta_norm"]),
        },
        "interpretation": {
            "passes_board_production_foresight_gradient": bool(
                finite_grad_rate > 0.99
                and nonzero_grad_rate > 0.99
                and score_improved_rate > args.pass_improved_rate
                and float(np.mean(merged["score_delta"])) > 0
            ),
            "scope": (
                "Production board Foresight gradient probe using state-trajectory conditioning. "
                "This verifies the scorer can backprop through board Foresight, but it is not a "
                "full DP denoising evaluation because board DP checkpoint is not used."
            ),
        },
        "rows": rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foresight_dir", default="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_smoke_0")
    parser.add_argument("--foresight_ckpt", default="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_smoke_0/foresight_best.ckpt")
    parser.add_argument("--scorer_ckpt", default="/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_final.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_episodes", type=int, default=4)
    parser.add_argument("--n_eval", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--score_window", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality_weight", type=float, default=None)
    parser.add_argument("--binary_weight", type=float, default=None)
    parser.add_argument("--reason_weight", type=float, default=None)
    parser.add_argument("--step_size", type=float, default=None)
    parser.add_argument("--max_total_delta", type=float, default=None)
    parser.add_argument("--smooth_weight", type=float, default=0.02)
    parser.add_argument("--no_clip", action="store_true")
    parser.add_argument("--pass_improved_rate", type=float, default=0.9)
    parser.add_argument("--output", default=str(OUT_DIR / "board_production_foresight_smoke_gradient_N64.json"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
