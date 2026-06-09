"""Board DP denoising -> production Foresight -> PTG scorer guidance smoke.

This closes the engineering loop for board wiping with the available smoke
checkpoints:

  DP denoising action -> board Foresight -> decoded tactile marker
      -> PTG board energy -> dscore/d(noisy action)

It is intentionally labeled as a smoke test because both the board DP and board
Foresight checkpoints used by default are short smoke checkpoints.  The result
proves the gradient interface and code path, not final deployment quality.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "diffusion") not in sys.path:
    sys.path.insert(0, str(ROOT / "diffusion"))

from network import ConditionalUnet1D
from diffusion.train_dp_tac_concat import FrozenTactileVAEEncoder, OfficialVisionEncoder
from TFAC_V5.eval_tac_energy_guided_denoising import summarize
from TFAC_V5.pretrain_latent_foresight import LatentForesightPretrainModel
from TFAC_V5.ptg_proxy_scorer_v2_runtime import PTGProxyScorerV2Runtime, TASK_TO_ID
from TFAC_V5.tac_quality_guidance_config import get_guidance_profile


OUT_DIR = Path("/home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke")


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


class BoardDPForesightChain:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
        self.dp_config = load_json(args.dp_config)
        self.foresight_config = load_json(Path(args.foresight_dir) / "args.json")
        self.foresight_stats = load_pickle(Path(args.foresight_dir) / "dataset_stats.pkl")
        self._load_dp()
        self._load_foresight()
        self.scorer = PTGProxyScorerV2Runtime(args.scorer_ckpt, device=str(self.device))
        freeze(self.scorer)

    def _load_dp(self):
        cfg = self.dp_config
        camera_names = cfg["camera_names"]
        if isinstance(camera_names, str):
            camera_names = camera_names.split(",")
        self.dp_camera_names = camera_names
        self.pred_horizon = int(cfg["pred_horizon"])
        self.obs_horizon = int(cfg.get("obs_horizon", 2))
        self.action_dim = int(cfg["action_dim"])
        self.dp_tac_history = int(cfg.get("tac_history", 8))
        self.dp_resize_shape = tuple(cfg["resize_shape"])
        self.dp_crop_shape = tuple(cfg["crop_shape"])

        self.dp_vision = OfficialVisionEncoder(camera_names).to(self.device)
        self.dp_tac_encoder = FrozenTactileVAEEncoder(
            cfg["vae_checkpoint"],
            latent_dim=int(cfg.get("vae_latent_dim", 16)),
            temporal_window=self.dp_tac_history,
        ).to(self.device)
        down_dims = cfg.get("down_dims", [128, 256])
        if isinstance(down_dims, str):
            down_dims = [int(x) for x in down_dims.split(",")]
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=int(cfg["global_cond_dim"]),
            diffusion_step_embed_dim=int(cfg.get("diffusion_step_embed_dim", 64)),
            down_dims=down_dims,
            kernel_size=5,
        ).to(self.device)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=int(cfg.get("num_train_timesteps", 20)),
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        ckpt = torch.load(self.args.dp_ckpt, map_location=self.device, weights_only=False)
        if "ema_net" in ckpt and not self.args.no_ema:
            self.noise_pred_net.load_state_dict(ckpt["ema_net"])
            self.dp_vision.load_state_dict(ckpt["ema_vis"])
            self.dp_weight_source = "ema"
        else:
            self.noise_pred_net.load_state_dict(ckpt["noise_pred_net"])
            self.dp_vision.load_state_dict(ckpt["vision_encoder"])
            self.dp_weight_source = "raw"
        freeze(self.noise_pred_net)
        freeze(self.dp_vision)
        freeze(self.dp_tac_encoder)

        ns = cfg["norm_stats"]
        self.dp_action_min = torch.tensor(ns["action_min"], dtype=torch.float32, device=self.device)
        self.dp_action_max = torch.tensor(ns["action_max"], dtype=torch.float32, device=self.device)
        self.dp_qpos_min = torch.tensor(ns["qpos_min"], dtype=torch.float32, device=self.device)
        self.dp_qpos_max = torch.tensor(ns["qpos_max"], dtype=torch.float32, device=self.device)
        self.image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _load_foresight(self):
        cfg = self.foresight_config
        camera_names = cfg["camera_names"]
        self.foresight_camera_names = camera_names
        self.foresight_chunk = int(cfg.get("chunk_size", 10))
        self.vae_window = int(cfg.get("tactile_vae_window", 8) or 8)
        self.use_state_trajectory = bool(cfg.get("use_state_trajectory", False))
        self.foresight = LatentForesightPretrainModel(
            camera_names=camera_names,
            cam_backbone_mapping={cam: 0 for cam in camera_names},
            hidden_dim=int(cfg["hidden_dim"]),
            state_dim=int(cfg.get("state_dim", 7)),
            foresight_layers=int(cfg.get("foresight_layers", 1)),
            foresight_nheads=int(cfg.get("foresight_nheads", 4)),
            foresight_dim_feedforward=int(cfg.get("foresight_dim_feedforward", 512)),
            dropout=float(cfg.get("dropout", 0.1)),
            tactile_mode=cfg.get("tactile_mode", "marker"),
            max_history=int(cfg.get("max_history", 8)),
            predict_horizon=int(cfg.get("predict_horizon", 1)),
            tactile_vae_ckpt=cfg.get("tactile_vae_ckpt"),
            tactile_vae_latent_dim=int(cfg.get("tactile_vae_latent_dim", 16)),
            use_delta_pred=bool(cfg.get("use_delta_pred", False)),
            residual_prediction=bool(cfg.get("residual_prediction", False)),
        ).to(self.device)
        state = torch.load(self.args.foresight_ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.foresight.load_state_dict(state, strict=False)
        freeze(self.foresight)

        ns = self.foresight_stats
        self.fs_qpos_mean = torch.tensor(ns["qpos_mean"], dtype=torch.float32, device=self.device)
        self.fs_qpos_std = torch.tensor(ns["qpos_std"], dtype=torch.float32, device=self.device)
        self.fs_action_mean = torch.tensor(ns["action_mean"], dtype=torch.float32, device=self.device)
        self.fs_action_std = torch.tensor(ns["action_std"], dtype=torch.float32, device=self.device)

    def dp_unnorm_action(self, action_norm):
        return (action_norm + 1) / 2 * (self.dp_action_max - self.dp_action_min) + self.dp_action_min

    def dp_norm_qpos(self, qpos_raw):
        return (qpos_raw - self.dp_qpos_min) / (self.dp_qpos_max - self.dp_qpos_min + 1e-8) * 2 - 1

    def fs_norm_qpos(self, qpos_raw):
        return (qpos_raw - self.fs_qpos_mean) / self.fs_qpos_std

    def fs_norm_action(self, action_raw):
        if self.use_state_trajectory:
            return self.fs_norm_qpos(action_raw)
        return (action_raw - self.fs_action_mean) / self.fs_action_std

    def preprocess_image(self, img_uint8, for_dp=True):
        img = torch.tensor(img_uint8.astype(np.float32) / 255.0).permute(2, 0, 1)
        img = self.image_normalize(img)
        if for_dp:
            img = transforms.functional.resize(img, self.dp_resize_shape)
            img = transforms.functional.center_crop(img, self.dp_crop_shape)
        return img

    def marker_history(self, marker_all, t, history):
        frames = []
        for k in range(history):
            idx = max(0, min(t - history + 1 + k, marker_all.shape[0] - 1))
            frames.append(marker_all[idx].astype(np.float32))
        return torch.tensor(np.stack(frames), dtype=torch.float32)

    @torch.no_grad()
    def build_obs_cond(self, f, t):
        obs_feats = []
        marker_all = f["observations/tac/left/marker_offset"][:]
        qpos_all = f["observations/proprio_joint"][:]
        for step in range(self.obs_horizon):
            t_obs = max(0, t - self.obs_horizon + 1 + step)
            imgs = {}
            for cam in self.dp_camera_names:
                imgs[cam] = self.preprocess_image(f[f"observations/images/{cam}"][t_obs], for_dp=True).to(self.device).unsqueeze(0)
            vf = self.dp_vision(imgs)
            marker = self.marker_history(marker_all, t_obs, self.dp_tac_history).to(self.device).unsqueeze(0)
            tf = self.dp_tac_encoder(marker)
            qpos = torch.tensor(qpos_all[t_obs], dtype=torch.float32, device=self.device).unsqueeze(0)
            obs_feats.append(torch.cat([vf, tf, self.dp_norm_qpos(qpos)], dim=-1))
        return torch.cat(obs_feats, dim=-1)

    def build_foresight_inputs(self, f, t, batch):
        marker_all = f["observations/tac/left/marker_offset"][:]
        images = []
        for cam in self.foresight_camera_names:
            if cam == "gelsight":
                marker_win = self.marker_history(marker_all, t, self.vae_window).to(self.device)
                images.append(marker_win.unsqueeze(0).expand(batch, *marker_win.shape))
            else:
                img = self.preprocess_image(f[f"observations/images/{cam}"][t], for_dp=False).to(self.device)
                images.append(img.unsqueeze(0).expand(batch, *img.shape))
        qpos_raw = torch.tensor(f["observations/proprio_joint"][t], dtype=torch.float32, device=self.device)
        qpos_norm = self.fs_norm_qpos(qpos_raw.unsqueeze(0)).expand(batch, -1)
        return images, qpos_norm

    def dp_step(self, noisy_action, timestep, obs_cond):
        noise_pred = self.noise_pred_net(sample=noisy_action, timestep=timestep, global_cond=obs_cond)
        return self.noise_scheduler.step(model_output=noise_pred, timestep=timestep, sample=noisy_action).prev_sample

    def score_norm_actions(self, f, t, action_norm, args):
        batch = action_norm.shape[0]
        action_raw = self.dp_unnorm_action(action_norm)
        fs_images, qpos_norm = self.build_foresight_inputs(f, t, batch)
        fs_action = action_raw[:, : self.foresight_chunk, :]
        fs_action_norm = self.fs_norm_action(fs_action)
        z_pred, _, _, _, _, _ = self.foresight(fs_images, fs_action_norm, future_images=None, qpos=qpos_norm)
        if z_pred.dim() == 3:
            z_pred = z_pred[:, -1]
        c = int(self.foresight_config.get("tactile_vae_latent_dim", 16))
        marker = self.foresight.tactile_vae.decoder(z_pred.reshape(batch, c, 3, 3))
        marker_seq = marker.unsqueeze(1).expand(-1, args.score_window, -1, -1, -1)
        task_id = torch.full((batch,), TASK_TO_ID["board"], dtype=torch.long, device=self.device)
        score = self.scorer.weighted_energy_score(
            marker_seq,
            right_marker_seq=marker_seq,
            joint_action_seq=action_raw[:, : args.score_window, :],
            task_id=task_id,
            quality_weight=args.quality_weight,
            binary_weight=args.binary_weight,
            reason_weight=args.reason_weight,
            clip=not args.no_clip,
        )
        if args.smooth_weight and action_raw.shape[1] >= 3:
            accel = action_raw[:, 2:] - 2 * action_raw[:, 1:-1] + action_raw[:, :-2]
            score = score - args.smooth_weight * torch.linalg.norm(accel, dim=-1).mean(dim=1)
        return score, action_raw, marker, z_pred

    def refine_clean_action(self, f, t, base_norm, args):
        x_base = base_norm.detach()
        x = x_base.clone()
        logs = []
        for step_idx in range(args.refine_steps):
            x_req = x.detach().clone().requires_grad_(True)
            score, _, _, _ = self.score_norm_actions(f, t, x_req, args)
            grad = torch.autograd.grad(score.sum(), x_req, retain_graph=False)[0]
            grad_norm = grad.flatten(1).norm(dim=1)
            grad_unit = grad / grad_norm.view(-1, 1, 1).clamp_min(1e-8)
            with torch.no_grad():
                proposal = x_req + args.clean_refine_scale * grad_unit
                delta = proposal - x_base
                if args.max_clean_norm_delta > 0:
                    delta_norm = delta.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-8)
                    proposal = x_base + delta * torch.clamp(args.max_clean_norm_delta / delta_norm, max=1.0)
                if args.clamp_norm_action:
                    proposal = proposal.clamp(-1.0, 1.0)
                score_new, _, _, _ = self.score_norm_actions(f, t, proposal, args)
                accept = score_new > score.detach()
                x = torch.where(accept.view(-1, 1, 1), proposal, x)
            logs.append(
                {
                    "step_idx": int(step_idx),
                    "score_mean": float(score.detach().mean().cpu()),
                    "score_after_mean": float(score_new.detach().mean().cpu()),
                    "grad_norm_mean": float(grad_norm.detach().mean().cpu()),
                    "accept_rate": float(accept.float().mean().cpu()),
                }
            )
        return x.detach(), logs

    def denoise(self, f, t, obs_cond, initial_noise, args, guided):
        batch = initial_noise.shape[0]
        obs_cond_k = obs_cond.expand(batch, -1)
        x = initial_noise.clone()
        self.noise_scheduler.set_timesteps(int(self.dp_config.get("num_inference_steps", 20)))
        timesteps = list(self.noise_scheduler.timesteps)
        guide_start = int(len(timesteps) * args.guide_start_frac)
        logs = []
        for step_idx, timestep in enumerate(timesteps):
            with torch.no_grad():
                x = self.dp_step(x, timestep, obs_cond_k)
                if args.clamp_norm_action:
                    x = x.clamp(-1.0, 1.0)
            if guided and step_idx >= guide_start and ((step_idx - guide_start) % args.guide_every == 0):
                x_req = x.detach().clone().requires_grad_(True)
                score, _, _, _ = self.score_norm_actions(f, t, x_req, args)
                grad = torch.autograd.grad(score.sum(), x_req, retain_graph=False)[0]
                grad_norm = grad.flatten(1).norm(dim=1)
                grad_unit = grad / grad_norm.view(-1, 1, 1).clamp_min(1e-8)
                with torch.no_grad():
                    proposal = x_req + args.guidance_scale * grad_unit
                    if args.max_norm_delta_per_step > 0:
                        delta = proposal - x
                        delta_norm = delta.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-8)
                        proposal = x + delta * torch.clamp(args.max_norm_delta_per_step / delta_norm, max=1.0)
                    if args.clamp_norm_action:
                        proposal = proposal.clamp(-1.0, 1.0)
                if args.accept_only_improved:
                    with torch.no_grad():
                        score_new, _, _, _ = self.score_norm_actions(f, t, proposal, args)
                        accept = score_new > score.detach()
                        x = torch.where(accept.view(-1, 1, 1), proposal, x)
                        accept_rate = float(accept.float().mean().cpu())
                else:
                    x = proposal.detach()
                    accept_rate = 1.0
                logs.append(
                    {
                        "step_idx": int(step_idx),
                        "timestep": int(timestep),
                        "score_mean": float(score.detach().mean().cpu()),
                        "grad_norm_mean": float(grad_norm.detach().mean().cpu()),
                        "accept_rate": accept_rate,
                    }
                )
        return x.detach(), logs


def episode_files(data_dir: Path):
    return sorted(data_dir.glob("episode_*.hdf5"))


def action_smoothness_np(actions):
    if actions.shape[1] < 3:
        return np.zeros(actions.shape[0], dtype=np.float32)
    accel = actions[:, 2:] - 2 * actions[:, 1:-1] + actions[:, :-2]
    return np.linalg.norm(accel, axis=-1).mean(axis=1)


def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    profile = get_guidance_profile("board")
    if args.quality_weight is None:
        args.quality_weight = profile.energy.quality
    if args.binary_weight is None:
        args.binary_weight = profile.energy.binary_margin
    if args.reason_weight is None:
        args.reason_weight = profile.energy.reason_margin

    chain = BoardDPForesightChain(args)
    files = episode_files(Path(args.data_dir))
    if args.n_episodes and len(files) > args.n_episodes:
        files = [files[i] for i in sorted(rng.choice(len(files), args.n_episodes, replace=False))]
    rows = []
    accum = {k: [] for k in ["base_score", "guided_score", "score_delta", "base_smooth", "guided_smooth", "norm_delta", "range_violation"]}
    guide_grad_norms = []
    guide_accept_rates = []
    for path in tqdm(files, desc="Board DP denoising full-chain smoke"):
        with h5py.File(path, "r") as f:
            T = len(f["observations/proprio_joint"])
            min_t = chain.obs_horizon - 1
            max_t = T - chain.pred_horizon - 1
            if max_t <= min_t:
                continue
            frame_indices = np.linspace(min_t, max_t, min(args.frames_per_episode, max_t - min_t + 1), dtype=int)
            for t in frame_indices:
                obs_cond = chain.build_obs_cond(f, int(t))
                init = torch.randn(args.K, chain.pred_horizon, chain.action_dim, device=chain.device)
                base_norm, _ = chain.denoise(f, int(t), obs_cond, init, args, guided=False)
                if args.mode == "denoising":
                    guided_norm, logs = chain.denoise(f, int(t), obs_cond, init, args, guided=True)
                elif args.mode == "clean_refine":
                    guided_norm, logs = chain.refine_clean_action(f, int(t), base_norm, args)
                else:
                    raise ValueError(args.mode)
                with torch.no_grad():
                    base_score, base_raw, _, _ = chain.score_norm_actions(f, int(t), base_norm, args)
                    guided_score, guided_raw, _, _ = chain.score_norm_actions(f, int(t), guided_norm, args)
                base_np = base_raw.detach().cpu().numpy()
                guided_np = guided_raw.detach().cpu().numpy()
                base_score_np = base_score.detach().cpu().numpy()
                guided_score_np = guided_score.detach().cpu().numpy()
                score_delta = guided_score_np - base_score_np
                norm_delta = torch.linalg.norm((guided_norm - base_norm).flatten(1), dim=1).detach().cpu().numpy()
                range_violation = torch.clamp(guided_norm.abs() - 1.0, min=0.0).amax(dim=(1, 2)).detach().cpu().numpy()
                base_smooth = action_smoothness_np(base_np)
                guided_smooth = action_smoothness_np(guided_np)
                accum["base_score"].append(base_score_np)
                accum["guided_score"].append(guided_score_np)
                accum["score_delta"].append(score_delta)
                accum["base_smooth"].append(base_smooth)
                accum["guided_smooth"].append(guided_smooth)
                accum["norm_delta"].append(norm_delta)
                accum["range_violation"].append(range_violation)
                guide_grad_norms.extend([x["grad_norm_mean"] for x in logs])
                guide_accept_rates.extend([x["accept_rate"] for x in logs])
                rows.append(
                    {
                        "episode": path.name,
                        "t": int(t),
                        "base_score_mean": float(base_score_np.mean()),
                        "guided_score_mean": float(guided_score_np.mean()),
                        "score_delta_mean": float(score_delta.mean()),
                        "guided_beats_base_rate": float(np.mean(guided_score_np > base_score_np)),
                        "norm_delta_mean": float(norm_delta.mean()),
                        "range_violation_max": float(range_violation.max()),
                        "n_guidance_steps": len(logs),
                    }
                )
                if args.n_eval and len(rows) >= args.n_eval:
                    break
        if args.n_eval and len(rows) >= args.n_eval:
            break

    if not rows:
        raise RuntimeError("No valid board frames evaluated")
    merged = {k: np.concatenate(v, axis=0) for k, v in accum.items()}
    score_improved_rate = float(np.mean(merged["score_delta"] > 0))
    result = {
        "config": vars(args),
        "scope": "Smoke full-chain with board DP/Foresight smoke checkpoints. Not final production quality.",
        "n_frames": len(rows),
        "n_action_samples": int(len(merged["score_delta"])),
        "summary": {
            "base_score": summarize(merged["base_score"]),
            "guided_score": summarize(merged["guided_score"]),
            "score_delta": summarize(merged["score_delta"]),
            "guided_beats_base_rate": score_improved_rate,
            "base_smoothness": summarize(merged["base_smooth"]),
            "guided_smoothness": summarize(merged["guided_smooth"]),
            "smoothness_delta": summarize(merged["guided_smooth"] - merged["base_smooth"]),
            "norm_action_delta": summarize(merged["norm_delta"]),
            "range_violation": summarize(merged["range_violation"]),
            "guide_grad_norm_mean_per_step": summarize(guide_grad_norms),
            "guide_accept_rate_per_step": summarize(guide_accept_rates),
        },
        "interpretation": {
            "passes_board_dp_full_chain_smoke": bool(
                score_improved_rate > args.pass_improved_rate
                and float(np.mean(merged["score_delta"])) > 0
                and float(np.max(merged["range_violation"])) <= 1e-6
            ),
            "meaning": "PTG board energy can guide DP actions through board Foresight in the smoke-checkpoint chain.",
            "remaining_gap": "Run the same check with full board DP and stronger board Foresight checkpoints before claiming production completion.",
        },
        "rows": rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="/home/chenshuai/data/dataset/260522_v8l_caheiban_flat_smoke4")
    parser.add_argument("--dp_config", default="/home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_smoke4/config.json")
    parser.add_argument("--dp_ckpt", default="/home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_smoke4/dp_final.pth")
    parser.add_argument("--foresight_dir", default="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_smoke_0")
    parser.add_argument("--foresight_ckpt", default="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_smoke_0/foresight_best.ckpt")
    parser.add_argument("--scorer_ckpt", default="/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_final.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--n_episodes", type=int, default=2)
    parser.add_argument("--frames_per_episode", type=int, default=2)
    parser.add_argument("--n_eval", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", default="clean_refine", choices=["clean_refine", "denoising"])
    parser.add_argument("--score_window", type=int, default=8)
    parser.add_argument("--quality_weight", type=float, default=None)
    parser.add_argument("--binary_weight", type=float, default=None)
    parser.add_argument("--reason_weight", type=float, default=None)
    parser.add_argument("--smooth_weight", type=float, default=0.02)
    parser.add_argument("--no_clip", action="store_true")
    parser.add_argument("--guide_start_frac", type=float, default=0.6)
    parser.add_argument("--guide_every", type=int, default=1)
    parser.add_argument("--guidance_scale", type=float, default=0.002)
    parser.add_argument("--max_norm_delta_per_step", type=float, default=0.01)
    parser.add_argument("--refine_steps", type=int, default=4)
    parser.add_argument("--clean_refine_scale", type=float, default=0.01)
    parser.add_argument("--max_clean_norm_delta", type=float, default=0.08)
    parser.add_argument("--accept_only_improved", action="store_true", default=True)
    parser.add_argument("--clamp_norm_action", action="store_true", default=True)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--pass_improved_rate", type=float, default=0.75)
    parser.add_argument("--output", default=str(OUT_DIR / "board_dp_clean_refine_full_chain_smoke_K4_N4.json"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
