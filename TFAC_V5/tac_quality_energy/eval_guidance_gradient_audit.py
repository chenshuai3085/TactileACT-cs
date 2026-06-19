#!/usr/bin/env python3
"""Audit TacQuality gradient guidance through Foresight.

This is not a rollout metric.  It checks whether the deployed-style guidance
chain has a usable differentiable signal:

    raw action chunk -> Foresight -> predicted marker -> TacQuality score
        -> d score / d action -> trust-region refinement
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import h5py
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as TVF

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.pretrain_latent_foresight_multistep import MultiStepLatentForesightModel
from TFAC_V5.pretrain_latent_foresight import LatentForesightPretrainModel
from TFAC_V5.tac_quality_energy.foresight_bridge import ForesightBridgeConfig, ForesightTacQualityBridge
from TFAC_V5.tac_quality_energy.serving_guidance import ActionNormalizer, load_rollout_arm_config
from TFAC_V5.tac_quality_energy.trust_region import TacQualityTrustRegionRefiner, TrustRegionConfig
from TFAC_V5.tac_quality_energy.ptg_proxy_runtime import PTGProxyScorerV2Runtime
from TFAC_V5.tac_quality_energy.insertion_runtime import InsertionRiskScorerRuntime
from TFAC_V5.tac_quality_energy.runtime import DistilledTacQualityEnergyRuntime
from TFAC_V5.tac_quality_energy.force_band_runtime import ForceBandTacQualityEnergyRuntime
from TFAC_V5.tac_quality_energy.board_ensemble_runtime import BoardForceBandEnsembleRuntime


DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_guidance_gradient_audit")
DEFAULT_BOARD_DATASET = Path("/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/wipe_pos_straight_z124_125_150_20260609")
DEFAULT_BOARD_FORESIGHT_DIR = Path("/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload")
DEFAULT_BOARD_FORESIGHT_CKPT = DEFAULT_BOARD_FORESIGHT_DIR / "foresight_best.ckpt"
DEFAULT_ROLLOUT_CONFIG = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/"
    "tac_quality_rollout_arm_configs_current_s12_good_margin_20260619.json"
)
IMG_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def freeze(module: torch.nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def summarize(x: torch.Tensor | np.ndarray | List[float]) -> Dict[str, Any]:
    arr = torch.as_tensor(x).detach().float().flatten()
    if arr.numel() == 0:
        return {"n": 0}
    return {
        "n": int(arr.numel()),
        "mean": float(arr.mean().cpu()),
        "std": float(arr.std(unbiased=False).cpu()),
        "min": float(arr.min().cpu()),
        "max": float(arr.max().cpu()),
    }


def marker_norm_stats(fs_cfg: Mapping[str, Any]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return the marker normalization used by the Foresight TactileVAE."""
    norm = fs_cfg.get("norm_stats", fs_cfg)
    mean = norm.get("marker_offset_mean")
    std = norm.get("marker_offset_std")
    if mean is None or std is None:
        vae_ckpt = fs_cfg.get("tactile_vae_ckpt")
        if vae_ckpt and Path(vae_ckpt).exists():
            ckpt = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
            vae_stats = ckpt.get("norm_stats", {})
            mean = vae_stats.get("mean", mean)
            std = vae_stats.get("std", std)
    if mean is None:
        mean = [0.2101736068725586, -0.6422404050827026]
    if std is None:
        std = [1.6805468797683716, 3.6716601848602295]
    return tuple(float(x) for x in mean), tuple(float(x) for x in std)


def preprocess_foresight_image(img_uint8: np.ndarray) -> torch.Tensor:
    """Match the single-frame image preprocessing used by insertion Foresight."""
    img = torch.as_tensor(img_uint8.astype(np.float32) / 255.0).permute(2, 0, 1)
    return IMG_NORM(img)


def list_episode_paths(dataset_dir: Path, max_episodes: int) -> List[Path]:
    paths = sorted(dataset_dir.glob("episode_*.hdf5"))
    if not paths:
        paths = sorted(dataset_dir.glob("*/episode_*.hdf5"))
    if max_episodes > 0:
        paths = paths[:max_episodes]
    if not paths:
        raise FileNotFoundError(f"No episode_*.hdf5 under {dataset_dir}")
    return paths


def load_foresight(foresight_dir: Path, foresight_ckpt: Path, device: torch.device):
    cfg = load_json(foresight_dir / "args.json")
    camera_names = cfg.get("camera_names", ["gelsight"])
    predict_horizon = int(cfg.get("predict_horizon", cfg.get("foresight_horizon", 1)))
    kwargs = dict(
        camera_names=camera_names,
        cam_backbone_mapping={cam: 0 for cam in camera_names},
        hidden_dim=int(cfg.get("hidden_dim", 512)),
        state_dim=int(cfg.get("state_dim", 7)),
        foresight_layers=int(cfg.get("foresight_layers", 3)),
        foresight_nheads=int(cfg.get("foresight_nheads", 8)),
        foresight_dim_feedforward=int(cfg.get("foresight_dim_feedforward", 2048)),
        dropout=float(cfg.get("dropout", 0.1)),
        tactile_mode=cfg.get("tactile_mode", "marker"),
        max_history=int(cfg.get("max_history", 8)),
        predict_horizon=predict_horizon,
        tactile_vae_ckpt=cfg.get("tactile_vae_ckpt"),
        tactile_vae_latent_dim=int(cfg.get("tactile_vae_latent_dim", 16)),
    )
    if predict_horizon > 1:
        model = MultiStepLatentForesightModel(**kwargs).to(device)
        kind = "multistep"
    else:
        model = LatentForesightPretrainModel(
            **kwargs,
            use_delta_pred=bool(cfg.get("use_delta_pred", False)),
            residual_prediction=bool(cfg.get("residual_prediction", False)),
        ).to(device)
        kind = "single_step"
    state = torch.load(foresight_ckpt, map_location=device, weights_only=False)
    if isinstance(state, Mapping) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    freeze(model)
    fs_norm = load_pickle(foresight_dir / "dataset_stats.pkl")
    return model, cfg, fs_norm, {"kind": kind, "missing": len(missing), "unexpected": len(unexpected)}


def load_scorer(runtime_name: str, checkpoint: str, device: str, ensemble_config: Mapping[str, Any] | None = None):
    if runtime_name == "PTGProxyScorerV2Runtime":
        return PTGProxyScorerV2Runtime(checkpoint, device=device)
    if runtime_name == "InsertionRiskScorerRuntime":
        return InsertionRiskScorerRuntime(checkpoint, device=device)
    if runtime_name == "DistilledTacQualityEnergyRuntime":
        return DistilledTacQualityEnergyRuntime(checkpoint, device=device)
    if runtime_name == "ForceBandTacQualityEnergyRuntime":
        return ForceBandTacQualityEnergyRuntime(checkpoint, device=device)
    if runtime_name == "BoardForceBandEnsembleRuntime":
        cfg = dict(ensemble_config or {})
        return BoardForceBandEnsembleRuntime(
            old_checkpoint=cfg.get("old_checkpoint", checkpoint),
            s12_checkpoint=cfg.get("s12_checkpoint", checkpoint),
            old_weight=float(cfg.get("old_weight", 0.95)),
            s12_weight=cfg.get("s12_weight"),
            old_mode=str(cfg.get("old_mode", "energy_clipped")),
            s12_mode=str(cfg.get("s12_mode", "energy_clipped")),
            device=device,
        )
    raise KeyError(f"Unsupported scorer runtime: {runtime_name}")


def refiner_config(arm: Mapping[str, Any]) -> TrustRegionConfig:
    ref = arm.get("refiner", {}).get("refinement", {})
    return TrustRegionConfig(
        steps=int(ref.get("refine_steps", 4)),
        step_size=float(ref.get("action_step", 0.0002)),
        max_total_delta=float(ref.get("max_total_delta", 0.02)),
        accept_only_improved=bool(ref.get("accept_only_improved", True)),
    )


def score_from_prediction(scorer, score_mode: str, profile_energy: Mapping[str, Any], tactile: Dict[str, torch.Tensor], action_raw: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
    if score_mode == "profile" and hasattr(scorer, "weighted_energy_score"):
        return scorer.weighted_energy_score(
            tactile["left_marker_seq"],
            right_marker_seq=tactile.get("right_marker_seq"),
            eef_action_seq=tactile.get("eef_action_seq"),
            joint_action_seq=action_raw,
            task_id=task_id,
            quality_weight=float(profile_energy.get("quality", 0.75)),
            binary_weight=float(profile_energy.get("binary_margin", 0.10)),
            reason_weight=float(profile_energy.get("reason_margin", 0.0)),
            clip=True,
        )
    # Match serving_guidance.EnergyGuidanceAdapter: runtimes without an explicit
    # weighted_energy_score fall back from profile to energy_clipped.
    fallback_mode = "energy_clipped" if score_mode == "profile" else score_mode
    return scorer.score(
        tactile["left_marker_seq"],
        right_marker_seq=tactile.get("right_marker_seq"),
        eef_action_seq=tactile.get("eef_action_seq"),
        joint_action_seq=action_raw,
        task_id=task_id,
        mode=fallback_mode,
    )


def marker_window(raw_marker: np.ndarray, start: int, window: int) -> np.ndarray:
    frames = []
    for i in range(window):
        t = max(0, start - (window - 1 - i))
        frames.append(raw_marker[t])
    return np.stack(frames).astype(np.float32)


def valid_starts(length: int, window: int, chunk: int, count: int, rng: np.random.Generator) -> np.ndarray:
    lo = window - 1
    hi = max(lo + 1, length - chunk - 1)
    if hi <= lo:
        return np.asarray([], dtype=np.int64)
    if count <= 0 or count >= (hi - lo):
        return np.arange(lo, hi, dtype=np.int64)
    return np.sort(rng.choice(np.arange(lo, hi, dtype=np.int64), size=count, replace=False))


def build_bridge(
    foresight,
    fs_norm: Mapping[str, Any],
    fs_cfg: Mapping[str, Any],
    qpos_raw: torch.Tensor,
    marker_window_raw: torch.Tensor,
    foresight_images: Iterable[torch.Tensor],
    task: str,
) -> ForesightTacQualityBridge:
    mean, std = marker_norm_stats(fs_cfg)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=marker_window_raw.device).view(1, 1, 1, 1, 2)
    std_t = torch.tensor(std, dtype=torch.float32, device=marker_window_raw.device).view(1, 1, 1, 1, 2)
    marker_norm = (marker_window_raw - mean_t) / std_t.clamp_min(1e-8)
    return ForesightTacQualityBridge(
        foresight,
        fs_norm,
        qpos_raw=qpos_raw,
        foresight_images=list(foresight_images),
        marker_window_norm=marker_norm,
        config=ForesightBridgeConfig(
            task=task,
            window=int(fs_cfg.get("tactile_vae_window", 8)),
            action_chunk=int(fs_cfg.get("chunk_size", 16)),
            latent_dim=int(fs_cfg.get("tactile_vae_latent_dim", 16)),
            marker_mean=mean,
            marker_std=std,
            residual_prediction=bool(fs_cfg.get("residual_prediction", False)),
        ),
    )


def sample_episode_windows(args: argparse.Namespace, fs_cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    rows: List[Dict[str, Any]] = []
    camera_names = list(fs_cfg.get("camera_names", ["gelsight"]))
    image_cameras = [cam for cam in camera_names if cam not in ("gelsight", "blank")]
    for path in list_episode_paths(Path(args.dataset_dir), args.max_episodes):
        try:
            with h5py.File(path, "r") as root:
                marker_path = f"observations/tac/{args.tac_side}/marker_offset"
                qpos_path = f"observations/{args.proprio_key}"
                if marker_path not in root or qpos_path not in root or args.action_key not in root:
                    continue
                marker = root[marker_path][()]
                qpos = root[qpos_path][()]
                action = root[args.action_key][()]
                image_arrays = {}
                missing_image = False
                for cam in image_cameras:
                    key = f"observations/images/{cam}"
                    if key not in root:
                        missing_image = True
                        break
                    image_arrays[cam] = root[key][()]
                if missing_image:
                    continue
        except OSError:
            continue
        length = min(len(marker), len(qpos), len(action))
        starts = valid_starts(length, args.window, args.action_chunk, args.samples_per_episode, rng)
        for start in starts:
            foresight_images = [
                preprocess_foresight_image(image_arrays[cam][int(start)])
                for cam in image_cameras
            ]
            rows.append(
                {
                    "path": str(path),
                    "start": int(start),
                    "qpos": qpos[start].astype(np.float32),
                    "marker_window": marker_window(marker, int(start), args.window),
                    "action": action[start : start + args.action_chunk].astype(np.float32),
                    "foresight_images": foresight_images,
                }
            )
    if args.max_samples > 0 and len(rows) > args.max_samples:
        idx = rng.choice(np.arange(len(rows)), size=args.max_samples, replace=False)
        rows = [rows[int(i)] for i in np.sort(idx)]
    if not rows:
        raise RuntimeError("No valid windows sampled for gradient audit")
    return rows


def run(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    rollout = load_rollout_arm_config(Path(args.rollout_arm_config))
    arm = rollout["tasks"][args.task][args.arm]
    scorer_runtime = args.scorer_runtime or arm["scorer_runtime"]
    scorer_checkpoint = args.scorer_checkpoint or arm["checkpoint"]["path"]
    ensemble_config = arm.get("ensemble", {})
    if args.ensemble_config:
        ensemble_config = json.loads(args.ensemble_config)
    scorer = load_scorer(scorer_runtime, scorer_checkpoint, str(device), ensemble_config=ensemble_config)
    score_mode = str(args.score_mode or arm.get("refiner", {}).get("score_mode", "energy_clipped"))
    profile_energy = arm.get("refiner", {}).get("energy", {})
    refiner = TacQualityTrustRegionRefiner(refiner_config(arm))
    foresight, fs_cfg, fs_norm, fs_info = load_foresight(Path(args.foresight_dir), Path(args.foresight_ckpt), device)
    windows = sample_episode_windows(args, fs_cfg)

    normalizer = ActionNormalizer(mode="identity")
    task_id_value = 1 if args.task == "board" else 0
    reports: List[Dict[str, Any]] = []
    for idx, row in enumerate(windows):
        action_raw = torch.tensor(row["action"], dtype=torch.float32, device=device).unsqueeze(0)
        qpos_raw = torch.tensor(row["qpos"], dtype=torch.float32, device=device).view(1, -1)
        marker_raw = torch.tensor(row["marker_window"], dtype=torch.float32, device=device).unsqueeze(0)
        foresight_images = [img.to(device).unsqueeze(0) for img in row.get("foresight_images", [])]
        if action_raw.shape[1] < args.action_horizon:
            continue
        if action_raw.shape[1] < args.action_chunk:
            pad = action_raw[:, -1:].expand(-1, args.action_chunk - action_raw.shape[1], -1)
            action_raw = torch.cat([action_raw, pad], dim=1)
        bridge = build_bridge(foresight, fs_norm, fs_cfg, qpos_raw, marker_raw, foresight_images, args.task)
        task_id = torch.full((1,), task_id_value, dtype=torch.long, device=device)

        def score_fn(candidate_raw: torch.Tensor) -> torch.Tensor:
            tactile = bridge(candidate_raw)
            return score_from_prediction(scorer, score_mode, profile_energy, tactile, candidate_raw, task_id)

        guided, report = refiner.refine(normalizer.denormalize(action_raw[:, : args.action_horizon]), score_fn)
        delta = guided - action_raw[:, : args.action_horizon]
        report.update(
            {
                "sample": int(idx),
                "episode": row["path"],
                "start": int(row["start"]),
                "action_delta_abs_max": float(delta.abs().max().detach().cpu()),
                "action_delta_norm": float(delta.flatten(1).norm(dim=1).mean().detach().cpu()),
            }
        )
        reports.append(report)

    score_delta = np.asarray([r["score_delta"]["mean"] for r in reports], dtype=np.float64)
    improved = np.asarray([r["improved_rate"] for r in reports], dtype=np.float64)
    finite = np.asarray([r["finite_grad_rate"] for r in reports], dtype=np.float64)
    positive = np.asarray([r["positive_grad_rate"] for r in reports], dtype=np.float64)
    accept = np.asarray([r["accept_rate"] for r in reports], dtype=np.float64)
    delta_norm = np.asarray([r["action_delta_norm"] for r in reports], dtype=np.float64)
    within = np.asarray([r["max_delta_within_trust_region"] for r in reports], dtype=bool)
    result = {
        "purpose": "TacQuality guidance-gradient audit through Foresight.",
        "task": args.task,
        "arm": args.arm,
        "device": str(device),
        "dataset_dir": args.dataset_dir,
        "n_samples": int(len(reports)),
        "foresight": {
            "dir": args.foresight_dir,
            "ckpt": args.foresight_ckpt,
            **fs_info,
        },
        "scorer": {
            "runtime": scorer_runtime,
            "checkpoint": scorer_checkpoint,
            "score_mode": score_mode,
            "profile_energy": profile_energy,
            "ensemble_config": ensemble_config if scorer_runtime == "BoardForceBandEnsembleRuntime" else None,
        },
        "summary": {
            "finite_grad_rate_mean": float(np.mean(finite)),
            "positive_grad_rate_mean": float(np.mean(positive)),
            "accept_rate_mean": float(np.mean(accept)),
            "improved_rate_mean": float(np.mean(improved)),
            "score_delta": summarize(score_delta),
            "action_delta_norm": summarize(delta_norm),
            "trust_region_pass_rate": float(np.mean(within)),
            "pass": bool(
                len(reports) > 0
                and np.mean(finite) >= args.min_finite_grad_rate
                and np.mean(positive) >= args.min_positive_grad_rate
                and np.mean(within) >= args.min_trust_region_pass_rate
                and np.mean(score_delta > 0.0) >= args.min_score_improve_rate
            ),
            "min_finite_grad_rate": args.min_finite_grad_rate,
            "min_positive_grad_rate": args.min_positive_grad_rate,
            "min_trust_region_pass_rate": args.min_trust_region_pass_rate,
            "min_score_improve_rate": args.min_score_improve_rate,
        },
        "reports": reports,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "guidance_gradient_audit.json"
    md_path = out_dir / "guidance_gradient_audit.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "summary": result["summary"]}, indent=2, ensure_ascii=False))
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    s = result["summary"]
    lines = [
        "# TacQuality Guidance-Gradient Audit",
        "",
        "This audit checks whether the deployed-style guidance chain provides usable gradients.",
        "It is not a real robot rollout metric.",
        "",
        "## Setup",
        "",
        f"- task: `{result['task']}`",
        f"- arm: `{result['arm']}`",
        f"- samples: `{result['n_samples']}`",
        f"- scorer: `{result['scorer']['runtime']}`",
        f"- score mode: `{result['scorer']['score_mode']}`",
        f"- foresight: `{result['foresight']['dir']}`",
        "",
        "## Summary",
        "",
        f"- pass: `{s['pass']}`",
        f"- finite grad rate mean: `{s['finite_grad_rate_mean']:.4f}`",
        f"- positive grad rate mean: `{s['positive_grad_rate_mean']:.4f}`",
        f"- accept rate mean: `{s['accept_rate_mean']:.4f}`",
        f"- improved rate mean: `{s['improved_rate_mean']:.4f}`",
        f"- trust region pass rate: `{s['trust_region_pass_rate']:.4f}`",
        f"- score delta mean: `{s['score_delta']['mean']:.6f}`",
        f"- action delta norm mean: `{s['action_delta_norm']['mean']:.6f}`",
        "",
        "## Evidence Boundary",
        "",
        "- This verifies gradient availability and bounded action refinement through Foresight.",
        "- It does not prove that guided robot rollouts improve contact force or task success.",
        "- The next required check is correlation between `score(Foresight(action))` and real contact-phase force/marker quality.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["board", "insertion"], default="board")
    parser.add_argument("--arm", default="marker_joint_s12_guided")
    parser.add_argument("--dataset_dir", default=str(DEFAULT_BOARD_DATASET))
    parser.add_argument("--foresight_dir", default=str(DEFAULT_BOARD_FORESIGHT_DIR))
    parser.add_argument("--foresight_ckpt", default=str(DEFAULT_BOARD_FORESIGHT_CKPT))
    parser.add_argument("--rollout_arm_config", default=str(DEFAULT_ROLLOUT_CONFIG))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--scorer_runtime", default=None)
    parser.add_argument("--scorer_checkpoint", default=None)
    parser.add_argument("--score_mode", default=None)
    parser.add_argument("--ensemble_config", default=None, help="JSON config for BoardForceBandEnsembleRuntime.")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--tac_side", default="left")
    parser.add_argument("--proprio_key", default="proprio_joint")
    parser.add_argument("--action_key", default="actions/joint_abs")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--action_chunk", type=int, default=16)
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument("--max_episodes", type=int, default=8)
    parser.add_argument("--samples_per_episode", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_finite_grad_rate", type=float, default=0.999)
    parser.add_argument("--min_positive_grad_rate", type=float, default=0.999)
    parser.add_argument("--min_trust_region_pass_rate", type=float, default=0.999)
    parser.add_argument("--min_score_improve_rate", type=float, default=0.80)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
