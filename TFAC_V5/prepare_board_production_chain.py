"""Prepare board-wiping production DP/Foresight training assets.

This does not train models.  It creates the missing production-chain entry
points for the board task:

  1. a flat symlink dataset that existing DP scripts can read;
  2. a board Foresight config using the original nested board dataset;
  3. a board DP+TactileVAE config/command using the flat symlink dataset;
  4. an audit JSON/markdown explaining the next commands.

The goal is to turn the current gap ("no board production DP/Foresight
checkpoint") into reproducible training commands.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import h5py


ROOT = Path(__file__).resolve().parents[1]
BOARD_DIR = Path("/home/chenshuai/data/dataset/260522_v8l_caheiban")
FLAT_DIR = Path("/home/chenshuai/data/dataset/260522_v8l_caheiban_flat")
OUT_DIR = Path("/home/chenshuai/Project/output/board_production_chain_setup")
FORESIGHT_CONFIG = ROOT / "TFAC_V5/config_pretrain_foresight_board_260522.json"


def episode_files(board_dir: Path) -> List[Path]:
    direct = sorted(board_dir.glob("episode_*.hdf5"))
    if direct:
        return direct
    return sorted((board_dir / "success").glob("episode_*.hdf5"))


def inspect_episode(path: Path) -> Dict[str, object]:
    required = [
        "observations/images/global",
        "observations/images/wrist",
        "observations/proprio_joint",
        "observations/proprio_eef",
        "observations/tac/left/marker_offset",
        "observations/tac/right/marker_offset",
        "actions/joint_abs",
        "actions/eef_abs",
    ]
    with h5py.File(path, "r") as f:
        shapes = {key: list(f[key].shape) for key in required if key in f}
        missing = [key for key in required if key not in f]
    return {"path": str(path), "missing": missing, "shapes": shapes}


def prepare_flat_dataset(src_files: List[Path], flat_dir: Path, force: bool = False) -> Dict[str, object]:
    flat_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    existing = 0
    links = []
    for new_id, src in enumerate(src_files):
        dst = flat_dir / f"episode_{new_id}.hdf5"
        if dst.exists() or dst.is_symlink():
            if force:
                dst.unlink()
            else:
                existing += 1
                links.append({"dst": str(dst), "src": os.path.realpath(dst)})
                continue
        os.symlink(src, dst)
        created += 1
        links.append({"dst": str(dst), "src": str(src)})
    return {
        "flat_dir": str(flat_dir),
        "n_links": len(links),
        "created": created,
        "existing": existing,
        "links_preview": links[:10],
    }


def board_foresight_config(args) -> Dict[str, object]:
    return {
        "save_dir": "/home/chenshuai/Project/output/foresight_ckpt",
        "name": "latent_foresight_board_260522",
        "dataset_dir": str(Path(args.board_dir)),
        "hidden_dim": 512,
        "batch_size": args.foresight_batch_size,
        "num_epochs": args.foresight_epochs,
        "chunk_size": 10,
        "lr": 4e-5,
        "weight_decay": 1e-4,
        "seed": args.seed,
        "dropout": 0.1,
        "gpu": int(args.gpu.split(",")[0]) if str(args.gpu).split(",")[0].lstrip("-").isdigit() else args.gpu,
        "foresight_layers": 3,
        "foresight_nheads": 8,
        "foresight_dim_feedforward": 2048,
        "foresight_horizon": 10,
        "predict_horizon": 1,
        "history_len": 1,
        "tactile_mode": "marker",
        "tactile_vae_ckpt": "/home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt",
        "tactile_vae_stats": "/home/chenshuai/Project/output/tactile_vae_full/tactile_norm_stats.json",
        "tactile_vae_latent_dim": 16,
        "tactile_vae_window": 8,
        "delta_weighted": True,
        "delta_alpha": 2.0,
        "camera_names": ["global", "wrist", "gelsight"],
        "state_dim": 7,
        "proprio_key": "proprio_joint",
        "action_key": "actions/joint_abs",
        "tac_side": "left",
        "tac_img_key": "img",
        "use_state_trajectory": True,
        "task": "board_wiping",
        "source_dataset": str(Path(args.board_dir)),
    }


def dp_command(args, flat_dir: Path) -> str:
    return " ".join(
        [
            "/home/chenshuai/miniconda3/envs/TactileACT/bin/python",
            "diffusion/train_dp_tac_concat.py",
            f"--dataset_dir {flat_dir}",
            "--save_dir /home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522",
            "--camera_names global,wrist",
            "--proprio_key proprio_joint",
            "--action_key actions/joint_abs",
            "--tac_side left",
            "--tac_history 8",
            "--vae_checkpoint /home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt",
            "--vae_latent_dim 16",
            f"--pred_horizon {args.dp_pred_horizon}",
            "--obs_horizon 2",
            "--n_action_steps 8",
            "--resize_shape 240,320",
            "--crop_shape 216,288",
            f"--epochs {args.dp_epochs}",
            f"--batch_size {args.dp_batch_size}",
            "--lr 1e-4",
            "--weight_decay 1e-6",
            "--warmup_steps 500",
            "--num_train_timesteps 100",
            "--num_inference_steps 100",
            "--diffusion_step_embed_dim 128",
            "--down_dims 512,1024,2048",
            f"--seed {args.seed}",
            "--save_freq 50",
            f"--gpu {args.gpu}",
        ]
    )


def foresight_command(config_path: Path) -> str:
    return " ".join(
        [
            "/home/chenshuai/miniconda3/envs/TactileACT/bin/python",
            "TFAC_V5/pretrain_latent_foresight.py",
            f"--config {config_path}",
        ]
    )


def write_markdown(result: Dict[str, object], path: Path) -> None:
    lines = [
        "# Board Production Chain Setup",
        "",
        "## Dataset",
        "",
        f"- source: `{result['board_dir']}`",
        f"- flat symlink dir: `{result['flat_dataset']['flat_dir']}`",
        f"- episodes: `{result['n_episodes']}`",
        "",
        "## Commands",
        "",
        "### Train Board Foresight",
        "",
        "```bash",
        result["commands"]["train_foresight"],
        "```",
        "",
        "### Train Board DP + TactileVAE",
        "",
        "```bash",
        result["commands"]["train_dp"],
        "```",
        "",
        "## After Training",
        "",
        "Run production full-chain guidance/refinement with:",
        "",
        "```text",
        "action -> board production Foresight -> PTG board energy -> dscore/daction",
        "```",
        "",
        "Do not count the existing board surrogate checkpoint as production Foresight/DP.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args):
    board_dir = Path(args.board_dir)
    flat_dir = Path(args.flat_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = episode_files(board_dir)
    if not files:
        raise RuntimeError(f"No episode_*.hdf5 found under {board_dir} or {board_dir / 'success'}")
    first = inspect_episode(files[0])
    if first["missing"]:
        raise RuntimeError(f"First board episode is missing required keys: {first['missing']}")
    flat = prepare_flat_dataset(files, flat_dir, force=args.force_links)

    cfg = board_foresight_config(args)
    FORESIGHT_CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=4), encoding="utf-8")
    commands = {
        "train_foresight": foresight_command(FORESIGHT_CONFIG),
        "train_dp": dp_command(args, flat_dir),
    }
    result = {
        "board_dir": str(board_dir),
        "n_episodes": len(files),
        "first_episode": first,
        "flat_dataset": flat,
        "foresight_config": str(FORESIGHT_CONFIG),
        "commands": commands,
        "outputs_expected": {
            "foresight_ckpt": "/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522/foresight_best.ckpt",
            "dp_ckpt_dir": "/home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522",
        },
        "scope": "Setup only. Training and production full-chain verification still need to be run.",
    }
    json_path = out_dir / "board_production_chain_setup.json"
    md_path = out_dir / "board_production_chain_setup.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board_dir", default=str(BOARD_DIR))
    parser.add_argument("--flat_dir", default=str(FLAT_DIR))
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--force_links", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--foresight_epochs", type=int, default=500)
    parser.add_argument("--foresight_batch_size", type=int, default=64)
    parser.add_argument("--dp_epochs", type=int, default=600)
    parser.add_argument("--dp_batch_size", type=int, default=32)
    parser.add_argument("--dp_pred_horizon", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
