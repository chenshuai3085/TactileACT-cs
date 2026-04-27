"""
Compare Foresight V1 vs V2: reconstruction quality + action sensitivity.

Usage:
  python TFAC_V5/compare_foresight_v1_v2.py
"""

import os
import sys
import glob

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from lightweight_foresight import LightweightForesight
from lightweight_foresight_v2 import LightweightForesightV2


DATA_ROOT = "/home/chenshuai/data/dataset"
DATASETS = ["260309", "260310"]
FORESIGHT_HORIZON = 10
CHUNK_SIZE = 20


def load_v1(device):
    ckpt = torch.load("/home/chenshuai/Project/output/lightweight_foresight/foresight_best.pt",
                       map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = LightweightForesight(
        hidden=args.get("hidden", 512), n_layers=args.get("n_layers", 4)
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_v2(device):
    ckpt = torch.load("/home/chenshuai/Project/output/lightweight_foresight_v2/foresight_v2_best.pt",
                       map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = LightweightForesightV2(
        hidden=args.get("hidden", 512), n_layers=args.get("n_layers", 4)
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_episodes(ds_name, split="success", max_ep=15):
    ds_dir = os.path.join(DATA_ROOT, ds_name, split)
    if not os.path.isdir(ds_dir):
        ds_dir = os.path.join(DATA_ROOT, ds_name)
    files = sorted(glob.glob(os.path.join(ds_dir, "*.hdf5")))[:max_ep]
    episodes = []
    for f in files:
        try:
            with h5py.File(f, "r") as h:
                episodes.append({
                    "qpos": h["observations/proprio_joint"][:].astype(np.float32),
                    "eef": h["observations/proprio_eef"][:].astype(np.float32),
                    "action": h["actions/joint_abs"][:].astype(np.float32),
                    "marker": h["observations/tac/left/marker_offset"][:].astype(np.float32),
                })
        except Exception:
            continue
    return episodes


@torch.no_grad()
def eval_model(model, episodes, device, name=""):
    h, cs = FORESIGHT_HORIZON, CHUNK_SIZE
    all_mse, all_cos = [], []

    for ep in episodes:
        T = ep["qpos"].shape[0]
        for t in range(0, T - max(cs, h), 10):
            if t + cs > T or t + h >= T:
                continue
            qpos = torch.from_numpy(ep["qpos"][t:t+1]).to(device)
            eef = torch.from_numpy(ep["eef"][t:t+1]).to(device)
            action = torch.from_numpy(ep["action"][t:t+cs][None]).to(device)
            mcur = torch.from_numpy(ep["marker"][t].flatten()[None]).to(device)
            mfut_gt = ep["marker"][t+h].flatten()

            pred = model(qpos, eef, action, mcur).cpu().numpy()[0]
            all_mse.append(((pred - mfut_gt) ** 2).mean())
            all_cos.append(np.dot(pred, mfut_gt) / (np.linalg.norm(pred) * np.linalg.norm(mfut_gt) + 1e-8))

    print(f"  {name}: MSE={np.mean(all_mse):.4f}±{np.std(all_mse):.4f}, "
          f"Cosine={np.mean(all_cos):.4f}±{np.std(all_cos):.4f} (N={len(all_mse)})")
    return all_mse, all_cos


@torch.no_grad()
def eval_action_sensitivity(model, episodes, device, name=""):
    h, cs = FORESIGHT_HORIZON, CHUNK_SIZE
    results = {s: [] for s in [0.3, 0.5, 1.0, 2.0]}

    for ep in episodes[:5]:
        T = ep["qpos"].shape[0]
        for t in range(max(0, T - 80), T - max(cs, h), 10):
            if t + cs > T or t + h >= T:
                continue
            qpos = torch.from_numpy(ep["qpos"][t:t+1]).to(device)
            eef = torch.from_numpy(ep["eef"][t:t+1]).to(device)
            action = torch.from_numpy(ep["action"][t:t+cs][None]).to(device)
            mcur = torch.from_numpy(ep["marker"][t].flatten()[None]).to(device)

            pred_gt = model(qpos, eef, action, mcur)
            for scale in [0.3, 0.5, 1.0, 2.0]:
                noise = torch.randn_like(action) * action.std() * scale
                pred_noisy = model(qpos, eef, action + noise, mcur)
                diff = (pred_gt - pred_noisy).norm().item()
                results[scale].append(diff)

    print(f"\n  {name} Action Sensitivity:")
    all_diffs, all_scales = [], []
    for scale in [0.3, 0.5, 1.0, 2.0]:
        d = np.array(results[scale])
        print(f"    scale={scale:.1f}: RMSE={d.mean():.4f}±{d.std():.4f}")
        all_diffs.extend(d.tolist())
        all_scales.extend([scale] * len(d))

    corr = np.corrcoef(all_scales, all_diffs)[0, 1]
    print(f"    Correlation: {corr:.3f} ({'GOOD' if corr > 0.3 else 'WEAK'})")
    return corr


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Foresight V1 vs V2 Comparison")
    print("=" * 60)

    v1, v1_ckpt = load_v1(device)
    v2, v2_ckpt = load_v2(device)
    print(f"V1: epoch={v1_ckpt['epoch']}, params={sum(p.numel() for p in v1.parameters()):,}")
    print(f"V2: epoch={v2_ckpt['epoch']}, params={sum(p.numel() for p in v2.parameters()):,}")

    print("\n--- Reconstruction Quality ---")
    episodes = []
    for ds in DATASETS:
        episodes.extend(load_episodes(ds, "success"))
    print(f"  ({len(episodes)} success episodes)")
    eval_model(v1, episodes, device, "V1")
    eval_model(v2, episodes, device, "V2")

    print("\n--- Action Sensitivity ---")
    corr_v1 = eval_action_sensitivity(v1, episodes, device, "V1")
    corr_v2 = eval_action_sensitivity(v2, episodes, device, "V2")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  V1 action sensitivity correlation: {corr_v1:.3f}")
    print(f"  V2 action sensitivity correlation: {corr_v2:.3f}")
    improvement = (corr_v2 - corr_v1) / max(abs(corr_v1), 0.01) * 100
    print(f"  Improvement: {improvement:+.1f}%")


if __name__ == "__main__":
    main()
