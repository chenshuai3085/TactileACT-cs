"""
Evaluate Lightweight Foresight quality for CQF readiness.

Tests:
1. Overall reconstruction quality (MSE, cosine sim)
2. Per-phase quality: approach vs insertion vs bounce
3. Action sensitivity: same state, different actions → different predictions?
4. Prediction stability: noisy action → meaningfully different prediction?

Usage:
  python TFAC_V5/eval_foresight_quality.py \
    --foresight_ckpt /home/chenshuai/Project/output/lightweight_foresight/foresight_best.pt
"""

import argparse
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from lightweight_foresight import LightweightForesight


DATA_ROOT = "/home/chenshuai/data/dataset"
DATASETS = ["260309", "260310", "260401", "260402", "260403", "260407"]
FORESIGHT_HORIZON = 10
CHUNK_SIZE = 20


def load_foresight(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = LightweightForesight(
        hidden=args.get("hidden", 512),
        n_layers=args.get("n_layers", 4),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded foresight from {ckpt_path} (epoch {ckpt.get('epoch', '?')})")
    print(f"  Val metrics: {ckpt.get('val_metrics', {})}")
    return model


def load_episodes(ds_name, split="success"):
    """Load episodes from a dataset split."""
    import glob
    ds_dir = os.path.join(DATA_ROOT, ds_name, split)
    if not os.path.isdir(ds_dir):
        ds_dir = os.path.join(DATA_ROOT, ds_name)
    files = sorted(glob.glob(os.path.join(ds_dir, "*.hdf5")))
    episodes = []
    for f in files[:20]:
        try:
            with h5py.File(f, "r") as h:
                episodes.append({
                    "qpos": h["observations/proprio_joint"][:].astype(np.float32),
                    "eef": h["observations/proprio_eef"][:].astype(np.float32),
                    "action": h["actions/joint_abs"][:].astype(np.float32),
                    "marker": h["observations/tac/left/marker_offset"][:].astype(np.float32),
                    "file": os.path.basename(f),
                })
        except Exception:
            continue
    return episodes


@torch.no_grad()
def eval_reconstruction(model, episodes, device, label=""):
    """Evaluate reconstruction quality across episodes."""
    h = FORESIGHT_HORIZON
    cs = CHUNK_SIZE
    all_mse, all_cos = [], []

    for ep in episodes:
        T = ep["qpos"].shape[0]
        for t in range(0, T - max(cs, h), 5):
            if t + cs > T or t + h >= T:
                continue
            qpos = torch.from_numpy(ep["qpos"][t:t+1]).to(device)
            eef = torch.from_numpy(ep["eef"][t:t+1]).to(device)
            action = torch.from_numpy(ep["action"][t:t+cs][None]).to(device)
            mcur = torch.from_numpy(ep["marker"][t].flatten()[None]).to(device)
            mfut_gt = ep["marker"][t+h].flatten()

            pred = model(qpos, eef, action, mcur).cpu().numpy()[0]

            mse = ((pred - mfut_gt) ** 2).mean()
            cos = np.dot(pred, mfut_gt) / (np.linalg.norm(pred) * np.linalg.norm(mfut_gt) + 1e-8)
            all_mse.append(mse)
            all_cos.append(cos)

    if all_mse:
        print(f"  {label}: N={len(all_mse)}, MSE={np.mean(all_mse):.4f}±{np.std(all_mse):.4f}, "
              f"Cosine={np.mean(all_cos):.4f}±{np.std(all_cos):.4f}")
    return all_mse, all_cos


@torch.no_grad()
def eval_action_sensitivity(model, episodes, device, n_variants=8):
    """Test: same state, different actions → different predictions?
    This is critical for CQF: if predictions don't change with action,
    CQF can't distinguish good vs bad actions."""
    h = FORESIGHT_HORIZON
    cs = CHUNK_SIZE
    pred_diffs = []
    pred_gt_diffs = []

    for ep in episodes[:5]:
        T = ep["qpos"].shape[0]
        # Pick frames from late episode (near insertion)
        for t in range(max(0, T - 80), T - max(cs, h), 10):
            if t + cs > T or t + h >= T:
                continue
            qpos = torch.from_numpy(ep["qpos"][t:t+1]).to(device)
            eef = torch.from_numpy(ep["eef"][t:t+1]).to(device)
            action_gt = torch.from_numpy(ep["action"][t:t+cs][None]).to(device)
            mcur = torch.from_numpy(ep["marker"][t].flatten()[None]).to(device)

            pred_gt = model(qpos, eef, action_gt, mcur).cpu().numpy()[0]

            preds = [pred_gt]
            for _ in range(n_variants):
                noise_scale = np.random.choice([0.3, 0.5, 1.0, 2.0])
                noise = torch.randn_like(action_gt) * action_gt.std() * noise_scale
                action_noisy = action_gt + noise
                pred_noisy = model(qpos, eef, action_noisy, mcur).cpu().numpy()[0]
                preds.append(pred_noisy)

                diff = np.sqrt(((pred_noisy - pred_gt) ** 2).mean())
                pred_diffs.append(diff)
                pred_gt_diffs.append(noise_scale)

    if pred_diffs:
        pred_diffs = np.array(pred_diffs)
        pred_gt_diffs = np.array(pred_gt_diffs)
        print(f"\n  Action Sensitivity (N={len(pred_diffs)}):")
        for scale in [0.3, 0.5, 1.0, 2.0]:
            mask = np.abs(pred_gt_diffs - scale) < 0.01
            if mask.any():
                d = pred_diffs[mask]
                print(f"    noise_scale={scale:.1f}: pred_RMSE={d.mean():.4f}±{d.std():.4f}")

        # Correlation between noise scale and prediction difference
        corr = np.corrcoef(pred_gt_diffs, pred_diffs)[0, 1]
        print(f"    Correlation(noise_scale, pred_diff) = {corr:.3f}")
        print(f"    → {'GOOD' if corr > 0.3 else 'WEAK'}: predictions {'do' if corr > 0.3 else 'do NOT'} vary meaningfully with action")
    return pred_diffs


@torch.no_grad()
def eval_phase_quality(model, device):
    """Evaluate quality separately for success (insertion) vs bounce episodes."""
    print("\n--- Per-Phase Quality ---")
    for ds_name in DATASETS[:3]:
        print(f"\nDataset: {ds_name}")
        for split in ["success", "bounce"]:
            episodes = load_episodes(ds_name, split)
            if episodes:
                eval_reconstruction(model, episodes, device, label=f"{split} ({len(episodes)} eps)")


@torch.no_grad()
def eval_temporal_progression(model, episodes, device):
    """Show how prediction quality changes through an episode.
    Insertion frames should have more tactile signal → harder to predict."""
    h = FORESIGHT_HORIZON
    cs = CHUNK_SIZE
    early_mse, mid_mse, late_mse = [], [], []

    for ep in episodes:
        T = ep["qpos"].shape[0]
        for t in range(0, T - max(cs, h), 5):
            if t + cs > T or t + h >= T:
                continue
            qpos = torch.from_numpy(ep["qpos"][t:t+1]).to(device)
            eef = torch.from_numpy(ep["eef"][t:t+1]).to(device)
            action = torch.from_numpy(ep["action"][t:t+cs][None]).to(device)
            mcur = torch.from_numpy(ep["marker"][t].flatten()[None]).to(device)
            mfut_gt = ep["marker"][t+h].flatten()

            pred = model(qpos, eef, action, mcur).cpu().numpy()[0]
            mse = ((pred - mfut_gt) ** 2).mean()

            frac = t / T
            if frac < 0.33:
                early_mse.append(mse)
            elif frac < 0.67:
                mid_mse.append(mse)
            else:
                late_mse.append(mse)

    print(f"\n  Temporal Progression:")
    print(f"    Early (0-33%):  MSE={np.mean(early_mse):.4f}±{np.std(early_mse):.4f} (N={len(early_mse)})")
    print(f"    Mid   (33-67%): MSE={np.mean(mid_mse):.4f}±{np.std(mid_mse):.4f} (N={len(mid_mse)})")
    print(f"    Late  (67-100%):MSE={np.mean(late_mse):.4f}±{np.std(late_mse):.4f} (N={len(late_mse)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--foresight_ckpt", type=str,
                        default="/home/chenshuai/Project/output/lightweight_foresight/foresight_best.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Lightweight Foresight Quality Evaluation")
    print("=" * 60)

    model = load_foresight(args.foresight_ckpt, device)

    # 1. Overall reconstruction on success episodes
    print("\n--- Overall Reconstruction (Success Episodes) ---")
    all_success = []
    for ds_name in DATASETS[:3]:
        eps = load_episodes(ds_name, "success")
        all_success.extend(eps)
    eval_reconstruction(model, all_success, device, label="Success")

    # 2. Overall reconstruction on bounce episodes
    print("\n--- Overall Reconstruction (Bounce Episodes) ---")
    all_bounce = []
    for ds_name in DATASETS[:3]:
        eps = load_episodes(ds_name, "bounce")
        all_bounce.extend(eps)
    eval_reconstruction(model, all_bounce, device, label="Bounce")

    # 3. Action sensitivity (critical for CQF)
    print("\n--- Action Sensitivity Test ---")
    eval_action_sensitivity(model, all_success[:5], device)

    # 4. Temporal progression
    print("\n--- Temporal Progression (Success) ---")
    eval_temporal_progression(model, all_success[:10], device)

    # 5. Per-phase quality
    eval_phase_quality(model, device)

    print("\n" + "=" * 60)
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
