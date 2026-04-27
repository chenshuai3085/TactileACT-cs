"""
Precompute CQF latent representations using LatentForesightPretrainModel.

For each CQF sample (from annotations.pkl), computes:
  - z_cur (144-dim): TactileVAE encoding of current marker window
  - z_future_gt (144-dim): TactileVAE encoding of future marker window
  - z_pred (144-dim): Foresight prediction of future latent
  - Also stores: qpos, eef, action_chunk(20), label

Output: {split}_samples.pt per split (train/val)

Usage:
  python TFAC_V5/precompute_cqf_latents.py \
    --foresight_ckpt /home/chenshuai/data/xiaomi_act/latent_foresight_pretrain_dw/foresight_best.ckpt \
    --output_dir /home/chenshuai/Project/output/cqf_latent_data
"""

import argparse
import json
import os
import pickle
import glob
import sys

import h5py
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from pretrain_latent_foresight import LatentForesightPretrainModel


DATA_ROOT = "/home/chenshuai/data/dataset"
SPLIT_DATASETS = ["260309", "260310", "260401", "260402", "260403", "260407"]
MERGED_DATASETS = [
    "0414", "260401_0402", "260408_0409", "260417",
    "0209-0210", "0331", "260309_0310",
]

FORESIGHT_HORIZON = 10
CQF_CHUNK_SIZE = 20


def load_foresight_model(ckpt_path, args_path, device):
    """Load LatentForesightPretrainModel from checkpoint."""
    with open(args_path) as f:
        config = json.load(f)

    camera_names = config['camera_names']
    cam_backbone_mapping = {cam: 0 for cam in camera_names}

    model = LatentForesightPretrainModel(
        camera_names=camera_names,
        cam_backbone_mapping=cam_backbone_mapping,
        hidden_dim=config['hidden_dim'],
        state_dim=config['state_dim'],
        foresight_layers=config.get('foresight_layers', 3),
        foresight_nheads=config.get('foresight_nheads', 8),
        foresight_dim_feedforward=config.get('foresight_dim_feedforward', 2048),
        dropout=config.get('dropout', 0.1),
        tactile_mode=config.get('tactile_mode', 'marker'),
        max_history=config.get('max_history', 8),
        predict_horizon=config.get('predict_horizon', 1),
        tactile_vae_ckpt=config.get('tactile_vae_ckpt'),
        tactile_vae_latent_dim=config.get('tactile_vae_latent_dim', 16),
        use_delta_pred=config.get('use_delta_pred', False),
        residual_prediction=config.get('residual_prediction', False),
    )

    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Loaded foresight model from {ckpt_path}")
    return model, config


def get_marker_window(marker_all, t, window_size, mo_mean, mo_std):
    """Get normalized marker window [t-W+1, ..., t] with edge padding."""
    T = marker_all.shape[0]
    frames = []
    for i in range(window_size):
        idx = max(0, t - (window_size - 1 - i))
        idx = min(idx, T - 1)
        frame = marker_all[idx].astype(np.float32)
        frame = (frame - mo_mean) / mo_std
        frames.append(torch.tensor(frame, dtype=torch.float32))
    return torch.stack(frames)  # (W, 9, 9, 2)


def preprocess_image(img_uint8):
    """Convert HDF5 image (H, W, 3) uint8 → (3, H, W) ImageNet-normalized tensor."""
    img = torch.tensor(img_uint8.astype(np.float32) / 255.0)
    img = img.permute(2, 0, 1)  # (3, H, W)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    return normalize(img)


def find_hdf5(ds_dir, ep_name):
    """Find episode HDF5 file."""
    fname = f"{ep_name}.hdf5"
    path = os.path.join(ds_dir, fname)
    if os.path.exists(path):
        return path
    for subdir in ["success", "bounce"]:
        path = os.path.join(ds_dir, subdir, fname)
        if os.path.exists(path):
            return path
    return None


@torch.no_grad()
def precompute_dataset(model, ds_name, config, device, batch_size=8,
                       n_perturb_per_pos=2, perturb_scales=(0.5, 1.0, 2.0),
                       vae_only=False):
    """Precompute latent representations for one dataset."""
    ds_dir = os.path.join(DATA_ROOT, ds_name)
    ann_path = os.path.join(ds_dir, "annotations.pkl")
    if not os.path.exists(ann_path):
        print(f"  Skip {ds_name}: no annotations.pkl")
        return []

    with open(ann_path, "rb") as f:
        ann = pickle.load(f)

    norm_stats = config['norm_stats']
    mo_mean = np.array(norm_stats['marker_offset_mean'], dtype=np.float32).reshape(1, 1, 2)
    mo_std = np.array(norm_stats['marker_offset_std'], dtype=np.float32).reshape(1, 1, 2)
    qpos_mean = np.array(norm_stats['qpos_mean'], dtype=np.float32)
    qpos_std = np.array(norm_stats['qpos_std'], dtype=np.float32)
    action_mean = np.array(norm_stats['action_mean'], dtype=np.float32)
    action_std = np.array(norm_stats['action_std'], dtype=np.float32)

    foresight_chunk = config.get('chunk_size', 10)
    vae_window = config.get('tactile_vae_window', 8)
    h = FORESIGHT_HORIZON
    cs = CQF_CHUNK_SIZE

    all_samples = []
    rng = np.random.RandomState(42)

    ep_items = [(k, v) for k, v in ann.items() if k != "_meta"]
    print(f"  {len(ep_items)} episodes to process")

    for ep_idx, (ep_name, ep_ann) in enumerate(ep_items):
        hdf5_path = find_hdf5(ds_dir, ep_name)
        if hdf5_path is None:
            continue

        labels = ep_ann["labels"]
        lifts = ep_ann["lifts"]
        T = len(labels)

        # Identify relevant frames
        frames = []
        for t in np.where(labels == 1)[0]:
            if t + cs <= T and t + h < T:
                frames.append((t, 1.0))
        lift_starts = [s for s, e, r in lifts]
        for t in np.where(labels == 2)[0]:
            if t + cs <= T and t + h < T:
                if any(t + h >= ls for ls in lift_starts):
                    frames.append((t, 0.0))

        if not frames:
            continue

        try:
            with h5py.File(hdf5_path, "r") as f:
                qpos_all = f["observations/proprio_joint"][:]
                eef_all = f["observations/proprio_eef"][:]
                action_all = f["actions/joint_abs"][:]
                marker_all = f["observations/tac/left/marker_offset"][:]

                has_global = "observations/images/global" in f
                has_wrist = "observations/images/wrist" in f

                for batch_start in range(0, len(frames), batch_size):
                    batch_frames = frames[batch_start:batch_start + batch_size]
                    B = len(batch_frames)

                    img_global_list, img_wrist_list, marker_win_list = [], [], []
                    action_list, qpos_list = [], []
                    marker_future_list = []
                    raw_list = []

                    for t, label in batch_frames:
                        # Marker window for current frame
                        marker_win_list.append(
                            get_marker_window(marker_all, t, vae_window, mo_mean, mo_std))

                        # Marker window for future frame
                        marker_future_list.append(
                            get_marker_window(marker_all, t + h, vae_window, mo_mean, mo_std))

                        if not vae_only:
                            # Vision images (only needed for foresight forward)
                            if has_global:
                                img_global_list.append(
                                    preprocess_image(f["observations/images/global"][t]))
                            else:
                                img_global_list.append(torch.zeros(3, 480, 640))
                            if has_wrist:
                                img_wrist_list.append(
                                    preprocess_image(f["observations/images/wrist"][t]))
                            else:
                                img_wrist_list.append(torch.zeros(3, 480, 640))

                            # Normalized action for foresight (chunk_size=10)
                            act = action_all[t:t + foresight_chunk].astype(np.float32)
                            if len(act) < foresight_chunk:
                                act = np.pad(act, ((0, foresight_chunk - len(act)), (0, 0)), mode='edge')
                            act_norm = (act - action_mean) / action_std
                            action_list.append(torch.tensor(act_norm, dtype=torch.float32))

                            # Normalized qpos
                            qp = qpos_all[t].astype(np.float32)
                            qp_norm = (qp - qpos_mean) / qpos_std
                            qpos_list.append(torch.tensor(qp_norm, dtype=torch.float32))

                        # Raw data for CQF (unnormalized)
                        act_20 = action_all[t:t + cs].astype(np.float32)
                        if len(act_20) < cs:
                            act_20 = np.pad(act_20, ((0, cs - len(act_20)), (0, 0)), mode='edge')
                        raw_list.append({
                            "qpos": qpos_all[t].astype(np.float32),
                            "eef": eef_all[t].astype(np.float32),
                            "action_chunk": act_20,
                            "label": label,
                        })

                    # Stack marker windows and move to device
                    cur_markers = torch.stack(marker_win_list).to(device)
                    future_markers = torch.stack(marker_future_list).to(device)

                    # Encode current and future markers through TactileVAE
                    z_cur_raw, _ = model.tactile_vae.encode_single_frame(cur_markers)
                    z_cur_flat = z_cur_raw.reshape(B, -1)
                    z_future_raw, _ = model.tactile_vae.encode_single_frame(future_markers)
                    z_future_flat = z_future_raw.reshape(B, -1)

                    z_cur_np = z_cur_flat.cpu().numpy()
                    z_fut_np = z_future_flat.cpu().numpy()

                    if not vae_only:
                        # Full foresight forward → z_pred
                        images = [
                            torch.stack(img_global_list).to(device),
                            torch.stack(img_wrist_list).to(device),
                            cur_markers,
                        ]
                        actions = torch.stack(action_list).to(device)
                        qpos_t = torch.stack(qpos_list).to(device)

                        t_hat, _, _, _, _, _ = model(
                            images, actions, qpos=qpos_t)
                        z_pred_np = t_hat.cpu().numpy()
                    else:
                        z_pred_np = z_fut_np.copy()

                    # Store base samples
                    for i, rd in enumerate(raw_list):
                        rd["z_cur"] = z_cur_np[i]
                        rd["z_future_gt"] = z_fut_np[i]
                        rd["z_pred"] = z_pred_np[i]
                        all_samples.append(rd)

                    # Perturbation negatives for positive samples
                    pos_indices = [i for i, (t, l) in enumerate(batch_frames) if l > 0.5]
                    if pos_indices and n_perturb_per_pos > 0:
                        global_action_std = np.mean([
                            a.std().item() for a in action_list])

                        for pi in pos_indices:
                            t_pos, _ = batch_frames[pi]
                            for j in range(n_perturb_per_pos):
                                scale = perturb_scales[j % len(perturb_scales)]
                                noise = rng.randn(foresight_chunk, 7).astype(np.float32)
                                act_noisy = action_list[pi].cpu().numpy() + \
                                    noise * global_action_std * scale

                                act_noisy_t = torch.tensor(
                                    act_noisy, dtype=torch.float32).unsqueeze(0).to(device)
                                perturb_images = [
                                    images[0][pi:pi+1],
                                    images[1][pi:pi+1],
                                    images[2][pi:pi+1],
                                ]
                                perturb_qpos = qpos_t[pi:pi+1]

                                z_pred_p, _, _, z_cur_p, _, _ = model(
                                    perturb_images, act_noisy_t, qpos=perturb_qpos)

                                # Unnormalize noisy action for CQF
                                act_noisy_raw = act_noisy * action_std + action_mean
                                act_20_noisy = np.zeros((cs, 7), dtype=np.float32)
                                act_20_noisy[:foresight_chunk] = act_noisy_raw
                                act_20_noisy[foresight_chunk:] = raw_list[pi]["action_chunk"][foresight_chunk:]

                                soft_label = max(0.0, 1.0 - scale * 0.3)
                                all_samples.append({
                                    "qpos": raw_list[pi]["qpos"],
                                    "eef": raw_list[pi]["eef"],
                                    "action_chunk": act_20_noisy,
                                    "z_cur": z_cur_p.cpu().numpy().squeeze(),
                                    "z_future_gt": z_fut_np[pi],
                                    "z_pred": z_pred_p.cpu().numpy().squeeze(),
                                    "label": soft_label,
                                    "is_perturb": True,
                                })

        except Exception as e:
            import traceback
            print(f"  Error processing {ep_name}: {e}")
            traceback.print_exc()
            continue

        if (ep_idx + 1) % 5 == 0 or ep_idx == len(ep_items) - 1:
            print(f"  [{ep_idx+1}/{len(ep_items)}] {len(all_samples)} samples so far")

    return all_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--foresight_ckpt", type=str,
                        default="/home/chenshuai/data/xiaomi_act/latent_foresight_pretrain_dw/foresight_best.ckpt")
    parser.add_argument("--foresight_dir", type=str, default=None,
                        help="Directory containing foresight checkpoint (auto-detects args.json)")
    parser.add_argument("--output_dir", type=str,
                        default="/home/chenshuai/Project/output/cqf_latent_data")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_perturb", type=int, default=2,
                        help="Perturbation negatives per positive sample")
    parser.add_argument("--no_perturb", action="store_true",
                        help="Skip perturbation negatives (Phase A only)")
    parser.add_argument("--vae_only", action="store_true",
                        help="Only compute VAE latents (z_cur, z_future_gt), skip foresight z_pred")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Only process these datasets (e.g. --datasets 260407)")
    args = parser.parse_args()

    if args.no_perturb:
        args.n_perturb = 0
    if args.vae_only:
        args.n_perturb = 0

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-detect foresight dir
    foresight_dir = args.foresight_dir
    if foresight_dir is None:
        foresight_dir = os.path.dirname(args.foresight_ckpt)
    args_path = os.path.join(foresight_dir, "args.json")

    model, config = load_foresight_model(args.foresight_ckpt, args_path, device)

    # Process datasets
    if args.datasets:
        all_datasets = args.datasets
    else:
        all_datasets = SPLIT_DATASETS + MERGED_DATASETS
    all_samples = []

    print(f"\nProcessing {len(all_datasets)} datasets...")
    for ds_name in all_datasets:
        print(f"\n--- {ds_name} ---")
        samples = precompute_dataset(
            model, ds_name, config, device,
            batch_size=args.batch_size,
            n_perturb_per_pos=args.n_perturb,
            vae_only=args.vae_only,
        )
        n_pos = sum(1 for s in samples if s["label"] > 0.5 and not s.get("is_perturb"))
        n_neg = sum(1 for s in samples if s["label"] <= 0.5 and not s.get("is_perturb"))
        n_perturb = sum(1 for s in samples if s.get("is_perturb"))
        print(f"  {len(samples)} samples (pos={n_pos}, neg={n_neg}, perturb={n_perturb})")
        all_samples.extend(samples)

    print(f"\nTotal: {len(all_samples)} samples")

    # Train/val split
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(all_samples))
    n_val = int(len(all_samples) * args.val_ratio)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_samples = [all_samples[i] for i in train_indices]
    val_samples = [all_samples[i] for i in val_indices]

    n_train_pos = sum(1 for s in train_samples if s["label"] > 0.5)
    n_val_pos = sum(1 for s in val_samples if s["label"] > 0.5)
    print(f"\nTrain: {len(train_samples)} (pos={n_train_pos}, neg={len(train_samples)-n_train_pos})")
    print(f"Val:   {len(val_samples)} (pos={n_val_pos}, neg={len(val_samples)-n_val_pos})")

    # Save
    train_path = os.path.join(args.output_dir, "train_samples.pt")
    val_path = os.path.join(args.output_dir, "val_samples.pt")
    torch.save(train_samples, train_path)
    torch.save(val_samples, val_path)
    print(f"\nSaved: {train_path} ({os.path.getsize(train_path)/1e6:.1f} MB)")
    print(f"       {val_path} ({os.path.getsize(val_path)/1e6:.1f} MB)")

    # Save config
    meta = {
        "foresight_ckpt": args.foresight_ckpt,
        "foresight_config": config,
        "tac_dim": 144,
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "datasets": all_datasets,
        "foresight_horizon": FORESIGHT_HORIZON,
        "cqf_chunk_size": CQF_CHUNK_SIZE,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print("\nDone!")


if __name__ == "__main__":
    main()
