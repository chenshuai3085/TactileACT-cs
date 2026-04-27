"""
CQFDataset: 从annotations.pkl + HDF5构造CQF训练样本。

Phase A (GT tactile): 正样本(insertion帧) + 负样本(pre_bounce帧)
  每个样本 = (qpos, eef, action_chunk, marker_cur, marker_future, label)

后续Phase B/C在训练脚本中替换marker_future为Foresight预测。
"""

import os
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


DATA_ROOT = "/home/chenshuai/data/dataset"

SPLIT_DATASETS = ["260309", "260310", "260401", "260402", "260403", "260407"]
MERGED_DATASETS = [
    "0414", "260401_0402", "260408_0409", "260417",
    "0209-0210", "0331", "260309_0310",
]

FORESIGHT_HORIZON = 10
CHUNK_SIZE = 20


class CQFDataset(Dataset):
    """
    CQF训练数据集 (Phase A: GT tactile)。

    正样本: label=1 (insertion) 帧
    负样本: label=2 (pre_bounce) 帧，且 t+h >= lift_start

    每个样本返回:
        qpos:          (7,)       当前关节角度
        eef:           (6,)       当前末端位姿
        action_chunk:  (20, 7)    要执行的action chunk
        marker_cur:    (9, 9, 2)  当前marker_offset
        marker_future: (9, 9, 2)  未来t+h的marker_offset (Phase A: GT)
        label:         float      1.0=好, 0.0=坏
    """

    def __init__(self, dataset_names=None, data_root=DATA_ROOT,
                 foresight_horizon=FORESIGHT_HORIZON, chunk_size=CHUNK_SIZE,
                 split="train", val_ratio=0.2, seed=42):
        super().__init__()
        self.foresight_horizon = foresight_horizon
        self.chunk_size = chunk_size

        if dataset_names is None:
            dataset_names = SPLIT_DATASETS + MERGED_DATASETS

        samples = []
        for ds_name in dataset_names:
            ds_dir = os.path.join(data_root, ds_name)
            ann_path = os.path.join(ds_dir, "annotations.pkl")
            if not os.path.exists(ann_path):
                continue
            with open(ann_path, "rb") as f:
                ann = pickle.load(f)

            ds_samples = self._extract_samples(ds_dir, ann)
            samples.extend(ds_samples)

        # train/val split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(samples))
        n_val = int(len(samples) * val_ratio)
        if split == "val":
            indices = indices[:n_val]
        else:
            indices = indices[n_val:]

        self.samples = [samples[i] for i in indices]

        n_pos = sum(1 for s in self.samples if s["label"] > 0.5)
        n_neg = len(self.samples) - n_pos
        print(f"CQFDataset ({split}): {len(self.samples)} samples "
              f"(pos={n_pos}, neg={n_neg}, ratio=1:{n_neg/max(n_pos,1):.1f})")

    def _extract_samples(self, ds_dir, ann):
        """从一个数据集提取所有有效样本。"""
        import h5py
        samples = []
        meta = ann.get("_meta", {})

        for ep_name, ep_ann in ann.items():
            if ep_name == "_meta":
                continue

            labels = ep_ann["labels"]
            lifts = ep_ann["lifts"]
            T = len(labels)

            # 找HDF5文件路径
            hdf5_path = self._find_hdf5(ds_dir, ep_name)
            if hdf5_path is None:
                continue

            try:
                with h5py.File(hdf5_path, "r") as f:
                    qpos_all = f["observations/proprio_joint"][:]    # (T, 7)
                    eef_all = f["observations/proprio_eef"][:]      # (T, 6)
                    action_all = f["actions/joint_abs"][:]           # (T, 7)
                    marker_all = f["observations/tac/left/marker_offset"][:]  # (T, 9, 9, 2)
            except Exception:
                continue

            h = self.foresight_horizon
            cs = self.chunk_size

            # 正样本: label=1 (insertion)
            pos_frames = np.where(labels == 1)[0]
            for t in pos_frames:
                if t + cs > T or t + h >= T:
                    continue
                samples.append({
                    "qpos": qpos_all[t].astype(np.float32),
                    "eef": eef_all[t].astype(np.float32),
                    "action_chunk": action_all[t:t+cs].astype(np.float32),
                    "marker_cur": marker_all[t].astype(np.float32),
                    "marker_future": marker_all[t+h].astype(np.float32),
                    "label": 1.0,
                })

            # 负样本: label=2 (pre_bounce), 且t+h >= lift_start
            neg_frames = np.where(labels == 2)[0]
            lift_starts = [s for s, e, r in lifts]

            for t in neg_frames:
                if t + cs > T or t + h >= T:
                    continue
                # 过滤: t+h必须落在某个lift的start之后
                future_in_lift = any(t + h >= ls for ls in lift_starts)
                if not future_in_lift:
                    continue
                samples.append({
                    "qpos": qpos_all[t].astype(np.float32),
                    "eef": eef_all[t].astype(np.float32),
                    "action_chunk": action_all[t:t+cs].astype(np.float32),
                    "marker_cur": marker_all[t].astype(np.float32),
                    "marker_future": marker_all[t+h].astype(np.float32),
                    "label": 0.0,
                })

        return samples

    def _find_hdf5(self, ds_dir, ep_name):
        """查找episode的HDF5文件，处理split和merged两种目录结构。"""
        fname = f"{ep_name}.hdf5"
        # 直接在目录下
        path = os.path.join(ds_dir, fname)
        if os.path.exists(path):
            return path
        # success/bounce子目录
        for subdir in ["success", "bounce"]:
            path = os.path.join(ds_dir, subdir, fname)
            if os.path.exists(path):
                return path
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "qpos": torch.from_numpy(s["qpos"]),
            "eef": torch.from_numpy(s["eef"]),
            "action_chunk": torch.from_numpy(s["action_chunk"]),
            "marker_cur": torch.from_numpy(s["marker_cur"]),
            "marker_future": torch.from_numpy(s["marker_future"]),
            "label": torch.tensor(s["label"], dtype=torch.float32),
        }

    def get_weighted_sampler(self):
        """返回WeightedRandomSampler，使每个batch正负比接近1:1。"""
        labels = np.array([s["label"] for s in self.samples])
        n_pos = (labels > 0.5).sum()
        n_neg = (labels <= 0.5).sum()

        weight_pos = 1.0 / max(n_pos, 1)
        weight_neg = 1.0 / max(n_neg, 1)

        weights = np.where(labels > 0.5, weight_pos, weight_neg)
        return WeightedRandomSampler(weights, num_samples=len(self.samples), replacement=True)


def build_cqf_dataloaders(batch_size=256, num_workers=4, **kwargs):
    """构建CQF训练和验证DataLoader。"""
    from torch.utils.data import DataLoader

    train_ds = CQFDataset(split="train", **kwargs)
    val_ds = CQFDataset(split="val", **kwargs)

    train_sampler = train_ds.get_weighted_sampler()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    print("Building CQF dataset...")
    train_ds = CQFDataset(split="train")
    val_ds = CQFDataset(split="val")

    sample = train_ds[0]
    print(f"\nSample shapes:")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} {v.dtype}")
        else:
            print(f"  {k}: {v}")

    # 统计
    train_labels = [s["label"] for s in train_ds.samples]
    val_labels = [s["label"] for s in val_ds.samples]
    print(f"\nTrain: {sum(l>0.5 for l in train_labels)} pos, {sum(l<=0.5 for l in train_labels)} neg")
    print(f"Val:   {sum(l>0.5 for l in val_labels)} pos, {sum(l<=0.5 for l in val_labels)} neg")
