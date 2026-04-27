"""
CQF Phase B Dataset: mixed GT + Foresight predicted tactile.

During Phase B, a fraction (gt_ratio, linearly decayed) of samples use GT future
tactile, while the rest use LightweightForesight predictions.

Also includes action-perturbation negative samples:
  - Take insertion frames, add noise to action
  - Use Foresight to predict what would happen with noisy action
  - These become additional negative samples with soft labels

Usage:
  Phase B (mixed):
    dataset = CQFPhaseBDataset(foresight_model, gt_ratio=0.5, ...)
  Phase C (pure foresight):
    dataset = CQFPhaseBDataset(foresight_model, gt_ratio=0.0, ...)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from cqf_dataset import CQFDataset


class CQFPhaseBDataset(Dataset):
    """
    Wraps CQFDataset for Phase B/C training.

    For each sample from the base dataset:
      - With probability gt_ratio: return GT marker_future (Phase A behavior)
      - With probability (1-gt_ratio): run Foresight to get predicted marker_future

    Additionally generates action-perturbation negatives from insertion frames.
    """

    def __init__(self, foresight_model, gt_ratio=0.5,
                 n_perturb_per_pos=2, perturb_scales=(0.5, 1.0, 2.0),
                 device="cuda:0", **base_kwargs):
        super().__init__()
        self.base_dataset = CQFDataset(**base_kwargs)
        self.foresight = foresight_model
        self.gt_ratio = gt_ratio
        self.n_perturb_per_pos = n_perturb_per_pos
        self.perturb_scales = perturb_scales
        self.device = device

        self._precompute_foresight()
        self._generate_perturbation_negatives()

    @torch.no_grad()
    def _precompute_foresight(self):
        """Pre-compute foresight predictions for all samples."""
        self.foresight.eval()
        batch_size = 512
        self.foresight_preds = []

        all_samples = self.base_dataset.samples
        for start in range(0, len(all_samples), batch_size):
            batch = all_samples[start:start+batch_size]
            qpos = torch.stack([torch.from_numpy(s["qpos"]) for s in batch]).to(self.device)
            eef = torch.stack([torch.from_numpy(s["eef"]) for s in batch]).to(self.device)
            action = torch.stack([torch.from_numpy(s["action_chunk"]) for s in batch]).to(self.device)
            mcur = torch.stack([torch.from_numpy(s["marker_cur"].flatten()) for s in batch]).to(self.device)

            pred = self.foresight(qpos, eef, action, mcur)
            self.foresight_preds.extend(pred.cpu().numpy())

        print(f"Pre-computed foresight predictions for {len(self.foresight_preds)} samples")

    @torch.no_grad()
    def _generate_perturbation_negatives(self):
        """Generate action-perturbation negative samples from insertion frames."""
        self.foresight.eval()
        self.perturb_samples = []

        pos_indices = [i for i, s in enumerate(self.base_dataset.samples)
                       if s["label"] > 0.5]

        rng = np.random.RandomState(123)
        action_stds = []
        for s in self.base_dataset.samples[:1000]:
            action_stds.append(s["action_chunk"].std())
        global_action_std = np.mean(action_stds)

        batch_size = 512
        perturb_inputs = []

        for idx in pos_indices:
            s = self.base_dataset.samples[idx]
            for j in range(self.n_perturb_per_pos):
                scale = self.perturb_scales[j % len(self.perturb_scales)]
                noise = rng.randn(*s["action_chunk"].shape).astype(np.float32)
                noisy_action = s["action_chunk"] + noise * global_action_std * scale

                soft_label = max(0.0, 1.0 - scale * 0.3)

                perturb_inputs.append({
                    "qpos": s["qpos"],
                    "eef": s["eef"],
                    "action_chunk": noisy_action,
                    "marker_cur": s["marker_cur"],
                    "label": soft_label,
                })

        for start in range(0, len(perturb_inputs), batch_size):
            batch = perturb_inputs[start:start+batch_size]
            qpos = torch.stack([torch.from_numpy(s["qpos"]) for s in batch]).to(self.device)
            eef = torch.stack([torch.from_numpy(s["eef"]) for s in batch]).to(self.device)
            action = torch.stack([torch.from_numpy(s["action_chunk"]) for s in batch]).to(self.device)
            mcur = torch.stack([torch.from_numpy(s["marker_cur"].flatten()) for s in batch]).to(self.device)

            pred = self.foresight(qpos, eef, action, mcur)
            preds = pred.cpu().numpy()

            for i, p in enumerate(batch):
                p["marker_future_foresight"] = preds[i]
                self.perturb_samples.append(p)

        print(f"Generated {len(self.perturb_samples)} perturbation negative samples")

    def __len__(self):
        return len(self.base_dataset) + len(self.perturb_samples)

    def __getitem__(self, idx):
        if idx < len(self.base_dataset):
            base_sample = self.base_dataset[idx]
            s = self.base_dataset.samples[idx]

            if np.random.random() < self.gt_ratio:
                marker_future = base_sample["marker_future"]
            else:
                marker_future = torch.from_numpy(
                    self.foresight_preds[idx].reshape(9, 9, 2))

            return {
                "qpos": base_sample["qpos"],
                "eef": base_sample["eef"],
                "action_chunk": base_sample["action_chunk"],
                "marker_cur": base_sample["marker_cur"],
                "marker_future": marker_future,
                "label": base_sample["label"],
            }
        else:
            pidx = idx - len(self.base_dataset)
            p = self.perturb_samples[pidx]
            return {
                "qpos": torch.from_numpy(p["qpos"]),
                "eef": torch.from_numpy(p["eef"]),
                "action_chunk": torch.from_numpy(p["action_chunk"]),
                "marker_cur": torch.from_numpy(p["marker_cur"]),
                "marker_future": torch.from_numpy(
                    p["marker_future_foresight"].reshape(9, 9, 2)),
                "label": torch.tensor(p["label"], dtype=torch.float32),
            }

    def get_weighted_sampler(self):
        """Weighted sampler for balanced positive/negative sampling."""
        labels = []
        for s in self.base_dataset.samples:
            labels.append(s["label"])
        for p in self.perturb_samples:
            labels.append(p["label"])

        labels = np.array(labels)
        n_pos = (labels > 0.5).sum()
        n_neg = (labels <= 0.5).sum()
        weight_pos = 1.0 / max(n_pos, 1)
        weight_neg = 1.0 / max(n_neg, 1)
        weights = np.where(labels > 0.5, weight_pos, weight_neg)
        return WeightedRandomSampler(weights, num_samples=len(self), replacement=True)
