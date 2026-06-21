#!/usr/bin/env python3
"""Offline gradient audit for force-aware board Foresight guidance.

This evaluates the part that matters for classifier/energy guidance:

    normalized action chunk -> force-aware Foresight -> tactile/force quality score
        -> d score / d action chunk -> trust-region action refinement

It is not a robot rollout metric.  It checks whether the learned score is
episode-heldout, differentiable with respect to action, and numerically usable
as a DP guidance signal.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.pretrain_latent_foresight_multistep_force import (  # noqa: E402
    ForceAwareForesightDataset,
    ForceAwareMultiStepForesightModel,
    LABEL_TO_REASON,
    parse_dataset_roots,
    scan_labeled_episodes,
    split_episodes,
)
from TFAC_V5.tac_quality_energy.trust_region import (  # noqa: E402
    TacQualityTrustRegionRefiner,
    TrustRegionConfig,
)
from utils import set_seed  # noqa: E402


DEFAULT_FORESIGHT_DIR = Path(
    "/home/chenshuai/Project/output/foresight_ckpt/"
    "latent_foresight_board_forceaware_multistep16_boardvae_e100_bs16_0"
)
DEFAULT_CKPT = DEFAULT_FORESIGHT_DIR / "foresight_force_best.ckpt"
DEFAULT_OUT_ROOT = Path("/home/chenshuai/Project/output/force_aware_foresight_guidance_audit")
REASON_NAMES = {
    0: "too_small",
    1: "good",
    2: "too_large",
    3: "oscillate",
}


@dataclass(frozen=True)
class ScoreWeights:
    band_margin: float = 1.0
    contact_logprob: float = 0.20
    force_center: float = 0.25
    force_smooth: float = 0.10
    action_smooth: float = 0.0


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def save_json(data: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def freeze(module: torch.nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def tensor_summary(values: Iterable[float] | np.ndarray | torch.Tensor) -> Dict[str, Any]:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p05": float(np.quantile(arr, 0.05)),
        "median": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 4) -> Dict[str, float]:
    pred = logits.argmax(dim=-1).reshape(-1)
    lab = labels.reshape(-1)
    acc = (pred == lab).float().mean().item()
    recalls = []
    for cls in range(num_classes):
        mask = lab == cls
        if mask.any():
            recalls.append((pred[mask] == cls).float().mean().item())
    return {"acc": float(acc), "balanced_acc": float(np.mean(recalls)) if recalls else 0.0}


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average ranks for ties.
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = ranks[order[start:end]].mean()
        start = end
    pos_ranks = ranks[labels == 1]
    auc = (pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return float(auc)


def action_smoothness(actions: torch.Tensor) -> torch.Tensor:
    if actions.shape[1] < 3:
        return torch.zeros(actions.shape[0], dtype=actions.dtype, device=actions.device)
    accel = actions[:, 2:] - 2.0 * actions[:, 1:-1] + actions[:, :-2]
    return torch.linalg.norm(accel, dim=-1).mean(dim=1)


def score_components(
    out: Mapping[str, torch.Tensor],
    action: torch.Tensor | None = None,
    weights: ScoreWeights = ScoreWeights(),
) -> Dict[str, torch.Tensor]:
    logits = out["force_band_logits"]
    risk_logits = torch.stack([logits[..., 0], logits[..., 2], logits[..., 3]], dim=-1)
    band_margin = logits[..., 1] - torch.logsumexp(risk_logits, dim=-1)
    prob = logits.softmax(dim=-1)
    contact_logprob = F.logsigmoid(out["contact_logits"])
    contact_prob = torch.sigmoid(out["contact_logits"])

    proxy = out["force_proxy_pred"]
    zeros = torch.zeros_like(proxy[..., 0])
    force_center_pen = F.smooth_l1_loss(proxy[..., 0], zeros, reduction="none")
    smooth_pen = (
        F.smooth_l1_loss(proxy[..., 2], zeros, reduction="none")
        + 0.5 * F.smooth_l1_loss(proxy[..., 3], zeros, reduction="none")
        + 0.5 * F.smooth_l1_loss(proxy[..., 5], zeros, reduction="none")
    )
    score = (
        weights.band_margin * band_margin.mean(dim=1)
        + weights.contact_logprob * contact_logprob.mean(dim=1)
        - weights.force_center * force_center_pen.mean(dim=1)
        - weights.force_smooth * smooth_pen.mean(dim=1)
    )
    if action is not None and weights.action_smooth > 0:
        score = score - weights.action_smooth * action_smoothness(action)

    return {
        "score": score,
        "band_margin": band_margin.mean(dim=1),
        "good_prob": prob[..., 1].mean(dim=1),
        "risk_prob": (prob[..., 0] + prob[..., 2] + prob[..., 3]).mean(dim=1),
        "contact_prob": contact_prob.mean(dim=1),
        "force_center_penalty": force_center_pen.mean(dim=1),
        "force_smooth_penalty": smooth_pen.mean(dim=1),
    }


def load_model(foresight_dir: Path, ckpt_path: Path, device: torch.device) -> Tuple[ForceAwareMultiStepForesightModel, Dict[str, Any], Dict[str, Any]]:
    cfg = load_json(foresight_dir / "args.json")
    stats = load_pickle(foresight_dir / "dataset_stats.pkl")
    meta = cfg.get("meta", stats["meta"])
    action_dim = int(meta["state_dim"] if cfg.get("use_state_trajectory", True) else meta["action_dim"])
    model = ForceAwareMultiStepForesightModel(
        state_dim=int(meta["state_dim"]),
        action_dim=action_dim,
        hidden_dim=int(cfg.get("hidden_dim", 512)),
        foresight_layers=int(cfg.get("foresight_layers", 3)),
        foresight_nheads=int(cfg.get("foresight_nheads", 8)),
        foresight_dim_feedforward=int(cfg.get("foresight_dim_feedforward", 2048)),
        dropout=float(cfg.get("dropout", 0.1)),
        tactile_vae_ckpt=cfg.get("tactile_vae_ckpt"),
        tactile_vae_latent_dim=int(cfg.get("tactile_vae_latent_dim", 16)),
        predict_horizon=int(cfg.get("predict_horizon", 16)),
        tactile_vae_window=int(cfg.get("tactile_vae_window", 8)),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"WARNING load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    freeze(model)
    return model, cfg, stats


def build_dataset(cfg: Mapping[str, Any], stats: Mapping[str, Any], split: str, samples_per_episode: int, split_seed: int):
    roots = parse_dataset_roots(dict(cfg))
    episodes = scan_labeled_episodes(roots)
    train_eps, val_eps = split_episodes(episodes, float(cfg.get("train_ratio", 0.9)), split_seed)
    if split == "train":
        selected = train_eps
    elif split == "all":
        selected = episodes
    else:
        selected = val_eps

    dataset = ForceAwareForesightDataset(
        selected,
        norm_stats=stats["norm_stats"],
        meta=stats["meta"],
        force_ref=stats["force_ref"],
        chunk_size=int(cfg.get("chunk_size", 16)),
        horizon=int(cfg.get("predict_horizon", 16)),
        tactile_vae_window=int(cfg.get("tactile_vae_window", 8)),
        samples_per_episode=samples_per_episode,
        phase_start_frac=float(cfg.get("phase_start_frac", 0.20)),
        phase_end_frac=float(cfg.get("phase_end_frac", 0.95)),
        contact_only=bool(cfg.get("contact_only", True)),
        use_state_trajectory=bool(cfg.get("use_state_trajectory", True)),
        preload=False,
    )
    return dataset, {"all": len(episodes), "train": len(train_eps), "val": len(val_eps), "selected": len(selected)}


def to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def raw_delta_norm(action_delta_normed: torch.Tensor, stats: Mapping[str, Any], use_state_trajectory: bool) -> torch.Tensor:
    key = "qpos_std" if use_state_trajectory else "action_std"
    std = torch.as_tensor(stats["norm_stats"][key], dtype=action_delta_normed.dtype, device=action_delta_normed.device)
    raw_delta = action_delta_normed * std.view(1, 1, -1)
    return raw_delta.flatten(1).norm(dim=1)


def plot_audit(records: Mapping[str, np.ndarray], out_path: Path) -> None:
    labels = records["binary_label"]
    score = records["base_score"]
    delta = records["score_delta"]
    grad = records["grad_norm"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(score[labels == 1], bins=30, alpha=0.65, label="good")
    axes[0].hist(score[labels == 0], bins=30, alpha=0.65, label="bad")
    axes[0].set_title("Base score by episode label")
    axes[0].set_xlabel("quality score")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].hist(delta, bins=30, color="#3b82f6", alpha=0.80)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_title("Score change after guidance")
    axes[1].set_xlabel("final - base")
    axes[1].grid(alpha=0.25)

    axes[2].hist(grad, bins=30, color="#16a34a", alpha=0.80)
    axes[2].set_title("Gradient norm")
    axes[2].set_xlabel("||d score / d action||")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_report(result: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Force-aware Foresight Guidance Audit",
        "",
        "目的：验证 board wiping 的 force-aware Foresight 评分是否能作为 DP 的可微梯度引导信号。",
        "",
        "## Setup",
        f"- split: `{result['setup']['split']}`",
        f"- samples: `{result['setup']['num_samples']}`",
        f"- foresight ckpt: `{result['setup']['ckpt']}`",
        f"- score weights: `{result['setup']['score_weights']}`",
        f"- trust region: `{result['setup']['trust_region']}`",
        "",
        "## Held-out Scorer Metrics",
        f"- force-band acc: `{result['scorer_metrics']['band_acc']:.4f}`",
        f"- force-band balanced acc: `{result['scorer_metrics']['band_balanced_acc']:.4f}`",
        f"- contact acc: `{result['scorer_metrics']['contact_acc']:.4f}`",
        f"- good-vs-bad score AUC: `{result['scorer_metrics']['score_good_bad_auc']}`",
        f"- good-vs-bad good-prob AUC: `{result['scorer_metrics']['good_prob_auc']}`",
        "",
        "## Gradient Guidance Audit",
        f"- finite grad rate: `{result['guidance_metrics']['finite_grad_rate']:.4f}`",
        f"- positive grad rate: `{result['guidance_metrics']['positive_grad_rate']:.4f}`",
        f"- improved rate: `{result['guidance_metrics']['improved_rate']:.4f}`",
        f"- trust-region pass: `{result['guidance_metrics']['trust_region_pass_rate']:.4f}`",
        f"- score delta mean: `{result['summaries']['score_delta']['mean']:.6f}`",
        f"- normalized action delta mean: `{result['summaries']['action_delta_norm']['mean']:.6f}`",
        f"- raw action delta mean: `{result['summaries']['raw_action_delta_norm']['mean']:.6f}`",
        "",
        "## Interpretation",
        result["interpretation"],
        "",
        "## Outputs",
        f"- JSON: `{result['paths']['json']}`",
        f"- Figure: `{result['paths']['figure']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(args.seed)
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    out_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)

    model, cfg, stats = load_model(args.foresight_dir, args.ckpt, device)
    split_seed = int(cfg.get("seed", 42))
    dataset, split_counts = build_dataset(cfg, stats, args.split, args.samples_per_episode, split_seed)
    if args.max_samples > 0 and args.max_samples < len(dataset):
        dataset = Subset(dataset, list(range(args.max_samples)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    weights = ScoreWeights(
        band_margin=args.w_band_margin,
        contact_logprob=args.w_contact,
        force_center=args.w_force_center,
        force_smooth=args.w_force_smooth,
        action_smooth=args.w_action_smooth,
    )
    tr_cfg = TrustRegionConfig(
        steps=args.refine_steps,
        step_size=args.action_step,
        max_total_delta=args.max_total_delta,
        accept_only_improved=True,
        min_grad_norm=1e-10,
    )
    refiner = TacQualityTrustRegionRefiner(tr_cfg)

    record: Dict[str, List[float]] = {
        "base_score": [],
        "final_score": [],
        "score_delta": [],
        "base_good_prob": [],
        "final_good_prob": [],
        "good_prob_delta": [],
        "base_contact_prob": [],
        "final_contact_prob": [],
        "grad_norm": [],
        "action_delta_norm": [],
        "raw_action_delta_norm": [],
        "finite_grad": [],
        "positive_grad": [],
        "improved": [],
        "trust_region_pass": [],
        "label_id": [],
        "binary_label": [],
    }
    all_band_logits, all_band_labels = [], []
    all_contact_logits, all_contact_labels = [], []
    per_refiner_logs: List[Dict[str, Any]] = []

    use_state_trajectory = bool(cfg.get("use_state_trajectory", True))

    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        marker_hist = batch["marker_hist"]
        qpos = batch["qpos"]
        action = batch["action"]

        def score_fn(candidate_action: torch.Tensor) -> torch.Tensor:
            out = model(marker_hist, qpos, candidate_action)
            return score_components(out, candidate_action, weights)["score"]

        x = action.detach().clone().requires_grad_(True)
        base_out = model(marker_hist, qpos, x)
        base_comp = score_components(base_out, x, weights)
        grad = torch.autograd.grad(base_comp["score"].sum(), x, retain_graph=False)[0]
        grad_norm = grad.flatten(1).norm(dim=1).detach()
        finite_grad = torch.isfinite(grad).flatten(1).all(dim=1).detach()
        positive_grad = (grad_norm > tr_cfg.min_grad_norm).detach()

        final_action, report = refiner.refine(action, score_fn)
        per_refiner_logs.append(report)

        with torch.no_grad():
            final_out = model(marker_hist, qpos, final_action)
            final_comp = score_components(final_out, final_action, weights)
            base_score = base_comp["score"].detach()
            final_score = final_comp["score"].detach()
            delta = final_action - action
            delta_norm = delta.flatten(1).norm(dim=1)
            raw_norm = raw_delta_norm(delta, stats, use_state_trajectory)
            trust_pass = delta_norm <= args.max_total_delta + 1e-6 if args.max_total_delta > 0 else torch.ones_like(delta_norm, dtype=torch.bool)

            label_id = batch["label_id"].detach()
            binary = (label_id == 1).long()

            record["base_score"].extend(base_score.cpu().tolist())
            record["final_score"].extend(final_score.cpu().tolist())
            record["score_delta"].extend((final_score - base_score).cpu().tolist())
            record["base_good_prob"].extend(base_comp["good_prob"].detach().cpu().tolist())
            record["final_good_prob"].extend(final_comp["good_prob"].detach().cpu().tolist())
            record["good_prob_delta"].extend((final_comp["good_prob"] - base_comp["good_prob"]).detach().cpu().tolist())
            record["base_contact_prob"].extend(base_comp["contact_prob"].detach().cpu().tolist())
            record["final_contact_prob"].extend(final_comp["contact_prob"].detach().cpu().tolist())
            record["grad_norm"].extend(grad_norm.cpu().tolist())
            record["action_delta_norm"].extend(delta_norm.cpu().tolist())
            record["raw_action_delta_norm"].extend(raw_norm.cpu().tolist())
            record["finite_grad"].extend(finite_grad.float().cpu().tolist())
            record["positive_grad"].extend(positive_grad.float().cpu().tolist())
            record["improved"].extend((final_score > base_score).float().cpu().tolist())
            record["trust_region_pass"].extend(trust_pass.float().cpu().tolist())
            record["label_id"].extend(label_id.cpu().tolist())
            record["binary_label"].extend(binary.cpu().tolist())

            all_band_logits.append(base_out["force_band_logits"].detach().cpu())
            all_band_labels.append(batch["future_force_band"].detach().cpu())
            all_contact_logits.append(base_out["contact_logits"].detach().cpu())
            all_contact_labels.append(batch["future_contact"].detach().cpu())

    arrays = {k: np.asarray(v) for k, v in record.items()}
    band_metrics = classification_metrics(torch.cat(all_band_logits), torch.cat(all_band_labels))
    contact_pred = (torch.sigmoid(torch.cat(all_contact_logits)) >= 0.5)
    contact_gt = torch.cat(all_contact_labels) >= 0.5
    contact_acc = float((contact_pred == contact_gt).float().mean().item())

    by_label: Dict[str, Any] = {}
    for label_id in sorted(set(arrays["label_id"].astype(int).tolist())):
        mask = arrays["label_id"].astype(int) == label_id
        by_label[REASON_NAMES.get(label_id, str(label_id))] = {
            "n": int(mask.sum()),
            "base_score": tensor_summary(arrays["base_score"][mask]),
            "score_delta": tensor_summary(arrays["score_delta"][mask]),
            "base_good_prob": tensor_summary(arrays["base_good_prob"][mask]),
            "good_prob_delta": tensor_summary(arrays["good_prob_delta"][mask]),
        }

    score_auc = binary_auc(arrays["base_score"], arrays["binary_label"])
    good_prob_auc = binary_auc(arrays["base_good_prob"], arrays["binary_label"])
    figure_path = out_dir / "audit_histograms.png"
    plot_audit(arrays, figure_path)

    result: Dict[str, Any] = {
        "setup": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "device": str(device),
            "foresight_dir": str(args.foresight_dir),
            "ckpt": str(args.ckpt),
            "split": args.split,
            "seed": int(args.seed),
            "split_seed": int(split_seed),
            "split_counts": split_counts,
            "num_samples": int(len(arrays["base_score"])),
            "score_weights": asdict(weights),
            "trust_region": asdict(tr_cfg),
            "max_samples": int(args.max_samples),
        },
        "force_ref": stats["force_ref"],
        "scorer_metrics": {
            "band_acc": band_metrics["acc"],
            "band_balanced_acc": band_metrics["balanced_acc"],
            "contact_acc": contact_acc,
            "score_good_bad_auc": score_auc,
            "good_prob_auc": good_prob_auc,
        },
        "guidance_metrics": {
            "finite_grad_rate": float(arrays["finite_grad"].mean()),
            "positive_grad_rate": float(arrays["positive_grad"].mean()),
            "improved_rate": float(arrays["improved"].mean()),
            "trust_region_pass_rate": float(arrays["trust_region_pass"].mean()),
        },
        "summaries": {
            "base_score": tensor_summary(arrays["base_score"]),
            "final_score": tensor_summary(arrays["final_score"]),
            "score_delta": tensor_summary(arrays["score_delta"]),
            "base_good_prob": tensor_summary(arrays["base_good_prob"]),
            "final_good_prob": tensor_summary(arrays["final_good_prob"]),
            "good_prob_delta": tensor_summary(arrays["good_prob_delta"]),
            "grad_norm": tensor_summary(arrays["grad_norm"]),
            "action_delta_norm": tensor_summary(arrays["action_delta_norm"]),
            "raw_action_delta_norm": tensor_summary(arrays["raw_action_delta_norm"]),
        },
        "by_label": by_label,
        "interpretation": (
            "This is a positive offline gradient audit if held-out band/contact metrics are high, "
            "finite/positive gradient rates are near 1.0, and trust-region refinement consistently "
            "raises predicted quality with small action deltas. It still does not replace real "
            "baseline-vs-guided rollout force-curve evaluation."
        ),
        "paths": {
            "json": str(out_dir / "audit_results.json"),
            "markdown": str(out_dir / "audit_report.md"),
            "figure": str(figure_path),
        },
        "refiner_batch_reports": per_refiner_logs[:5],
    }
    save_json(result, out_dir / "audit_results.json")
    write_report(result, out_dir / "audit_report.md")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foresight_dir", type=Path, default=DEFAULT_FORESIGHT_DIR)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--samples_per_episode", type=int, default=12)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--refine_steps", type=int, default=4)
    parser.add_argument("--action_step", type=float, default=0.02)
    parser.add_argument("--max_total_delta", type=float, default=0.08)
    parser.add_argument("--w_band_margin", type=float, default=1.0)
    parser.add_argument("--w_contact", type=float, default=0.20)
    parser.add_argument("--w_force_center", type=float, default=0.25)
    parser.add_argument("--w_force_smooth", type=float, default=0.10)
    parser.add_argument("--w_action_smooth", type=float, default=0.0)
    args = parser.parse_args()

    result = run(args)
    print(json.dumps({
        "out": result["paths"],
        "scorer_metrics": result["scorer_metrics"],
        "guidance_metrics": result["guidance_metrics"],
        "score_delta": result["summaries"]["score_delta"],
        "grad_norm": result["summaries"]["grad_norm"],
    }, indent=2))


if __name__ == "__main__":
    main()
