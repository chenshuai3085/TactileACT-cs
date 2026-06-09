"""Evaluate whether tactile quality scorers are suitable for DP guidance.

This script is deliberately not a reranking benchmark.  It checks two
requirements for a scorer that will be used as classifier guidance:

1. It must classify/score contact quality correctly under episode-level splits
   or on the held-out labels saved by the training scripts.
2. Its score must be a useful potential function: non-saturated, finite
   gradients, and local gradient ascent should increase the quality score.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, r2_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.insertion_risk_scorer_runtime import InsertionRiskScorerRuntime  # noqa: E402
from TFAC_V5.ptg_proxy_scorer_v2_runtime import PTGProxyScorerV2Runtime  # noqa: E402


OUT_DIR = Path("/home/chenshuai/Project/output/scorer_guidance_suitability")
INSERTION_DATA = Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_features.npz")
PTG_V2_DATA = Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_features.npz")


def summarize(x: Iterable[float]) -> dict:
    arr = np.asarray(list(x), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def corr(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def binary_metrics(y_binary: np.ndarray, p_good: np.ndarray) -> dict:
    valid = y_binary >= 0
    y = y_binary[valid].astype(np.int64)
    p = p_good[valid]
    pred = (p >= 0.5).astype(np.int64)
    return {
        "n_valid": int(valid.sum()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
    }


def reason_metrics(y_reason: np.ndarray, reason_prob: np.ndarray) -> dict:
    pred = np.argmax(reason_prob, axis=1)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_reason, pred)),
        "macro_f1": float(f1_score(y_reason, pred, average="macro")),
    }


def score_diagnostics(scores: Dict[str, np.ndarray], quality: np.ndarray, y_binary: np.ndarray) -> dict:
    out = {}
    for name, value in scores.items():
        value = np.asarray(value, dtype=np.float64)
        valid = y_binary >= 0
        good = value[y_binary == 1]
        bad = value[y_binary == 0]
        margin = float(good.mean() - bad.mean()) if len(good) and len(bad) else None
        out[name] = {
            "summary": summarize(value),
            "corr_with_quality": corr(value, quality),
            "good_minus_bad_mean": margin,
            "valid_binary_auc": float(roc_auc_score(y_binary[valid], value[valid]))
            if valid.any() and len(np.unique(y_binary[valid])) == 2
            else None,
        }
    return out


def saturation_metrics(p_good: np.ndarray, quality_score: np.ndarray) -> dict:
    return {
        "p_good_lt_0p02": float(np.mean(p_good < 0.02)),
        "p_good_gt_0p98": float(np.mean(p_good > 0.98)),
        "p_good_saturated_total": float(np.mean((p_good < 0.02) | (p_good > 0.98))),
        "quality_lt_0p02": float(np.mean(quality_score < 0.02)),
        "quality_gt_0p98": float(np.mean(quality_score > 0.98)),
        "quality_saturated_total": float(np.mean((quality_score < 0.02) | (quality_score > 0.98))),
    }


def insertion_forward_batches(runtime, marker, action, batch_size: int, modes: list[str]) -> dict:
    outs = {"p_good": [], "risk_prob": [], "quality_score": [], "reason_prob": []}
    scores = {m: [] for m in modes}
    with torch.no_grad():
        for start in range(0, len(marker), batch_size):
            m = torch.from_numpy(marker[start : start + batch_size]).to(runtime.device)
            a = torch.from_numpy(action[start : start + batch_size]).to(runtime.device)
            out = runtime.forward(m, a)
            for key in outs:
                outs[key].append(out[key].detach().cpu().numpy())
            for mode in modes:
                scores[mode].append(runtime.score(m, a, mode=mode).detach().cpu().numpy())
    return {
        **{k: np.concatenate(v, axis=0) for k, v in outs.items()},
        "scores": {k: np.concatenate(v, axis=0) for k, v in scores.items()},
    }


def insertion_gradient_probe(runtime, marker, action, reason, modes: list[str], n_per_class: int, step_size: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rows = {}
    for mode in modes:
        mode_rows = []
        for cls in np.unique(reason):
            idx = np.flatnonzero(reason == cls)
            if len(idx) == 0:
                continue
            take = rng.choice(idx, min(n_per_class, len(idx)), replace=False)
            m0 = torch.tensor(marker[take], dtype=torch.float32, device=runtime.device, requires_grad=True)
            a0 = torch.tensor(action[take], dtype=torch.float32, device=runtime.device, requires_grad=True)
            score0 = runtime.score(m0, a0, mode=mode)
            grad_m, grad_a = torch.autograd.grad(score0.sum(), [m0, a0], retain_graph=False)
            grad_a_rms = grad_a.flatten(1).norm(dim=1) / math.sqrt(float(grad_a[0].numel()))
            grad_m_rms = grad_m.flatten(1).norm(dim=1) / math.sqrt(float(grad_m[0].numel()))
            with torch.no_grad():
                denom = grad_a.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-8)
                a1 = a0 + step_size * grad_a / denom
                score1 = runtime.score(m0, a1, mode=mode)
            mode_rows.append(
                {
                    "reason": int(cls),
                    "n": int(len(take)),
                    "score_before": summarize(score0.detach().cpu().numpy()),
                    "score_after_action_step": summarize(score1.detach().cpu().numpy()),
                    "score_delta_mean": float((score1 - score0).detach().cpu().mean()),
                    "improved_rate": float(((score1 - score0) > 0).detach().cpu().float().mean()),
                    "grad_action_rms": summarize(grad_a_rms.detach().cpu().numpy()),
                    "grad_marker_rms": summarize(grad_m_rms.detach().cpu().numpy()),
                    "finite_grad_rate": float(
                        torch.isfinite(grad_a).flatten(1).all(dim=1).float().mean().detach().cpu()
                    ),
                }
            )
        rows[mode] = mode_rows
    return rows


def evaluate_insertion(args) -> dict:
    data = np.load(args.insertion_data, allow_pickle=True)
    marker = data["marker"].astype(np.float32)
    action = data["action"].astype(np.float32)
    reason = data["reason"].astype(np.int64)
    binary = data["binary"].astype(np.int64)
    quality = data["quality"].astype(np.float32)

    runtime = InsertionRiskScorerRuntime(args.insertion_ckpt, device=args.device)
    modes = ["quality", "p_good", "log_p_good", "neg_risk", "risk_guidance", "energy", "energy_clipped"]
    pred = insertion_forward_batches(runtime, marker, action, args.batch_size, modes)
    p_good = pred["p_good"]
    quality_score = pred["quality_score"]
    reason_prob = pred["reason_prob"]
    result = {
        "kind": "insertion_risk_runtime",
        "data": {
            "n": int(len(marker)),
            "reason_counts": {str(k): int(v) for k, v in zip(*np.unique(reason, return_counts=True))},
            "binary_counts_with_neutral": {str(k): int(v) for k, v in zip(*np.unique(binary, return_counts=True))},
        },
        "classification": {
            "binary_from_p_good": binary_metrics(binary, p_good),
            "reason": reason_metrics(reason, reason_prob),
            "quality_corr": corr(quality_score, quality),
            "quality_r2": float(r2_score(quality, quality_score)),
        },
        "saturation": saturation_metrics(p_good, quality_score),
        "score_diagnostics": score_diagnostics(pred["scores"], quality, binary),
        "gradient_probe": insertion_gradient_probe(
            runtime,
            marker,
            action,
            reason,
            modes=["quality", "log_p_good", "risk_guidance", "energy", "energy_clipped"],
            n_per_class=args.grad_n_per_class,
            step_size=args.grad_step_size,
            seed=args.seed,
        ),
    }
    return result


def evaluate_ptg_v2_feature(args) -> dict:
    data = np.load(args.ptg_v2_data, allow_pickle=True)
    X = data["X"].astype(np.float32)
    task_id = data["task_id"].astype(np.int64)
    task = data["task"]
    reason = data["reason"].astype(np.int64)
    binary = data["binary"].astype(np.int64)
    quality = data["quality"].astype(np.float32)

    runtime = PTGProxyScorerV2Runtime(args.ptg_v2_ckpt, device=args.device)
    feat = torch.tensor(X, dtype=torch.float32, device=runtime.device)
    tid = torch.tensor(task_id, dtype=torch.long, device=runtime.device)
    feat_norm = (feat - runtime.scaler_mean.view(1, -1)) / (runtime.scaler_scale.view(1, -1) + 1e-8)
    with torch.no_grad():
        raw = runtime.model(feat_norm, tid)
        p_good = torch.softmax(raw["binary_logits"], dim=-1)[:, 1].cpu().numpy()
        reason_prob = torch.softmax(raw["reason_logits"], dim=-1).cpu().numpy()
        quality_logit_t = raw["quality"]
        quality_score = torch.sigmoid(quality_logit_t).cpu().numpy()
        good_logit_margin_t = raw["binary_logits"][:, 1] - raw["binary_logits"][:, 0]
        bad_reason_logits_t = torch.stack(
            [
                raw["reason_logits"][:, 0],
                torch.logsumexp(raw["reason_logits"][:, 2:], dim=-1),
            ],
            dim=-1,
        )
        reason_logit_margin_t = raw["reason_logits"][:, 1] - torch.logsumexp(bad_reason_logits_t, dim=-1)
        energy_t = quality_logit_t + 0.25 * good_logit_margin_t + 0.25 * reason_logit_margin_t
        quality_logit = quality_logit_t.cpu().numpy()
        good_logit_margin = good_logit_margin_t.cpu().numpy()
        reason_logit_margin = reason_logit_margin_t.cpu().numpy()
        energy = energy_t.cpu().numpy()
    scores = {
        "quality": quality_score,
        "quality_logit": quality_logit,
        "p_good": p_good,
        "log_p_good": np.log(np.clip(p_good, 1e-8, 1.0)),
        "reason_good": reason_prob[:, 1],
        "good_logit_margin": good_logit_margin,
        "reason_logit_margin": reason_logit_margin,
        "energy": energy,
        "energy_clipped": np.tanh(energy / 4.0) * 4.0,
        "guidance_like": quality_score + 0.25 * np.log(np.clip(p_good, 1e-8, 1.0)) + 0.25 * reason_prob[:, 1],
    }

    grad_rows = {}
    for task_name in ["insertion", "board", "mixed"]:
        if task_name == "mixed":
            idx = np.arange(len(X))
        else:
            idx = np.flatnonzero(task == task_name)
        if len(idx) > args.grad_feature_n:
            idx = np.random.default_rng(args.seed).choice(idx, args.grad_feature_n, replace=False)
        x_probe = torch.tensor(X[idx], dtype=torch.float32, device=runtime.device, requires_grad=True)
        t_probe = torch.tensor(task_id[idx], dtype=torch.long, device=runtime.device)
        x_norm = (x_probe - runtime.scaler_mean.view(1, -1)) / (runtime.scaler_scale.view(1, -1) + 1e-8)
        raw_probe = runtime.model(x_norm, t_probe)
        p = torch.softmax(raw_probe["binary_logits"], dim=-1)[:, 1]
        q = torch.sigmoid(raw_probe["quality"])
        r_good = torch.softmax(raw_probe["reason_logits"], dim=-1)[:, 1]
        score = q + 0.25 * torch.log(p.clamp_min(1e-8)) + 0.25 * r_good
        grad = torch.autograd.grad(score.sum(), x_probe, retain_graph=False)[0]
        with torch.no_grad():
            denom = grad.norm(dim=1, keepdim=True).clamp_min(1e-8)
            x_after = x_probe + args.grad_step_size * grad / denom
            raw_after = runtime.model((x_after - runtime.scaler_mean.view(1, -1)) / (runtime.scaler_scale.view(1, -1) + 1e-8), t_probe)
            p_after = torch.softmax(raw_after["binary_logits"], dim=-1)[:, 1]
            q_after = torch.sigmoid(raw_after["quality"])
            r_after = torch.softmax(raw_after["reason_logits"], dim=-1)[:, 1]
            score_after = q_after + 0.25 * torch.log(p_after.clamp_min(1e-8)) + 0.25 * r_after
        grad_rows[task_name] = {
            "n": int(len(idx)),
            "grad_feature_rms": summarize((grad.norm(dim=1) / math.sqrt(X.shape[1])).detach().cpu().numpy()),
            "score_delta_mean": float((score_after - score).detach().cpu().mean()),
            "improved_rate": float(((score_after - score) > 0).detach().cpu().float().mean()),
        }

    per_task = {}
    for task_name in np.unique(task):
        idx = task == task_name
        per_task[str(task_name)] = {
            "n": int(idx.sum()),
            "binary": binary_metrics(binary[idx], p_good[idx]),
            "reason": reason_metrics(reason[idx], reason_prob[idx]),
            "quality_corr": corr(quality_score[idx], quality[idx]),
            "quality_r2": float(r2_score(quality[idx], quality_score[idx])),
            "saturation": saturation_metrics(p_good[idx], quality_score[idx]),
        }

    return {
        "kind": "ptg_proxy_scorer_v2_feature_space",
        "data": {
            "n": int(len(X)),
            "feature_dim": int(X.shape[1]),
            "task_counts": {str(k): int(v) for k, v in zip(*np.unique(task, return_counts=True))},
        },
        "mixed": {
            "binary": binary_metrics(binary, p_good),
            "reason": reason_metrics(reason, reason_prob),
            "quality_corr": corr(quality_score, quality),
            "quality_r2": float(r2_score(quality, quality_score)),
            "saturation": saturation_metrics(p_good, quality_score),
            "score_diagnostics": score_diagnostics(scores, quality, binary),
        },
        "per_task": per_task,
        "feature_gradient_probe": grad_rows,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--grad_n_per_class", type=int, default=64)
    parser.add_argument("--grad_feature_n", type=int, default=512)
    parser.add_argument("--grad_step_size", type=float, default=0.02)
    parser.add_argument("--insertion_data", default=str(INSERTION_DATA))
    parser.add_argument("--ptg_v2_data", default=str(PTG_V2_DATA))
    parser.add_argument("--insertion_ckpt", default="/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt")
    parser.add_argument("--ptg_v2_ckpt", default="/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_final.pt")
    parser.add_argument("--output", default=str(OUT_DIR / "guidance_suitability_eval.json"))
    return parser.parse_args()


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "config": vars(args),
        "evaluation_definition": {
            "accuracy": "classification/quality metrics from saved episode-level scorer datasets",
            "guidance": "score saturation + finite gradients + local gradient-ascent score improvement",
            "not_used_as_primary_metric": "DP candidate reranking",
        },
        "insertion_risk_runtime": evaluate_insertion(args),
        "ptg_proxy_v2_feature_space": evaluate_ptg_v2_feature(args),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    short = {
        "output": str(out_path),
        "insertion": {
            "binary_auc": result["insertion_risk_runtime"]["classification"]["binary_from_p_good"]["auc"],
            "reason_macro_f1": result["insertion_risk_runtime"]["classification"]["reason"]["macro_f1"],
            "quality_corr": result["insertion_risk_runtime"]["classification"]["quality_corr"],
            "saturation": result["insertion_risk_runtime"]["saturation"],
        },
        "ptg_v2_mixed": {
            "binary_auc": result["ptg_proxy_v2_feature_space"]["mixed"]["binary"]["auc"],
            "reason_macro_f1": result["ptg_proxy_v2_feature_space"]["mixed"]["reason"]["macro_f1"],
            "quality_corr": result["ptg_proxy_v2_feature_space"]["mixed"]["quality_corr"],
            "saturation": result["ptg_proxy_v2_feature_space"]["mixed"]["saturation"],
        },
    }
    print(json.dumps(short, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(parse_args())
