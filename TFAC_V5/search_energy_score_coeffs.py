"""Search interpretable energy-score coefficients for tactile guidance.

The scorer must be both a good classifier and a usable DP guidance potential.
This script searches coefficients for logit-space energy scores instead of
optimizing reranking:

  energy = wq * quality_logit + wb * binary_margin + wr * reason_margin

Metrics:
  - quality correlation: continuous quality ordering
  - binary AUC: good/bad classification
  - good-bad margin: separation strength
  - dynamic range: enough signal but not explosive
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.insertion_risk_scorer_runtime import InsertionRiskScorerRuntime  # noqa: E402
from TFAC_V5.ptg_proxy_scorer_v2_runtime import PTGProxyScorerV2Runtime  # noqa: E402


OUT_DIR = Path("/home/chenshuai/Project/output/scorer_guidance_suitability")


def corr(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def score_row(energy, quality, binary, target_range=8.0):
    valid = binary >= 0
    auc = float(roc_auc_score(binary[valid], energy[valid])) if len(np.unique(binary[valid])) == 2 else 0.0
    qcorr = corr(energy, quality)
    good = energy[binary == 1]
    bad = energy[binary == 0]
    margin = float(good.mean() - bad.mean()) if len(good) and len(bad) else 0.0
    erange = float(np.percentile(energy, 99) - np.percentile(energy, 1))
    range_score = math.exp(-abs(math.log((erange + 1e-6) / target_range)))
    objective = 0.45 * max(qcorr, 0.0) + 0.35 * auc + 0.10 * np.tanh(margin / 4.0) + 0.10 * range_score
    return {
        "objective": float(objective),
        "quality_corr": float(qcorr),
        "binary_auc": auc,
        "good_bad_margin": margin,
        "p01_p99_range": erange,
        "range_score": float(range_score),
    }


def top_rows(rows, n=20):
    return sorted(rows, key=lambda r: r["metrics"]["objective"], reverse=True)[:n]


def make_grid():
    vals_q = [0.5, 0.75, 1.0, 1.25, 1.5]
    vals_b = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75]
    vals_r = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75]
    for wq in vals_q:
        for wb in vals_b:
            for wr in vals_r:
                yield wq, wb, wr


@torch.no_grad()
def insertion_logits(args):
    data = np.load(args.insertion_data, allow_pickle=True)
    marker = data["marker"].astype(np.float32)
    action = data["action"].astype(np.float32)
    quality = data["quality"].astype(np.float32)
    binary = data["binary"].astype(np.int64)
    runtime = InsertionRiskScorerRuntime(args.insertion_ckpt, device=args.device)
    out = {"quality": quality, "binary": binary, "qlogit": [], "bmargin": [], "rmargin": []}
    for start in range(0, len(marker), args.batch_size):
        m = torch.from_numpy(marker[start : start + args.batch_size]).to(runtime.device)
        a = torch.from_numpy(action[start : start + args.batch_size]).to(runtime.device)
        pred = runtime.forward(m, a)
        out["qlogit"].append(pred["quality_logit"].cpu().numpy())
        out["bmargin"].append(pred["good_logit_margin"].cpu().numpy())
        out["rmargin"].append(pred["reason_logit_margin"].cpu().numpy())
    return {k: np.concatenate(v) if isinstance(v, list) else v for k, v in out.items()}


@torch.no_grad()
def ptg_v2_logits(args):
    data = np.load(args.ptg_v2_data, allow_pickle=True)
    X = data["X"].astype(np.float32)
    task = data["task"]
    task_id = data["task_id"].astype(np.int64)
    quality = data["quality"].astype(np.float32)
    binary = data["binary"].astype(np.int64)
    runtime = PTGProxyScorerV2Runtime(args.ptg_v2_ckpt, device=args.device)
    feat = torch.tensor(X, dtype=torch.float32, device=runtime.device)
    tid = torch.tensor(task_id, dtype=torch.long, device=runtime.device)
    feat_norm = (feat - runtime.scaler_mean.view(1, -1)) / (runtime.scaler_scale.view(1, -1) + 1e-8)
    raw = runtime.model(feat_norm, tid)
    qlogit = raw["quality"].cpu().numpy()
    bmargin = (raw["binary_logits"][:, 1] - raw["binary_logits"][:, 0]).cpu().numpy()
    bad_reason_logits = torch.stack(
        [
            raw["reason_logits"][:, 0],
            torch.logsumexp(raw["reason_logits"][:, 2:], dim=-1),
        ],
        dim=-1,
    )
    rmargin = (raw["reason_logits"][:, 1] - torch.logsumexp(bad_reason_logits, dim=-1)).cpu().numpy()
    return {"quality": quality, "binary": binary, "task": task, "qlogit": qlogit, "bmargin": bmargin, "rmargin": rmargin}


def search_one(logits, name, mask=None):
    if mask is None:
        mask = np.ones_like(logits["quality"], dtype=bool)
    quality = logits["quality"][mask]
    binary = logits["binary"][mask]
    q = logits["qlogit"][mask]
    b = logits["bmargin"][mask]
    r = logits["rmargin"][mask]
    rows = []
    for wq, wb, wr in make_grid():
        energy = wq * q + wb * b + wr * r
        metrics = score_row(energy, quality, binary)
        rows.append({"name": name, "weights": {"quality": wq, "binary_margin": wb, "reason_margin": wr}, "metrics": metrics})
    return top_rows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--insertion_data", default="/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_features.npz")
    parser.add_argument("--ptg_v2_data", default="/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_features.npz")
    parser.add_argument("--insertion_ckpt", default="/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt")
    parser.add_argument("--ptg_v2_ckpt", default="/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_final.pt")
    parser.add_argument("--output", default=str(OUT_DIR / "energy_coeff_search.json"))
    return parser.parse_args()


def main(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    insertion = insertion_logits(args)
    ptg = ptg_v2_logits(args)
    result = {
        "definition": "energy = wq*quality_logit + wb*binary_margin + wr*reason_margin",
        "objective": "0.45*quality_corr + 0.35*binary_auc + 0.10*tanh(margin/4) + 0.10*range_score",
        "insertion_top": search_one(insertion, "insertion"),
        "ptg_v2_mixed_top": search_one(ptg, "ptg_v2_mixed"),
        "ptg_v2_board_top": search_one(ptg, "ptg_v2_board", mask=ptg["task"] == "board"),
        "ptg_v2_insertion_top": search_one(ptg, "ptg_v2_insertion", mask=ptg["task"] == "insertion"),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v[0] for k, v in result.items() if k.endswith("_top")}, ensure_ascii=False, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main(parse_args())
