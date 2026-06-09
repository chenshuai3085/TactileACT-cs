"""Evaluate interpretable score formulas on cached DP candidates.

This script does not resample DP actions.  It reads the cached features from
train_dp_candidate_ranker.py and searches simple, deployable formulas:

    z(quality) - a*z(action_speed) - b*z(action_accel) - c*z(marker_delta)

The goal is to test whether phase-local normalization and smoothness penalties
make the scorer more useful for DP candidate selection.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold


OUT_DIR = Path("/home/chenshuai/Project/output/dp_candidate_ranker")


FEATURES = {
    "p_good": 0,
    "log_p_good": 1,
    "quality": 2,
    "hybrid": 3,
    "t4_weak": 4,
    "t4_good": 5,
    "t4_risk": 6,
    "t4_rough": 7,
    # marker proxy starts at 8
    "marker_delta_mean": 20,
    "marker_delta_p90": 21,
    "centroid_delta_mean": 22,
    "mag_delta_mean": 23,
    "marker_first_last_l2": 25,
    # action proxy starts at 26
    "action_speed_mean": 28,
    "action_speed_std": 29,
    "action_speed_p90": 30,
    "action_accel_mean": 31,
    "action_accel_p90": 32,
    "action_first_last_l2": 33,
    "action_abs_delta_mean": 34,
    "action_abs_delta_max": 35,
}


def group_z(x, groups):
    out = np.zeros_like(x, dtype=np.float64)
    for g in np.unique(groups):
        idx = groups == g
        vals = x[idx].astype(np.float64)
        out[idx] = (vals - vals.mean()) / (vals.std() + 1e-8)
    return out


def group_minmax_good(x, groups):
    out = np.zeros_like(x, dtype=np.float64)
    for g in np.unique(groups):
        idx = groups == g
        vals = x[idx].astype(np.float64)
        out[idx] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
    return out


def select_metrics(score, l1, groups):
    selected, random_sel, oracle, corr = [], [], [], []
    wins = 0
    rng = np.random.default_rng(123)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        best = idx[np.argmax(score[idx])]
        rnd = idx[int(rng.integers(len(idx)))]
        selected.append(l1[best])
        random_sel.append(l1[rnd])
        oracle.append(l1[idx].min())
        wins += int(l1[best] < l1[rnd])
        if np.std(score[idx]) > 1e-8 and np.std(l1[idx]) > 1e-8:
            corr.append(float(np.corrcoef(score[idx], -l1[idx])[0, 1]))
        else:
            corr.append(0.0)
    selected = np.asarray(selected)
    random_sel = np.asarray(random_sel)
    oracle = np.asarray(oracle)
    return {
        "selected_l1_mean": float(selected.mean()),
        "random_l1_mean": float(random_sel.mean()),
        "oracle_l1_mean": float(oracle.mean()),
        "beats_random": float(wins / len(selected)),
        "score_l1_corr_mean": float(np.mean(corr)),
        "oracle_gap_ratio": float((selected.mean() - oracle.mean()) / max(oracle.mean(), 1e-8)),
    }


def formula_score(Z, params):
    base = Z[params["base"]]
    score = base.copy()
    for name, weight in params["penalties"].items():
        if weight == 0:
            continue
        score -= weight * Z[name]
    for name, weight in params.get("bonuses", {}).items():
        if weight == 0:
            continue
        score += weight * Z[name]
    return score


def search_formulas(Z, l1, groups, train_groups):
    train_mask = np.isin(groups, train_groups)
    bases = ["quality", "hybrid", "p_good"]
    penalty_sets = [
        [],
        ["action_speed_p90"],
        ["action_accel_p90"],
        ["action_abs_delta_max"],
        ["marker_delta_p90"],
        ["action_speed_p90", "action_accel_p90"],
        ["action_abs_delta_max", "marker_delta_p90"],
        ["action_speed_p90", "action_accel_p90", "marker_delta_p90"],
    ]
    weights = [0.25, 0.5, 1.0]
    best = None
    rows = []
    for base in bases:
        for penalty_names in penalty_sets:
            weight_grid = [()] if not penalty_names else product(weights, repeat=len(penalty_names))
            for ws in weight_grid:
                params = {
                    "base": base,
                    "penalties": dict(zip(penalty_names, ws)),
                    "bonuses": {"t4_good": 0.25},
                }
                score = formula_score(Z, params)
                metrics = select_metrics(score[train_mask], l1[train_mask], groups[train_mask])
                row = {"params": params, "metrics": metrics}
                rows.append(row)
                key = (metrics["selected_l1_mean"], -metrics["beats_random"])
                if best is None or key < (best["metrics"]["selected_l1_mean"], -best["metrics"]["beats_random"]):
                    best = row
    return best, rows


def evaluate(args):
    data = np.load(args.cache, allow_pickle=True)
    X = data["X"].astype(np.float64)
    l1 = data["l1"].astype(np.float64)
    groups = data["groups"].astype(np.int64)
    z_features = {
        name: group_z(X[:, idx], groups)
        for name, idx in FEATURES.items()
    }

    cv = GroupKFold(n_splits=min(args.folds, len(np.unique(groups))))
    fold_rows = []
    for fold, (tr, te) in enumerate(cv.split(X, l1, groups)):
        train_groups = np.unique(groups[tr])
        best, _ = search_formulas(z_features, l1, groups, train_groups)
        score = formula_score(z_features, best["params"])
        test_metrics = select_metrics(score[te], l1[te], groups[te])
        base_quality = select_metrics(X[te, FEATURES["quality"]], l1[te], groups[te])
        fold_rows.append(
            {
                "fold": fold,
                "best_params": best["params"],
                "train_metrics": best["metrics"],
                "test_metrics": test_metrics,
                "base_quality_test": base_quality,
            }
        )
        print(f"fold {fold}: {test_metrics}, params={best['params']}", flush=True)

    def agg(key):
        out = {}
        for metric in fold_rows[0][key]:
            vals = [r[key][metric] for r in fold_rows]
            out[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        return out

    result = {
        "cache": args.cache,
        "n": int(len(X)),
        "n_groups": int(len(np.unique(groups))),
        "folds": fold_rows,
        "formula": agg("test_metrics"),
        "base_quality": agg("base_quality_test"),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"formula_eval_{Path(args.cache).stem}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"formula": result["formula"], "base_quality": result["base_quality"]}, indent=2))
    print(f"Saved {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        default="/home/chenshuai/Project/output/dp_candidate_ranker/dp_candidates_K32_N120_seed42.npz",
    )
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
