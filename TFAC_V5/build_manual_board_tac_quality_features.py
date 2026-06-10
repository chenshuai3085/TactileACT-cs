"""Build TacQuality proxy features with manual board-wiping labels.

This keeps the socket insertion labels unchanged, but replaces the old
force-threshold board weak labels with user-provided board label directories.
The output schema matches train_distilled_tac_quality_energy.py:

  X, reason, binary, quality, task, task_id, groups, sample_ids

Current manual board labels are partial:
  - good_smooth: positive wiping, force suitable and smooth;
  - too_light_unclean: negative wiping, z too high / force too small / unclean.

Future too-heavy or unstable-force directories can be added without changing
the downstream multi-head energy model.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.eval_board_quality_label_schemes import action_proxy_features, marker_proxy_features  # noqa: E402
from TFAC_V5.evaluate_marker_proxy_scorer import insertion_t4  # noqa: E402


INSERTION_DIR = Path("/home/chenshuai/data/dataset/0414")
BOARD_POS_DIR = Path("/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/wipe_pos_straight_z124_125_150_20260609")
BOARD_NEG_LIGHT_DIR = Path("/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/z_too_high")
OUT_DIR = Path("/home/chenshuai/Project/output/manual_board_tac_quality_features")

TASK_TO_ID = {"insertion": 0, "board": 1}
REASON_NAMES = {
    0: "weak_or_no_contact_too_light",
    1: "good_stable_smooth",
    2: "excessive_or_risk_too_heavy",
    3: "impact_or_rough_force",
    4: "rough_motion_or_unstable",
}
BOARD_REASON_DIRS = {
    "good_smooth": 1,
    "too_light_unclean": 0,
}


def deterministic_seed(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def pad_window(seq: np.ndarray, window: int) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32)
    if len(seq) >= window:
        return seq[-window:]
    pad = np.repeat(seq[:1], window - len(seq), axis=0)
    return np.concatenate([pad, seq], axis=0)


def feature_vector(left_seq: np.ndarray, right_seq: np.ndarray, eef_seq: np.ndarray, joint_seq: np.ndarray) -> np.ndarray:
    left = marker_proxy_features(left_seq)
    right = marker_proxy_features(right_seq)
    diff = np.abs(left - right)
    eef = action_proxy_features(eef_seq)
    joint = action_proxy_features(joint_seq)
    return np.concatenate([left, right, diff, eef, joint]).astype(np.float32)


def force_mag(force6: np.ndarray) -> np.ndarray:
    return np.linalg.norm(force6[:, :3], axis=1)


def contact_segments(force6: np.ndarray, min_len: int, q: float) -> List[Tuple[int, int]]:
    f = force_mag(force6)
    if len(f) == 0:
        return []
    threshold = max(float(np.quantile(f, q)), float(np.median(f) + 0.25 * np.std(f)))
    mask = f >= threshold
    segs: List[Tuple[int, int]] = []
    start = None
    for i, value in enumerate(mask.tolist() + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start >= min_len:
                segs.append((start, i))
            start = None
    if not segs:
        margin = max(0, len(f) // 5)
        if len(f) - 2 * margin >= min_len:
            segs = [(margin, len(f) - margin)]
    return segs


def insertion_reason(ep_type: str, raw_label: int, frame_idx: int, lifts) -> int:
    t4 = insertion_t4(ep_type, raw_label, frame_idx, lifts)
    if ep_type != "bounce" and t4 in (2, 3):
        t4 = 1 if raw_label == 1 else 0
    if t4 == 0:
        return 0
    if t4 == 1:
        return 1
    if t4 == 2:
        return 2
    return 3


def reason_targets(reason: int, task: str) -> Tuple[int, float]:
    if reason == 1:
        return 1, 1.0
    if reason == 0:
        # In insertion this is neutral/no contact; in board this is bad
        # too-light wiping.  Keep insertion neutral, board negative.
        return (-1, 0.25) if task == "insertion" else (0, 0.0)
    if reason == 2:
        return 0, 0.03
    if reason == 3:
        return 0, 0.0
    if reason == 4:
        return 0, 0.0
    raise ValueError(reason)


def build_insertion_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    annotations = pickle.load(open(Path(args.insertion_dir) / "annotations.pkl", "rb"))
    rows: List[Dict[str, Any]] = []
    for ep_name, info in sorted(annotations.items()):
        if not isinstance(info, dict) or info.get("type") not in {"success", "bounce"}:
            continue
        path = Path(args.insertion_dir) / f"{ep_name}.hdf5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:
            left = f["observations/tac/left/marker_offset"][:]
            right = f["observations/tac/right/marker_offset"][:] if "observations/tac/right/marker_offset" in f else left
            eef = f["actions/eef_abs"][:]
            joint = f["actions/joint_abs"][:]
        labels = np.asarray(info["labels"])
        n = min(len(left), len(right), len(eef), len(joint), len(labels))
        indices = np.arange(args.insertion_window - 1, n)
        if args.max_insertion_per_episode and len(indices) > args.max_insertion_per_episode:
            rng = np.random.default_rng(deterministic_seed(ep_name))
            indices = np.sort(rng.choice(indices, args.max_insertion_per_episode, replace=False))
        ep_type = info.get("type")
        lifts = info.get("lifts", [])
        for frame_idx in indices:
            raw = int(labels[frame_idx])
            reason = insertion_reason(ep_type, raw, int(frame_idx), lifts)
            binary, quality = reason_targets(reason, "insertion")
            start = max(0, int(frame_idx) - args.insertion_window + 1)
            rows.append(
                {
                    "sample_id": f"insertion/{ep_name}/{frame_idx}",
                    "task": "insertion",
                    "episode": ep_name,
                    "reason": reason,
                    "binary": binary,
                    "quality": quality,
                    "raw_label": raw,
                    "feature": feature_vector(
                        pad_window(left[start : frame_idx + 1], args.insertion_window),
                        pad_window(right[start : frame_idx + 1], args.insertion_window),
                        pad_window(eef[start : frame_idx + 1], args.insertion_window),
                        pad_window(joint[start : frame_idx + 1], args.insertion_window),
                    ),
                }
            )
    return rows


def build_board_dir_samples(args: argparse.Namespace, name: str, root: Path, reason: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*.hdf5")):
        with h5py.File(path, "r") as f:
            left = f["observations/tac/left/marker_offset"][:]
            right = f["observations/tac/right/marker_offset"][:]
            eef = f["actions/eef_abs"][:]
            joint = f["actions/joint_abs"][:]
            ft = f["ft"][:]
        n = min(len(left), len(right), len(eef), len(joint), len(ft))
        segs = contact_segments(ft[:n], min_len=max(args.board_window, 8), q=args.board_contact_quantile)
        before = len(rows)
        for seg_start, seg_end in segs:
            for start in range(seg_start, max(seg_start + 1, seg_end - args.board_window + 1), args.board_stride):
                end = min(seg_end, start + args.board_window)
                if end - start < max(8, args.board_window // 2):
                    continue
                binary, quality = reason_targets(reason, "board")
                rows.append(
                    {
                        "sample_id": f"board/{name}/{path.stem}/{start}_{end}",
                        "task": "board",
                        "episode": f"{name}/{path.stem}",
                        "reason": reason,
                        "binary": binary,
                        "quality": quality,
                        "raw_label": name,
                        "feature": feature_vector(
                            left[start:end],
                            right[start:end],
                            eef[start:end],
                            joint[start:end],
                        ),
                        "force_mean": float(force_mag(ft[start:end]).mean()),
                    }
                )
        summary.append(
            {
                "episode": f"{name}/{path.stem}",
                "n_steps": int(n),
                "segments": [[int(a), int(b)] for a, b in segs],
                "n_windows": int(len(rows) - before),
            }
        )
    return rows, summary


def build_board_samples(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    dirs = {
        "good_smooth": Path(args.board_positive_dir),
        "too_light_unclean": Path(args.board_negative_light_dir),
    }
    samples: List[Dict[str, Any]] = []
    summaries = {}
    for name, root in dirs.items():
        reason = BOARD_REASON_DIRS[name]
        rows, summary = build_board_dir_samples(args, name, root, reason)
        samples.extend(rows)
        summaries[name] = summary
    meta = {
        "label_source": "manual_dirs",
        "board_dirs": {name: str(path) for name, path in dirs.items()},
        "reason_dirs": BOARD_REASON_DIRS,
        "contact_only": True,
        "contact_quantile": args.board_contact_quantile,
        "board_window": args.board_window,
        "board_stride": args.board_stride,
        "episode_contact_summary_head": {k: v[:20] for k, v in summaries.items()},
        "label_scope_note": (
            "Manual board labels are partial. Current negative samples cover too_light/unclean only; "
            "future too_heavy and rough/unstable negative directories should be added as reason classes 2/4."
        ),
    }
    return samples, meta


def build(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    insertion = build_insertion_samples(args)
    board, board_meta = build_board_samples(args)
    samples = insertion + board
    if not samples:
        raise RuntimeError("No samples built")

    X = np.stack([s["feature"] for s in samples]).astype(np.float32)
    reason = np.array([s["reason"] for s in samples], dtype=np.int64)
    binary = np.array([s["binary"] for s in samples], dtype=np.int64)
    quality = np.array([s["quality"] for s in samples], dtype=np.float32)
    task = np.array([s["task"] for s in samples])
    task_id = np.array([TASK_TO_ID[s["task"]] for s in samples], dtype=np.int64)
    groups = np.array([f"{s['task']}::{s['episode']}" for s in samples])
    sample_ids = np.array([s["sample_id"] for s in samples])

    npz_path = out_dir / "manual_board_tac_quality_features.npz"
    np.savez_compressed(
        npz_path,
        X=X,
        reason=reason,
        binary=binary,
        quality=quality,
        task=task,
        task_id=task_id,
        groups=groups,
        sample_ids=sample_ids,
    )

    csv_path = out_dir / "manual_board_tac_quality_samples.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "task", "episode", "reason", "reason_name", "binary", "quality", "raw_label"],
        )
        writer.writeheader()
        for s in samples:
            writer.writerow(
                {
                    "sample_id": s["sample_id"],
                    "task": s["task"],
                    "episode": s["episode"],
                    "reason": s["reason"],
                    "reason_name": REASON_NAMES[s["reason"]],
                    "binary": s["binary"],
                    "quality": s["quality"],
                    "raw_label": s["raw_label"],
                }
            )

    meta = {
        "purpose": "TacQuality proxy features with unchanged insertion labels and manual board contact-window labels.",
        "feature_dim": int(X.shape[1]),
        "feature_blocks": ["left_marker", "right_marker", "lr_absdiff", "eef_action", "joint_action"],
        "task_to_id": TASK_TO_ID,
        "reason_names": REASON_NAMES,
        "insertion_dir": str(args.insertion_dir),
        "board_meta": board_meta,
        "counts": {
            "n": int(len(X)),
            "task_counts": {str(k): int(v) for k, v in Counter(task.tolist()).items()},
            "reason_counts": {str(k): int(v) for k, v in Counter(reason.tolist()).items()},
            "binary_counts_with_neutral": {str(k): int(v) for k, v in Counter(binary.tolist()).items()},
            "n_groups": int(len(np.unique(groups))),
        },
        "outputs": {"npz": str(npz_path), "csv": str(csv_path)},
    }
    meta_path = out_dir / "manual_board_tac_quality_features_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"npz": str(npz_path), "meta": str(meta_path), **meta["counts"]}, ensure_ascii=False, indent=2))
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--insertion_dir", type=Path, default=INSERTION_DIR)
    parser.add_argument("--board_positive_dir", type=Path, default=BOARD_POS_DIR)
    parser.add_argument("--board_negative_light_dir", type=Path, default=BOARD_NEG_LIGHT_DIR)
    parser.add_argument("--output_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--insertion_window", type=int, default=8)
    parser.add_argument("--max_insertion_per_episode", type=int, default=260)
    parser.add_argument("--board_window", type=int, default=32)
    parser.add_argument("--board_stride", type=int, default=32)
    parser.add_argument("--board_contact_quantile", type=float, default=0.60)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
