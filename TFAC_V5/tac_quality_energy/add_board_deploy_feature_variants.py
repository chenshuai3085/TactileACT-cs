#!/usr/bin/env python3
"""Add deploy-contract board feature variants to an existing feature cache.

The original predicted-domain cache included ``marker_action`` features built
from predicted marker + joint action proxy + dataset ``eef_abs`` proxy.  The
serving-time guidance contract only guarantees the DP joint action chunk and
the Foresight-predicted marker.  This utility adds joint-only variants without
rerunning the expensive Foresight feature builder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def add_variants(input_path: Path, output_path: Path) -> dict:
    data = np.load(input_path, allow_pickle=True)
    payload = {key: data[key] for key in data.files}

    if "marker_action" not in payload or "left_marker_action" not in payload:
        raise KeyError("Expected marker_action and left_marker_action in cache")

    marker_action = payload["marker_action"].astype(np.float32)
    left_marker_action = payload["left_marker_action"].astype(np.float32)
    gt_marker_action = payload.get("gt_marker_action")

    # marker_action = pred_both(54) + joint_proxy(10) + eef_proxy(10)
    # left_marker_action = pred_left(18) + joint_proxy(10) + eef_proxy(10)
    payload["marker_joint_action"] = marker_action[:, :64].astype(np.float32)
    payload["left_marker_joint_action"] = left_marker_action[:, :28].astype(np.float32)
    if gt_marker_action is not None:
        payload["gt_marker_joint_action"] = gt_marker_action[:, :64].astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    result = {
        "input": str(input_path),
        "output": str(output_path),
        "added": {
            "marker_joint_action": list(payload["marker_joint_action"].shape),
            "left_marker_joint_action": list(payload["left_marker_joint_action"].shape),
            "gt_marker_joint_action": list(payload["gt_marker_joint_action"].shape)
            if "gt_marker_joint_action" in payload
            else None,
        },
        "contract": "predicted marker proxy + joint action proxy only; no dataset eef_abs proxy",
    }
    (output_path.parent / "add_deploy_feature_variants_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/home/chenshuai/Project/output/board_predicted_domain_force_band_features_20260618/"
            "board_predicted_domain_force_band_features.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/chenshuai/Project/output/board_predicted_domain_force_band_features_20260618_deploy/"
            "board_predicted_domain_force_band_features_deploy.npz"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    add_variants(parse_args().input, parse_args().output)
