"""Server-side rollout logging for real robot board-wiping tests.

The deployment server receives the real observation sent by ``ws_client.py`` and
knows the action it returns.  This logger records one directory per rollout so
force-curve evaluation can run even when logs are collected only on the GPU
server side.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def rollout_group_name(arm: str) -> str:
    lowered = str(arm).lower()
    if "baseline" in lowered or lowered in {"base", "no_guidance"}:
        return "baseline"
    if "guided" in lowered or lowered in {"ptg", "ptg_guided"}:
        return "guided"
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(arm))


class ServerRolloutLogger:
    """Record one server-side robot rollout.

    The CSV schema intentionally keeps the force column names used by
    ``eval_board_force_rollouts.py`` and adds qpos/eef/action columns for
    trajectory inspection.
    """

    def __init__(
        self,
        root_dir: str | Path,
        *,
        episode: int,
        host: str,
        port: int,
        task: str,
        arm: str,
        server_metadata: Mapping[str, Any],
    ) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        group = rollout_group_name(arm)
        self.root = Path(root_dir).expanduser()
        self.trial_dir = self.root / group / f"{ts}_port{int(port)}_episode{int(episode):04d}"
        self.trial_dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.records: list[dict[str, float | int]] = []
        self.metadata: dict[str, Any] = {
            "created_at": ts,
            "log_side": "server",
            "episode": int(episode),
            "trial": int(episode),
            "host": str(host),
            "port": int(port),
            "task": str(task),
            "arm": str(arm),
            "group": group,
            "server_metadata": _jsonable(dict(server_metadata)),
            "schema": {
                "ft": "robot wrist force/torque if present; first 3 dims are Fx,Fy,Fz",
                "left_force6d": "left tactile force6d if present",
                "right_force6d": "right tactile force6d if present",
                "qpos": "robot proprioception sent by client",
                "eef": "end-effector pose sent by client if present",
                "action": "server action returned to client at this step",
            },
        }

    @staticmethod
    def _vec(value: Any, n: int, fill: float = float("nan")) -> list[float]:
        if value is None:
            return [fill] * n
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        out = [float(x) for x in arr[:n]]
        if len(out) < n:
            out.extend([fill] * (n - len(out)))
        return out

    @staticmethod
    def _mag3(values: list[float]) -> float:
        arr = np.asarray(values[:3], dtype=np.float64)
        if not np.isfinite(arr).all():
            return float("nan")
        return float(np.linalg.norm(arr))

    @staticmethod
    def _tac_side(obs: Mapping[str, Any], side: str) -> Mapping[str, Any]:
        tac = obs.get("tac", {})
        if not isinstance(tac, Mapping):
            return {}
        side_data = tac.get(side, {})
        return side_data if isinstance(side_data, Mapping) else {}

    @staticmethod
    def _marker_proxy(side_data: Mapping[str, Any]) -> dict[str, float]:
        marker = side_data.get("marker_offset")
        if marker is None:
            return {}
        arr = np.asarray(marker, dtype=np.float32)
        if arr.ndim < 1:
            return {}
        mag = np.linalg.norm(arr.reshape(-1, arr.shape[-1]), axis=-1)
        finite = mag[np.isfinite(mag)]
        if len(finite) == 0:
            return {}
        return {
            "marker_mag_mean": float(finite.mean()),
            "marker_mag_max": float(finite.max()),
            "marker_contact_area": float((finite > 1e-6).mean()),
        }

    @staticmethod
    def _report_scalars(report: Mapping[str, Any] | None) -> dict[str, float | int]:
        if not isinstance(report, Mapping):
            return {}
        out: dict[str, float | int] = {}
        for key, value in report.items():
            if isinstance(value, (bool, np.bool_)):
                out[f"guidance_{key}"] = int(value)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                v = float(value)
                if np.isfinite(v):
                    out[f"guidance_{key}"] = v
        return out

    def record(
        self,
        *,
        step: int,
        obs: Mapping[str, Any],
        action: Any | None,
        action_norm: Any | None = None,
        guidance_report: Mapping[str, Any] | None = None,
    ) -> None:
        ft = self._vec(obs.get("ft"), 6)
        left_data = self._tac_side(obs, "left")
        right_data = self._tac_side(obs, "right")
        left = self._vec(left_data.get("force6d"), 6)
        right = self._vec(right_data.get("force6d"), 6)

        now = time.time()
        row: dict[str, float | int] = {
            "step": int(step),
            "wall_time": float(now),
            "t": float(now - self.t0),
            "ft_fx": ft[0],
            "ft_fy": ft[1],
            "ft_fz": ft[2],
            "ft_tx": ft[3],
            "ft_ty": ft[4],
            "ft_tz": ft[5],
            "ft_f_mag": self._mag3(ft),
            "left_fx": left[0],
            "left_fy": left[1],
            "left_fz": left[2],
            "left_tx": left[3],
            "left_ty": left[4],
            "left_tz": left[5],
            "left_f_mag": self._mag3(left),
            "right_fx": right[0],
            "right_fy": right[1],
            "right_fz": right[2],
            "right_tx": right[3],
            "right_ty": right[4],
            "right_tz": right[5],
            "right_f_mag": self._mag3(right),
        }

        for prefix, values in [
            ("qpos", self._vec(obs.get("qpos"), 16)),
            ("eef", self._vec(obs.get("eef"), 16)),
            ("action", self._vec(action, 32)),
            ("action_norm", self._vec(action_norm, 32)),
        ]:
            for i, value in enumerate(values):
                if np.isfinite(value):
                    row[f"{prefix}_{i}"] = value

        for side_name, side_data in [("left", left_data), ("right", right_data)]:
            for key, value in self._marker_proxy(side_data).items():
                row[f"{side_name}_{key}"] = value

        row.update(self._report_scalars(guidance_report))
        self.records.append(row)

    @staticmethod
    def _finite_array(records: list[dict[str, Any]], key: str) -> np.ndarray:
        vals = np.array([r.get(key, np.nan) for r in records], dtype=np.float64)
        return vals[np.isfinite(vals)]

    def _summarize(self) -> dict[str, Any]:
        summary: dict[str, Any] = {"n_samples": int(len(self.records))}
        for key in ("ft_fz", "ft_f_mag", "left_fz", "left_f_mag", "right_fz", "right_f_mag"):
            vals = self._finite_array(self.records, key)
            if len(vals):
                d = np.diff(vals) if len(vals) > 1 else np.array([], dtype=np.float64)
                summary[key] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "p05": float(np.percentile(vals, 5)),
                    "p50": float(np.percentile(vals, 50)),
                    "p95": float(np.percentile(vals, 95)),
                    "delta_abs_mean": float(np.abs(d).mean()) if len(d) else 0.0,
                    "delta_abs_p95": float(np.percentile(np.abs(d), 95)) if len(d) else 0.0,
                }
        return summary

    def _write_csv(self) -> Path:
        csv_path = self.trial_dir / "force_trace.csv"
        keys = sorted({k for row in self.records for k in row})
        preferred = [
            "step", "t", "wall_time",
            "ft_fx", "ft_fy", "ft_fz", "ft_f_mag",
            "left_fx", "left_fy", "left_fz", "left_f_mag",
            "right_fx", "right_fy", "right_fz", "right_f_mag",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=preferred + [k for k in keys if k not in preferred])
            writer.writeheader()
            writer.writerows(self.records)
        return csv_path

    def _write_npz(self) -> Path:
        npz_path = self.trial_dir / "force_trace.npz"
        arrays = {}
        if self.records:
            keys = sorted({k for row in self.records for k in row})
            for key in keys:
                arrays[key] = np.array([row.get(key, np.nan) for row in self.records], dtype=np.float32)
        np.savez_compressed(npz_path, **arrays)
        return npz_path

    def _write_plot(self) -> str | None:
        if not self.records:
            return None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            logging.warning("[server-rollout-log] matplotlib unavailable, skip plot: %s", exc)
            return None

        t = np.array([r["t"] for r in self.records], dtype=np.float64)
        plot_path = self.trial_dir / "force_curve.png"
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for key, label in [
            ("ft_fz", "robot ft Fz"),
            ("left_fz", "left tactile Fz"),
            ("right_fz", "right tactile Fz"),
        ]:
            y = np.array([r.get(key, np.nan) for r in self.records], dtype=np.float64)
            if np.isfinite(y).any():
                axes[0].plot(t, y, label=label, linewidth=1.5)
        axes[0].set_ylabel("Fz")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        for key, label in [
            ("ft_f_mag", "robot |Fxyz|"),
            ("left_f_mag", "left tactile |Fxyz|"),
            ("right_f_mag", "right tactile |Fxyz|"),
        ]:
            y = np.array([r.get(key, np.nan) for r in self.records], dtype=np.float64)
            if np.isfinite(y).any():
                axes[1].plot(t, y, label=label, linewidth=1.5)
        axes[1].set_xlabel("time (s)")
        axes[1].set_ylabel("|Fxyz|")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        return str(plot_path)

    def finalize(self, *, steps: int, stop_reason: str) -> dict[str, Any]:
        self.metadata["steps"] = int(steps)
        self.metadata["stop_reason"] = str(stop_reason)
        self.metadata["summary"] = self._summarize()
        csv_path = self._write_csv()
        npz_path = self._write_npz()
        plot_path = self._write_plot()
        self.metadata["artifacts"] = {
            "force_trace_csv": str(csv_path),
            "force_trace_npz": str(npz_path),
            "force_curve_png": plot_path,
        }
        meta_path = self.trial_dir / "metadata.json"
        meta_path.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("[server-rollout-log] saved rollout to %s", self.trial_dir)
        return self.metadata
