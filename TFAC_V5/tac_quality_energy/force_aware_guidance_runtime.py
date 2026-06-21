"""Force-aware Foresight score adapter for board DP guidance.

This runtime is separate from the marker-only TacQuality bridge.  It scores
action chunks directly through the force-aware Foresight heads:

    action/state chunk -> future force/contact prediction -> quality energy

The score stays differentiable with respect to the candidate action chunk and
is meant for bounded trust-region guidance, not candidate reranking.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F

from TFAC_V5.pretrain_latent_foresight_multistep_force import ForceAwareMultiStepForesightModel
from TFAC_V5.tac_quality_energy.trust_region import (
    TacQualityTrustRegionRefiner,
    TrustRegionConfig,
    summarize_tensor,
)


DEFAULT_FORCE_AWARE_DIR = Path(
    "/home/chenshuai/Project/output/foresight_ckpt/"
    "latent_foresight_board_forceaware_multistep16_boardvae_e100_bs16_0"
)
DEFAULT_FORCE_AWARE_CKPT = DEFAULT_FORCE_AWARE_DIR / "foresight_force_best.ckpt"


@dataclass(frozen=True)
class ForceAwareScoreWeights:
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


def freeze(module: torch.nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def _action_smoothness(action: torch.Tensor) -> torch.Tensor:
    if action.shape[1] < 3:
        return torch.zeros(action.shape[0], dtype=action.dtype, device=action.device)
    accel = action[:, 2:] - 2.0 * action[:, 1:-1] + action[:, :-2]
    return torch.linalg.norm(accel, dim=-1).mean(dim=1)


def _score_components(
    out: Mapping[str, torch.Tensor],
    action: Optional[torch.Tensor],
    weights: ForceAwareScoreWeights,
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
        score = score - weights.action_smooth * _action_smoothness(action)
    return {
        "score": score,
        "band_margin": band_margin.mean(dim=1),
        "good_prob": prob[..., 1].mean(dim=1),
        "risk_prob": (prob[..., 0] + prob[..., 2] + prob[..., 3]).mean(dim=1),
        "contact_prob": contact_prob.mean(dim=1),
        "force_center_penalty": force_center_pen.mean(dim=1),
        "force_smooth_penalty": smooth_pen.mean(dim=1),
    }


class ForceAwareForesightGuidanceRuntime(torch.nn.Module):
    """Load force-aware Foresight and expose a differentiable board quality score."""

    def __init__(
        self,
        ckpt_path: str | Path = DEFAULT_FORCE_AWARE_CKPT,
        *,
        foresight_dir: str | Path | None = None,
        device: str | torch.device = "cuda:0",
        weights: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        self.ckpt_path = Path(ckpt_path)
        self.foresight_dir = Path(foresight_dir) if foresight_dir is not None else self.ckpt_path.parent
        self.device_name = str(device)
        self.config = load_json(self.foresight_dir / "args.json")
        self.stats = load_pickle(self.foresight_dir / "dataset_stats.pkl")
        meta = self.config.get("meta", self.stats["meta"])
        action_dim = int(meta["state_dim"] if self.config.get("use_state_trajectory", True) else meta["action_dim"])
        model = ForceAwareMultiStepForesightModel(
            state_dim=int(meta["state_dim"]),
            action_dim=action_dim,
            hidden_dim=int(self.config.get("hidden_dim", 512)),
            foresight_layers=int(self.config.get("foresight_layers", 3)),
            foresight_nheads=int(self.config.get("foresight_nheads", 8)),
            foresight_dim_feedforward=int(self.config.get("foresight_dim_feedforward", 2048)),
            dropout=float(self.config.get("dropout", 0.1)),
            tactile_vae_ckpt=self.config.get("tactile_vae_ckpt"),
            tactile_vae_latent_dim=int(self.config.get("tactile_vae_latent_dim", 16)),
            predict_horizon=int(self.config.get("predict_horizon", 16)),
            tactile_vae_window=int(self.config.get("tactile_vae_window", 8)),
        ).to(device)
        ckpt = torch.load(self.ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[force-aware-guidance] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
        freeze(model)
        self.model = model
        self.weights = ForceAwareScoreWeights(**dict(weights or {}))
        norm = self.stats["norm_stats"]
        self.register_buffer("qpos_mean", torch.as_tensor(norm["qpos_mean"], dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("qpos_std", torch.as_tensor(norm["qpos_std"], dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("marker_mean", torch.as_tensor(norm["marker_offset_mean"], dtype=torch.float32, device=device).view(1, 1, 1, 1, 2), persistent=False)
        self.register_buffer("marker_std", torch.as_tensor(norm["marker_offset_std"], dtype=torch.float32, device=device).view(1, 1, 1, 1, 2), persistent=False)
        self.use_state_trajectory = bool(self.config.get("use_state_trajectory", True))

    @property
    def horizon(self) -> int:
        return int(self.config.get("predict_horizon", 16))

    @property
    def chunk_size(self) -> int:
        return int(self.config.get("chunk_size", 16))

    @property
    def tactile_window(self) -> int:
        return int(self.config.get("tactile_vae_window", 8))

    def normalize_qpos(self, qpos_raw: torch.Tensor) -> torch.Tensor:
        return (qpos_raw - self.qpos_mean.view(1, -1)) / self.qpos_std.view(1, -1).clamp_min(1e-8)

    def normalize_action_as_qpos(self, action_raw: torch.Tensor) -> torch.Tensor:
        return (action_raw - self.qpos_mean.view(1, 1, -1)) / self.qpos_std.view(1, 1, -1).clamp_min(1e-8)

    def normalize_marker(self, marker_window_raw: torch.Tensor) -> torch.Tensor:
        return (marker_window_raw - self.marker_mean) / self.marker_std.clamp_min(1e-8)

    def _prepare_inputs(
        self,
        action_raw: torch.Tensor,
        qpos_raw: torch.Tensor,
        marker_window_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = action_raw.shape[0]
        qpos = qpos_raw.to(action_raw.device, action_raw.dtype).view(1, -1)
        if qpos.shape[0] == 1 and batch > 1:
            qpos = qpos.expand(batch, -1)
        marker = marker_window_raw.to(action_raw.device, action_raw.dtype)
        if marker.shape[0] == 1 and batch > 1:
            marker = marker.expand(batch, *marker.shape[1:])
        action_chunk = action_raw[:, : self.chunk_size, :]
        return self.normalize_marker(marker), self.normalize_qpos(qpos), self.normalize_action_as_qpos(action_chunk)

    def forward_score(
        self,
        action_raw: torch.Tensor,
        *,
        qpos_raw: torch.Tensor,
        marker_window_raw: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        marker_norm, qpos_norm, action_norm = self._prepare_inputs(action_raw, qpos_raw, marker_window_raw)
        out = self.model(marker_norm, qpos_norm, action_norm)
        components = _score_components(out, action_norm, self.weights)
        return components if return_components else components["score"]

    def summary(self) -> Dict[str, Any]:
        return {
            "runtime": type(self).__name__,
            "ckpt_path": str(self.ckpt_path),
            "foresight_dir": str(self.foresight_dir),
            "score_weights": asdict(self.weights),
            "chunk_size": self.chunk_size,
            "horizon": self.horizon,
            "tactile_window": self.tactile_window,
            "use_state_trajectory": self.use_state_trajectory,
            "force_ref": self.stats.get("force_ref", {}),
        }


class ForceAwareBoardGuidanceAdapter:
    """Trust-region action refiner around ForceAwareForesightGuidanceRuntime."""

    def __init__(
        self,
        runtime: ForceAwareForesightGuidanceRuntime,
        refiner_config: TrustRegionConfig,
        *,
        action_normalizer,
    ):
        self.runtime = runtime
        self.refiner = TacQualityTrustRegionRefiner(refiner_config)
        self.action_normalizer = action_normalizer

    def guide_final_action(
        self,
        action_norm: torch.Tensor,
        *,
        qpos_raw: torch.Tensor,
        marker_window_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        action_raw = self.action_normalizer.denormalize(action_norm).detach()

        def score_fn(candidate_raw: torch.Tensor) -> torch.Tensor:
            return self.runtime.forward_score(
                candidate_raw,
                qpos_raw=qpos_raw,
                marker_window_raw=marker_window_raw,
            )

        guided_raw, report = self.refiner.refine(action_raw, score_fn)
        guided_norm = self.action_normalizer.normalize(guided_raw)
        with torch.no_grad():
            base_comp = self.runtime.forward_score(
                action_raw,
                qpos_raw=qpos_raw,
                marker_window_raw=marker_window_raw,
                return_components=True,
            )
            final_comp = self.runtime.forward_score(
                guided_raw,
                qpos_raw=qpos_raw,
                marker_window_raw=marker_window_raw,
                return_components=True,
            )
        report.update(
            {
                "task": "board",
                "score_mode": "force_aware_quality",
                "adapter_policy": "force_aware_foresight_trust_region_refinement",
                "scorer_runtime": type(self.runtime).__name__,
                "raw_action_delta": summarize_tensor((guided_raw - action_raw).flatten(1).norm(dim=1)),
                "normalized_action_delta": summarize_tensor((guided_norm - action_norm).flatten(1).norm(dim=1)),
                "action_normalizer": self.action_normalizer.summary(),
                "force_aware_runtime": self.runtime.summary(),
                "base_components": {k: summarize_tensor(v) for k, v in base_comp.items()},
                "final_components": {k: summarize_tensor(v) for k, v in final_comp.items()},
                "integration_contract": {
                    "score_input": "raw action tensor + current qpos + raw marker history",
                    "score_output": "future force/contact quality energy from force-aware Foresight heads",
                    "guidance_location": "after DP denoising has produced clean/final action",
                    "reranking": False,
                    "every_step_ddpm_guidance": False,
                    "recompute_foresight_each_guidance_call": True,
                },
            }
        )
        return guided_norm.detach(), report
