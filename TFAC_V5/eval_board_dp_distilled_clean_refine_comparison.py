"""Compare PTGProxyV2 and distilled TacQualityEnergy in board DP full-chain.

This reuses the board DP -> production Foresight -> scorer code path from
eval_board_dp_denoising_full_chain_smoke.py and runs clean-action refinement
with two scorer runtimes on the same frames and random seeds.

Scope: production-like smoke chain with available board DP/Foresight checkpoints.
It is stronger than surrogate guidance, but still not a robot rollout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.distilled_tac_quality_energy_runtime import DistilledTacQualityEnergyRuntime  # noqa: E402
from TFAC_V5.eval_board_dp_denoising_full_chain_smoke import (  # noqa: E402
    BoardDPForesightChain,
    action_smoothness_np,
    episode_files,
)
from TFAC_V5.eval_tac_energy_guided_denoising import summarize  # noqa: E402
from TFAC_V5.ptg_proxy_scorer_v2_runtime import PTGProxyScorerV2Runtime, TASK_TO_ID  # noqa: E402
from TFAC_V5.tac_quality_guidance_config import get_guidance_profile  # noqa: E402


OUT_DIR = Path("/home/chenshuai/Project/output/board_dp_distilled_clean_refine_comparison")


class ComparableBoardDPForesightChain(BoardDPForesightChain):
    def set_scorer(self, scorer, scorer_name: str) -> None:
        self.scorer = scorer
        self.scorer_name = scorer_name
        self.scorer.eval()
        for p in self.scorer.parameters():
            p.requires_grad_(False)

    def score_norm_actions(self, f, t, action_norm, args):
        batch = action_norm.shape[0]
        action_raw = self.dp_unnorm_action(action_norm)
        fs_images, qpos_norm = self.build_foresight_inputs(f, t, batch)
        fs_action = action_raw[:, : self.foresight_chunk, :]
        fs_action_norm = self.fs_norm_action(fs_action)
        z_pred, _, _, _, _, _ = self.foresight(fs_images, fs_action_norm, future_images=None, qpos=qpos_norm)
        if z_pred.dim() == 3:
            z_pred = z_pred[:, -1]
        c = int(self.foresight_config.get("tactile_vae_latent_dim", 16))
        marker = self.foresight.tactile_vae.decoder(z_pred.reshape(batch, c, 3, 3))
        marker_seq = marker.unsqueeze(1).expand(-1, args.score_window, -1, -1, -1)
        task_id = torch.full((batch,), TASK_TO_ID["board"], dtype=torch.long, device=self.device)
        action_seq = action_raw[:, : args.score_window, :]
        if self.scorer_name == "ptg_proxy_v2":
            score = self.scorer.weighted_energy_score(
                marker_seq,
                right_marker_seq=marker_seq,
                joint_action_seq=action_seq,
                task_id=task_id,
                quality_weight=args.quality_weight,
                binary_weight=args.binary_weight,
                reason_weight=args.reason_weight,
                clip=not args.no_clip,
            )
        elif self.scorer_name == "distilled_energy":
            score = self.scorer.score(
                marker_seq,
                right_marker_seq=marker_seq,
                joint_action_seq=action_seq,
                task_id=task_id,
                mode="energy_clipped" if not args.no_clip else "energy",
            )
        else:
            raise ValueError(self.scorer_name)
        if args.smooth_weight and action_raw.shape[1] >= 3:
            accel = action_raw[:, 2:] - 2 * action_raw[:, 1:-1] + action_raw[:, :-2]
            score = score - args.smooth_weight * torch.linalg.norm(accel, dim=-1).mean(dim=1)
        return score, action_raw, marker, z_pred


def collect_frames(chain, args):
    rng = np.random.default_rng(args.seed)
    files = episode_files(Path(args.data_dir))
    if args.n_episodes and len(files) > args.n_episodes:
        files = [files[i] for i in sorted(rng.choice(len(files), args.n_episodes, replace=False))]
    frames = []
    for path in files:
        with h5py.File(path, "r") as f:
            T = len(f["observations/proprio_joint"])
            min_t = chain.obs_horizon - 1
            max_t = T - chain.pred_horizon - 1
            if max_t <= min_t:
                continue
            frame_indices = np.linspace(min_t, max_t, min(args.frames_per_episode, max_t - min_t + 1), dtype=int)
            for t in frame_indices:
                frames.append((path, int(t)))
                if args.n_eval and len(frames) >= args.n_eval:
                    return frames
    return frames


def run_one(chain, args, scorer_name: str, scorer, frames):
    chain.set_scorer(scorer, scorer_name)
    accum = {k: [] for k in ["base_score", "guided_score", "score_delta", "base_smooth", "guided_smooth", "norm_delta", "range_violation"]}
    guide_grad_norms = []
    guide_accept_rates = []
    rows = []
    for path, t in tqdm(frames, desc=f"{scorer_name} board DP clean-refine"):
        with h5py.File(path, "r") as f:
            torch.manual_seed(args.seed + len(rows))
            obs_cond = chain.build_obs_cond(f, int(t))
            init = torch.randn(args.K, chain.pred_horizon, chain.action_dim, device=chain.device)
            base_norm, _ = chain.denoise(f, int(t), obs_cond, init, args, guided=False)
            guided_norm, logs = chain.refine_clean_action(f, int(t), base_norm, args)
            with torch.no_grad():
                base_score, base_raw, _, _ = chain.score_norm_actions(f, int(t), base_norm, args)
                guided_score, guided_raw, _, _ = chain.score_norm_actions(f, int(t), guided_norm, args)
            base_np = base_raw.detach().cpu().numpy()
            guided_np = guided_raw.detach().cpu().numpy()
            base_score_np = base_score.detach().cpu().numpy()
            guided_score_np = guided_score.detach().cpu().numpy()
            score_delta = guided_score_np - base_score_np
            norm_delta = torch.linalg.norm((guided_norm - base_norm).flatten(1), dim=1).detach().cpu().numpy()
            range_violation = torch.clamp(guided_norm.abs() - 1.0, min=0.0).amax(dim=(1, 2)).detach().cpu().numpy()
            base_smooth = action_smoothness_np(base_np)
            guided_smooth = action_smoothness_np(guided_np)
            accum["base_score"].append(base_score_np)
            accum["guided_score"].append(guided_score_np)
            accum["score_delta"].append(score_delta)
            accum["base_smooth"].append(base_smooth)
            accum["guided_smooth"].append(guided_smooth)
            accum["norm_delta"].append(norm_delta)
            accum["range_violation"].append(range_violation)
            guide_grad_norms.extend([x["grad_norm_mean"] for x in logs])
            guide_accept_rates.extend([x["accept_rate"] for x in logs])
            rows.append(
                {
                    "episode": path.name,
                    "t": int(t),
                    "base_score_mean": float(base_score_np.mean()),
                    "guided_score_mean": float(guided_score_np.mean()),
                    "score_delta_mean": float(score_delta.mean()),
                    "guided_beats_base_rate": float(np.mean(guided_score_np > base_score_np)),
                    "norm_delta_mean": float(norm_delta.mean()),
                    "range_violation_max": float(range_violation.max()),
                    "n_guidance_steps": len(logs),
                }
            )
    if not rows:
        raise RuntimeError(f"No rows for {scorer_name}")
    merged = {k: np.concatenate(v, axis=0) for k, v in accum.items()}
    score_improved_rate = float(np.mean(merged["score_delta"] > 0))
    return {
        "scorer": scorer_name,
        "n_frames": len(rows),
        "n_action_samples": int(len(merged["score_delta"])),
        "summary": {
            "base_score": summarize(merged["base_score"]),
            "guided_score": summarize(merged["guided_score"]),
            "score_delta": summarize(merged["score_delta"]),
            "guided_beats_base_rate": score_improved_rate,
            "base_smoothness": summarize(merged["base_smooth"]),
            "guided_smoothness": summarize(merged["guided_smooth"]),
            "smoothness_delta": summarize(merged["guided_smooth"] - merged["base_smooth"]),
            "norm_action_delta": summarize(merged["norm_delta"]),
            "range_violation": summarize(merged["range_violation"]),
            "guide_grad_norm_mean_per_step": summarize(guide_grad_norms),
            "guide_accept_rate_per_step": summarize(guide_accept_rates),
        },
        "passes_board_dp_clean_refine_smoke": bool(
            score_improved_rate > args.pass_improved_rate
            and float(np.mean(merged["score_delta"])) > 0
            and float(np.max(merged["range_violation"])) <= 1e-6
        ),
        "rows": rows,
    }


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Board DP Distilled Clean-Refine Comparison",
        "",
        "Production-like board DP/Foresight smoke-chain comparison. This is not a robot rollout.",
        "",
        f"- overall_pass: `{result['overall_pass']}`",
        f"- n_frames: `{result['n_frames']}`",
        "",
        "| scorer | pass | improved | score delta mean | smooth delta mean | norm delta p95 | range violation max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ["ptg_proxy_v2", "distilled_energy"]:
        row = result[key]
        s = row["summary"]
        lines.append(
            f"| {key} | {row['passes_board_dp_clean_refine_smoke']} | {s['guided_beats_base_rate']:.4f} | "
            f"{s['score_delta']['mean']:.6f} | {s['smoothness_delta']['mean']:.6f} | "
            f"{s['norm_action_delta']['p95']:.6f} | {s['range_violation']['max']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Scores are scorer-internal and should not be compared by absolute value across scorers.",
            "- Compare pass/fail, improved rate, action delta, and smoothness side effects.",
            "- This smoke chain is stronger than surrogate guidance but weaker than production rollout.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    profile = get_guidance_profile("board")
    if args.quality_weight is None:
        args.quality_weight = profile.energy.quality
    if args.binary_weight is None:
        args.binary_weight = profile.energy.binary_margin
    if args.reason_weight is None:
        args.reason_weight = profile.energy.reason_margin

    chain = ComparableBoardDPForesightChain(args)
    frames = collect_frames(chain, args)
    if not frames:
        raise RuntimeError("No valid board frames evaluated")
    ptg = PTGProxyScorerV2Runtime(args.ptg_ckpt, device=str(chain.device))
    distilled = DistilledTacQualityEnergyRuntime(args.distilled_ckpt, device=str(chain.device))
    ptg_result = run_one(chain, args, "ptg_proxy_v2", ptg, frames)
    distilled_result = run_one(chain, args, "distilled_energy", distilled, frames)
    result = {
        "purpose": "Compare PTGProxyV2 and distilled TacQualityEnergy in board DP/Foresight clean-action refinement.",
        "scope": "Production-like smoke chain with available board DP/Foresight checkpoints; not robot rollout.",
        "config": vars(args),
        "chain": {
            "dp_variant": chain.dp_variant,
            "dp_uses_feature_cache": chain.dp_uses_feature_cache,
            "dp_weight_source": chain.dp_weight_source,
            "dp_feature_encoder_source": chain.dp_feature_encoder_source,
        },
        "n_frames": len(frames),
        "ptg_proxy_v2": ptg_result,
        "distilled_energy": distilled_result,
        "overall_pass": bool(
            ptg_result["passes_board_dp_clean_refine_smoke"]
            and distilled_result["passes_board_dp_clean_refine_smoke"]
        ),
        "recommendation": (
            "Distilled energy passes the board DP/Foresight clean-refine smoke gate; keep as ablation candidate."
            if distilled_result["passes_board_dp_clean_refine_smoke"]
            else "Distilled energy should not replace PTGProxyV2 in board DP clean-refine."
        ),
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "board_dp_distilled_clean_refine_comparison.json"
    md_path = out_dir / "board_dp_distilled_clean_refine_comparison.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "overall_pass": result["overall_pass"],
                "ptg": {
                    "pass": ptg_result["passes_board_dp_clean_refine_smoke"],
                    "improved": ptg_result["summary"]["guided_beats_base_rate"],
                    "delta": ptg_result["summary"]["score_delta"]["mean"],
                },
                "distilled": {
                    "pass": distilled_result["passes_board_dp_clean_refine_smoke"],
                    "improved": distilled_result["summary"]["guided_beats_base_rate"],
                    "delta": distilled_result["summary"]["score_delta"]["mean"],
                },
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="/home/chenshuai/data/dataset/260522_v8l_caheiban_flat_smoke4")
    parser.add_argument("--dp_config", default="/home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_smoke4/config.json")
    parser.add_argument("--dp_ckpt", default="/home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_smoke4/dp_final.pth")
    parser.add_argument("--foresight_dir", default="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_smoke_0")
    parser.add_argument("--foresight_ckpt", default="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_smoke_0/foresight_best.ckpt")
    parser.add_argument("--scorer_ckpt", default="/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_final.pt")
    parser.add_argument("--ptg_ckpt", default="/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_final.pt")
    parser.add_argument("--distilled_ckpt", default="/home/chenshuai/Project/output/distilled_tac_quality_energy/distilled_tac_quality_energy_final.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--n_episodes", type=int, default=2)
    parser.add_argument("--frames_per_episode", type=int, default=2)
    parser.add_argument("--n_eval", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", default="clean_refine", choices=["clean_refine"])
    parser.add_argument("--score_window", type=int, default=8)
    parser.add_argument("--quality_weight", type=float, default=None)
    parser.add_argument("--binary_weight", type=float, default=None)
    parser.add_argument("--reason_weight", type=float, default=None)
    parser.add_argument("--smooth_weight", type=float, default=0.02)
    parser.add_argument("--no_clip", action="store_true")
    parser.add_argument("--guide_start_frac", type=float, default=0.6)
    parser.add_argument("--guide_every", type=int, default=1)
    parser.add_argument("--guidance_scale", type=float, default=0.002)
    parser.add_argument("--max_norm_delta_per_step", type=float, default=0.01)
    parser.add_argument("--refine_steps", type=int, default=4)
    parser.add_argument("--clean_refine_scale", type=float, default=0.01)
    parser.add_argument("--max_clean_norm_delta", type=float, default=0.08)
    parser.add_argument("--accept_only_improved", action="store_true", default=True)
    parser.add_argument("--clamp_norm_action", action="store_true", default=True)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--pass_improved_rate", type=float, default=0.75)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
