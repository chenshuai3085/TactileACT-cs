"""
TFM 预测质量分析脚本。

评估维度：
1. 逐 horizon 分析 (h=4, 8, 12)
2. Mask vs No-mask 对比（纯视觉 vs 视觉+触觉）
3. Naive baseline 对比（直接复制当前触觉作为预测）
4. Horizon × Mask/No-mask 交叉分析

Usage:
    cd /path/to/TactileACT-cs
    python -m tactile_foresight.evaluation.analyze_tfm \
        --checkpoint /home/chenshuai/Project/output/tfm_xiaomi/tfm_best.pth \
        --config /home/chenshuai/Project/output/tfm_xiaomi/tfm_config.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tactile_foresight.models.tfm import TactileForesightModel


def parse_args():
    p = argparse.ArgumentParser(description="TFM 预测质量分析")
    p.add_argument("--checkpoint", type=str,
                    default="/home/chenshuai/Project/output/tfm_xiaomi/tfm_best.pth")
    p.add_argument("--config", type=str,
                    default="/home/chenshuai/Project/output/tfm_xiaomi/tfm_config.json")
    p.add_argument("--output_dir", type=str, default=None,
                    help="结果保存目录，默认与 checkpoint 同目录")
    p.add_argument("--samples_per_episode", type=int, default=50,
                    help="每条验证轨迹采样的时间步数（越多越稳定）")
    p.add_argument("--batch_size", type=int, default=256)
    return p.parse_args()

def load_val_episodes(config: dict):
    """复现训练时的 train/val split，返回 val episode ids。"""
    start = config.get("start_episode", 0)
    num = config["num_episodes"]
    seed = config.get("seed", 42)
    episode_ids = list(range(start, start + num))

    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(len(episode_ids))
    split = int(0.8 * len(episode_ids))
    val_ids = [episode_ids[i] for i in shuffled[split:]]
    train_ids = [episode_ids[i] for i in shuffled[:split]]
    return train_ids, val_ids


def load_features(episode_ids, feature_dir):
    """加载指定 episode 的预提取特征。"""
    all_feats = {}
    for eid in episode_ids:
        path = os.path.join(feature_dir, f"episode_{eid}_dino.pt")
        all_feats[eid] = torch.load(path, map_location="cpu")
    return all_feats


def build_eval_samples(all_feats, episode_ids, camera_names, horizons,
                       samples_per_episode, seed=123):
    """为每个 (horizon, masked) 组合构建确定性评估样本。

    返回 dict: (horizon, masked) -> list of (v_feat, t_cur, t_fut)
    """
    rng = np.random.RandomState(seed)
    sample_feat = all_feats[episode_ids[0]]
    ep_len = sample_feat["tactile"].shape[0]
    max_h = max(horizons)
    max_t = ep_len - max_h  # 可采样的最大时间步 (exclusive)

    samples = {}
    for h in horizons:
        for masked in [True, False]:
            samples[(h, masked)] = []

    for eid in episode_ids:
        feats = all_feats[eid]
        timesteps = rng.randint(0, max_t, size=samples_per_episode)
        cam_choices = rng.choice(camera_names, size=samples_per_episode)

        for t, cam in zip(timesteps, cam_choices):
            v = feats[f"vision_{cam}"][t].float()
            t_cur = feats["tactile"][t].float()
            for h in horizons:
                t_fut = feats["tactile"][min(t + h, ep_len - 1)].float()
                for masked in [True, False]:
                    samples[(h, masked)].append((v, t_cur, t_fut))

    return samples

@torch.no_grad()
def evaluate_model(model, samples, device, batch_size=256):
    """对每个 (horizon, masked) 组合计算指标。

    返回 dict: (horizon, masked) -> {cos_sim, mse, count}
    """
    model.pred_head.eval()
    results = {}

    for (h, masked), sample_list in samples.items():
        all_cos = []
        all_mse = []
        all_baseline_cos = []
        all_baseline_mse = []

        # 分 batch 处理
        for i in range(0, len(sample_list), batch_size):
            batch = sample_list[i:i + batch_size]
            v_feat = torch.stack([s[0] for s in batch]).to(device)
            t_cur = torch.stack([s[1] for s in batch]).to(device)
            t_fut = torch.stack([s[2] for s in batch]).to(device)
            B = v_feat.shape[0]

            horizon_t = torch.full((B,), h, dtype=torch.long, device=device)
            mask_t = torch.full((B,), masked, dtype=torch.bool, device=device)

            # 模型预测
            pred = model.predict(v_feat, t_cur, horizon_t, mask_t)

            # 指标
            cos = F.cosine_similarity(pred, t_fut, dim=-1)  # (B,)
            mse = ((pred - t_fut) ** 2).mean(dim=-1)         # (B,)
            all_cos.append(cos.cpu())
            all_mse.append(mse.cpu())

            # Naive baseline: 直接复制当前触觉
            base_cos = F.cosine_similarity(t_cur, t_fut, dim=-1)
            base_mse = ((t_cur - t_fut) ** 2).mean(dim=-1)
            all_baseline_cos.append(base_cos.cpu())
            all_baseline_mse.append(base_mse.cpu())

        all_cos = torch.cat(all_cos)
        all_mse = torch.cat(all_mse)
        all_baseline_cos = torch.cat(all_baseline_cos)
        all_baseline_mse = torch.cat(all_baseline_mse)

        results[(h, masked)] = {
            "cos_sim_mean": all_cos.mean().item(),
            "cos_sim_std": all_cos.std().item(),
            "mse_mean": all_mse.mean().item(),
            "mse_std": all_mse.std().item(),
            "baseline_cos_sim_mean": all_baseline_cos.mean().item(),
            "baseline_cos_sim_std": all_baseline_cos.std().item(),
            "baseline_mse_mean": all_baseline_mse.mean().item(),
            "baseline_mse_std": all_baseline_mse.std().item(),
            "count": len(sample_list),
        }

    return results

def print_results(results, horizons):
    """打印格式化的结果表格。"""
    sep = "=" * 95

    # --- 1. 交叉分析表 ---
    print(f"\n{sep}")
    print("  TFM 预测质量分析 — Horizon × Mask/No-mask 交叉表")
    print(sep)
    print(f"{'Horizon':>8} | {'Mask':>6} | {'CosSim':>12} | {'MSE':>12} | "
          f"{'Baseline CosSim':>16} | {'Baseline MSE':>13}")
    print("-" * 95)

    for h in horizons:
        for masked in [False, True]:
            r = results[(h, masked)]
            mask_str = "有" if not masked else "无"
            print(
                f"  h={h:<4} | 触觉{mask_str} | "
                f"{r['cos_sim_mean']:.4f}±{r['cos_sim_std']:.4f} | "
                f"{r['mse_mean']:.4f}±{r['mse_std']:.4f} | "
                f"{r['baseline_cos_sim_mean']:.4f}±{r['baseline_cos_sim_std']:.4f} | "
                f"{r['baseline_mse_mean']:.4f}±{r['baseline_mse_std']:.4f}"
            )
        if h != horizons[-1]:
            print("-" * 95)

    # --- 2. 逐 horizon 汇总（mask + no-mask 平均）---
    print(f"\n{sep}")
    print("  逐 Horizon 汇总（mask + no-mask 平均）")
    print(sep)
    print(f"{'Horizon':>8} | {'模型 CosSim':>14} | {'模型 MSE':>12} | "
          f"{'Baseline CosSim':>16} | {'Baseline MSE':>13} | {'CosSim 提升':>12}")
    print("-" * 95)

    for h in horizons:
        r_nomask = results[(h, False)]
        r_mask = results[(h, True)]
        avg_cos = (r_nomask["cos_sim_mean"] + r_mask["cos_sim_mean"]) / 2
        avg_mse = (r_nomask["mse_mean"] + r_mask["mse_mean"]) / 2
        avg_base_cos = (r_nomask["baseline_cos_sim_mean"] + r_mask["baseline_cos_sim_mean"]) / 2
        avg_base_mse = (r_nomask["baseline_mse_mean"] + r_mask["baseline_mse_mean"]) / 2
        delta = avg_cos - avg_base_cos
        print(
            f"  h={h:<4} | {avg_cos:>14.4f} | {avg_mse:>12.4f} | "
            f"{avg_base_cos:>16.4f} | {avg_base_mse:>13.4f} | "
            f"{delta:>+12.4f}"
        )

    # --- 3. Mask vs No-mask 汇总（所有 horizon 平均）---
    print(f"\n{sep}")
    print("  Mask vs No-mask 汇总（所有 horizon 平均）")
    print(sep)
    print(f"{'条件':>10} | {'CosSim':>12} | {'MSE':>12} | {'样本数':>8}")
    print("-" * 55)

    for masked in [False, True]:
        cos_vals = [results[(h, masked)]["cos_sim_mean"] for h in horizons]
        mse_vals = [results[(h, masked)]["mse_mean"] for h in horizons]
        counts = [results[(h, masked)]["count"] for h in horizons]
        label = "纯视觉(masked)" if masked else "视觉+触觉"
        print(
            f"  {label:<8} | {np.mean(cos_vals):>12.4f} | "
            f"{np.mean(mse_vals):>12.4f} | {sum(counts):>8d}"
        )

    print(sep)

def results_to_serializable(results):
    """将 (horizon, masked) key 转为 JSON 可序列化格式。"""
    out = {}
    for (h, masked), v in results.items():
        key = f"h{h}_{'masked' if masked else 'unmasked'}"
        out[key] = v
    return out


def main():
    args = parse_args()

    # --- 加载配置 ---
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = json.load(f)
        print(f"[分析] 从配置文件加载: {args.config}")
    else:
        print(f"[分析] 配置文件不存在，使用默认参数")
        config = {
            "feature_dir": "/home/chenshuai/data/dataset/260309_0310_dino_features",
            "num_episodes": 337, "start_episode": 0,
            "camera_names": "global,wrist", "horizons": "4,8,12",
            "hidden_dim": 512, "num_layers": 4, "nheads": 8,
            "dropout": 0.1, "seed": 42,
        }

    horizons = [int(h) for h in str(config["horizons"]).split(",")]
    camera_names = str(config["camera_names"]).split(",")
    feature_dir = config["feature_dir"]

    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[分析] device={device}")

    # --- 构建模型并加载权重 ---
    print("[分析] 构建 TFM 模型...")
    model = TactileForesightModel(
        dino_dim=768,
        horizons=horizons,
        hidden_dim=config.get("hidden_dim", 512),
        num_layers=config.get("num_layers", 4),
        nheads=config.get("nheads", 8),
        dropout=config.get("dropout", 0.1),
        device=device,
    )
    model.load_model(args.checkpoint)
    print(f"[分析] 加载权重: {args.checkpoint}")
    print(f"[分析] 可训练参数: {model.num_trainable_params():,}")

    # --- 获取验证集 ---
    train_ids, val_ids = load_val_episodes(config)
    print(f"[分析] 训练集: {len(train_ids)} episodes, 验证集: {len(val_ids)} episodes")

    # --- 加载特征 ---
    print(f"[分析] 加载验证集特征 ({len(val_ids)} episodes)...")
    t0 = time.time()
    all_feats = load_features(val_ids, feature_dir)
    print(f"[分析] 特征加载完成，耗时 {time.time()-t0:.1f}s")

    # --- 构建评估样本 ---
    print(f"[分析] 构建评估样本 (每 episode {args.samples_per_episode} 个)...")
    samples = build_eval_samples(
        all_feats, val_ids, camera_names, horizons,
        args.samples_per_episode,
    )
    total = sum(len(v) for v in samples.values())
    print(f"[分析] 总评估样本数: {total}")

    # --- 评估 ---
    print("[分析] 开始评估...")
    t0 = time.time()
    results = evaluate_model(model, samples, device, args.batch_size)
    print(f"[分析] 评估完成，耗时 {time.time()-t0:.1f}s")

    # --- 打印结果 ---
    print_results(results, horizons)

    # --- 保存 JSON ---
    save_path = os.path.join(output_dir, "tfm_analysis.json")
    with open(save_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "config": args.config,
            "num_val_episodes": len(val_ids),
            "samples_per_episode": args.samples_per_episode,
            "horizons": horizons,
            "results": results_to_serializable(results),
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[分析] 结果已保存到: {save_path}")


if __name__ == "__main__":
    main()
