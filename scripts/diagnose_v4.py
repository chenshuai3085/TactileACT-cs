"""
TFAC V4 训练诊断脚本
诊断 5 个问题:
1. A2 vs A1: refinement decoder 是否真的改善了 action?
2. Foresight 质量: 预测的触觉 vs GT 有多远?
3. Gate 分布: fusion gate 是否合理?
4. KL / latent space: CVAE 是否有意义?
5. 过拟合: train vs val gap 分析
"""

import torch
import torch.nn.functional as F
import numpy as np
import os, sys, json, pickle, re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from TFAC_V4.tfac_policy import TFACPolicyV4
from TFAC_V4.dataset import ForesightEpisodicDataset
from utils import get_norm_stats, load_meta_data, set_seed

CKPT_DIR = '/home/chenshuai/data/xiaomi_act/v4_ldg_9'
CONFIG_PATH = os.path.join(CKPT_DIR, 'args.json')


def load_model_and_data():
    with open(CONFIG_PATH, 'r') as f:
        args = json.load(f)

    dataset_dir = args['dataset_dir']
    save_dir = args['save_dir']
    chunk_size = args['chunk_size']

    meta_data = load_meta_data(dataset_dir, save_dir=save_dir, config_overrides=args)
    camera_names = meta_data['camera_names']
    state_dim = meta_data['state_dim']
    proprio_key = meta_data['proprio_key']
    action_key = meta_data['action_key']
    tac_side = meta_data['tac_side']
    tac_img_key = meta_data['tac_img_key']

    tactile_mode = args.get('tactile_mode', 'marker')
    norm_stats = get_norm_stats(dataset_dir, meta_data['num_episodes'], chunk_size=0,
                                proprio_key=proprio_key, action_key=action_key,
                                tactile_mode=tactile_mode, tac_side=tac_side)

    # Load pretrained backbones (CLIP) if needed
    pretrained_backbones = None
    cam_backbone_mapping = None
    if args.get('backbone') == 'clip_backbone':
        try:
            from clip_pretraining_xiaomi import modified_resnet18
        except ImportError:
            from clip_pretraining import modified_resnet18
        vision_model = modified_resnet18()
        cam_backbone_mapping = {cam_name: 0 for cam_name in camera_names}
        # For diagnosis we don't need pretrained weights — they're in the checkpoint
        pretrained_backbones = [vision_model]

    # Build policy
    policy = TFACPolicyV4(
        state_dim=state_dim,
        hidden_dim=args['hidden_dim'],
        position_embedding_type=args['position_embedding'],
        lr_backbone=args['lr_backbone'],
        masks=args['masks'],
        backbone_type=args['backbone'],
        dilation=args['dilation'],
        dropout=args['dropout'],
        nheads=args['nheads'],
        dim_feedforward=args['dim_feedforward'],
        num_enc_layers=args['enc_layers'],
        num_dec_layers=args['dec_layers'],
        pre_norm=args['pre_norm'],
        num_queries=chunk_size,
        camera_names=camera_names,
        z_dimension=args['z_dimension'],
        lr=args['lr'],
        weight_decay=args['weight_decay'],
        kl_weight=args['kl_weight'],
        foresight_layers=args.get('foresight_layers', 3),
        foresight_nheads=args.get('foresight_nheads', 4),
        foresight_dim_feedforward=args.get('foresight_dim_feedforward', 2048),
        proj_dim=args.get('proj_dim', 128),
        contrastive_temperature=args.get('contrastive_temperature', 0.07),
        curriculum_ratio=args.get('curriculum_ratio', 0.75),
        lambda_draft=args.get('lambda_draft', 0.3),
        lambda_latent=args.get('lambda_latent', 0.3),
        lambda_obs=args.get('lambda_obs', 0.2),
        lambda_consistency=args.get('lambda_consistency', 0.3),
        lambda_contrastive_spatial=args.get('lambda_contrastive_spatial', 0.2),
        lambda_contrastive_temporal=args.get('lambda_contrastive_temporal', 0.1),
        lambda_contrastive_gt=args.get('lambda_contrastive_gt', 0.1),
        lambda_dynamics=args.get('lambda_dynamics', 0.1),
        lambda_sampling=args.get('lambda_sampling', 0.5),
        num_dec_layers_draft=args.get('dec_layers_draft', None),
        marker_encoder_type=args.get('marker_encoder_type', 'spatial'),
        fusion_mode=args.get('fusion_mode', 'contact_gate'),
        a2_init=args.get('a2_init', 'residual'),
        max_history=args.get('max_history', 8),
        predict_horizon=args.get('predict_horizon', 1),
        sampling_steps=args.get('sampling_steps', 3),
        n_tac_tokens=args.get('n_tac_tokens', 9),
        pretrained_backbones=pretrained_backbones,
        cam_backbone_mapping=cam_backbone_mapping,
    )

    # Load best checkpoint
    ckpt_path = os.path.join(CKPT_DIR, 'policy_best.ckpt')
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # The checkpoint is from policy_core.state_dict(), which is the full TFACPolicyV4
    policy.load_state_dict(state_dict)
    policy.cuda()
    policy.eval()
    print(f"Loaded checkpoint from {ckpt_path}")

    # Build validation dataset
    num_episodes = meta_data['num_episodes']
    episode_files = sorted(
        [f for f in os.listdir(dataset_dir)
         if f.startswith('episode_') and f.endswith('.hdf5')]
    )
    episode_ids = np.array([
        int(re.search(r'episode_(\d+)', f).group(1)) for f in episode_files
    ])
    set_seed(1)
    shuffled_indices = np.random.permutation(len(episode_ids))
    split = int(0.8 * len(episode_ids))
    val_indices = episode_ids[shuffled_indices[split:]]

    foresight_horizon = args.get('foresight_horizon', 8)
    history_len = args.get('history_len', 1)
    multi_frame_vision = args.get('multi_frame_vision', False)

    val_dataset = ForesightEpisodicDataset(
        val_indices, dataset_dir, camera_names, norm_stats,
        chunk_size=chunk_size, foresight_horizon=foresight_horizon,
        proprio_key=proprio_key, action_key=action_key,
        tac_side=tac_side, tac_img_key=tac_img_key,
        tactile_mode=tactile_mode, history_len=history_len,
        multi_frame_vision=multi_frame_vision)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        pin_memory=True, num_workers=2)

    return policy, val_loader, args


def diagnose():
    policy, val_loader, args = load_model_and_data()
    model = policy.model

    # Collect diagnostics over N batches
    N_BATCHES = 20
    results = {
        'l1_a1': [], 'l1_a2': [], 'l1_delta': [],
        'a1_a2_cosine': [], 'a1_a2_l2': [],
        'obs_mse': [], 'latent_mse': [],
        'gate_mem': [], 'gate_a1': [], 'gate_traj': [],
        'residual_alpha': [],
        'mu_norm': [], 'logvar_mean': [], 'z_entropy': [],
        'kl_per_sample': [],
        'foresight_per_frame_mse': [],
        'consistency_sim_pos': [], 'consistency_sim_neg': [],
    }

    print(f"\nRunning diagnostics over {N_BATCHES} val batches...\n")

    with torch.inference_mode():
        for batch_idx, data in enumerate(val_loader):
            if batch_idx >= N_BATCHES:
                break

            image_data, qpos_data, action_data, is_pad, future_image_data, history_image_data = data
            qpos_data = qpos_data.cuda()
            action_data = action_data.cuda()
            is_pad = is_pad.cuda()
            image_data = tuple(img.cuda() for img in image_data)
            future_image_data = tuple(img.cuda() for img in future_image_data)
            history_image_data = tuple(h.cuda() for h in history_image_data)

            # Run full forward pass (training mode to get all outputs)
            (a1_hat, a2_hat, t_hat_obs, v_hat_future,
             v_gt_feat, t_gt_feat, t_embed_future, t_current_global,
             (mu, logvar), z_dynamics_next) = model(
                qpos_data, image_data, action_data, is_pad,
                future_image_data, use_predicted_future=False,
                history_images=history_image_data)

            pad_mask = ~is_pad.unsqueeze(-1)

            # ===== 1. A1 vs A2 comparison =====
            l1_a1 = (F.l1_loss(a1_hat, action_data, reduction='none') * pad_mask).mean().item()
            l1_a2 = (F.l1_loss(a2_hat, action_data, reduction='none') * pad_mask).mean().item()
            results['l1_a1'].append(l1_a1)
            results['l1_a2'].append(l1_a2)
            results['l1_delta'].append(l1_a1 - l1_a2)

            # A1-A2 similarity
            a1_flat = a1_hat.reshape(-1, a1_hat.shape[-1])
            a2_flat = a2_hat.reshape(-1, a2_hat.shape[-1])
            cosine = F.cosine_similarity(a1_flat, a2_flat, dim=-1).mean().item()
            l2_dist = (a1_hat - a2_hat).pow(2).mean().sqrt().item()
            results['a1_a2_cosine'].append(cosine)
            results['a1_a2_l2'].append(l2_dist)

            # ===== 2. Foresight quality =====
            if t_gt_feat is not None and t_hat_obs is not None:
                obs_mse = F.mse_loss(t_hat_obs, t_gt_feat).item()
                results['obs_mse'].append(obs_mse)

                # Per-frame MSE
                if t_gt_feat.dim() == 5:
                    per_frame = []
                    for h in range(t_gt_feat.shape[1]):
                        frame_mse = F.mse_loss(t_hat_obs[:, h], t_gt_feat[:, h]).item()
                        per_frame.append(frame_mse)
                    results['foresight_per_frame_mse'].append(per_frame)

            if t_embed_future is not None and t_gt_feat is not None:
                if t_gt_feat.dim() == 5 and t_embed_future.dim() == 4:
                    B, H = t_gt_feat.shape[:2]
                    gt_flat = t_gt_feat.reshape(B * H, 9, 9, 2)
                    if model.is_spatial_encoder:
                        gt_tokens = model.marker_encoder(gt_flat).permute(1, 0, 2).view(B, H, 9, -1)
                    else:
                        gt_enc = model.marker_encoder(gt_flat)
                        gt_tokens = gt_enc.view(B, H, 1, -1)
                    latent_mse = F.mse_loss(t_embed_future, gt_tokens).item()
                    results['latent_mse'].append(latent_mse)

            # ===== 3. Gate distribution =====
            if hasattr(model, 'contact_fusion') and hasattr(model.contact_fusion, '_last_gate_means'):
                gm, ga, gf = model.contact_fusion._last_gate_means
                results['gate_mem'].append(gm)
                results['gate_a1'].append(ga)
                results['gate_traj'].append(gf)

            # Residual alpha
            if hasattr(model, 'residual_alpha'):
                alpha = torch.sigmoid(model.residual_alpha).item()
                results['residual_alpha'].append(alpha)

            # ===== 4. KL / Latent space =====
            results['mu_norm'].append(mu.norm(dim=-1).mean().item())
            results['logvar_mean'].append(logvar.mean().item())

            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl_per_sample = kl_per_dim.sum(dim=-1).mean().item()
            results['kl_per_sample'].append(kl_per_sample)

            # ===== 5. Consistency: dynamics vs foresight agreement =====
            if t_embed_future is not None and z_dynamics_next is not None:
                if t_embed_future.dim() == 4:
                    foresight_global = t_embed_future[:, -1].mean(dim=1)
                else:
                    foresight_global = t_embed_future.mean(dim=1)

                z_dyn_norm = F.normalize(z_dynamics_next, dim=-1)
                z_fore_norm = F.normalize(foresight_global, dim=-1)
                sim_pos = (z_dyn_norm * z_fore_norm).sum(dim=-1).mean().item()
                z_neg = torch.roll(z_fore_norm, shifts=1, dims=0)
                sim_neg = (z_dyn_norm * z_neg).sum(dim=-1).mean().item()
                results['consistency_sim_pos'].append(sim_pos)
                results['consistency_sim_neg'].append(sim_neg)

            if (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx + 1}/{N_BATCHES}")

    # ===== Print Report =====
    print("\n" + "=" * 70)
    print("TFAC V4 诊断报告 (best checkpoint @ epoch 80)")
    print("=" * 70)

    print("\n--- 1. A1 vs A2 (Draft vs Final Action) ---")
    mean_l1_a1 = np.mean(results['l1_a1'])
    mean_l1_a2 = np.mean(results['l1_a2'])
    mean_delta = np.mean(results['l1_delta'])
    improvement_pct = mean_delta / mean_l1_a1 * 100 if mean_l1_a1 > 0 else 0
    print(f"  L1 A1 (draft):   {mean_l1_a1:.5f}")
    print(f"  L1 A2 (final):   {mean_l1_a2:.5f}")
    print(f"  Improvement:     {mean_delta:.5f} ({improvement_pct:.1f}%)")
    print(f"  A1-A2 cosine:    {np.mean(results['a1_a2_cosine']):.5f}")
    print(f"  A1-A2 L2 dist:   {np.mean(results['a1_a2_l2']):.5f}")
    if improvement_pct < 5:
        print(f"  ⚠ A2 改进不到 5%，foresight→fusion→refinement 路径贡献很小!")
    elif improvement_pct > 15:
        print(f"  ✓ A2 显著改进, refinement 有效")
    else:
        print(f"  ~ A2 有一定改进但不显著")

    print("\n--- 2. Foresight 触觉预测质量 ---")
    if results['obs_mse']:
        print(f"  Observation MSE (pred vs GT): {np.mean(results['obs_mse']):.6f}")
    if results['latent_mse']:
        print(f"  Latent MSE (embed vs GT enc): {np.mean(results['latent_mse']):.6f}")
    if results['foresight_per_frame_mse']:
        avg_per_frame = np.mean(results['foresight_per_frame_mse'], axis=0)
        print(f"  Per-frame MSE (t+1 to t+{len(avg_per_frame)}):")
        for i, mse in enumerate(avg_per_frame):
            bar = '█' * int(mse * 500)
            print(f"    t+{i+1}: {mse:.6f} {bar}")

    print("\n--- 3. Gate 分布 ---")
    if results['gate_mem']:
        print(f"  gate_mem  (memory):    {np.mean(results['gate_mem']):.4f}")
        print(f"  gate_a1   (draft act): {np.mean(results['gate_a1']):.4f}")
        print(f"  gate_traj (tactile):   {np.mean(results['gate_traj']):.4f}")
        if np.mean(results['gate_traj']) < 0.2:
            print(f"  ⚠ gate_traj < 0.2, 触觉对 fusion 贡献过小!")
    if results['residual_alpha']:
        print(f"  residual_alpha:        {np.mean(results['residual_alpha']):.4f}")
        alpha = np.mean(results['residual_alpha'])
        if alpha < 0.3:
            print(f"  ⚠ alpha < 0.3, 残差连接几乎不生效, A2 ≈ A1")

    print("\n--- 4. KL / Latent Space ---")
    print(f"  mu norm (should >0 if useful):  {np.mean(results['mu_norm']):.5f}")
    print(f"  logvar mean (should ~0):        {np.mean(results['logvar_mean']):.5f}")
    print(f"  KL per sample:                  {np.mean(results['kl_per_sample']):.5f}")
    if np.mean(results['kl_per_sample']) < 0.01:
        print(f"  ⚠ KL ≈ 0, CVAE latent 完全崩塌! z 恒等于 prior, 没有编码有用信息")
    elif np.mean(results['kl_per_sample']) < 0.1:
        print(f"  ~ KL 很低, latent 信息量很少")

    print("\n--- 5. Consistency (Dynamics vs Foresight) ---")
    if results['consistency_sim_pos']:
        sim_pos = np.mean(results['consistency_sim_pos'])
        sim_neg = np.mean(results['consistency_sim_neg'])
        print(f"  sim(dynamics, foresight) positive: {sim_pos:.4f}")
        print(f"  sim(dynamics, foresight) negative: {sim_neg:.4f}")
        print(f"  margin (pos - neg):                {sim_pos - sim_neg:.4f}")
        if sim_pos - sim_neg < 0.1:
            print(f"  ⚠ Dynamics 和 Foresight 预测不一致, 两条路径没有对齐")
        elif sim_pos > 0.8:
            print(f"  ✓ 两条预测路径高度一致")

    print("\n--- 6. 综合诊断 ---")
    issues = []
    if improvement_pct < 5:
        issues.append("A2 refinement 无效 (核心问题)")
    if np.mean(results.get('kl_per_sample', [1])) < 0.01:
        issues.append("KL 崩塌, CVAE 无意义")
    if results['gate_traj'] and np.mean(results['gate_traj']) < 0.2:
        issues.append("触觉 gate 权重过低")
    if results['residual_alpha'] and np.mean(results['residual_alpha']) < 0.3:
        issues.append("残差 alpha 过小, A2≈A1")
    if results['obs_mse'] and np.mean(results['obs_mse']) > 0.01:
        issues.append("触觉预测 MSE 偏高")

    if issues:
        print("  发现的问题:")
        for i, issue in enumerate(issues):
            print(f"    {i+1}. {issue}")
    else:
        print("  未发现严重问题")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    diagnose()
