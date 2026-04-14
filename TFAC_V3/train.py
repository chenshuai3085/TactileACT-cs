"""
TFAC 训练脚本, 基于 imitate_episodes.py 改写。
支持课程学习: 前 75% epochs 用 GT future tactile, 后 25% 用预测的。
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
import json

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import get_norm_stats, compute_dict_mean, set_seed, detach_dict, load_meta_data
from TFAC_V3.dataset import ForesightEpisodicDataset
from TFAC_V3.tfac_policy import TFACPolicy

from typing import List, Dict, Any

def main(args):
    save_dir = args['save_dir']
    model_name = args['name']
    batch_size = args['batch_size']
    num_epochs = args['num_epochs']
    chunk_size = args['chunk_size']
    seed = args['seed']
    gpu = args['gpu']

    ckpt_dir = os.path.join(save_dir, model_name)
    dataset_dir = args.get('dataset_dir') or os.path.join(save_dir, 'data')
    assert os.path.exists(save_dir), f'{save_dir} does not exist.'
    assert os.path.exists(dataset_dir), f'{dataset_dir} does not exist.'

    # 自动推断 meta_data (优先 config > HDF5 推断 > meta_data.json fallback)
    meta_data = load_meta_data(dataset_dir, save_dir=save_dir, config_overrides=args)
    num_episodes = meta_data['num_episodes']
    camera_names = meta_data['camera_names']
    state_dim = meta_data['state_dim']
    proprio_key = meta_data['proprio_key']
    action_key = meta_data['action_key']
    tac_side = meta_data['tac_side']
    tac_img_key = meta_data['tac_img_key']

    norm_stats = get_norm_stats(dataset_dir, num_episodes, chunk_size=0,
                                proprio_key=proprio_key, action_key=action_key,
                                tactile_mode=args.get('tactile_mode', 'image'),
                                tac_side=tac_side)
    args['norm_stats'] = {k: v.tolist() for k, v in norm_stats.items()}

    set_seed(seed)
    gpu_ids = args.get('gpu_ids', None)
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(g) for g in gpu_ids)
        # DataParallel device_ids 是相对于 CUDA_VISIBLE_DEVICES 的逻辑编号
        args['gpu_ids'] = list(range(len(gpu_ids)))
    elif gpu != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # --- Backbone setup (标准 ImageNet ResNet18, 在 TFACPolicy 内自动加载) ---
    tactile_mode = args.get('tactile_mode', 'image')

    # --- Build TFAC policy ---
    policy = TFACPolicy(
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
        # TFAC specific
        foresight_layers=args.get('foresight_layers', 2),
        foresight_nheads=args.get('foresight_nheads', 4),
        foresight_dim_feedforward=args.get('foresight_dim_feedforward', 2048),
        proj_dim=args.get('proj_dim', 128),
        contrastive_temperature=args.get('contrastive_temperature', 0.07),
        curriculum_ratio=args.get('curriculum_ratio', 0.75),
        lambda_draft=args.get('lambda_draft', 0.5),
        lambda_foresight=args.get('lambda_foresight', 1.0),
        lambda_foresight_vis=args.get('lambda_foresight_vis', 0.3),
        lambda_contrastive=args.get('lambda_contrastive', 0.1),
        lambda_contrastive_gt=args.get('lambda_contrastive_gt', 0.0),
        num_dec_layers_draft=args.get('dec_layers_draft', None),
        foresight_change_weight=args.get('foresight_change_weight', False),
        # V4 modularity
        tactile_mode=args.get('tactile_mode', 'image'),
        marker_encoder_type=args.get('marker_encoder_type', 'conv2d'),
        fusion_mode=args.get('fusion_mode', 'gate'),
        foresight_tac_decoder=args.get('foresight_tac_decoder', 'linear'),
        spatial_tac_dec_layers=args.get('spatial_tac_dec_layers', 3),
        a2_init=args.get('a2_init', 'zero'),
        predict_horizon=args.get('predict_horizon', 1),
        sampling_steps=args.get('sampling_steps', 0),
        lambda_sampling=args.get('lambda_sampling', 0.5),
    )
    policy.cuda()

    # --- Load pretrained foresight weights (Stage 2) ---
    pretrain_ckpt = args.get('pretrain_foresight_ckpt', None)
    if pretrain_ckpt and os.path.exists(pretrain_ckpt):
        print(f"Loading pretrained foresight from: {pretrain_ckpt}")
        pretrain_state = torch.load(pretrain_ckpt, map_location='cuda')

        # Map pretrain model keys → TFAC model keys
        key_mapping = {
            'input_proj.': 'model.input_proj.',
            'marker_encoder.': 'model.marker_encoder.',
            'marker_pos_embed': 'model.marker_pos_embed',
            'foresight.': 'model.foresight.',
        }

        model_state = policy.model.state_dict()
        loaded_keys = []
        for pt_key, pt_val in pretrain_state.items():
            # Skip frozen backbone weights
            if pt_key.startswith('backbone.'):
                continue

            mapped_key = None
            for prefix, target_prefix in key_mapping.items():
                if pt_key.startswith(prefix):
                    mapped_key = target_prefix + pt_key[len(prefix):]
                    break

            if mapped_key and mapped_key in model_state:
                if model_state[mapped_key].shape == pt_val.shape:
                    model_state[mapped_key] = pt_val
                    loaded_keys.append(mapped_key)
                else:
                    print(f"  Shape mismatch: {mapped_key} "
                          f"({model_state[mapped_key].shape} vs {pt_val.shape}), skipping")

        policy.model.load_state_dict(model_state)
        print(f"  Loaded {len(loaded_keys)} pretrained weight tensors")

        # Set lower lr for foresight modules
        foresight_lr_scale = args.get('foresight_lr_scale', 0.1)
        print(f"  Foresight lr scale: {foresight_lr_scale}x")

        # Rebuild optimizer with separate param groups
        foresight_param_names = set()
        for name, _ in policy.model.named_parameters():
            for prefix in ['input_proj.', 'marker_encoder.', 'marker_pos_embed', 'foresight.']:
                if name.startswith(prefix):
                    foresight_param_names.add(name)

        foresight_params = []
        other_params = []
        backbone_params = []
        for name, param in policy.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone' in name:
                backbone_params.append(param)
            elif name in foresight_param_names:
                foresight_params.append(param)
            else:
                other_params.append(param)

        base_lr = args['lr']
        policy.optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': base_lr},
            {'params': foresight_params, 'lr': base_lr * foresight_lr_scale},
            {'params': backbone_params, 'lr': args['lr_backbone']},
        ], weight_decay=args.get('weight_decay', 1e-4))
        print(f"  Optimizer: other_params lr={base_lr}, "
              f"foresight lr={base_lr * foresight_lr_scale}, "
              f"backbone lr={args['lr_backbone']}")
    elif pretrain_ckpt:
        print(f"WARNING: pretrain_foresight_ckpt not found: {pretrain_ckpt}")

    # --- Checkpoint dir ---
    if os.path.exists(ckpt_dir):
        n = 0
        while os.path.exists(ckpt_dir + f'_{n}'):
            n += 1
        print(f'Warning: {ckpt_dir} exists. Using {ckpt_dir}_{n}')
        ckpt_dir = ckpt_dir + f'_{n}'
    os.makedirs(ckpt_dir)

    combo_dict = {**args, **meta_data}
    with open(os.path.join(ckpt_dir, 'args.json'), 'w') as f:
        json.dump(combo_dict, f, indent=4)

    # --- Dataset ---
    foresight_horizon = args.get('foresight_horizon', 8)
    predict_horizon = args.get('predict_horizon', 1)
    if predict_horizon > 1 and predict_horizon != foresight_horizon:
        raise ValueError(
            f"predict_horizon={predict_horizon} must equal foresight_horizon={foresight_horizon} "
            f"when predict_horizon > 1 (multi-frame prediction requires matching horizons)")
    train_ratio = 0.8
    # 从实际文件名提取 episode 编号，兼容编号不连续的情况
    import re
    episode_files = sorted(
        [f for f in os.listdir(dataset_dir)
         if f.startswith('episode_') and f.endswith('.hdf5')]
    )
    episode_ids = np.array([
        int(re.search(r'episode_(\d+)', f).group(1)) for f in episode_files
    ])
    shuffled_indices = np.random.permutation(len(episode_ids))
    split = int(train_ratio * len(episode_ids))
    train_indices = episode_ids[shuffled_indices[:split]]
    val_indices = episode_ids[shuffled_indices[split:]]

    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(norm_stats, f)

    history_len = args.get('history_len', 1)
    multi_frame_vision = args.get('multi_frame_vision', False)
    dataset_kwargs = dict(proprio_key=proprio_key, action_key=action_key,
                          tac_side=tac_side, tac_img_key=tac_img_key,
                          tactile_mode=tactile_mode, history_len=history_len,
                          multi_frame_vision=multi_frame_vision)
    train_dataset = ForesightEpisodicDataset(
        train_indices, dataset_dir, camera_names, norm_stats,
        chunk_size=chunk_size, foresight_horizon=foresight_horizon, **dataset_kwargs)
    val_dataset = ForesightEpisodicDataset(
        val_indices, dataset_dir, camera_names, norm_stats,
        chunk_size=chunk_size, foresight_horizon=foresight_horizon, **dataset_kwargs)

    # 预加载模式: 数据已在内存, 少量 worker 即可; 否则多 worker 加速 I/O
    # persistent_workers 避免每 epoch 重建 worker
    n_workers = 2 if train_dataset.cache else 8
    prefetch = 2 if train_dataset.cache else 8
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  pin_memory=True, num_workers=n_workers,
                                  prefetch_factor=prefetch, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                pin_memory=True, num_workers=n_workers,
                                prefetch_factor=prefetch, persistent_workers=True)

    # --- Multi-GPU DataParallel ---
    gpu_ids = args.get('gpu_ids', None)
    if gpu_ids and len(gpu_ids) > 1:
        policy = torch.nn.DataParallel(policy, device_ids=gpu_ids)
        print(f'Using DataParallel on GPUs: {gpu_ids}')
    elif gpu_ids and len(gpu_ids) == 1:
        print(f'Using single GPU: {gpu_ids[0]}')

    # --- Training loop ---
    best_ckpt_info = train_tfac(
        policy=policy,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=num_epochs,
        ckpt_dir=ckpt_dir,
        seed=seed,
    )

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val l1_final {min_val_loss:.6f} @ epoch {best_epoch}')


def train_tfac(policy, train_dataloader, val_dataloader,
               num_epochs, ckpt_dir, seed):
    plot_freq = 50

    # DataParallel 兼容: 统一访问底层 policy
    is_dp = isinstance(policy, torch.nn.DataParallel)
    policy_core = policy.module if is_dp else policy

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    def _prepare_data(data):
        """将 dataloader 输出转移到 GPU。
        images 类数据转为 tuple of tensors (DataParallel 会递归 scatter tuple 中每个 tensor)。
        """
        image_data, qpos_data, action_data, is_pad, future_image_data, history_image_data = data
        qpos_data = qpos_data.cuda()
        action_data = action_data.cuda()
        is_pad = is_pad.cuda()
        # list → tuple, 每个 tensor 独立 .cuda()
        # DataParallel scatter 递归处理 tuple, 沿 dim=0 (batch) 分片每个 tensor
        image_data = tuple(img.cuda() for img in image_data)
        future_image_data = tuple(img.cuda() for img in future_image_data)
        history_image_data = tuple(h.cuda() for h in history_image_data)
        return image_data, qpos_data, action_data, is_pad, future_image_data, history_image_data

    def _reduce_dict(forward_dict):
        """DataParallel gather 后每个 loss 可能是 (N_gpu,)，取 mean 变标量。"""
        return {k: v.mean() if v.dim() > 0 else v for k, v in forward_dict.items()}

    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')

        # --- Validation ---
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                image_data, qpos_data, action_data, is_pad, future_image_data, history_image_data = _prepare_data(data)

                forward_dict = policy(qpos_data, image_data, action_data, is_pad,
                                      future_images=future_image_data,
                                      epoch=epoch, total_epochs=num_epochs,
                                      history_images=history_image_data)
                forward_dict = _reduce_dict(forward_dict)
                epoch_dicts.append(forward_dict)

            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            # Select best ckpt by l1_final (action quality), not total loss
            epoch_val_l1_final = epoch_summary['l1_final']
            if epoch_val_l1_final < min_val_loss:
                min_val_loss = epoch_val_l1_final
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy_core.state_dict()))
                print(f'*** New best at epoch {epoch}, val l1_final: {min_val_loss:.5f} ***')

        print(f'Val loss: {epoch_val_loss:.5f}, l1_final: {epoch_val_l1_final:.5f} (best: epoch {best_ckpt_info[0]}, l1_final {best_ckpt_info[1]:.5f})')
        summary_string = ' '.join(f'{k}: {v.item():.4f}' for k, v in epoch_summary.items())
        print(summary_string)

        # --- Training ---
        policy.train()
        policy_core.optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            image_data, qpos_data, action_data, is_pad, future_image_data, history_image_data = _prepare_data(data)

            forward_dict = policy(qpos_data, image_data, action_data, is_pad,
                                  future_images=future_image_data,
                                  epoch=epoch, total_epochs=num_epochs,
                                  history_images=history_image_data)
            forward_dict = _reduce_dict(forward_dict)

            loss = forward_dict['loss']
            loss.backward()
            policy_core.optimizer.step()
            policy_core.optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))

        n_batches = len(train_dataloader)
        epoch_summary = compute_dict_mean(
            train_history[n_batches * epoch:n_batches * (epoch + 1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ' '.join(f'{k}: {v.item():.4f}' for k, v in epoch_summary.items())
        print(summary_string)

        # gate 权重已通过 loss_dict 返回 (gate_mem/gate_a1/gate_fut)，无需额外打印

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy_core.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, 'policy_last.ckpt')
    torch.save(policy_core.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished: Seed {seed}, best val l1_final {min_val_loss:.6f} at epoch {best_epoch}')

    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    with open(os.path.join(ckpt_dir, 'train_history.pkl'), 'wb') as f:
        pickle.dump(train_history, f)
    with open(os.path.join(ckpt_dir, 'validation_history.pkl'), 'wb') as f:
        pickle.dump(validation_history, f)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    n_val = len(validation_history)
    n_train = len(train_history)
    if n_val == 0 or n_train == 0:
        return
    # train_history 是 per-batch, validation_history 是 per-epoch
    # 将 train 的 x 轴映射到 epoch 刻度
    batches_per_epoch = n_train // n_val if n_val > 0 else n_train
    train_x = np.arange(n_train) / max(batches_per_epoch, 1)
    val_x = np.arange(n_val)
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [s[key].item() for s in train_history]
        val_values = [s[key].item() for s in validation_history]
        plt.plot(train_x, train_values, label='train', alpha=0.6)
        plt.plot(val_x, val_values, label='val', linewidth=2)
        plt.xlabel('Epoch')
        plt.legend()
        plt.title(key)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to JSON config')
    cli_args = parser.parse_args()

    with open(cli_args.config, 'r') as f:
        config = json.load(f)

    main(config)
