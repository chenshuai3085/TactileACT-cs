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

from utils import get_norm_stats, compute_dict_mean, set_seed, detach_dict
from TFAC_V2.dataset import ForesightEpisodicDataset
from TFAC_V2.tfac_policy import TFACPolicy

from typing import List, Dict, Any

FREEZE_TACTILE = True


def main(args):
    save_dir = args['save_dir']
    model_name = args['name']
    batch_size = args['batch_size']
    num_epochs = args['num_epochs']
    chunk_size = args['chunk_size']
    seed = args['seed']
    gpu = args['gpu']

    ckpt_dir = os.path.join(save_dir, model_name)
    dataset_dir = os.path.join(save_dir, 'data')
    assert os.path.exists(save_dir), f'{save_dir} does not exist.'

    with open(os.path.join(save_dir, 'meta_data.json'), 'r') as f:
        meta_data = json.load(f)
    # Auto-detect episode count from dataset_dir
    actual_episodes = len([fname for fname in os.listdir(dataset_dir)
        if fname.startswith('episode_') and fname.endswith('.hdf5')])
    num_episodes = actual_episodes if actual_episodes > 0 else meta_data['num_episodes']
    if actual_episodes != meta_data['num_episodes']:
        print(f'Warning: meta_data says {meta_data["num_episodes"]} episodes, '
              f'but found {actual_episodes} in {dataset_dir}. Using {num_episodes}.')
    camera_names = meta_data['camera_names']
    state_dim = meta_data['state_dim']
    proprio_key = meta_data.get('proprio_key', 'qpos')
    action_key = meta_data.get('action_key', 'action')
    tac_side = meta_data.get('tac_side', 'left')
    tac_img_key = meta_data.get('tac_img_key', 'img')

    norm_stats = get_norm_stats(dataset_dir, num_episodes, chunk_size=0,
                                proprio_key=proprio_key, action_key=action_key,
                                tactile_mode=args.get('tactile_mode', 'image'),
                                tac_side=tac_side)
    args['norm_stats'] = {k: v.tolist() for k, v in norm_stats.items()}

    set_seed(seed)
    if gpu != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # --- Load pretrained backbones ---
    tactile_mode = args.get('tactile_mode', 'image')
    if args['backbone'] == "clip_backbone":
        try:
            from clip_pretraining_xiaomi import modified_resnet18
        except ImportError:
            from clip_pretraining import modified_resnet18
        vision_model = modified_resnet18()
        camera_backbone_mapping = {cam_name: 0 for cam_name in camera_names}

        if tactile_mode == 'image':
            # Need gelsight backbone only in image mode
            gelsight_model = modified_resnet18()
            camera_backbone_mapping['gelsight'] = 1

            if args['gelsight_backbone_path'] != 'none' and args['vision_backbone_path'] != 'none':
                vision_model.load_state_dict(torch.load(args['vision_backbone_path']))
                gelsight_model.load_state_dict(torch.load(args['gelsight_backbone_path']))
            elif args['gelsight_backbone_path'] != 'none' or args['vision_backbone_path'] != 'none':
                raise ValueError('Both vision and gelsight backbones must be specified if one is specified.')

            if FREEZE_TACTILE:
                gelsight_model.requires_grad_(False)
                print("Freezing tactile backbone")
            pretrained_backbones = [vision_model, gelsight_model]
        else:
            # marker mode: gelsight uses MarkerEncoder, only need vision backbone
            camera_backbone_mapping['gelsight'] = 0  # placeholder, not used
            if args.get('vision_backbone_path', 'none') != 'none':
                vision_model.load_state_dict(torch.load(args['vision_backbone_path']))
            pretrained_backbones = [vision_model]
            print(f"Marker mode: skipping gelsight backbone, using {args.get('marker_encoder_type', 'conv2d')} encoder")
    else:
        pretrained_backbones = None
        camera_backbone_mapping = None

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
        pretrained_backbones=pretrained_backbones,
        cam_backbone_mapping=camera_backbone_mapping,
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
    )
    policy.cuda()

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
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(norm_stats, f)

    history_len = args.get('history_len', 1)
    dataset_kwargs = dict(proprio_key=proprio_key, action_key=action_key,
                          tac_side=tac_side, tac_img_key=tac_img_key,
                          tactile_mode=tactile_mode, history_len=history_len)
    train_dataset = ForesightEpisodicDataset(
        train_indices, dataset_dir, camera_names, norm_stats,
        chunk_size=chunk_size, foresight_horizon=foresight_horizon, **dataset_kwargs)
    val_dataset = ForesightEpisodicDataset(
        val_indices, dataset_dir, camera_names, norm_stats,
        chunk_size=chunk_size, foresight_horizon=foresight_horizon, **dataset_kwargs)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  pin_memory=True, num_workers=8, prefetch_factor=8)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                pin_memory=True, num_workers=8, prefetch_factor=8)

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


def train_tfac(policy: TFACPolicy, train_dataloader, val_dataloader,
               num_epochs, ckpt_dir, seed):
    plot_freq = 50

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')

        # --- Validation ---
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                image_data, qpos_data, action_data, is_pad, future_image_data, history_image_data = data

                qpos_data = qpos_data.cuda()
                image_data = [img.cuda() for img in image_data]
                action_data = action_data.cuda()
                is_pad = is_pad.cuda()
                future_image_data = [img.cuda() for img in future_image_data]
                history_image_data = [h.cuda() for h in history_image_data]

                forward_dict = policy(qpos_data, image_data, action_data, is_pad,
                                      future_images=future_image_data,
                                      epoch=epoch, total_epochs=num_epochs,
                                      history_images=history_image_data)
                epoch_dicts.append(forward_dict)

            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            # Select best ckpt by l1_final (action quality), not total loss
            epoch_val_l1_final = epoch_summary['l1_final']
            if epoch_val_l1_final < min_val_loss:
                min_val_loss = epoch_val_l1_final
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
                print(f'*** New best at epoch {epoch}, val l1_final: {min_val_loss:.5f} ***')

        print(f'Val loss: {epoch_val_loss:.5f}, l1_final: {epoch_val_l1_final:.5f} (best: epoch {best_ckpt_info[0]}, l1_final {best_ckpt_info[1]:.5f})')
        summary_string = ' '.join(f'{k}: {v.item():.4f}' for k, v in epoch_summary.items())
        print(summary_string)

        # --- Training ---
        policy.train()
        policy.optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            image_data, qpos_data, action_data, is_pad, future_image_data, history_image_data = data

            qpos_data = qpos_data.cuda()
            image_data = [img.cuda() for img in image_data]
            action_data = action_data.cuda()
            is_pad = is_pad.cuda()
            future_image_data = [img.cuda() for img in future_image_data]
            history_image_data = [h.cuda() for h in history_image_data]

            forward_dict = policy(qpos_data, image_data, action_data, is_pad,
                                  future_images=future_image_data,
                                  epoch=epoch, total_epochs=num_epochs,
                                  history_images=history_image_data)

            loss = forward_dict['loss']
            loss.backward()
            policy.optimizer.step()
            policy.optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))

        n_batches = len(train_dataloader)
        epoch_summary = compute_dict_mean(
            train_history[n_batches * epoch:n_batches * (epoch + 1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ' '.join(f'{k}: {v.item():.4f}' for k, v in epoch_summary.items())
        print(summary_string)

        # 打印 gate 权重分布 (仅 gate fusion 模式)
        if hasattr(policy.model, 'gated_fusion'):
            gate_means = policy.model.gated_fusion._last_gate_means
            print(f'Gate weights: memory={gate_means[0]:.3f}, a1={gate_means[1]:.3f}, future={gate_means[2]:.3f}')

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, 'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

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
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [s[key].item() for s in train_history]
        val_values = [s[key].item() for s in validation_history]
        plt.plot(np.linspace(1, num_epochs - 1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(1, num_epochs - 1, len(validation_history)), val_values, label='val')
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
