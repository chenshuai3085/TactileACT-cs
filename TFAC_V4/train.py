"""
TFAC V4 Training Script.
Based on V3 train.py, adapted for TFACPolicyV4 imports and new config params.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
import argparse
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for nohup/background
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
import json

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import get_norm_stats, compute_dict_mean, set_seed, detach_dict, load_meta_data
from TFAC_V4.dataset import ForesightEpisodicDataset
from TFAC_V4.tfac_policy import TFACPolicyV4

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

    meta_data = load_meta_data(dataset_dir, save_dir=save_dir, config_overrides=args)
    num_episodes = meta_data['num_episodes']
    camera_names = meta_data['camera_names']
    state_dim = meta_data['state_dim']
    proprio_key = meta_data['proprio_key']
    action_key = meta_data['action_key']
    tac_side = meta_data['tac_side']
    tac_img_key = meta_data['tac_img_key']

    tactile_mode = args.get('tactile_mode', 'marker')
    norm_stats = get_norm_stats(dataset_dir, num_episodes, chunk_size=0,
                                proprio_key=proprio_key, action_key=action_key,
                                tactile_mode=tactile_mode,
                                tac_side=tac_side)
    args['norm_stats'] = {k: v.tolist() for k, v in norm_stats.items()}

    set_seed(seed)
    gpu_ids = args.get('gpu_ids', None)
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(g) for g in gpu_ids)
        args['gpu_ids'] = list(range(len(gpu_ids)))
    elif gpu != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # --- Load pretrained backbones (CLIP) ---
    pretrained_backbones = None
    cam_backbone_mapping = None
    if args.get('backbone') == 'clip_backbone':
        try:
            from clip_pretraining_xiaomi import modified_resnet18
        except ImportError:
            from clip_pretraining import modified_resnet18
        vision_model = modified_resnet18()
        cam_backbone_mapping = {cam_name: 0 for cam_name in camera_names}

        vision_path = args.get('vision_backbone_path', 'none')
        if vision_path != 'none' and os.path.exists(vision_path):
            vision_model.load_state_dict(torch.load(vision_path, map_location='cpu'))
            print(f"Loaded CLIP vision backbone from {vision_path}")
        else:
            print("WARNING: clip_backbone mode but no pretrained weights loaded!")
        pretrained_backbones = [vision_model]

    # --- Build V4 policy ---
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
        # Foresight
        foresight_layers=args.get('foresight_layers', 3),
        foresight_nheads=args.get('foresight_nheads', 4),
        foresight_dim_feedforward=args.get('foresight_dim_feedforward', 2048),
        proj_dim=args.get('proj_dim', 128),
        contrastive_temperature=args.get('contrastive_temperature', 0.07),
        # Curriculum
        curriculum_ratio=args.get('curriculum_ratio', 0.75),
        # V4 loss weights
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
        # V4 specific
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
    predict_horizon = args.get('predict_horizon', 1)
    if predict_horizon > 1 and predict_horizon != foresight_horizon:
        raise ValueError(
            f"predict_horizon={predict_horizon} must equal foresight_horizon={foresight_horizon} "
            f"when predict_horizon > 1")

    train_ratio = 0.8
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
    contrastive_vision_indices = args.get('contrastive_vision_indices', None)
    dataset_kwargs = dict(proprio_key=proprio_key, action_key=action_key,
                          tac_side=tac_side, tac_img_key=tac_img_key,
                          tactile_mode=tactile_mode, history_len=history_len,
                          multi_frame_vision=multi_frame_vision,
                          contrastive_vision_indices=contrastive_vision_indices)
    train_dataset = ForesightEpisodicDataset(
        train_indices, dataset_dir, camera_names, norm_stats,
        chunk_size=chunk_size, foresight_horizon=foresight_horizon, **dataset_kwargs)
    val_dataset = ForesightEpisodicDataset(
        val_indices, dataset_dir, camera_names, norm_stats,
        chunk_size=chunk_size, foresight_horizon=foresight_horizon, **dataset_kwargs)

    n_workers = 2 if train_dataset.cache else 8
    prefetch = 2 if train_dataset.cache else 8
    def _worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  pin_memory=True, num_workers=n_workers,
                                  prefetch_factor=prefetch, persistent_workers=True,
                                  worker_init_fn=_worker_init_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                pin_memory=True, num_workers=n_workers,
                                prefetch_factor=prefetch, persistent_workers=True,
                                worker_init_fn=_worker_init_fn)

    # --- Multi-GPU ---
    gpu_ids = args.get('gpu_ids', None)
    if gpu_ids and len(gpu_ids) > 1:
        policy = torch.nn.DataParallel(policy, device_ids=gpu_ids)
        print(f'Using DataParallel on GPUs: {gpu_ids}')

    # --- Training loop ---
    val_freq = args.get('val_freq', 5)
    grad_accum_steps = args.get('grad_accum_steps', 1)
    best_ckpt_info = train_tfac_v4(
        policy=policy,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=num_epochs,
        ckpt_dir=ckpt_dir,
        seed=seed,
        val_freq=val_freq,
        grad_accum_steps=grad_accum_steps,
    )

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val l1_final {min_val_loss:.6f} @ epoch {best_epoch}')


def train_tfac_v4(policy, train_dataloader, val_dataloader,
                   num_epochs, ckpt_dir, seed, val_freq=5,
                   grad_accum_steps=1):

    is_dp = isinstance(policy, torch.nn.DataParallel)
    policy_core = policy.module if is_dp else policy

    train_history = []
    validation_history = []
    val_epochs = []
    min_val_loss = np.inf
    best_ckpt_info = None

    def _prepare_data(data):
        image_data, qpos_data, action_data, is_pad, future_image_data, history_image_data = data
        qpos_data = qpos_data.cuda()
        action_data = action_data.cuda()
        is_pad = is_pad.cuda()
        image_data = tuple(img.cuda() for img in image_data)
        future_image_data = tuple(img.cuda() for img in future_image_data)
        history_image_data = tuple(h.cuda() for h in history_image_data)
        return image_data, qpos_data, action_data, is_pad, future_image_data, history_image_data

    def _reduce_dict(forward_dict):
        return {k: v.mean() if v.dim() > 0 else v for k, v in forward_dict.items()}

    print(f'Validation frequency: every {val_freq} epoch(s)')

    # AMP: mixed precision for memory efficiency
    scaler = torch.cuda.amp.GradScaler()
    print('Using AMP (automatic mixed precision)')

    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')

        # --- Validation ---
        run_val = (epoch % val_freq == 0) or (epoch == num_epochs - 1)
        if run_val:
            with torch.inference_mode():
                policy.eval()
                epoch_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    image_data, qpos_data, action_data, is_pad, future_image_data, history_image_data = _prepare_data(data)
                    with torch.cuda.amp.autocast():
                        forward_dict = policy(qpos_data, image_data, action_data, is_pad,
                                              future_images=future_image_data,
                                              epoch=epoch, total_epochs=num_epochs,
                                              history_images=history_image_data)
                    forward_dict = _reduce_dict(forward_dict)
                    epoch_dicts.append(detach_dict(forward_dict))

                epoch_summary = compute_dict_mean(epoch_dicts)
                validation_history.append(epoch_summary)
                val_epochs.append(epoch)

                epoch_val_loss = epoch_summary['loss']
                epoch_val_l1_final = epoch_summary['l1_final']
                if epoch_val_l1_final < min_val_loss:
                    min_val_loss = epoch_val_l1_final
                    # Move state_dict to CPU to save GPU memory (~490MB)
                    cpu_state = {k: v.cpu() for k, v in policy_core.state_dict().items()}
                    best_ckpt_info = (epoch, min_val_loss, cpu_state)
                    # Save best checkpoint immediately to survive crashes
                    best_ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
                    torch.save(best_ckpt_info[2], best_ckpt_path)
                    print(f'*** New best at epoch {epoch}, val l1_final: {min_val_loss:.5f} (saved) ***')

            print(f'Val loss: {epoch_val_loss:.5f}, l1_final: {epoch_val_l1_final:.5f} '
                  f'(best: epoch {best_ckpt_info[0]}, l1_final {best_ckpt_info[1]:.5f})')
            summary_string = ' '.join(f'{k}: {v.item():.4f}' for k, v in epoch_summary.items())
            print(summary_string)

        # --- Training ---
        torch.cuda.empty_cache()
        policy.train()
        policy_core.optimizer.zero_grad()
        accum_steps = grad_accum_steps
        for batch_idx, data in enumerate(train_dataloader):
            image_data, qpos_data, action_data, is_pad, future_image_data, history_image_data = _prepare_data(data)

            with torch.cuda.amp.autocast():
                forward_dict = policy(qpos_data, image_data, action_data, is_pad,
                                      future_images=future_image_data,
                                      epoch=epoch, total_epochs=num_epochs,
                                      history_images=history_image_data)
                forward_dict = _reduce_dict(forward_dict)
                loss = forward_dict['loss'] / accum_steps

            scaler.scale(loss).backward()
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                scaler.unscale_(policy_core.optimizer)
                nn.utils.clip_grad_norm_(policy_core.model.parameters(), max_norm=10.0)
                scaler.step(policy_core.optimizer)
                scaler.update()
                policy_core.optimizer.zero_grad()
                torch.cuda.empty_cache()  # aggressive defrag after every optimizer step
            forward_dict['loss'] = forward_dict['loss'].detach()  # un-scaled for logging
            train_history.append(detach_dict(forward_dict))

        n_batches = len(train_dataloader)
        epoch_summary = compute_dict_mean(
            train_history[n_batches * epoch:n_batches * (epoch + 1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ' '.join(f'{k}: {v.item():.4f}' for k, v in epoch_summary.items())
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy_core.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, val_epochs, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, 'policy_last.ckpt')
    torch.save(policy_core.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished: Seed {seed}, best val l1_final {min_val_loss:.6f} at epoch {best_epoch}')

    plot_history(train_history, validation_history, val_epochs, num_epochs, ckpt_dir, seed)

    with open(os.path.join(ckpt_dir, 'train_history.pkl'), 'wb') as f:
        pickle.dump(train_history, f)
    with open(os.path.join(ckpt_dir, 'validation_history.pkl'), 'wb') as f:
        pickle.dump({'history': validation_history, 'epochs': val_epochs}, f)

    return best_ckpt_info


def plot_history(train_history, validation_history, val_epochs, num_epochs, ckpt_dir, seed):
    n_val = len(validation_history)
    n_train = len(train_history)
    if n_val == 0 or n_train == 0:
        return
    actual_epochs = max(val_epochs[-1], 1) if val_epochs else num_epochs
    batches_per_epoch = n_train / max(actual_epochs, 1)
    train_x = np.arange(n_train) / max(batches_per_epoch, 1)
    val_x = np.array(val_epochs) if val_epochs else np.arange(n_val)
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
