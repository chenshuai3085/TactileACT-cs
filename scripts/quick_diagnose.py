"""Quick diagnostic script for TFAC checkpoints.
Usage: python scripts/quick_diagnose.py <checkpoint_dir> [--epoch N]
"""
import argparse
import os
import json
import torch
import numpy as np
from pathlib import Path


def parse_plots_data(plot_dir):
    """Try to extract data from saved numpy arrays or parse from training."""
    # Look for saved training history
    for fname in ['train_history.json', 'val_history.json', 'training_log.json']:
        fpath = os.path.join(plot_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                return json.load(f)
    return None


def analyze_checkpoint(ckpt_path, config_path=None):
    """Analyze a checkpoint for key diagnostic metrics."""
    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Check if it's a full state dict or has metadata
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
        if 'epoch' in ckpt:
            print(f"Epoch: {ckpt['epoch']}")
        if 'best_val_loss' in ckpt:
            print(f"Best val loss: {ckpt['best_val_loss']:.6f}")
    else:
        state = ckpt

    # Count parameters
    total_params = sum(v.numel() for v in state.values())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    # Check key components
    components = {}
    for k in state.keys():
        parts = k.split('.')
        if len(parts) >= 2:
            comp = parts[0] + '.' + parts[1]
        else:
            comp = parts[0]
        if comp not in components:
            components[comp] = 0
        components[comp] += state[k].numel()

    print(f"\nComponent parameter counts:")
    for comp, count in sorted(components.items(), key=lambda x: -x[1])[:15]:
        print(f"  {comp}: {count/1e6:.2f}M")

    # Check for key diagnostic values
    if 'model.residual_alpha' in state:
        alpha = state['model.residual_alpha'].item()
        print(f"\nResidual alpha: {alpha:.4f} (sigmoid={torch.sigmoid(torch.tensor(alpha)).item():.4f})")

    if 'model.contrastive.log_temp' in state:
        log_temp = state['model.contrastive.log_temp'].item()
        print(f"Contrastive temperature: {np.exp(log_temp):.4f}")

    # Check foresight pos embed
    if 'model.foresight_pos_embed' in state:
        fpe = state['model.foresight_pos_embed']
        print(f"\nForesight pos_embed: shape={list(fpe.shape)}, norm={fpe.norm():.4f}")

    # Check marker pos embed
    if 'model.marker_pos_embed' in state:
        mpe = state['model.marker_pos_embed']
        print(f"Marker pos_embed: shape={list(mpe.shape)}, norm={mpe.norm():.4f}")

    # Check gate values if present
    gate_keys = [k for k in state.keys() if 'gate' in k.lower() or 'fusion' in k.lower()]
    if gate_keys:
        print(f"\nGate/Fusion parameters:")
        for k in gate_keys[:10]:
            v = state[k]
            print(f"  {k}: shape={list(v.shape)}, norm={v.norm():.4f}")

    # Action head comparison
    draft_head = state.get('model.action_head_draft.weight')
    final_head = state.get('model.action_head_final.weight')
    if draft_head is not None and final_head is not None:
        cos_sim = torch.nn.functional.cosine_similarity(
            draft_head.flatten().unsqueeze(0),
            final_head.flatten().unsqueeze(0)
        ).item()
        print(f"\nAction head similarity (draft vs final): cosine={cos_sim:.4f}")

    # Query embed comparison
    draft_q = state.get('model.query_embed_draft.weight')
    final_q = state.get('model.query_embed_final.weight')
    if draft_q is not None and final_q is not None:
        cos_sim = torch.nn.functional.cosine_similarity(
            draft_q.flatten().unsqueeze(0),
            final_q.flatten().unsqueeze(0)
        ).item()
        print(f"Query embed similarity (draft vs final): cosine={cos_sim:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_dir', help='Directory with checkpoints')
    parser.add_argument('--config', help='Config file path')
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir

    # Analyze best checkpoint
    best_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    if os.path.exists(best_path):
        analyze_checkpoint(best_path, args.config)

    # List all epoch checkpoints
    epoch_ckpts = sorted([
        f for f in os.listdir(ckpt_dir)
        if f.startswith('policy_epoch_') and f.endswith('.ckpt')
    ])
    if epoch_ckpts:
        print(f"\nAvailable epoch checkpoints: {len(epoch_ckpts)}")
        for f in epoch_ckpts[-3:]:
            print(f"  {f}")


if __name__ == '__main__':
    main()
