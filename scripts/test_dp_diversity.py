"""
Test DP diversity & quality: generate K trajectories from different noise seeds,
measure EE spread (diversity) and EE error (quality).

Also test ACT diversity for fair comparison.

Usage:
  conda run -n TactileACT python scripts/test_dp_diversity.py \
    --dp_dir /home/chenshuai/Project/output/dp_joint7 \
    --act_dir /home/chenshuai/data/xiaomi_act/act_kl001_clip1099 \
    --dataset_dir /home/chenshuai/data/dataset/260309_0310 \
    --K 16 --episodes 50,150,250
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
import h5py
from torchvision import transforms
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'diffusion'))
from network import ConditionalUnet1D

try:
    from clip_pretraining_xiaomi import modified_resnet18
except ImportError:
    from clip_pretraining import modified_resnet18


# ==================== FK ====================
def rpy_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]])

def make_transform(xyz, rpy):
    T = np.eye(4); T[:3,:3] = rpy_matrix(*rpy); T[:3,3] = xyz; return T

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(4); R[0,0]=c; R[0,1]=-s; R[1,0]=s; R[1,1]=c; return R

JOINTS = [
    {"xyz":[0,0,0.2405],"rpy":[0,0,0]},
    {"xyz":[0,0,0],     "rpy":[1.5708,-1.5708,0]},
    {"xyz":[0.256,0,0], "rpy":[0,0,1.5708]},
    {"xyz":[0,-0.21,0], "rpy":[1.5708,0,0]},
    {"xyz":[0,0,0],     "rpy":[-1.5708,0,0]},
    {"xyz":[0,-0.144,0],"rpy":[1.5708,0,0]},
]

def fk(joint_deg):
    T = np.eye(4)
    for i, jt in enumerate(JOINTS):
        T = T @ make_transform(jt["xyz"], jt["rpy"]) @ rot_z(np.radians(joint_deg[i]))
    return T[:3, 3]

def joints_to_ee(joints):
    return np.array([fk(joints[t, :6]) for t in range(joints.shape[0])])


# ==================== DP Inference ====================
class VisionEncoder(nn.Module):
    def __init__(self, camera_names):
        super().__init__()
        self.camera_names = camera_names
        self.shared_vision = nn.Sequential(modified_resnet18(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.gelsight_encoder = nn.Sequential(modified_resnet18(), nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, images_list):
        features = []
        for i, cam in enumerate(self.camera_names):
            if cam == 'gelsight':
                features.append(self.gelsight_encoder(images_list[i]))
            else:
                features.append(self.shared_vision(images_list[i]))
        return torch.cat(features, dim=-1)


@torch.no_grad()
def dp_infer_K(noise_pred_net, vision_encoder, obs_cond, noise_scheduler,
               action_dim, pred_horizon, K, device):
    """Generate K action trajectories from K different noise seeds."""
    noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)

    # K different initial noises
    noisy = torch.randn(K, pred_horizon, action_dim, device=device)
    obs_cond_K = obs_cond.expand(K, -1)

    for t in noise_scheduler.timesteps:
        noise_pred = noise_pred_net(noisy, t, global_cond=obs_cond_K)
        noisy = noise_scheduler.step(noise_pred, t, noisy).prev_sample

    return noisy  # (K, pred_horizon, action_dim)


# ==================== ACT Inference ====================
def load_act_policy(ckpt_dir, device):
    with open(os.path.join(ckpt_dir, 'args.json'), 'r') as f:
        args = json.load(f)
    from policy import ACTPolicy
    camera_names = args['camera_names']
    backbone_type = args.get('backbone', 'resnet18')
    pretrained_backbones = None
    camera_backbone_mapping = None
    if backbone_type == 'clip_backbone':
        try:
            from clip_pretraining_xiaomi import modified_resnet18 as mr18
        except ImportError:
            from clip_pretraining import modified_resnet18 as mr18
        vision_model = mr18()
        gelsight_model = mr18()
        camera_backbone_mapping = {cam: 0 for cam in camera_names}
        camera_backbone_mapping['gelsight'] = 1
        pretrained_backbones = [vision_model, gelsight_model]

    policy = ACTPolicy(
        state_dim=args.get('state_dim', 7),
        hidden_dim=args['hidden_dim'],
        position_embedding_type=args.get('position_embedding', 'sine'),
        lr_backbone=args.get('lr_backbone', 1e-5),
        masks=args.get('masks', False),
        backbone_type=backbone_type,
        dilation=args.get('dilation', False),
        dropout=args.get('dropout', 0.1),
        nheads=args['nheads'],
        dim_feedforward=args['dim_feedforward'],
        num_enc_layers=args['enc_layers'],
        num_dec_layers=args['dec_layers'],
        pre_norm=args.get('pre_norm', False),
        num_queries=args['chunk_size'],
        camera_names=camera_names,
        z_dimension=args.get('z_dimension', 32),
        lr=args['lr'],
        weight_decay=args.get('weight_decay', 1e-4),
        kl_weight=args['kl_weight'],
        pretrained_backbones=pretrained_backbones,
        cam_backbone_mapping=camera_backbone_mapping,
    )
    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    if not os.path.exists(ckpt_path):
        # fallback: find latest epoch checkpoint
        import glob
        epoch_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'policy_epoch_*_seed_*.ckpt')))
        if epoch_ckpts:
            ckpt_path = epoch_ckpts[-1]
            print(f"  Using fallback checkpoint: {os.path.basename(ckpt_path)}")
        else:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    policy.load_state_dict(ckpt, strict=False)
    policy.to(device).eval()
    return policy, args


@torch.no_grad()
def act_infer_K(policy, qpos, images, K, z_dim, T_temp, device):
    """Generate K trajectories from K different z samples."""
    results = []
    # z=0 baseline
    a0 = policy(qpos, images).cpu().squeeze(0).numpy()
    results.append(a0)
    # K-1 z-sampled
    for _ in range(K - 1):
        z = (T_temp * torch.randn(z_dim)).numpy()
        a = policy(qpos, images, z=z).cpu().squeeze(0).numpy()
        results.append(a)
    return np.array(results)  # (K, chunk, action_dim)


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dp_dir', type=str, default=None)
    parser.add_argument('--act_dir', type=str, default=None)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--K', type=int, default=16)
    parser.add_argument('--episodes', type=str, default='50,150,250')
    parser.add_argument('--T', type=float, default=3.0, help='ACT z temperature')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--pred_interval', type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    test_episodes = [int(x) for x in args.episodes.split(',')]
    img_norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    results = {}

    # ==================== DP Test ====================
    if args.dp_dir and os.path.exists(os.path.join(args.dp_dir, 'dp_best.pth')):
        print("=" * 60)
        print("Testing DP diversity...")
        with open(os.path.join(args.dp_dir, 'config.json')) as f:
            dp_cfg = json.load(f)

        camera_names = dp_cfg['camera_names'].split(',') if isinstance(dp_cfg['camera_names'], str) else dp_cfg['camera_names']
        obs_horizon = dp_cfg.get('obs_horizon', 2)
        pred_horizon = dp_cfg['pred_horizon']
        action_dim = dp_cfg['action_dim']
        down_dims = dp_cfg['down_dims']
        ns = dp_cfg['norm_stats']
        action_min = np.array(ns['action_min'])
        action_max = np.array(ns['action_max'])
        qpos_min = np.array(ns['qpos_min'])
        qpos_max = np.array(ns['qpos_max'])

        vision_encoder = VisionEncoder(camera_names).to(device)
        noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=dp_cfg['global_cond_dim'],
            diffusion_step_embed_dim=dp_cfg.get('diffusion_step_embed_dim', 128),
            down_dims=down_dims, kernel_size=5,
        ).to(device)

        ckpt = torch.load(os.path.join(args.dp_dir, 'dp_best.pth'), map_location='cpu')
        # use EMA weights if available
        if 'ema_vis' in ckpt:
            vision_encoder.load_state_dict(ckpt['ema_vis'])
            print("Using EMA vision weights")
        else:
            vision_encoder.load_state_dict(ckpt['vision_encoder'])
        if 'ema_net' in ckpt:
            noise_pred_net.load_state_dict(ckpt['ema_net'])
            print("Using EMA noise net weights")
        else:
            noise_pred_net.load_state_dict(ckpt['noise_pred_net'])

        vision_encoder.eval()
        noise_pred_net.eval()

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=dp_cfg.get('num_train_timesteps', 100),
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True, prediction_type='epsilon',
        )

        def unnorm_action(a_norm):
            return (a_norm + 1) / 2 * (action_max - action_min) + action_min

        dp_results = []
        for ep_id in test_episodes:
            path = os.path.join(args.dataset_dir, f'episode_{ep_id}.hdf5')
            if not os.path.exists(path):
                print(f"  Episode {ep_id} not found, skip")
                continue

            with h5py.File(path, 'r') as f:
                ep_len = f[f'observations/{dp_cfg["proprio_key"]}'].shape[0]
                all_qpos = f[f'observations/{dp_cfg["proprio_key"]}'][()]
                all_action = f[f'/{dp_cfg["action_key"]}'][()]

            gt_ee = joints_to_ee(all_action)
            pred_starts = list(range(0, ep_len - pred_horizon, args.pred_interval))

            ep_spreads, ep_errors, ep_best_k = [], [], []
            for t in pred_starts:
                with h5py.File(path, 'r') as f:
                    obs_feats_list = []
                    for ot in range(obs_horizon):
                        tt = max(0, t - obs_horizon + 1 + ot)
                        imgs = []
                        for cam in camera_names:
                            if cam == 'gelsight':
                                img = f[f'observations/tac/{dp_cfg["tac_side"]}/{dp_cfg["tac_img_key"]}'][tt]
                            else:
                                img = f[f'observations/images/{cam}'][tt]
                            img_t = torch.tensor(img, dtype=torch.float32).permute(2,0,1) / 255.0
                            img_t = img_norm(img_t).unsqueeze(0).to(device)
                            imgs.append(img_t)

                        vis_feat = vision_encoder(imgs)
                        qpos_t = all_qpos[tt]
                        qpos_norm = (qpos_t - qpos_min) / (qpos_max - qpos_min + 1e-8) * 2 - 1
                        qpos_t_dev = torch.tensor(qpos_norm, dtype=torch.float32).unsqueeze(0).to(device)
                        obs_feats_list.append(torch.cat([vis_feat, qpos_t_dev], dim=-1))

                    obs_cond = torch.cat(obs_feats_list, dim=-1)

                # generate K trajectories
                actions_K = dp_infer_K(noise_pred_net, vision_encoder, obs_cond,
                                       noise_scheduler, action_dim, pred_horizon, args.K, device)
                actions_K = actions_K.cpu().numpy()

                # unnormalize and compute EE
                ee_K = []
                for k in range(args.K):
                    a_raw = unnorm_action(actions_K[k])
                    ee_K.append(joints_to_ee(a_raw))
                ee_K = np.array(ee_K)  # (K, pred_horizon, 3)

                gt_chunk = gt_ee[t:t+pred_horizon]
                spread = ee_K.std(axis=0).mean() * 1000  # mm
                per_k_errors = [np.linalg.norm(ee_K[k] - gt_chunk, axis=1).mean() * 1000
                                for k in range(args.K)]
                error = np.mean(per_k_errors)
                best_k_error = np.min(per_k_errors)
                ep_spreads.append(spread)
                ep_errors.append(error)
                ep_best_k.append(best_k_error)

            avg_spread = np.mean(ep_spreads)
            avg_error = np.mean(ep_errors)
            avg_best_k = np.mean(ep_best_k)
            print(f"  DP ep{ep_id}: spread={avg_spread:.2f}mm, error={avg_error:.2f}mm, "
                  f"best-of-K={avg_best_k:.2f}mm, improvement={((avg_error-avg_best_k)/avg_error*100):.1f}%")
            dp_results.append({'ep': ep_id, 'spread': avg_spread, 'error': avg_error, 'best_k': avg_best_k})

        results['dp'] = dp_results

    # ==================== ACT Test ====================
    has_act_ckpt = args.act_dir and (
        os.path.exists(os.path.join(args.act_dir, 'policy_best.ckpt')) or
        len([f for f in os.listdir(args.act_dir) if f.startswith('policy_epoch_')]) > 0
    ) if args.act_dir else False
    if has_act_ckpt:
        print("=" * 60)
        print("Testing ACT diversity...")
        policy, act_args = load_act_policy(args.act_dir, device)
        camera_names_act = act_args['camera_names']
        chunk_size = act_args['chunk_size']
        z_dim = act_args.get('z_dimension', 32)
        norm_stats = act_args['norm_stats']
        action_mean = np.array(norm_stats['action_mean'])
        action_std = np.array(norm_stats['action_std'])
        qpos_mean = torch.tensor(norm_stats['qpos_mean'], dtype=torch.float32)
        qpos_std = torch.tensor(norm_stats['qpos_std'], dtype=torch.float32)

        act_results = []
        for ep_id in test_episodes:
            path = os.path.join(args.dataset_dir, f'episode_{ep_id}.hdf5')
            if not os.path.exists(path):
                print(f"  Episode {ep_id} not found, skip")
                continue

            with h5py.File(path, 'r') as f:
                ep_len = f[f'observations/{act_args.get("proprio_key", "proprio_joint")}'].shape[0]
                all_qpos = f[f'observations/{act_args.get("proprio_key", "proprio_joint")}'][()]
                all_action = f[f'/{act_args.get("action_key", "actions/joint_abs")}'][()]

                gt_ee = joints_to_ee(all_action)
                pred_starts = list(range(0, ep_len - chunk_size, args.pred_interval))

                ep_spreads, ep_errors, ep_best_k = [], [], []
                for t in pred_starts:
                    qpos_t = torch.tensor(all_qpos[t], dtype=torch.float32).unsqueeze(0)
                    qpos_t = ((qpos_t - qpos_mean) / qpos_std).to(device)

                    images_t = []
                    for cam in camera_names_act:
                        if cam == 'gelsight':
                            tac_side = act_args.get('tac_side', 'left')
                            tac_key = act_args.get('tac_img_key', 'img')
                            img = f[f'observations/tac/{tac_side}/{tac_key}'][t]
                        elif cam == 'blank':
                            img = np.zeros((200, 266, 3), dtype=np.uint8)
                        else:
                            img = f[f'observations/images/{cam}'][t]
                        img_t = torch.tensor(img, dtype=torch.float32).permute(2,0,1) / 255.0
                        img_t = img_norm(img_t).unsqueeze(0).to(device)
                        images_t.append(img_t)

                    actions_K = act_infer_K(policy, qpos_t, images_t, args.K, z_dim, args.T, device)
                    actions_K_raw = actions_K * action_std + action_mean

                    ee_K = np.array([joints_to_ee(actions_K_raw[k]) for k in range(args.K)])
                    gt_chunk = gt_ee[t:t+chunk_size]
                    spread = ee_K.std(axis=0).mean() * 1000
                    per_k_errors = [np.linalg.norm(ee_K[k] - gt_chunk, axis=1).mean() * 1000
                                    for k in range(args.K)]
                    error = np.mean(per_k_errors)
                    best_k_error = np.min(per_k_errors)
                    ep_spreads.append(spread)
                    ep_errors.append(error)
                    ep_best_k.append(best_k_error)

            avg_spread = np.mean(ep_spreads)
            avg_error = np.mean(ep_errors)
            avg_best_k = np.mean(ep_best_k)
            print(f"  ACT ep{ep_id}: spread={avg_spread:.2f}mm, error={avg_error:.2f}mm, "
                  f"best-of-K={avg_best_k:.2f}mm, improvement={((avg_error-avg_best_k)/avg_error*100):.1f}%")
            act_results.append({'ep': ep_id, 'spread': avg_spread, 'error': avg_error, 'best_k': avg_best_k})

        results['act'] = act_results

    # ==================== Summary ====================
    print("\n" + "=" * 60)
    print("SUMMARY: Diversity (spread) vs Quality (error) vs Best-of-K")
    print("=" * 60)
    for method, res_list in results.items():
        spreads = [r['spread'] for r in res_list]
        errors = [r['error'] for r in res_list]
        best_ks = [r['best_k'] for r in res_list]
        avg_spread = np.mean(spreads)
        avg_error = np.mean(errors)
        avg_best_k = np.mean(best_ks)
        improvement = (avg_error - avg_best_k) / avg_error * 100

        print(f"\n{method.upper()}:")
        print(f"  Avg spread:        {avg_spread:.2f} mm")
        print(f"  Avg error (mean):  {avg_error:.2f} mm")
        print(f"  Avg error (best-K):{avg_best_k:.2f} mm")
        print(f"  Reranking gain:    {improvement:.1f}%  ({avg_error:.2f} → {avg_best_k:.2f} mm)")
        print(f"  Spread/Error:      {avg_spread/avg_error*100:.1f}%")
        for r in res_list:
            print(f"    ep{r['ep']}: spread={r['spread']:.2f}, err={r['error']:.2f}, "
                  f"best-K={r['best_k']:.2f}, gain={((r['error']-r['best_k'])/r['error']*100):.1f}%")

    print("\n" + "=" * 60)
    print("RERANKING VIABILITY ASSESSMENT")
    print("=" * 60)
    for method, res_list in results.items():
        avg_spread = np.mean([r['spread'] for r in res_list])
        avg_error = np.mean([r['error'] for r in res_list])
        avg_best_k = np.mean([r['best_k'] for r in res_list])
        gain = (avg_error - avg_best_k) / avg_error * 100
        if avg_spread > 3 and avg_error < 15 and gain > 15:
            verdict = "VIABLE"
        elif avg_spread > 1 and gain > 5:
            verdict = "MARGINAL"
        else:
            verdict = "NOT VIABLE"
        print(f"  {method.upper()}: spread={avg_spread:.2f}mm, error={avg_error:.2f}mm, "
              f"best-K gain={gain:.1f}% → {verdict}")


if __name__ == '__main__':
    main()
