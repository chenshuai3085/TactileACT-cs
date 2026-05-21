"""
Diagnose: Score using only z_int (9 dims, force proxy) vs full z (144 dims).

Hypothesis: z_int has more discrimination because it directly relates to
contact force, which is what differs between candidates.

Also tests: per-dimension variance analysis to find which dims are most
discriminative across K candidates.

Usage:
    python scripts/diagnose_zint_scoring.py --gpu 0
"""
import os, sys, json, pickle, argparse
import numpy as np
import torch
import copy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'diffusion'))
sys.path.insert(0, os.path.join(ROOT, 'TFAC_V5'))

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from network import ConditionalUnet1D, get_resnet, replace_bn_with_gn
from pretrain_latent_foresight import LatentForesightPretrainModel
from tactile_vae import build_tactile_vae
import torchvision.transforms as transforms
import h5py


class OfficialVisionEncoder(torch.nn.Module):
    def __init__(self, camera_names):
        super().__init__()
        self.camera_names = camera_names
        base = get_resnet('resnet18')
        replace_bn_with_gn(base, features_per_group=16)
        self.encoders = torch.nn.ModuleDict()
        for cam in camera_names:
            self.encoders[cam] = copy.deepcopy(base)
        self.feat_dim = 512
    def forward(self, images_dict):
        features = []
        for cam in self.camera_names:
            features.append(self.encoders[cam](images_dict[cam]))
        return torch.cat(features, dim=-1)


class FrozenTactileVAEEncoder(torch.nn.Module):
    TAC_MEAN = np.array([0.2102, -0.6422], dtype=np.float32)
    TAC_STD = np.array([1.6805, 3.6717], dtype=np.float32)
    def __init__(self, vae_ckpt, latent_dim=16, temporal_window=8):
        super().__init__()
        self.vae = build_tactile_vae(latent_dim=latent_dim, temporal_window=temporal_window)
        if vae_ckpt and os.path.exists(vae_ckpt):
            ckpt = torch.load(vae_ckpt, map_location='cpu', weights_only=False)
            sd = ckpt.get('model_state_dict', ckpt)
            self.vae.load_state_dict(sd)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.feat_dim = latent_dim * 3 * 3
        self.register_buffer('tac_mean', torch.tensor(self.TAC_MEAN))
        self.register_buffer('tac_std', torch.tensor(self.TAC_STD))
    @torch.no_grad()
    def forward(self, marker_seq):
        m = (marker_seq - self.tac_mean) / self.tac_std
        if m.dim() == 4: m = m.unsqueeze(1)
        z, _ = self.vae.encode_single_frame(m)
        return z.flatten(1)


def load_foresight(foresight_dir, joint_ckpt_path, device):
    with open(os.path.join(foresight_dir, 'args.json')) as f:
        fs_config = json.load(f)
    camera_names = fs_config.get('camera_names', ['global', 'wrist', 'gelsight'])
    model = LatentForesightPretrainModel(
        camera_names=camera_names,
        cam_backbone_mapping={cam: 0 for cam in camera_names},
        hidden_dim=fs_config.get('hidden_dim', 512),
        state_dim=fs_config.get('state_dim', 7),
        foresight_layers=fs_config.get('foresight_layers', 3),
        foresight_nheads=fs_config.get('foresight_nheads', 8),
        foresight_dim_feedforward=fs_config.get('foresight_dim_feedforward', 2048),
        dropout=fs_config.get('dropout', 0.1),
        tactile_mode=fs_config.get('tactile_mode', 'marker'),
        max_history=fs_config.get('max_history', 8),
        predict_horizon=fs_config.get('predict_horizon', 1),
        tactile_vae_ckpt=fs_config.get('tactile_vae_ckpt'),
        tactile_vae_latent_dim=fs_config.get('tactile_vae_latent_dim', 16),
        use_delta_pred=fs_config.get('use_delta_pred', False),
        residual_prediction=fs_config.get('residual_prediction', False),
    ).to(device)
    ckpt = torch.load(joint_ckpt_path, map_location=device, weights_only=False)
    if 'foresight' in ckpt:
        model.load_state_dict(ckpt['foresight'], strict=False)
    model.eval()
    with open(os.path.join(foresight_dir, 'dataset_stats.pkl'), 'rb') as f:
        stats = pickle.load(f)
    fs_norm = {k: torch.tensor(stats[k], dtype=torch.float32, device=device)
               for k in ['action_mean', 'action_std', 'qpos_mean', 'qpos_std']}
    return model, fs_config, fs_norm


def load_sample(dataset_dir, ep_idx, ts, cams, tac_history, device):
    path = os.path.join(dataset_dir, f'episode_{ep_idx}.hdf5')
    resize_tf = transforms.Resize((240, 320))
    crop_tf = transforms.CenterCrop((216, 288))
    img_norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    with h5py.File(path, 'r') as f:
        qpos = f['observations/proprio_joint'][()].astype(np.float32)
        marker = f['observations/tac/left/marker_offset'][()].astype(np.float32)
        ep_len = qpos.shape[0]
        imgs = {}
        for cam in cams:
            imgs[cam] = f[f'observations/images/{cam}'][()]
    obs_imgs = {}
    for cam in cams:
        img = torch.from_numpy(imgs[cam][ts]).float().div_(255).permute(2,0,1)
        obs_imgs[cam] = img_norm(crop_tf(resize_tf(img))).unsqueeze(0).to(device)
    frames = [marker[max(0, ts-tac_history+1+k)] for k in range(tac_history)]
    mh = torch.from_numpy(np.stack(frames)).unsqueeze(0).to(device)
    qp = torch.from_numpy(qpos[ts]).unsqueeze(0).to(device)
    return obs_imgs, mh, qp, ts/(ep_len-1), ep_len


@torch.no_grad()
def sample_k(net, sched, obs_cond, adim, ph, K, dev):
    sched.set_timesteps(100)
    oc = obs_cond.expand(K, -1)
    a = torch.randn((K, ph, adim), device=dev)
    for t in sched.timesteps:
        pred = net(a, t.unsqueeze(0).expand(K).to(dev), global_cond=oc)
        a = sched.step(pred, t, a).prev_sample
    return a


@torch.no_grad()
def foresight_predict(model, cfg, norm, cands, obs_imgs, mh, qp, amin, amax, dev):
    K = cands.shape[0]
    cs = cfg.get('chunk_size', 10)
    raw = (cands+1)/2*(amax-amin)+amin
    fs_a = (raw - norm['action_mean'])/norm['action_std']
    fs_a = fs_a[:, :cs, :]
    fs_cams = cfg.get('camera_names', ['global','wrist','gelsight'])
    fs_imgs = []
    for cam in fs_cams:
        if cam == 'gelsight':
            fs_imgs.append(mh.expand(K,-1,-1,-1,-1))
        elif cam in obs_imgs:
            fs_imgs.append(obs_imgs[cam].expand(K,-1,-1,-1))
    qfs = ((qp - norm['qpos_mean'])/norm['qpos_std']).expand(K,-1)
    zp, _, _, zc, _, _ = model(fs_imgs, fs_a, future_images=None, qpos=qfs)
    if zp.dim() == 3: zp = zp[:, -1, :]
    return zp, zc[:1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--K', type=int, default=16)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    K = args.K

    joint_ckpt_dir = '/home/chenshuai/Project/output/ckpt/dp_foresight_joint_0209'
    joint_ckpt_path = os.path.join(joint_ckpt_dir, 'dp_topk_ep130_loss0.0029.pth')
    foresight_dir = '/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_0209'
    dataset_dir = '/home/chenshuai/data/dataset/0209-0210_truncated'
    vae_ckpt = '/home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt'
    demo_stats_path = '/home/chenshuai/Project/output/demo_z_stats.pkl'

    with open(os.path.join(joint_ckpt_dir, 'config.json')) as f:
        config = json.load(f)
    camera_names = config['camera_names']
    if isinstance(camera_names, str): camera_names = camera_names.split(',')
    action_dim = config['action_dim']
    pred_horizon = config['pred_horizon']
    obs_horizon = config.get('obs_horizon', 2)
    tac_history = config.get('tac_history', 8)
    ns = config['norm_stats']
    amin = torch.tensor(ns['action_min'], dtype=torch.float32, device=device)
    amax = torch.tensor(ns['action_max'], dtype=torch.float32, device=device)
    qmin = np.array(ns['qpos_min'], dtype=np.float32)
    qmax = np.array(ns['qpos_max'], dtype=np.float32)

    print("Loading models...")
    vis_cams = [c for c in camera_names if c != 'gelsight']
    vis_enc = OfficialVisionEncoder(vis_cams).to(device)
    tac_enc = FrozenTactileVAEEncoder(vae_ckpt, 16, tac_history).to(device)
    net = ConditionalUnet1D(
        input_dim=action_dim, global_cond_dim=config['global_cond_dim'],
        diffusion_step_embed_dim=config.get('diffusion_step_embed_dim', 128),
        down_dims=config['down_dims'], kernel_size=5,
    ).to(device)

    ckpt = torch.load(joint_ckpt_path, map_location=device, weights_only=False)
    vis_enc.load_state_dict(ckpt.get('ema_vis', ckpt.get('vision_encoder')))
    net.load_state_dict(ckpt.get('ema_net', ckpt.get('noise_pred_net')))
    vis_enc.eval(); net.eval(); tac_enc.eval()

    foresight, fs_cfg, fs_norm = load_foresight(foresight_dir, joint_ckpt_path, device)
    sched = DDPMScheduler(num_train_timesteps=config.get('num_train_timesteps',100),
                          beta_schedule='squaredcos_cap_v2', clip_sample=True,
                          prediction_type='epsilon')

    # Load demo stats
    with open(demo_stats_path, 'rb') as f:
        demo_data = pickle.load(f)
    n_bins = demo_data['n_bins']

    # Extract z_int stats (first 9 dims = channel 0 of 16×3×3 flattened)
    # z is stored as (16, 3, 3) flattened to (144,)
    # z_int = z[0, :, :] = first 9 values (channel 0)
    # z_pat = z[1:16, :, :] = remaining 135 values
    stage_mu_int = [torch.tensor(s['mu'][:9], dtype=torch.float32, device=device) for s in demo_data['stage_stats']]
    stage_std_int = [torch.tensor(s['std'][:9], dtype=torch.float32, device=device) for s in demo_data['stage_stats']]
    stage_mu_full = [torch.tensor(s['mu'], dtype=torch.float32, device=device) for s in demo_data['stage_stats']]
    stage_std_full = [torch.tensor(s['std'], dtype=torch.float32, device=device) for s in demo_data['stage_stats']]

    print(f"\n{'='*70}")
    print("Z_INT (9-dim force) vs FULL Z (144-dim) SCORING COMPARISON")
    print(f"{'='*70}")

    test_cases = [(0,50), (0,130), (0,180), (5,80), (5,130),
                  (10,130), (10,180), (30,130), (30,180), (50,130), (50,180)]

    results = {'naive':[], 'full_staged':[], 'int_staged':[], 'int_delta':[],
               'int_demo_delta':[], 'per_dim_top5':[]}

    for ep_idx, ts in test_cases:
        try:
            obs_imgs, mh, qp, progress, ep_len = load_sample(
                dataset_dir, ep_idx, ts, vis_cams, tac_history, device)
        except:
            continue

        with torch.no_grad():
            vf = vis_enc(obs_imgs)
            tf = tac_enc(mh)
            qn = torch.from_numpy((qp.cpu().numpy()-qmin)/(qmax-qmin+1e-8)*2-1).float().to(device)
            oc = torch.cat([vf, tf, qn], dim=-1)
            if obs_horizon == 2: oc = torch.cat([oc, oc], dim=-1)

        cands = sample_k(net, sched, oc, action_dim, pred_horizon, K, device)
        zp, zc = foresight_predict(foresight, fs_cfg, fs_norm, cands,
                                    obs_imgs, mh, qp, amin, amax, device)

        # Split z into z_int and z_pat
        zp_int = zp[:, :9]   # (K, 9) - force proxy
        zp_pat = zp[:, 9:]   # (K, 135) - pattern
        zc_int = zc[:, :9]   # (1, 9)

        bin_idx = min(int(progress * n_bins), n_bins - 1)
        mu_int = stage_mu_int[bin_idx]
        std_int = stage_std_int[bin_idx]
        mu_full = stage_mu_full[bin_idx]
        std_full = stage_std_full[bin_idx]

        # === Scoring methods ===
        # 1. Naive: -||z_pred - z_cur|| (full 144 dims)
        s_naive = -torch.norm(zp - zc.expand_as(zp), dim=-1)

        # 2. Full staged: -mean(|z - mu| / std) on all 144 dims
        s_full = -((zp - mu_full)/std_full).abs().mean(dim=-1)

        # 3. z_int staged: -mean(|z_int - mu_int| / std_int) on 9 dims only
        s_int_staged = -((zp_int - mu_int)/std_int).abs().mean(dim=-1)

        # 4. z_int delta: -||z_int_pred - z_int_cur|| (force change)
        s_int_delta = -torch.norm(zp_int - zc_int.expand_as(zp_int), dim=-1)

        # 5. z_int demo-delta: predicted force change vs expected demo change
        # Expected: how much does z_int typically change from current stage to next?
        # Use difference between current bin mu and next bin mu as "expected delta"
        next_bin = min(bin_idx + 1, n_bins - 1)
        expected_delta = stage_mu_int[next_bin] - stage_mu_int[bin_idx]  # (9,)
        actual_delta = zp_int - zc_int  # (K, 9)
        s_int_demo_delta = -((actual_delta - expected_delta)/std_int).abs().mean(dim=-1)

        # Per-dimension analysis: which dims vary most across K?
        per_dim_std = zp.std(dim=0)  # (144,) std across K candidates
        top5_dims = per_dim_std.argsort(descending=True)[:5].cpu().numpy()
        top5_in_int = sum(1 for d in top5_dims if d < 9)

        # Compute relative ranges
        def rel_range(s):
            r = (s.max()-s.min()).item()
            m = abs(s.mean().item()) + 1e-8
            return r, r/m*100

        rn = rel_range(s_naive)
        rf = rel_range(s_full)
        ri = rel_range(s_int_staged)
        rid = rel_range(s_int_delta)
        ridd = rel_range(s_int_demo_delta)

        results['naive'].append(rn)
        results['full_staged'].append(rf)
        results['int_staged'].append(ri)
        results['int_delta'].append(rid)
        results['int_demo_delta'].append(ridd)
        results['per_dim_top5'].append(top5_in_int)

        print(f"\nEp{ep_idx} ts={ts} prog={progress:.2f} bin={bin_idx}")
        print(f"  {'Method':<20} {'Range':<10} {'RelRange%':<10} {'Best':<5}")
        print(f"  {'-'*50}")
        print(f"  {'Naive(full delta)':<20} {rn[0]:<10.4f} {rn[1]:<10.2f} {s_naive.argmax().item()}")
        print(f"  {'Full-staged(144d)':<20} {rf[0]:<10.4f} {rf[1]:<10.2f} {s_full.argmax().item()}")
        print(f"  {'Int-staged(9d)':<20} {ri[0]:<10.4f} {ri[1]:<10.2f} {s_int_staged.argmax().item()}")
        print(f"  {'Int-delta(9d)':<20} {rid[0]:<10.4f} {rid[1]:<10.2f} {s_int_delta.argmax().item()}")
        print(f"  {'Int-demo-delta(9d)':<20} {ridd[0]:<10.4f} {ridd[1]:<10.2f} {s_int_demo_delta.argmax().item()}")
        print(f"  Top-5 variant dims: {top5_dims} ({top5_in_int}/5 in z_int)")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'AvgRange':<12} {'AvgRelRange%':<12}")
    print("-"*50)
    for name in ['naive', 'full_staged', 'int_staged', 'int_delta', 'int_demo_delta']:
        r = results[name]
        if r:
            avg_r = np.mean([x[0] for x in r])
            avg_rel = np.mean([x[1] for x in r])
            print(f"  {name:<20} {avg_r:<12.5f} {avg_rel:<12.3f}%")

    # How many of the top-variant dims are in z_int?
    avg_top5_int = np.mean(results['per_dim_top5'])
    print(f"\n  Avg top-5 variant dims in z_int (out of 5): {avg_top5_int:.1f}")
    print(f"  (If >2.5, z_int carries most of the discrimination)")


if __name__ == '__main__':
    main()
