"""
Evaluate multi-dimensional CQV scoring on DP candidates.

Scoring dimensions:
  1. Smoothness: -||z_pred - z_cur|| normalized by phase delta stats
  2. Safety: penalty if z_int exceeds phase p95 bound
  3. Phase-fit: standardized distance to phase demo distribution

Compares:
  - naive: raw -||z_pred - z_cur||
  - smoothness_only: phase-normalized delta
  - safety_only: force bound check
  - fit_only: demo distribution distance
  - multi_dim: weighted combination (0.7/0.2/0.1)
  - multi_dim_v2: (0.5/0.3/0.2)

Also computes "oracle" ranking using GT future tactile to validate that
scoring CAN distinguish candidates.

Usage:
    python scripts/eval_multidim_scoring.py --gpu 0
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
            self.vae.load_state_dict(ckpt.get('model_state_dict', ckpt))
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
        images = {}
        for cam in cams:
            images[cam] = f[f'observations/images/{cam}'][()]
    obs_imgs = {}
    for cam in cams:
        img = torch.from_numpy(images[cam][ts]).float().div_(255).permute(2,0,1)
        obs_imgs[cam] = img_norm(crop_tf(resize_tf(img))).unsqueeze(0).to(device)
    frames = [marker[max(0, ts-tac_history+1+k)] for k in range(tac_history)]
    mh = torch.from_numpy(np.stack(frames)).unsqueeze(0).to(device)
    qp = torch.from_numpy(qpos[ts]).unsqueeze(0).to(device)
    return obs_imgs, mh, qp, ep_len, marker


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


class MultiDimScorer:
    """Multi-dimensional CQV scorer using phase statistics."""

    def __init__(self, phase_stats, device):
        self.device = device
        self.phase_stats = phase_stats
        # Pre-convert to tensors
        self.phase_tensors = {}
        for pid, stats in phase_stats.items():
            if stats is None:
                continue
            self.phase_tensors[pid] = {
                'delta_norm_mu': stats['delta_norm_mu'],
                'delta_norm_std': stats['delta_norm_std'],
                'z_int_norm_p95': stats['z_int_norm_p95'],
                'z_mu': torch.tensor(stats['z_mu'], dtype=torch.float32, device=device),
                'z_std': torch.tensor(stats['z_std'], dtype=torch.float32, device=device),
            }

    def score_naive(self, z_pred, z_cur):
        """Raw delta scoring."""
        return -torch.norm(z_pred - z_cur.expand_as(z_pred), dim=-1)

    def score_smoothness(self, z_pred, z_cur, phase):
        """Phase-normalized smoothness: how abnormal is the predicted change?"""
        delta = torch.norm(z_pred - z_cur.expand_as(z_pred), dim=-1)  # (K,)
        if phase not in self.phase_tensors:
            return -delta
        mu = self.phase_tensors[phase]['delta_norm_mu']
        std = self.phase_tensors[phase]['delta_norm_std']
        # Negative z-score: how many stds above expected delta
        return -(delta - mu) / (std + 1e-8)

    def score_safety(self, z_pred, phase):
        """Penalty for exceeding force safety bound."""
        z_int = z_pred[:, :9]  # (K, 9)
        z_int_norm = torch.norm(z_int, dim=-1)  # (K,)
        if phase not in self.phase_tensors:
            return torch.zeros(z_pred.shape[0], device=self.device)
        bound = self.phase_tensors[phase]['z_int_norm_p95']
        # 0 if within bound, negative penalty if exceeded
        penalty = torch.clamp(z_int_norm - bound, min=0)
        return -penalty

    def score_fit(self, z_pred, phase):
        """Standardized distance to phase demo distribution."""
        if phase not in self.phase_tensors:
            return torch.zeros(z_pred.shape[0], device=self.device)
        mu = self.phase_tensors[phase]['z_mu']
        std = self.phase_tensors[phase]['z_std']
        deviation = ((z_pred - mu) / std).abs().mean(dim=-1)
        return -deviation

    def score_multidim(self, z_pred, z_cur, phase, w1=0.7, w2=0.2, w3=0.1):
        """Weighted multi-dimensional score."""
        s1 = self.score_smoothness(z_pred, z_cur, phase)
        s2 = self.score_safety(z_pred, phase)
        s3 = self.score_fit(z_pred, phase)
        # Normalize each to ~same scale before weighting
        # s1 is in z-score units (~0 mean), s2 is 0 or negative, s3 is negative
        return w1 * s1 + w2 * s2 + w3 * s3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--K', type=int, default=16)
    parser.add_argument('--n_episodes', type=int, default=10)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    K = args.K

    # Paths
    joint_ckpt_dir = '/home/chenshuai/Project/output/ckpt/dp_foresight_joint_0209'
    joint_ckpt_path = os.path.join(joint_ckpt_dir, 'dp_topk_ep130_loss0.0029.pth')
    foresight_dir = '/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_0209'
    dataset_dir = '/home/chenshuai/data/dataset/0209-0210_truncated'
    vae_ckpt = '/home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt'
    phase_stats_path = '/home/chenshuai/Project/output/phase_scoring_stats.pkl'
    ann_path = os.path.join(dataset_dir, 'annotations.pkl')

    # Load config
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

    # Load models
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
    sched = DDPMScheduler(num_train_timesteps=config.get('num_train_timesteps', 100),
                          beta_schedule='squaredcos_cap_v2', clip_sample=True,
                          prediction_type='epsilon')

    # Load phase stats + annotations
    with open(phase_stats_path, 'rb') as f:
        phase_data = pickle.load(f)
    with open(ann_path, 'rb') as f:
        annotations = pickle.load(f)

    scorer = MultiDimScorer(phase_data['phase_stats'], device)

    # Also load VAE for oracle scoring (encode GT future tactile)
    vae_for_oracle = build_tactile_vae(latent_dim=16, temporal_window=tac_history)
    vae_ckpt_data = torch.load(vae_ckpt, map_location='cpu', weights_only=False)
    vae_for_oracle.load_state_dict(vae_ckpt_data.get('model_state_dict', vae_ckpt_data))
    vae_for_oracle = vae_for_oracle.to(device).eval()
    TAC_MEAN = torch.tensor([0.2102, -0.6422], device=device)
    TAC_STD = torch.tensor([1.6805, 3.6717], device=device)

    print(f"\n{'='*70}")
    print(f"MULTI-DIMENSIONAL SCORING EVALUATION (K={K})")
    print(f"{'='*70}")

    # Test on multiple episodes and phases
    foresight_horizon = 10
    test_episodes = list(range(0, min(50, args.n_episodes * 5), 5))[:args.n_episodes]

    all_results = {m: [] for m in ['naive', 'smoothness', 'safety', 'fit',
                                    'multi_07_02_01', 'multi_05_03_02']}
    oracle_correlations = []  # correlation between predicted and oracle rankings
    phase_breakdown = {0: [], 1: [], 2: []}  # results by phase

    n_tests = 0
    for ep_idx in test_episodes:
        ep_key = f'episode_{ep_idx}'
        if ep_key not in annotations:
            continue
        ann = annotations[ep_key]
        labels = ann['labels']

        # Test at multiple timesteps within this episode
        ep_len_ann = len(labels)
        test_ts_list = [int(ep_len_ann * p) for p in [0.2, 0.4, 0.6, 0.8]]

        for ts in test_ts_list:
            if ts >= ep_len_ann - foresight_horizon - 1:
                continue
            phase = int(labels[ts])
            if phase > 2:  # skip lift/reposition (rare, not scored)
                continue

            try:
                obs_imgs, mh, qp, ep_len, raw_marker = load_sample(
                    dataset_dir, ep_idx, ts, vis_cams, tac_history, device)
            except:
                continue

            # Encode obs
            with torch.no_grad():
                vf = vis_enc(obs_imgs)
                tf = tac_enc(mh)
                qn = torch.from_numpy(
                    (qp.cpu().numpy()-qmin)/(qmax-qmin+1e-8)*2-1
                ).float().to(device)
                oc = torch.cat([vf, tf, qn], dim=-1)
                if obs_horizon == 2:
                    oc = torch.cat([oc, oc], dim=-1)

            # Sample K candidates
            cands = sample_k(net, sched, oc, action_dim, pred_horizon, K, device)

            # Foresight predict
            zp, zc = foresight_predict(foresight, fs_cfg, fs_norm, cands,
                                        obs_imgs, mh, qp, amin, amax, device)

            # Oracle: encode GT future tactile
            future_t = min(ts + foresight_horizon, ep_len - 1)
            fut_frames = []
            for k in range(tac_history):
                idx = max(0, future_t - tac_history + 1 + k)
                idx = min(idx, ep_len - 1)
                fut_frames.append(raw_marker[idx])
            fut_window = np.stack(fut_frames)  # (8, 9, 9, 2)
            fut_t = torch.from_numpy(fut_window).unsqueeze(0).to(device)
            fut_norm = (fut_t - TAC_MEAN) / TAC_STD
            with torch.no_grad():
                z_gt, _ = vae_for_oracle.encode_single_frame(fut_norm)
                z_gt_flat = z_gt.flatten(1)  # (1, 144)

            # Oracle score: -||z_pred - z_gt|| (closer to GT = better)
            oracle_scores = -torch.norm(zp - z_gt_flat.expand_as(zp), dim=-1)

            # All scoring methods
            scores = {
                'naive': scorer.score_naive(zp, zc),
                'smoothness': scorer.score_smoothness(zp, zc, phase),
                'safety': scorer.score_safety(zp, phase),
                'fit': scorer.score_fit(zp, phase),
                'multi_07_02_01': scorer.score_multidim(zp, zc, phase, 0.7, 0.2, 0.1),
                'multi_05_03_02': scorer.score_multidim(zp, zc, phase, 0.5, 0.3, 0.2),
            }

            # Compute metrics
            oracle_ranking = oracle_scores.argsort(descending=True)
            oracle_best = oracle_ranking[0].item()

            for name, s in scores.items():
                pred_best = s.argmax().item()
                # Does this method pick the same candidate as oracle?
                matches_oracle = (pred_best == oracle_best)
                # Rank of oracle's best in this method's ranking
                pred_ranking = s.argsort(descending=True)
                oracle_best_rank = (pred_ranking == oracle_best).nonzero(as_tuple=True)[0].item()
                # Score range
                score_range = (s.max() - s.min()).item()
                rel_range = score_range / (abs(s.mean().item()) + 1e-8) * 100

                all_results[name].append({
                    'matches_oracle': matches_oracle,
                    'oracle_rank': oracle_best_rank,
                    'range': score_range,
                    'rel_range': rel_range,
                    'phase': phase,
                })

            # Kendall tau correlation with oracle
            from scipy.stats import kendalltau
            for name, s in scores.items():
                tau, _ = kendalltau(oracle_scores.cpu().numpy(), s.cpu().numpy())
                if name == 'naive':
                    oracle_correlations.append(tau)

            n_tests += 1
            if n_tests <= 5:  # Print first few detailed
                phase_name = {0:'approach', 1:'insertion', 2:'pre_bounce'}[phase]
                print(f"\n  Ep{ep_idx} ts={ts} phase={phase_name}")
                print(f"    Oracle best: candidate #{oracle_best}")
                print(f"    {'Method':<18} {'Best':<5} {'Match?':<7} {'OracleRank':<11} {'Range':<8} {'RelRange%':<9}")
                for name, s in scores.items():
                    pb = s.argmax().item()
                    match = "✓" if pb == oracle_best else "✗"
                    pred_ranking = s.argsort(descending=True)
                    orank = (pred_ranking == oracle_best).nonzero(as_tuple=True)[0].item()
                    rng = (s.max()-s.min()).item()
                    rr = rng / (abs(s.mean().item())+1e-8)*100
                    print(f"    {name:<18} #{pb:<4} {match:<7} {orank:<11} {rng:<8.4f} {rr:<9.2f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY ({n_tests} test points)")
    print(f"{'='*70}")
    print(f"{'Method':<18} {'OracleMatch%':<13} {'AvgOracleRank':<14} {'AvgRelRange%':<12}")
    print("-"*60)

    for name in ['naive', 'smoothness', 'safety', 'fit', 'multi_07_02_01', 'multi_05_03_02']:
        r = all_results[name]
        if not r:
            continue
        match_rate = np.mean([x['matches_oracle'] for x in r]) * 100
        avg_orank = np.mean([x['oracle_rank'] for x in r])
        avg_rel = np.mean([x['rel_range'] for x in r])
        print(f"  {name:<18} {match_rate:<13.1f} {avg_orank:<14.2f} {avg_rel:<12.2f}")

    # Breakdown by phase
    print(f"\n  Oracle match rate by phase:")
    for phase_id in [0, 1, 2]:
        phase_name = {0:'approach', 1:'insertion', 2:'pre_bounce'}[phase_id]
        for name in ['naive', 'smoothness', 'multi_07_02_01']:
            r = [x for x in all_results[name] if x['phase'] == phase_id]
            if r:
                match_rate = np.mean([x['matches_oracle'] for x in r]) * 100
                print(f"    {name:<18} @ {phase_name:<12}: {match_rate:.1f}% oracle match")

    # Kendall tau with oracle
    if oracle_correlations:
        print(f"\n  Naive ↔ Oracle Kendall τ: mean={np.mean(oracle_correlations):.3f}, "
              f"std={np.std(oracle_correlations):.3f}")


if __name__ == '__main__':
    main()
