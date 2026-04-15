"""
TFACPolicyV4: Think -> Dream -> Verify -> Act.

Loss composition (V4):
  L = l1_final                              (1.0)
    + lambda_draft * l1_draft               (0.3)
    + lambda_latent * latent_loss           (0.3)  C2: latent space prediction
    + lambda_obs * obs_dynamic_loss         (0.2)  C2: dynamic-weighted observation loss
    + lambda_consistency * consistency_loss  (0.3)  C3: tactile consistency verification
    + lambda_contrastive_spatial * spatial   (0.2)  C5: spatial contrastive
    + lambda_contrastive_temporal * temporal (0.1)  C5: temporal contrastive
    + lambda_contrastive_gt * gt_contrastive(0.1)  C5: GT contrastive
    + kl_weight * kl_loss                   (1.0)  CVAE regularization
    + lambda_dynamics * dynamics_aux_loss   (0.1)  C3: dynamics model auxiliary loss
    + lambda_sampling * sampling_loss       (0.5)  autoregressive rollout
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from detr.models.backbone import Backbone, Joiner, PositionEmbeddingSine, PositionEmbeddingLearned
from policy import kl_divergence
from TFAC_V4.tfac_model import TFACModelV4


class TFACPolicyV4(nn.Module):
    def __init__(self,
                 state_dim: int,
                 hidden_dim: int,
                 position_embedding_type: str,
                 lr_backbone: float,
                 masks: bool,
                 backbone_type: str,
                 dilation: bool,
                 dropout: float,
                 nheads: int,
                 dim_feedforward: int,
                 num_enc_layers: int,
                 num_dec_layers: int,
                 pre_norm: bool,
                 num_queries: int,
                 camera_names,
                 z_dimension: int,
                 lr: float,
                 weight_decay: float,
                 kl_weight: float,
                 # foresight params
                 foresight_layers: int = 2,
                 foresight_nheads: int = 4,
                 foresight_dim_feedforward: int = 2048,
                 # contrastive params
                 proj_dim: int = 128,
                 contrastive_temperature: float = 0.07,
                 # curriculum & loss weights
                 curriculum_ratio: float = 0.75,
                 lambda_draft: float = 0.3,
                 lambda_latent: float = 0.3,
                 lambda_obs: float = 0.2,
                 lambda_consistency: float = 0.3,
                 lambda_contrastive_spatial: float = 0.2,
                 lambda_contrastive_temporal: float = 0.1,
                 lambda_contrastive_gt: float = 0.1,
                 lambda_dynamics: float = 0.1,
                 lambda_sampling: float = 0.5,
                 num_dec_layers_draft: int = None,
                 # V4 specific
                 marker_encoder_type: str = "spatial",
                 fusion_mode: str = "contact_gate",
                 a2_init: str = "residual",
                 max_history: int = 8,
                 predict_horizon: int = 1,
                 sampling_steps: int = 3,
                 n_tac_tokens: int = 9,
                 ):
        super().__init__()

        self.predict_horizon = predict_horizon
        self.sampling_steps = sampling_steps
        self.fusion_mode = fusion_mode
        self.marker_encoder_type = marker_encoder_type

        # --- Build backbone ---
        cam_backbone_mapping = {cam_name: 0 for cam_name in camera_names}

        N_steps = hidden_dim // 2
        if position_embedding_type in ('v2', 'sine'):
            position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
        elif position_embedding_type in ('v3', 'learned'):
            position_embedding = PositionEmbeddingLearned(N_steps)
        else:
            raise ValueError(f"not supported {position_embedding_type}")

        train_backbone = lr_backbone > 0
        backbone = Backbone(name=backbone_type,
                            train_backbone=train_backbone,
                            return_interm_layers=masks,
                            dilation=dilation)
        backbone_model = Joiner(backbone, position_embedding)
        backbone_model.num_channels = backbone.num_channels
        backbones = [backbone_model]

        # --- Build TFACModelV4 ---
        self.model = TFACModelV4(
            backbones=backbones,
            state_dim=state_dim,
            num_queries=num_queries,
            camera_names=camera_names,
            z_dimension=z_dimension,
            cam_backbone_mapping=cam_backbone_mapping,
            hidden_dim=hidden_dim,
            nhead=nheads,
            num_enc_layers=num_enc_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            normalize_before=pre_norm,
            num_dec_layers=num_dec_layers,
            num_dec_layers_draft=num_dec_layers_draft,
            foresight_layers=foresight_layers,
            foresight_nheads=foresight_nheads,
            foresight_dim_feedforward=foresight_dim_feedforward,
            proj_dim=proj_dim,
            contrastive_temperature=contrastive_temperature,
            marker_encoder_type=marker_encoder_type,
            fusion_mode=fusion_mode,
            a2_init=a2_init,
            max_history=max_history,
            predict_horizon=predict_horizon,
            n_tac_tokens=n_tac_tokens,
        )

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("TFAC V4 number of parameters: %.2fM" % (n_parameters / 1e6,))
        self.model.cuda()

        # --- Optimizer ---
        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters()
                        if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.model.named_parameters()
                        if "backbone" in n and p.requires_grad],
             "lr": lr_backbone},
        ]
        self.optimizer = torch.optim.AdamW(param_dicts, lr=lr, weight_decay=weight_decay)

        # --- Loss weights ---
        self.kl_weight = kl_weight
        self.lambda_draft = lambda_draft
        self.lambda_latent = lambda_latent
        self.lambda_obs = lambda_obs
        self.lambda_consistency = lambda_consistency
        self.lambda_contrastive_spatial = lambda_contrastive_spatial
        self.lambda_contrastive_temporal = lambda_contrastive_temporal
        self.lambda_contrastive_gt = lambda_contrastive_gt
        self.lambda_dynamics = lambda_dynamics
        self.lambda_sampling = lambda_sampling
        self.curriculum_ratio = curriculum_ratio

        # Contrastive vision indices for multi-frame
        if predict_horizon > 1:
            self.contrastive_vision_indices = [predict_horizon // 2 - 1, predict_horizon - 1]
        else:
            self.contrastive_vision_indices = None

        print(f'TFAC V4 | KL={kl_weight}, curriculum={curriculum_ratio}, '
              f'predict_horizon={predict_horizon}, sampling_steps={sampling_steps}, '
              f'encoder={marker_encoder_type}, fusion={fusion_mode}, a2_init={a2_init}')

    def forward(self, qpos, images, actions=None, is_pad=None,
                future_images=None, epoch=None, total_epochs=None,
                ignore_latent=False, history_images=None):
        """
        Training: returns loss_dict
        Inference: returns a2_hat (B, chunk_size, action_dim)
        """
        # DataParallel compatibility
        if isinstance(images, tuple):
            images = list(images)
        if future_images is not None and isinstance(future_images, tuple):
            future_images = list(future_images)
        if history_images is not None and isinstance(history_images, tuple):
            history_images = list(history_images)

        if actions is not None:
            # Curriculum learning
            use_predicted = False
            if epoch is not None and total_epochs is not None:
                use_predicted = (epoch >= total_epochs * self.curriculum_ratio)

            (a1_hat, a2_hat, t_hat_obs, v_hat_future,
             v_gt_feat, t_gt_feat, t_embed_future, t_current_global,
             (mu, logvar), z_dynamics_next) = self.model(
                qpos, images, actions, is_pad, future_images, use_predicted,
                history_images=history_images)

            loss_dict = {}
            pad_mask = ~is_pad.unsqueeze(-1)

            # ---- Action losses ----
            l1_draft = (F.l1_loss(a1_hat, actions, reduction='none') * pad_mask).mean()
            l1_final = (F.l1_loss(a2_hat, actions, reduction='none') * pad_mask).mean()
            loss_dict['l1_draft'] = l1_draft
            loss_dict['l1_final'] = l1_final

            # ---- C2: Latent space prediction loss ----
            # t_embed_future: (B, H, 9, D) or (B, 9, D)
            # t_gt_feat: (B, H, 9, 9, 2) or (B, 9, 9, 2)
            loss_latent = torch.tensor(0.0, device=qpos.device)
            if t_gt_feat is not None and t_embed_future is not None:
                # Encode GT marker_offset → latent tokens for comparison
                if t_gt_feat.dim() == 5 and t_embed_future.dim() == 4:
                    # Multi-frame: (B, H, 9, 9, 2) → encode each frame
                    B, H = t_gt_feat.shape[:2]
                    gt_flat = t_gt_feat.reshape(B * H, 9, 9, 2)
                    if self.model.is_spatial_encoder:
                        gt_tokens = self.model.marker_encoder(gt_flat)  # (9, B*H, D)
                        gt_tokens = gt_tokens.permute(1, 0, 2).view(B, H, 9, -1)  # (B, H, 9, D)
                    else:
                        gt_enc = self.model.marker_encoder(gt_flat)  # (B*H, D)
                        gt_tokens = gt_enc.view(B, H, 1, -1)  # (B, H, 1, D)
                    loss_latent = F.smooth_l1_loss(t_embed_future, gt_tokens)
                elif t_gt_feat.dim() == 4 and t_embed_future.dim() == 3:
                    # Single-frame: (B, 9, 9, 2) → encode
                    if self.model.is_spatial_encoder:
                        gt_tokens = self.model.marker_encoder(t_gt_feat)  # (9, B, D)
                        gt_tokens = gt_tokens.permute(1, 0, 2)  # (B, 9, D)
                    else:
                        gt_tokens = self.model.marker_encoder(t_gt_feat).unsqueeze(1)  # (B, 1, D)
                    loss_latent = F.smooth_l1_loss(t_embed_future, gt_tokens)
            loss_dict['latent'] = loss_latent

            # ---- C2: Dynamic-Aware Weighted Observation Loss ----
            loss_obs = torch.tensor(0.0, device=qpos.device)
            if t_gt_feat is not None:
                if t_gt_feat.dim() == 5 and t_hat_obs.dim() == 5:
                    # Multi-frame: (B, H, 9, 9, 2)
                    per_frame_loss = F.smooth_l1_loss(
                        t_hat_obs, t_gt_feat, reduction='none').mean(dim=(2, 3, 4))  # (B, H)

                    # Dynamic weighting: frames with larger change get higher weight
                    with torch.no_grad():
                        # Change magnitude: ||gt[h] - gt[h-1]|| for h>0, ||gt[0] - current|| for h=0
                        changes = []
                        for h in range(t_gt_feat.shape[1]):
                            if h == 0:
                                # current tactile: from images
                                change_h = t_gt_feat[:, 0].pow(2).mean(dim=(1, 2, 3))
                            else:
                                diff = t_gt_feat[:, h] - t_gt_feat[:, h - 1]
                                change_h = diff.pow(2).mean(dim=(1, 2, 3))
                            changes.append(change_h)
                        change_tensor = torch.stack(changes, dim=1)  # (B, H)
                        # Normalize: weight = sqrt(change / mean_change)
                        weight = (change_tensor / (change_tensor.mean(dim=1, keepdim=True) + 1e-8)).sqrt()
                        weight = weight / (weight.mean() + 1e-8)

                    loss_obs = (weight * per_frame_loss).mean()
                elif t_gt_feat.dim() == 4 and t_hat_obs.dim() == 4:
                    # Single-frame: (B, 9, 9, 2)
                    loss_obs = F.smooth_l1_loss(t_hat_obs, t_gt_feat)
            loss_dict['obs'] = loss_obs

            # ---- C2: Vision foresight loss ----
            loss_foresight_vis = torch.tensor(0.0, device=qpos.device)
            if v_gt_feat is not None:
                v_gt_last = v_gt_feat[:, -1] if v_gt_feat.dim() == 3 else v_gt_feat
                loss_foresight_vis = F.mse_loss(v_hat_future, v_gt_last)
            loss_dict['foresight_vis'] = loss_foresight_vis

            # ---- C3: Tactile Consistency Loss ----
            # z_dynamics_next should match foresight's predicted tactile (global)
            loss_consistency = torch.tensor(0.0, device=qpos.device)
            if t_embed_future is not None:
                # Get foresight's global prediction
                if t_embed_future.dim() == 4:
                    # (B, H, 9, D) → last frame, mean over patches → (B, D)
                    foresight_global = t_embed_future[:, -1].mean(dim=1)
                else:
                    # (B, 9, D) → mean over patches → (B, D)
                    foresight_global = t_embed_future.mean(dim=1)

                # Margin ranking: foresight and dynamics should agree
                # sim(dynamics_pred, foresight_pred) > sim(dynamics_pred, random_foresight)
                z_dyn_norm = F.normalize(z_dynamics_next, dim=-1)
                z_foresight_norm = F.normalize(foresight_global, dim=-1)

                # Positive: same-sample pair
                sim_pos = (z_dyn_norm * z_foresight_norm).sum(dim=-1)  # (B,)

                # Negative: shifted foresight (different sample in batch)
                z_neg = torch.roll(z_foresight_norm, shifts=1, dims=0)
                sim_neg = (z_dyn_norm * z_neg).sum(dim=-1)  # (B,)

                # Margin ranking loss: want sim_pos > sim_neg by margin 0.2
                loss_consistency = F.relu(0.2 - (sim_pos - sim_neg)).mean()
            loss_dict['consistency'] = loss_consistency

            # ---- C3: Dynamics auxiliary loss ----
            # Dynamics model should predict next-step tactile that matches GT
            loss_dynamics = torch.tensor(0.0, device=qpos.device)
            if t_gt_feat is not None and z_dynamics_next is not None:
                # GT global: encode GT last frame → (B, D)
                if t_gt_feat.dim() == 5:
                    gt_last = t_gt_feat[:, -1]  # (B, 9, 9, 2)
                else:
                    gt_last = t_gt_feat
                if self.model.is_spatial_encoder:
                    with torch.no_grad():
                        gt_tokens = self.model.marker_encoder(gt_last)  # (9, B, D)
                        gt_global = gt_tokens.mean(dim=0)  # (B, D)
                else:
                    with torch.no_grad():
                        gt_global = self.model.marker_encoder(gt_last)  # (B, D)
                loss_dynamics = F.mse_loss(z_dynamics_next, gt_global)
            loss_dict['dynamics'] = loss_dynamics

            # ---- C5: Multi-level Contrastive ----
            # Spatial contrastive: per-patch t_embed vs v_gt
            loss_contrastive_spatial = torch.tensor(0.0, device=qpos.device)
            if v_gt_feat is not None and t_embed_future is not None:
                if t_embed_future.dim() == 4:
                    # Multi-frame: use last frame's patches
                    t_last_patches = t_embed_future[:, -1]  # (B, 9, D)
                    v_gt_last = v_gt_feat[:, -1] if v_gt_feat.dim() == 3 else v_gt_feat
                    loss_contrastive_spatial = self.model.contrastive.forward_spatial(
                        v_gt_last, t_last_patches)
                elif t_embed_future.dim() == 3:
                    v_gt_single = v_gt_feat[:, -1] if v_gt_feat.dim() == 3 else v_gt_feat
                    loss_contrastive_spatial = self.model.contrastive.forward_spatial(
                        v_gt_single, t_embed_future)
            loss_dict['contrastive_spatial'] = loss_contrastive_spatial

            # Temporal contrastive
            loss_contrastive_temporal = torch.tensor(0.0, device=qpos.device)
            if t_embed_future is not None and t_embed_future.dim() == 4:
                # (B, H, 9, D) → mean over patches → (B, H, D)
                t_traj = t_embed_future.mean(dim=2)
                loss_contrastive_temporal = self.model.contrastive.forward_temporal(t_traj)
            loss_dict['contrastive_temporal'] = loss_contrastive_temporal

            # GT contrastive: GT tactile vs GT vision (no prediction noise)
            loss_contrastive_gt = torch.tensor(0.0, device=qpos.device)
            if (self.lambda_contrastive_gt > 0
                    and v_gt_feat is not None and t_gt_feat is not None):
                if t_gt_feat.dim() == 5:
                    gt_last = t_gt_feat[:, -1]  # (B, 9, 9, 2)
                else:
                    gt_last = t_gt_feat
                if self.model.is_spatial_encoder:
                    with torch.no_grad():
                        gt_tokens = self.model.marker_encoder(gt_last)  # (9, B, D)
                        gt_global_t = gt_tokens.mean(dim=0)
                else:
                    with torch.no_grad():
                        gt_global_t = self.model.marker_encoder(gt_last)
                v_gt_last = v_gt_feat[:, -1] if v_gt_feat.dim() == 3 else v_gt_feat
                loss_contrastive_gt = self.model.contrastive.forward_global(
                    v_gt_last, gt_global_t)
            loss_dict['contrastive_gt'] = loss_contrastive_gt

            # ---- KL loss ----
            total_kld, _, _ = kl_divergence(mu, logvar)
            loss_dict['kl'] = total_kld[0]

            # ---- Sampling loss ----
            loss_sampling = torch.tensor(0.0, device=qpos.device)
            if (self.training
                    and self.sampling_steps > 0 and self.lambda_sampling > 0
                    and self.predict_horizon > 1
                    and t_gt_feat is not None):
                loss_sampling = self._compute_sampling_loss(
                    qpos, images, a1_hat.detach(), t_gt_feat,
                    history_images=history_images)
            loss_dict['sampling'] = loss_sampling

            # ---- Total loss ----
            loss = (l1_final
                    + self.lambda_draft * l1_draft
                    + self.lambda_latent * loss_latent
                    + self.lambda_obs * loss_obs
                    + self.lambda_consistency * loss_consistency
                    + self.lambda_contrastive_spatial * loss_contrastive_spatial
                    + self.lambda_contrastive_temporal * loss_contrastive_temporal
                    + self.lambda_contrastive_gt * loss_contrastive_gt
                    + self.lambda_dynamics * loss_dynamics
                    + self.lambda_sampling * loss_sampling
                    + 0.3 * loss_foresight_vis
                    + self.kl_weight * total_kld[0])
            loss_dict['loss'] = loss

            # Gate weights for logging
            if self.fusion_mode == "contact_gate":
                gm, ga, gf = self.model.contact_fusion._last_gate_means
                loss_dict['gate_mem'] = torch.tensor(gm, device=qpos.device)
                loss_dict['gate_a1'] = torch.tensor(ga, device=qpos.device)
                loss_dict['gate_traj'] = torch.tensor(gf, device=qpos.device)
            elif self.fusion_mode == "gate":
                gm, ga, gf = self.model.gated_fusion._last_gate_means
                loss_dict['gate_mem'] = torch.tensor(gm, device=qpos.device)
                loss_dict['gate_a1'] = torch.tensor(ga, device=qpos.device)
                loss_dict['gate_traj'] = torch.tensor(gf, device=qpos.device)

            # Residual alpha for logging
            if hasattr(self.model, 'residual_alpha'):
                loss_dict['residual_alpha'] = torch.sigmoid(self.model.residual_alpha).detach()

            return loss_dict

        else:
            # Inference: Think -> Dream -> Act
            a1_hat, a2_hat, _, _, _, _, _, _, _, _ = self.model(
                qpos, images, history_images=history_images)
            return a2_hat

    def _compute_sampling_loss(self, qpos, images, a1_detached, t_gt,
                               history_images=None):
        """
        Sampling loss: autoregressive rollout S steps.
        Reuses cached backbone features from main forward pass.
        """
        S = min(self.sampling_steps, self.predict_horizon)
        bs = qpos.size(0)

        cache = self.model._fwd_cache
        src = cache['src']
        n_vision = cache['n_vision']
        n_tactile = cache['n_tactile']
        v_tokens = src[:n_vision]
        t_tokens = src[n_vision:]

        hist_src = cache['hist_src']
        use_hist = hist_src is not None

        total_loss = 0.0
        current_t_input = t_tokens

        H = self.predict_horizon
        if S >= H:
            frame_indices = list(range(H))
        else:
            frame_indices = [round(i * (H - 1) / (S - 1)) for i in range(S)]

        for step, s in enumerate(frame_indices):
            if use_hist:
                v_in = hist_src[:, :n_vision]
                t_in = hist_src[:, n_vision:].clone()
                t_in[-1] = current_t_input
            else:
                v_in = v_tokens
                t_in = current_t_input

            t_hat_obs, _, _ = self.model.foresight(v_in, t_in, a1_detached, n_vision,
                                                     proprio=qpos)
            # t_hat_obs: (B, H, 9, 9, 2)
            if t_hat_obs.dim() == 5:
                step_loss = F.smooth_l1_loss(t_hat_obs[:, s], t_gt[:, s])
            else:
                step_loss = F.smooth_l1_loss(t_hat_obs, t_gt)
            total_loss = total_loss + step_loss

            # Prepare input for next step: encode predicted tactile
            with torch.no_grad():
                if t_hat_obs.dim() == 5:
                    pred_tac = t_hat_obs[:, s].detach()  # (B, 9, 9, 2)
                else:
                    pred_tac = t_hat_obs.detach()
                if self.model.is_spatial_encoder:
                    pred_tokens = self.model.marker_encoder(pred_tac)  # (9, B, D)
                    current_t_input = pred_tokens
                else:
                    pred_encoded = self.model.marker_encoder(pred_tac)  # (B, D)
                    current_t_input = pred_encoded.unsqueeze(0).expand(n_tactile, -1, -1)

        return total_loss / S

    def configure_optimizers(self):
        return self.optimizer
