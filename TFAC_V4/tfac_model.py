"""
TFACModelV4: Think -> Dream -> Verify -> Act

Core innovations over V3:
  C1: SpatialMarkerEncoder → 9 tokens (marker_encoder.py)
  C2: Latent prediction + Dynamic-Aware Weighted Loss (foresight)
  C3: Tactile Consistency Loss — action→tactile causal verification
  C4: Temporal Differential Residual Action — A2 = A1 + alpha*deltaA
  C5: Multi-level Contrastive Learning (foresight_transformer.py)

Architecture:
  1. Backbone + SpatialMarkerEncoder → [V_tokens; T_tokens(×9); latent; proprio]
  2. Shared VT-Encoder → memory
  3. Decoder_draft → A1 (draft action)
  4. ForesightTransformerV4: predict future tactile (latent + observation)
  5. TactileConsistencyVerifier: verify A1 is consistent with predicted tactile
  6. ContactAwareFusion: memory + A1 + trajectory summary → enriched memory
  7. Decoder_final → A2 = A1 + alpha * deltaA (residual refinement)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from typing import Tuple, List, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from detr.models.transformer import (
    TransformerEncoder, TransformerEncoderLayer,
    TransformerDecoder, TransformerDecoderLayer,
)
from TFAC_V4.foresight_transformer import ForesightTransformerV4, ForesightContrastiveV4
from TFAC_V4.marker_encoder import build_marker_encoder, LTDEncoder


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


# ---------------------------------------------------------------------------
# C3: Tactile Dynamics Model — for Consistency Loss
# ---------------------------------------------------------------------------

class TactileDynamicsModel(nn.Module):
    """
    Lightweight tactile dynamics model: (z_tac, a_chunk_summary) → z_next.

    Used in Consistency Loss (C3): verify that the action-conditioned predicted
    tactile is consistent with one-step dynamics from current tactile.

    This creates an independent causal path: action → dynamics → predicted tactile,
    which must match the foresight prediction. If A1 is bad, both paths disagree.
    """

    def __init__(self, d_model: int = 512, action_dim: int = 7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, z_tac: torch.Tensor, action_summary: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_tac: (B, D) — current tactile feature (global pooled)
            action_summary: (B, action_dim) — action chunk summary (mean or learned)
        Returns:
            z_next: (B, D) — predicted next tactile feature
        """
        x = torch.cat([z_tac, action_summary], dim=-1)  # (B, D+action_dim)
        return self.mlp(x)  # (B, D)


# ---------------------------------------------------------------------------
# C4: Tactile Trajectory Encoder
# ---------------------------------------------------------------------------

class TactileTrajectoryEncoder(nn.Module):
    """
    Encode predicted tactile trajectory (B, H, D) into a summary vector (B, D).

    Uses temporal differential: emphasize changes between frames.
    Input: H frames of global tactile features (from embed_predictor mean-pooled).
    """

    def __init__(self, d_model: int = 512):
        super().__init__()
        # diff features: (B, H-1, D) + first frame (B, 1, D) → (B, H, D) → projection
        self.temporal_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.aggregate = nn.Sequential(
            nn.Linear(d_model, d_model),
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectory: (B, H, D) — H frames of tactile features
        Returns:
            summary: (B, D)
        """
        B, H, D = trajectory.shape
        if H == 1:
            return self.aggregate(self.temporal_proj(trajectory[:, 0]))

        # Differential: [frame0, frame1-frame0, frame2-frame1, ...]
        diffs = trajectory[:, 1:] - trajectory[:, :-1]  # (B, H-1, D)
        first_frame = trajectory[:, :1]  # (B, 1, D)
        diff_seq = torch.cat([first_frame, diffs], dim=1)  # (B, H, D)

        # Per-frame projection + mean pooling
        projected = self.temporal_proj(diff_seq)  # (B, H, D)
        pooled = projected.mean(dim=1)  # (B, D)
        return self.aggregate(pooled)  # (B, D)


# ---------------------------------------------------------------------------
# C4: Contact-Aware Fusion
# ---------------------------------------------------------------------------

class ContactAwareFusion(nn.Module):
    """
    3-way gating fusion with contact-aware weighting.

    Unlike V3's GatedFusion which uses fixed memory/A1/future_tac,
    V4 fuses memory + A1_feat + trajectory_summary, with a contact-aware
    alpha that modulates tactile contribution based on proprio force signal.
    """

    def __init__(self, d_model: int, state_dim: int = 7):
        super().__init__()
        self.d_model = d_model

        # 3-way gate projection
        self.gate_proj = nn.Linear(d_model * 3, d_model * 3)
        self.memory_proj = nn.Linear(d_model, d_model)
        self.a1_proj = nn.Linear(d_model, d_model)
        self.traj_proj = nn.Linear(d_model, d_model)

        # Contact-aware alpha: proprio → scalar weight for tactile contribution
        self.contact_alpha = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, memory: torch.Tensor, a1_feat: torch.Tensor,
                traj_summary: torch.Tensor, proprio: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            memory:       (S, B, D)
            a1_feat:      (B, D)
            traj_summary: (B, D) — from TactileTrajectoryEncoder
            proprio:      (B, state_dim) — for contact-aware alpha
        Returns:
            fused: (S, B, D)
        """
        S, B, D = memory.shape
        a1_exp = a1_feat.unsqueeze(0).expand(S, -1, -1)
        traj_exp = traj_summary.unsqueeze(0).expand(S, -1, -1)

        # 3-way gate
        concat = torch.cat([memory, a1_exp, traj_exp], dim=-1)  # (S, B, 3D)
        gate_logits = self.gate_proj(concat).view(S, B, 3, D)
        gates = torch.softmax(gate_logits, dim=2)

        g_mem = gates[:, :, 0, :]
        g_a1 = gates[:, :, 1, :]
        g_traj = gates[:, :, 2, :]

        # Contact-aware alpha: scale tactile gate
        if proprio is not None:
            alpha = self.contact_alpha(proprio)  # (B, 1)
            alpha = alpha.unsqueeze(0)  # (1, B, 1)
            g_traj = g_traj * (0.5 + 0.5 * alpha)  # min 0.5, max 1.0

        fused = (g_mem * self.memory_proj(memory)
                 + g_a1 * self.a1_proj(a1_exp)
                 + g_traj * self.traj_proj(traj_exp))

        # Save gate means for logging
        self._last_gate_means = (
            g_mem.mean().item(),
            g_a1.mean().item(),
            g_traj.mean().item(),
        )

        return fused


# ---------------------------------------------------------------------------
# TFACModelV4
# ---------------------------------------------------------------------------

class TFACModelV4(nn.Module):
    """
    Full TFAC V4 model: Think -> Dream -> Verify -> Act.

    Returns (training):
        a1_hat, a2_hat, t_hat_obs, v_hat_future,
        v_gt_feat, t_gt_feat, t_embed_future, t_current_global,
        (mu, logvar), z_dynamics_next
    """

    def __init__(self, backbones, state_dim, num_queries, camera_names,
                 z_dimension, cam_backbone_mapping,
                 # shared encoder
                 hidden_dim=512, nhead=8, num_enc_layers=4,
                 dim_feedforward=2048, dropout=0.1, activation="relu",
                 normalize_before=False,
                 # decoder
                 num_dec_layers=7, num_dec_layers_draft=None,
                 # foresight
                 foresight_layers=2, foresight_nheads=4,
                 foresight_dim_feedforward=2048,
                 # contrastive
                 proj_dim=128, contrastive_temperature=0.07,
                 # V4 modularity
                 marker_encoder_type="spatial",
                 fusion_mode="contact_gate",
                 a2_init="residual",
                 max_history=8,
                 predict_horizon=1,
                 n_tac_tokens=9):
        super().__init__()

        self.num_queries = num_queries
        self.camera_names = camera_names
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.cam_backbone_mapping = cam_backbone_mapping
        self.tactile_mode = "marker"  # V4 always uses marker mode
        self.fusion_mode = fusion_mode
        self.a2_init = a2_init
        self.predict_horizon = predict_horizon
        self.n_tac_tokens = n_tac_tokens
        self.marker_encoder_type = marker_encoder_type

        # ---- Shared backbone ----
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
        else:
            self.backbones = None

        # ---- Marker encoder (V4: spatial by default → 9 tokens) ----
        self.marker_encoder = build_marker_encoder(marker_encoder_type, hidden_dim=hidden_dim)
        self.is_spatial_encoder = (marker_encoder_type == 'spatial')

        if self.is_spatial_encoder:
            # 9 learnable position embeddings for 9 patch tokens
            self.marker_pos_embed = nn.Parameter(torch.randn(n_tac_tokens, 1, hidden_dim) * 0.02)
        else:
            # Fallback: 1 token (V3 compatible)
            self.marker_pos_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # ---- CVAE encoder ----
        self.latent_dim = z_dimension
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(state_dim, hidden_dim)
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.register_buffer('pos_table',
                             get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim))

        cvae_enc_layer = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward,
                                                  dropout, activation, normalize_before)
        cvae_enc_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.cvae_encoder = TransformerEncoder(cvae_enc_layer, num_enc_layers, cvae_enc_norm)

        # ---- Shared VT encoder ----
        vt_enc_layer = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        vt_enc_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.vt_encoder = TransformerEncoder(vt_enc_layer, num_enc_layers, vt_enc_norm)

        # ---- Decoder draft (A1) ----
        _n_dec_draft = num_dec_layers_draft if num_dec_layers_draft is not None else num_dec_layers
        dec_layer_draft = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward,
                                                   dropout, activation, normalize_before)
        dec_norm_draft = nn.LayerNorm(hidden_dim)
        self.decoder_draft = TransformerDecoder(dec_layer_draft, _n_dec_draft, dec_norm_draft)
        self.query_embed_draft = nn.Embedding(num_queries, hidden_dim)
        self.action_head_draft = nn.Linear(hidden_dim, state_dim)

        # ---- Decoder final (A2) ----
        dec_layer_final = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward,
                                                   dropout, activation, normalize_before)
        dec_norm_final = nn.LayerNorm(hidden_dim)
        self.decoder_final = TransformerDecoder(dec_layer_final, num_dec_layers, dec_norm_final)
        self.query_embed_final = nn.Embedding(num_queries, hidden_dim)
        self.action_head_final = nn.Linear(hidden_dim, state_dim)

        # ---- Common projections ----
        self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

        # ---- C2: Foresight V4 ----
        n_tac = n_tac_tokens if self.is_spatial_encoder else 1
        self.foresight = ForesightTransformerV4(
            d_model=hidden_dim, action_dim=state_dim,
            num_layers=foresight_layers, nhead=foresight_nheads,
            dim_feedforward=foresight_dim_feedforward, dropout=dropout,
            max_history=max_history,
            predict_horizon=predict_horizon,
            state_dim=state_dim,
            n_tac_tokens=n_tac,
        )

        # ---- C5: Contrastive V4 ----
        self.contrastive = ForesightContrastiveV4(
            feat_dim=hidden_dim, proj_dim=proj_dim,
            temperature=contrastive_temperature)

        # ---- C3: Tactile Dynamics Model ----
        self.dynamics_model = TactileDynamicsModel(
            d_model=hidden_dim, action_dim=state_dim)

        # ---- C4: Trajectory Encoder ----
        self.trajectory_encoder = TactileTrajectoryEncoder(d_model=hidden_dim)

        # ---- Fusion ----
        if fusion_mode == "contact_gate":
            self.contact_fusion = ContactAwareFusion(hidden_dim, state_dim=state_dim)
            self.a1_pool_proj = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        elif fusion_mode == "gate":
            # V3-style gate (for ablation)
            self.gated_fusion = GatedFusionV3Compat(hidden_dim)
            self.a1_pool_proj = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        # ---- C4: Residual action refinement ----
        if a2_init == "residual":
            self.a1_to_hidden = nn.Linear(state_dim, hidden_dim)
            # Learnable residual scaling (starts small)
            self.residual_alpha = nn.Parameter(torch.tensor(0.1))
        elif a2_init == "a1_refine":
            self.a1_to_hidden = nn.Linear(state_dim, hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if 'backbone' in name or 'log_temp' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _encode_images(self, images):
        """
        Backbone encode all camera images.

        For spatial marker encoder:
          gelsight → SpatialMarkerEncoder → (9, B, D) tokens
        For legacy encoders:
          gelsight → MarkerEncoder → (1, B, D) token

        Returns:
            src:       (N_total, B, D)
            pos:       (N_total, B, D)
            n_vision:  int
            n_tactile: int
        """
        all_cam_features = []
        all_cam_pos = []
        n_vision = 0
        n_tactile = 0

        for cam_id, cam_name in enumerate(self.camera_names):
            if cam_name == 'gelsight':
                if self.is_spatial_encoder:
                    # SpatialMarkerEncoder: (B, 9, 9, 2) → (9, B, D)
                    proj = self.marker_encoder(images[cam_id])  # (9, B, D)
                    pos_flat = self.marker_pos_embed  # (9, 1, D)
                    n_tactile += proj.shape[0]  # 9
                else:
                    # Legacy: (B, 9, 9, 2) → (B, D) → (1, B, D)
                    marker_feat = self.marker_encoder(images[cam_id])
                    proj = marker_feat.unsqueeze(0)
                    pos_flat = self.marker_pos_embed
                    n_tactile += 1
            else:
                features, pos = self.backbones[self.cam_backbone_mapping[cam_name]](images[cam_id])
                features = features[0]
                pos = pos[0]
                proj = self.input_proj(features).flatten(2)  # (B, D, N_spatial)
                pos_flat = pos.flatten(2)

                n_tokens = proj.size(2)
                if cam_name == 'blank':
                    n_vision += n_tokens
                else:
                    n_vision += n_tokens

                proj = proj.permute(2, 0, 1)
                pos_flat = pos_flat.permute(2, 0, 1)

            all_cam_features.append(proj)
            all_cam_pos.append(pos_flat)

        src = torch.cat(all_cam_features, dim=0)
        pos = torch.cat(all_cam_pos, dim=0)

        return src, pos, n_vision, n_tactile

    def _cvae_encode(self, qpos, actions, is_pad):
        """CVAE encoder: training → infer z from actions; inference → z=0."""
        bs = qpos.size(0)
        if actions is not None:
            action_embed = self.encoder_action_proj(actions)
            qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)
            cls_embed = self.cls_embed.weight.unsqueeze(0).expand(bs, -1, -1)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], dim=1)
            encoder_input = encoder_input.permute(1, 0, 2)

            cls_joint_is_pad = torch.full((bs, 2), False, device=qpos.device)
            is_pad_full = torch.cat([cls_joint_is_pad, is_pad], dim=1)

            pos_embed = self.pos_table.clone().detach().permute(1, 0, 2)

            encoder_output = self.cvae_encoder(encoder_input, pos=pos_embed,
                                                src_key_padding_mask=is_pad_full)
            encoder_output = encoder_output[0]
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
        else:
            mu = logvar = None
            latent_sample = torch.zeros(bs, self.latent_dim, device=qpos.device)

        latent_input = self.latent_out_proj(latent_sample)
        return latent_input, mu, logvar

    def _run_decoder(self, decoder, query_embed, action_head, memory, pos, bs,
                     tgt_init=None):
        """Run a single decoder to generate action sequence."""
        q_embed = query_embed.weight.unsqueeze(1).expand(-1, bs, -1)
        tgt = tgt_init if tgt_init is not None else torch.zeros_like(q_embed)
        hs = decoder(tgt, memory, pos=pos, query_pos=q_embed)
        hs = hs[0].permute(1, 0, 2)
        a_hat = action_head(hs)
        return a_hat

    def _encode_history(self, history_images, skip_last=False):
        """Encode past k frames through backbone. Returns (end, N_total, B, D)."""
        num_cams = len(self.camera_names)
        k = history_images[0].shape[1]
        end = k - 1 if skip_last else k
        B = history_images[0].shape[0]

        if end == 0:
            return torch.empty(0, device=history_images[0].device)

        batched_images = []
        for cam_idx in range(num_cams):
            hist = history_images[cam_idx][:, :end]
            batched_images.append(hist.reshape(B * end, *hist.shape[2:]))

        with torch.no_grad():
            src, pos, n_v, n_t = self._encode_images(batched_images)

        N_total = src.shape[0]
        D = src.shape[2]
        hist_src = src.view(N_total, B, end, D).permute(2, 0, 1, 3)
        return hist_src

    def _compute_gt_future_features(self, future_images):
        """
        Compute GT future features for foresight loss targets.

        For marker mode:
          tactile GT: (B, H, 9, 9, 2) or (B, 9, 9, 2) — raw marker_offset
          vision GT: (B, D) or (B, H, D) — backbone pooled features

        Returns:
            v_gt_feat: (B, D) or (B, H, D)
            t_gt_feat: (B, H, 9, 9, 2) or (B, 9, 9, 2)
        """
        v_feats = []
        t_feat = None
        is_multi_frame_vision = False

        with torch.no_grad():
            for cam_id, cam_name in enumerate(self.camera_names):
                if cam_name == 'gelsight':
                    t_feat = future_images[cam_id]
                else:
                    img = future_images[cam_id]
                    if img.dim() == 5:
                        is_multi_frame_vision = True
                        B, H = img.shape[:2]
                        img_flat = img.view(B * H, *img.shape[2:])
                        features, _ = self.backbones[self.cam_backbone_mapping[cam_name]](img_flat)
                        features = features[0]
                        proj = self.input_proj(features).flatten(2)
                        pooled = proj.mean(dim=2)
                        v_feats.append(pooled.view(B, H, -1))
                    else:
                        features, _ = self.backbones[self.cam_backbone_mapping[cam_name]](img)
                        features = features[0]
                        proj = self.input_proj(features).flatten(2)
                        pooled = proj.mean(dim=2)
                        v_feats.append(pooled)

        if v_feats:
            if is_multi_frame_vision:
                v_gt_feat = torch.stack(v_feats, dim=0).mean(dim=0)
            else:
                v_gt_feat = torch.stack(v_feats, dim=0).mean(dim=0)
        else:
            v_gt_feat = torch.zeros(future_images[0].size(0), self.hidden_dim,
                                     device=future_images[0].device)

        # Handle single-frame prediction with multi-frame GT data:
        # Dataset always loads all horizon frames for gelsight marker mode,
        # but when predict_horizon=1 we only need the last frame.
        if t_feat is not None and self.predict_horizon == 1 and t_feat.dim() == 5:
            t_feat = t_feat[:, -1]  # (B, H, 9, 9, 2) → (B, 9, 9, 2)

        if t_feat is None:
            bs = future_images[0].size(0)
            if self.predict_horizon > 1:
                t_feat = torch.zeros(bs, self.predict_horizon, 9, 9, 2,
                                      device=future_images[0].device)
            else:
                t_feat = torch.zeros(bs, 9, 9, 2, device=future_images[0].device)

        return v_gt_feat, t_feat

    def forward(self, qpos, images, actions=None, is_pad=None,
                future_images=None, use_predicted_future=False,
                history_images=None):
        """
        Full forward pass: Think -> Dream -> Verify -> Act.

        Returns (training):
            a1_hat, a2_hat, t_hat_obs, v_hat_future,
            v_gt_feat, t_gt_feat, t_embed_future, t_current_global,
            (mu, logvar), z_dynamics_next
        """
        is_training = actions is not None
        bs = qpos.size(0)

        # ---- 1. CVAE → latent ----
        latent_input, mu, logvar = self._cvae_encode(qpos, actions, is_pad)

        # ---- 2. Backbone → image/tactile tokens ----
        src, pos, n_vision, n_tactile = self._encode_images(images)

        # ---- 3. Prepend [latent, proprio] ----
        proprio_input = self.input_proj_robot_state(qpos)
        addition_input = torch.stack([latent_input, proprio_input], dim=0)
        src_full = torch.cat([addition_input, src], dim=0)

        add_pos = self.additional_pos_embed.weight.unsqueeze(1).expand(-1, bs, -1)
        pos_expanded = pos.expand(-1, bs, -1) if pos.size(1) == 1 else pos
        pos_full = torch.cat([add_pos, pos_expanded], dim=0)

        # ---- 4. Shared encoder → memory ----
        memory = self.vt_encoder(src_full, pos=pos_full)

        # ---- 5. Decoder draft → A1 ----
        a1_hat = self._run_decoder(self.decoder_draft, self.query_embed_draft,
                                    self.action_head_draft, memory, pos_full, bs)

        # ---- 6. Foresight (Dream phase) ----
        v_tokens = src[:n_vision]
        t_tokens = src[n_vision:]

        has_history = history_images is not None and history_images[0].shape[1] > 1
        hist_src_full = None
        if has_history:
            hist_src = self._encode_history(history_images, skip_last=True)
            hist_src_full = torch.cat([hist_src, src.unsqueeze(0)], dim=0)
            v_tokens_hist = hist_src_full[:, :n_vision]
            t_tokens_hist = hist_src_full[:, n_vision:]

        # Cache for sampling loss
        self._fwd_cache = {
            'src': src, 'n_vision': n_vision, 'n_tactile': n_tactile,
            'hist_src': hist_src_full,
        }

        if has_history:
            t_hat_obs, v_hat_future, t_embed_future = self.foresight(
                v_tokens_hist, t_tokens_hist, a1_hat.detach(), n_vision,
                proprio=qpos)
        else:
            t_hat_obs, v_hat_future, t_embed_future = self.foresight(
                v_tokens, t_tokens, a1_hat.detach(), n_vision,
                proprio=qpos)

        # ---- 6b. Current tactile global feature (for dynamics) ----
        if self.is_spatial_encoder:
            t_current_global = t_tokens.mean(dim=0)  # (B, D) mean over 9 tokens
        else:
            t_current_global = t_tokens.mean(dim=0)  # (B, D)

        # ---- 7. Dynamics prediction (Verify phase — C3) ----
        action_summary = a1_hat.detach().mean(dim=1)  # (B, action_dim)
        z_dynamics_next = self.dynamics_model(t_current_global, action_summary)  # (B, D)

        # ---- 8. GT future features (training) ----
        v_gt_feat = t_gt_feat = None
        if is_training and future_images is not None:
            v_gt_feat, t_gt_feat = self._compute_gt_future_features(future_images)

        # ---- 9. Curriculum: choose tactile tokens for fusion ----
        # use_predicted_future=False (first 75%): use GT tactile → stronger gate signal
        # use_predicted_future=True  (last 25%):  use predicted tactile from foresight
        if is_training and not use_predicted_future and t_gt_feat is not None:
            # Encode GT future tactile through SpatialMarkerEncoder
            if self.is_spatial_encoder:
                gt_tokens = self.marker_encoder(t_gt_feat)  # (9, B, D)
                fusion_tokens = gt_tokens.permute(1, 0, 2)  # (B, 9, D)
            else:
                gt_enc = self.marker_encoder(t_gt_feat)  # (B, D)
                fusion_tokens = gt_enc.unsqueeze(1)  # (B, 1, D)
        else:
            # Use predicted foresight tokens
            fusion_tokens = t_embed_future  # (B, 9, D) or (B, H, 9, D)

        # ---- 9b. Trajectory summary for fusion ----
        if fusion_tokens.dim() == 4:
            # Multi-frame: (B, H, 9, D) → mean over patches → (B, H, D)
            traj_global = fusion_tokens.mean(dim=2)
        else:
            # Single-frame: (B, 9, D) → mean → (B, D)
            traj_global = fusion_tokens.mean(dim=1).unsqueeze(1)  # (B, 1, D)

        traj_summary = self.trajectory_encoder(traj_global)  # (B, D)

        # ---- 10. Fusion → enriched memory (Act phase) ----
        a1_pooled = self.a1_pool_proj(a1_hat.detach().mean(dim=1))

        if self.fusion_mode == "contact_gate":
            fused_memory = self.contact_fusion(
                memory, a1_pooled, traj_summary, proprio=qpos)
        elif self.fusion_mode == "gate":
            fused_memory = self.gated_fusion(memory, a1_pooled, traj_summary)

        # ---- 11. Decoder final → A2 (residual refinement) ----
        tgt_init = None
        if self.a2_init in ("residual", "a1_refine"):
            tgt_init = self.a1_to_hidden(a1_hat.detach()).permute(1, 0, 2)

        delta_or_a2 = self._run_decoder(
            self.decoder_final, self.query_embed_final,
            self.action_head_final, fused_memory, pos_full, bs,
            tgt_init=tgt_init)

        if self.a2_init == "residual":
            # C4: A2 = A1 + alpha * deltaA
            alpha = torch.sigmoid(self.residual_alpha)
            a2_hat = a1_hat.detach() + alpha * delta_or_a2
        else:
            a2_hat = delta_or_a2

        return (a1_hat, a2_hat, t_hat_obs, v_hat_future,
                v_gt_feat, t_gt_feat, t_embed_future, t_current_global, (mu, logvar),
                z_dynamics_next)


# ---------------------------------------------------------------------------
# V3-compatible GatedFusion (for ablation)
# ---------------------------------------------------------------------------

class GatedFusionV3Compat(nn.Module):
    """V3 GatedFusion, retained for ablation only."""

    def __init__(self, d_model: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model * 3, d_model * 3)
        self.memory_proj = nn.Linear(d_model, d_model)
        self.a1_proj = nn.Linear(d_model, d_model)
        self.future_proj = nn.Linear(d_model, d_model)

    def forward(self, memory, a1_feat, future_feat):
        S, B, D = memory.shape
        a1_exp = a1_feat.unsqueeze(0).expand(S, -1, -1)
        fut_exp = future_feat.unsqueeze(0).expand(S, -1, -1)

        concat = torch.cat([memory, a1_exp, fut_exp], dim=-1)
        gate_logits = self.gate_proj(concat).view(S, B, 3, D)
        gates = torch.softmax(gate_logits, dim=2)

        g_mem = gates[:, :, 0, :]
        g_a1 = gates[:, :, 1, :]
        g_fut = gates[:, :, 2, :]

        fused = (g_mem * self.memory_proj(memory)
                 + g_a1 * self.a1_proj(a1_exp)
                 + g_fut * self.future_proj(fut_exp))

        self._last_gate_means = (g_mem.mean().item(), g_a1.mean().item(), g_fut.mean().item())
        return fused
