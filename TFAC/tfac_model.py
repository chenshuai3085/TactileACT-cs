"""
TFACModel: TactileForesight ACT 核心模型。
- 共享 backbone + input_proj + CVAE encoder + Transformer encoder
- 独立 decoder_draft (A1) / decoder_final (A2)
- ForesightTransformer: A1.detach() 条件预测未来 V/T
- GatedFusion: 当前 memory + A1.detach() + T̂_future 门控融合 → Decoder₂ memory
"""

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from typing import Tuple, List, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from detr.models.transformer import (
    TransformerEncoder, TransformerEncoderLayer,
    TransformerDecoder, TransformerDecoderLayer,
)
from TFAC.foresight_transformer import ForesightTransformer, ForesightContrastive


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


class GatedFusion(nn.Module):
    """
    门控融合: current memory + A1 feat + future feat → enriched memory for Decoder₂。
    对 memory 中每个 token, gate 决定保留原始 vs 融合后的表示。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model * 3, d_model)
        self.value_proj = nn.Linear(d_model * 3, d_model)

    def forward(self, memory: torch.Tensor, a1_feat: torch.Tensor,
                future_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            memory:      (S, B, D) — shared encoder output
            a1_feat:     (B, D)    — pooled A1 feature (detached)
            future_feat: (B, D)    — T̂_future (or GT future tactile)
        Returns:
            fused: (S, B, D) — enriched memory for Decoder₂
        """
        S = memory.size(0)
        a1_exp = a1_feat.unsqueeze(0).expand(S, -1, -1)       # (S, B, D)
        fut_exp = future_feat.unsqueeze(0).expand(S, -1, -1)   # (S, B, D)

        concat = torch.cat([memory, a1_exp, fut_exp], dim=-1)  # (S, B, 3D)
        gate = torch.sigmoid(self.gate_proj(concat))            # (S, B, D)
        value = self.value_proj(concat)                          # (S, B, D)

        fused = gate * memory + (1.0 - gate) * value            # (S, B, D)
        return fused


class TFACModel(nn.Module):
    """
    完整 TFAC 模型。

    forward 返回:
        a1_hat:       (B, chunk_size, action_dim)
        a2_hat:       (B, chunk_size, action_dim)
        t_hat_future: (B, D)
        v_hat_future: (B, D)
        t_gt_feat:    (B, D)  — GT future tactile feature (训练时)
        v_gt_feat:    (B, D)  — GT future vision feature (训练时)
        (mu, logvar): CVAE latent
    """

    def __init__(self, backbones, state_dim, num_queries, camera_names,
                 z_dimension, cam_backbone_mapping,
                 # shared encoder params
                 hidden_dim=512, nhead=8, num_enc_layers=4,
                 dim_feedforward=2048, dropout=0.1, activation="relu",
                 normalize_before=False,
                 # decoder params
                 num_dec_layers=7,
                 # foresight params
                 foresight_layers=2, foresight_nheads=4,
                 foresight_dim_feedforward=2048,
                 # contrastive params
                 proj_dim=128, contrastive_temperature=0.07):
        super().__init__()

        self.num_queries = num_queries
        self.camera_names = camera_names
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.cam_backbone_mapping = cam_backbone_mapping

        # ---- Shared backbone ----
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
        else:
            self.backbones = None

        # ---- Shared CVAE encoder ----
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

        # ---- Shared vision/tactile encoder ----
        vt_enc_layer = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        vt_enc_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.vt_encoder = TransformerEncoder(vt_enc_layer, num_enc_layers, vt_enc_norm)

        # ---- Decoder draft (A1) ----
        dec_layer_draft = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward,
                                                   dropout, activation, normalize_before)
        dec_norm_draft = nn.LayerNorm(hidden_dim)
        self.decoder_draft = TransformerDecoder(dec_layer_draft, num_dec_layers, dec_norm_draft)
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
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)  # [latent, proprio]

        # ---- Foresight ----
        self.foresight = ForesightTransformer(
            d_model=hidden_dim, action_dim=state_dim,
            num_layers=foresight_layers, nhead=foresight_nheads,
            dim_feedforward=foresight_dim_feedforward, dropout=dropout)

        # ---- Contrastive ----
        self.contrastive = ForesightContrastive(
            feat_dim=hidden_dim, proj_dim=proj_dim,
            temperature=contrastive_temperature)

        # ---- Gated fusion for Decoder₂ ----
        self.gated_fusion = GatedFusion(hidden_dim)
        # A1 pooling: project action chunk → single feature for gating
        self.a1_pool_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            # skip backbone (may be pretrained) and contrastive log_temp
            if 'backbone' in name or 'log_temp' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _encode_images(self, images):
        """
        Backbone 编码所有相机图像。
        Args:
            images: list of (B, C, H, W), len = num_cameras
        Returns:
            src:       (N_total, B, D) — 所有 cam 的 spatial tokens concat
            pos:       (N_total, B, D) — 对应的 position embedding
            n_vision:  int — vision tokens 数量 (非 gelsight)
            n_tactile: int — tactile tokens 数量 (gelsight)
        """
        all_cam_features = []
        all_cam_pos = []
        n_vision = 0
        n_tactile = 0

        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[self.cam_backbone_mapping[cam_name]](images[cam_id])
            features = features[0]  # last layer
            pos = pos[0]
            proj = self.input_proj(features).flatten(2)  # (B, D, N_spatial)
            pos_flat = pos.flatten(2)                      # (B, D, N_spatial)

            n_tokens = proj.size(2)
            if cam_name == 'gelsight':
                n_tactile += n_tokens
            else:
                n_vision += n_tokens

            all_cam_features.append(proj)
            all_cam_pos.append(pos_flat)

        src = torch.cat(all_cam_features, dim=2)  # (B, D, N_total)
        pos = torch.cat(all_cam_pos, dim=2)        # (B, D, N_total)

        # permute to (N, B, D)
        src = src.permute(2, 0, 1)
        pos = pos.permute(2, 0, 1)

        return src, pos, n_vision, n_tactile

    def _cvae_encode(self, qpos, actions, is_pad):
        """CVAE encoder: 训练时从 action sequence 推断 z, 推理时 z=0."""
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
            encoder_output = encoder_output[0]  # CLS output
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
        else:
            mu = logvar = None
            latent_sample = torch.zeros(bs, self.latent_dim, device=qpos.device)

        latent_input = self.latent_out_proj(latent_sample)  # (B, D)
        return latent_input, mu, logvar

    def _run_decoder(self, decoder, query_embed, action_head, memory, pos, bs):
        """运行单个 decoder 生成动作序列。"""
        q_embed = query_embed.weight.unsqueeze(1).expand(-1, bs, -1)  # (num_queries, B, D)
        tgt = torch.zeros_like(q_embed)

        hs = decoder(tgt, memory, pos=pos, query_pos=q_embed)  # (1, num_queries, B, D)
        hs = hs[0].permute(1, 0, 2)  # (B, num_queries, D)
        a_hat = action_head(hs)       # (B, num_queries, action_dim)
        return a_hat

    def _compute_gt_future_features(self, future_images):
        """
        用 backbone 计算 t+h 时刻图像的 pooled feature (用于 MSE target)。
        Returns:
            v_gt_feat: (B, D) — vision cameras mean-pooled
            t_gt_feat: (B, D) — tactile (gelsight) mean-pooled
        """
        v_feats = []
        t_feat = None

        with torch.no_grad():
            for cam_id, cam_name in enumerate(self.camera_names):
                features, _ = self.backbones[self.cam_backbone_mapping[cam_name]](future_images[cam_id])
                features = features[0]
                proj = self.input_proj(features).flatten(2)  # (B, D, N)
                pooled = proj.mean(dim=2)  # (B, D)

                if cam_name == 'gelsight':
                    t_feat = pooled
                else:
                    v_feats.append(pooled)

        # vision: average across all vision cameras
        if v_feats:
            v_gt_feat = torch.stack(v_feats, dim=0).mean(dim=0)  # (B, D)
        else:
            v_gt_feat = torch.zeros_like(t_feat)

        if t_feat is None:
            t_feat = torch.zeros_like(v_gt_feat)

        return v_gt_feat, t_feat

    def forward(self, qpos, images, actions=None, is_pad=None,
                future_images=None, use_predicted_future=False):
        """
        Args:
            qpos:    (B, state_dim)
            images:  list of (B, C, H, W), len=num_cameras
            actions: (B, chunk_size, action_dim) or None (inference)
            is_pad:  (B, chunk_size) or None
            future_images: list of (B, C, H, W) at t+h, or None (inference)
            use_predicted_future: bool — 课程学习: True 用预测, False 用 GT
        """
        is_training = actions is not None
        bs = qpos.size(0)

        # ---- 1. CVAE encoder → latent z ----
        latent_input, mu, logvar = self._cvae_encode(qpos, actions, is_pad)

        # ---- 2. Backbone → image tokens ----
        src, pos, n_vision, n_tactile = self._encode_images(images)
        # src: (N_total, B, D), pos: (N_total, B, D)

        # ---- 3. Prepend [latent, proprio] to src ----
        proprio_input = self.input_proj_robot_state(qpos)  # (B, D)
        addition_input = torch.stack([latent_input, proprio_input], dim=0)  # (2, B, D)
        src_full = torch.cat([addition_input, src], dim=0)  # (2+N_total, B, D)

        add_pos = self.additional_pos_embed.weight.unsqueeze(1).expand(-1, bs, -1)  # (2, B, D)
        pos_expanded = pos.expand(-1, bs, -1) if pos.size(1) == 1 else pos
        pos_full = torch.cat([add_pos, pos_expanded], dim=0)  # (2+N_total, B, D)

        # ---- 4. Shared encoder → memory ----
        memory = self.vt_encoder(src_full, pos=pos_full)  # (2+N_total, B, D)

        # ---- 5. Decoder draft → A1 ----
        a1_hat = self._run_decoder(self.decoder_draft, self.query_embed_draft,
                                    self.action_head_draft, memory, pos_full, bs)
        # a1_hat: (B, chunk_size, action_dim)

        # ---- 6. Foresight prediction (A1 detached) ----
        # 用 src (backbone 直接输出) 而非 memory, 与 GT target 保持同一特征空间
        v_tokens = src[:n_vision]   # (N_v, B, D)
        t_tokens = src[n_vision:]   # (N_t, B, D)

        t_hat_future, v_hat_future = self.foresight(
            v_tokens, t_tokens, a1_hat.detach(), n_vision)
        # t_hat_future: (B, D), v_hat_future: (B, D)

        # ---- 7. GT future features (训练时) ----
        v_gt_feat = t_gt_feat = None
        if is_training and future_images is not None:
            v_gt_feat, t_gt_feat = self._compute_gt_future_features(future_images)

        # ---- 8. 课程学习: 选择 future tactile ----
        if is_training:
            if use_predicted_future:
                future_tac_for_decoder = t_hat_future
            else:
                # 用 GT (但让梯度通过, 以便 decoder 学会利用 future info)
                future_tac_for_decoder = t_gt_feat if t_gt_feat is not None else t_hat_future
        else:
            future_tac_for_decoder = t_hat_future

        # ---- 9. Gated fusion → enriched memory for Decoder₂ ----
        a1_pooled = self.a1_pool_proj(a1_hat.detach().mean(dim=1))  # (B, D)
        fused_memory = self.gated_fusion(memory, a1_pooled, future_tac_for_decoder)
        # fused_memory: (2+N_total, B, D)

        # ---- 10. Decoder final → A2 ----
        a2_hat = self._run_decoder(self.decoder_final, self.query_embed_final,
                                    self.action_head_final, fused_memory, pos_full, bs)
        # a2_hat: (B, chunk_size, action_dim)

        return a1_hat, a2_hat, t_hat_future, v_hat_future, v_gt_feat, t_gt_feat, (mu, logvar)
