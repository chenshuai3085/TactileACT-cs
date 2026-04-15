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
from TFAC_V3.foresight_transformer import ForesightTransformer, ForesightContrastive
from TFAC_V3.marker_encoder import build_marker_encoder, LTDEncoder


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
    三路独立加权门控融合: memory + A1 feat + future feat → enriched memory。
    softmax 产生三路权重 (和为1), 可直接观测各路贡献。
    """

    def __init__(self, d_model: int):
        super().__init__()
        # 输入三路 concat, 输出 3 个 gate logits (逐维度)
        self.gate_proj = nn.Linear(d_model * 3, d_model * 3)
        # 各路先过 projection 对齐表示空间
        self.memory_proj = nn.Linear(d_model, d_model)
        self.a1_proj = nn.Linear(d_model, d_model)
        self.future_proj = nn.Linear(d_model, d_model)

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
        S, B, D = memory.shape
        a1_exp = a1_feat.unsqueeze(0).expand(S, -1, -1)       # (S, B, D)
        fut_exp = future_feat.unsqueeze(0).expand(S, -1, -1)   # (S, B, D)

        # 计算三路 gate: softmax 保证和为 1
        concat = torch.cat([memory, a1_exp, fut_exp], dim=-1)   # (S, B, 3D)
        gate_logits = self.gate_proj(concat)                     # (S, B, 3D)
        gate_logits = gate_logits.view(S, B, 3, D)              # (S, B, 3, D)
        gates = torch.softmax(gate_logits, dim=2)                # (S, B, 3, D)

        g_mem = gates[:, :, 0, :]    # (S, B, D)
        g_a1  = gates[:, :, 1, :]    # (S, B, D)
        g_fut = gates[:, :, 2, :]    # (S, B, D)

        # 各路投影后加权求和
        fused = (g_mem * self.memory_proj(memory)
                 + g_a1 * self.a1_proj(a1_exp)
                 + g_fut * self.future_proj(fut_exp))  # (S, B, D)

        # 保存 gate 均值供 log 观测
        self._last_gate_means = (
            g_mem.mean().item(),
            g_a1.mean().item(),
            g_fut.mean().item(),
        )
        # 逐样本权重: 对 S 和 D 维取均值 → (B, 3)
        self._last_gate_per_sample = torch.stack([
            g_mem.mean(dim=(0, 2)),  # (B,)
            g_a1.mean(dim=(0, 2)),
            g_fut.mean(dim=(0, 2)),
        ], dim=-1)  # (B, 3)

        return fused


class TFACModel(nn.Module):
    """
    完整 TFAC 模型。

    forward 返回:
        a1_hat:       (B, chunk_size, action_dim)
        a2_hat:       (B, chunk_size, action_dim)
        t_hat_future: image mode: (B, D); marker mode: (B, 9, 9, 2) raw prediction
        v_hat_future: (B, D)
        t_gt_feat:    image mode: (B, D) embedding; marker mode: (B, 9, 9, 2) raw GT
        v_gt_feat:    (B, D) — GT future vision feature (训练时)
        t_hat_encoded:(B, D) — MarkerEncoder(t_hat_future) for fusion/contrastive (marker mode only)
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
                 num_dec_layers_draft=None,  # None = same as num_dec_layers
                 # foresight params
                 foresight_layers=2, foresight_nheads=4,
                 foresight_dim_feedforward=2048,
                 # contrastive params
                 proj_dim=128, contrastive_temperature=0.07,
                 # V4 modularity switches
                 tactile_mode="image",        # "image" | "marker"
                 marker_encoder_type="conv2d", # "conv2d" | "pointnet"
                 fusion_mode="gate",           # "gate" | "ltd" | "token"
                 foresight_tac_decoder="linear",   # "linear" | "spatial"
                 spatial_tac_dec_layers=3,         # 2 (old ckpt) | 3 (new)
                 a2_init="zero",                    # "zero" | "a1_refine"
                 max_history=8,                     # max history frames for temporal pos embed
                 predict_horizon=1):                # multi-frame prediction horizon
        super().__init__()

        self.num_queries = num_queries
        self.camera_names = camera_names
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.cam_backbone_mapping = cam_backbone_mapping
        self.tactile_mode = tactile_mode
        self.fusion_mode = fusion_mode
        self.a2_init = a2_init
        self.predict_horizon = predict_horizon

        # ---- Shared backbone ----
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
        else:
            self.backbones = None

        # ---- Marker encoder (tactile_mode="marker") ----
        self.marker_encoder = None
        if tactile_mode == "marker":
            self.marker_encoder = build_marker_encoder(marker_encoder_type, hidden_dim=hidden_dim)
            # Learned position embedding for marker token (1 token)
            self.marker_pos_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

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
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)  # [latent, proprio]

        # ---- Foresight ----
        # marker mode: 预测 raw marker_offset (9*9*2=162 dim)
        # image mode: 预测 embedding (hidden_dim)
        tactile_out_dim = 9 * 9 * 2 if tactile_mode == "marker" else hidden_dim
        self.tactile_out_dim = tactile_out_dim

        self.foresight = ForesightTransformer(
            d_model=hidden_dim, action_dim=state_dim,
            num_layers=foresight_layers, nhead=foresight_nheads,
            dim_feedforward=foresight_dim_feedforward, dropout=dropout,
            tactile_out_dim=tactile_out_dim,
            tactile_decoder_type=foresight_tac_decoder,
            spatial_tac_dec_layers=spatial_tac_dec_layers,
            max_history=max_history,
            predict_horizon=predict_horizon)

        # ---- Contrastive ----
        self.contrastive = ForesightContrastive(
            feat_dim=hidden_dim, proj_dim=proj_dim,
            temperature=contrastive_temperature)

        # ---- Fusion for Decoder₂ ----
        if fusion_mode == "gate":
            self.gated_fusion = GatedFusion(hidden_dim)
            # A1 pooling: project action chunk → single feature for gating
            self.a1_pool_proj = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        elif fusion_mode == "ltd":
            self.ltd_encoder = LTDEncoder(hidden_dim)
            # LTD conditioning: project (B, D) → (S, B, D) via learned scale+shift (FiLM)
            self.ltd_scale = nn.Linear(hidden_dim, hidden_dim)
            self.ltd_shift = nn.Linear(hidden_dim, hidden_dim)
        elif fusion_mode == "token":
            # Append predicted future tactile as extra memory token
            self.foresight_pos_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        # ---- A2 init from A1 ----
        if a2_init == "a1_refine":
            self.a1_to_hidden = nn.Linear(state_dim, hidden_dim)

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
        tactile_mode="marker" 时, gelsight 位用 MarkerEncoder 编码。

        Args:
            images: list, len = num_cameras
                    - vision cameras: (B, C, H, W) image tensor
                    - gelsight (image mode): (B, C, H, W) image tensor
                    - gelsight (marker mode): (B, 9, 9, 2) marker_offset tensor
        Returns:
            src:       (N_total, B, D) — 所有 cam 的 tokens concat
            pos:       (N_total, B, D) — 对应的 position embedding
            n_vision:  int — vision tokens 数量 (非 gelsight)
            n_tactile: int — tactile tokens 数量 (gelsight)
        """
        all_cam_features = []
        all_cam_pos = []
        n_vision = 0
        n_tactile = 0

        for cam_id, cam_name in enumerate(self.camera_names):
            if cam_name == 'gelsight' and self.tactile_mode == 'marker':
                # MarkerEncoder: (B, 9, 9, 2) → (B, D) → (1, B, D)
                marker_feat = self.marker_encoder(images[cam_id])  # (B, D)
                proj = marker_feat.unsqueeze(0)  # (1, B, D)
                # pos 保持 B=1, 与 backbone pos 一致, 在 forward() 中统一 expand
                pos_flat = self.marker_pos_embed  # (1, 1, D)
                n_tactile += 1
            else:
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

                # permute to (N, B, D) for concat
                proj = proj.permute(2, 0, 1)      # (N, B, D)
                pos_flat = pos_flat.permute(2, 0, 1)  # (N, B, D)

            all_cam_features.append(proj)
            all_cam_pos.append(pos_flat)

        src = torch.cat(all_cam_features, dim=0)  # (N_total, B, D)
        pos = torch.cat(all_cam_pos, dim=0)        # (N_total, B, D)

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

    def _run_decoder(self, decoder, query_embed, action_head, memory, pos, bs,
                     tgt_init=None):
        """运行单个 decoder 生成动作序列。"""
        q_embed = query_embed.weight.unsqueeze(1).expand(-1, bs, -1)  # (num_queries, B, D)
        tgt = tgt_init if tgt_init is not None else torch.zeros_like(q_embed)

        hs = decoder(tgt, memory, pos=pos, query_pos=q_embed)  # (1, num_queries, B, D)
        hs = hs[0].permute(1, 0, 2)  # (B, num_queries, D)
        a_hat = action_head(hs)       # (B, num_queries, action_dim)
        return a_hat

    def _encode_history(self, history_images, skip_last=False):
        """
        对过去 k 帧图像分别过 backbone，返回各帧的 src tokens。

        Args:
            history_images: list of num_cameras tensors, each (k, C, H, W)
                            or marker mode (k, 9, 9, 2)
            skip_last: 跳过最后一帧 (当前帧), 避免浪费计算 (会被 src 替换)
        Returns:
            hist_src: (k, N_total, B, D) or (k-1, ...) if skip_last
        """
        num_cams = len(self.camera_names)
        k = history_images[0].shape[1]  # history length (B, k, C, H, W)
        end = k - 1 if skip_last else k

        results = []
        with torch.no_grad():
            for t in range(end):
                # Extract frame t from each camera: list of (B, C, H, W) or (B, 9, 9, 2)
                frame_images = [history_images[cam_idx][:, t] for cam_idx in range(num_cams)]
                src, pos, n_v, n_t = self._encode_images(frame_images)
                results.append(src)

        return torch.stack(results)  # (end, N_total, B, D)

    def _compute_gt_future_features(self, future_images):
        """
        计算 GT future features (用于 foresight loss target)。

        image mode: backbone → mean-pool → (B, D) embedding
        marker mode: gelsight 直接返回 raw marker_offset
            - 多帧: (B, H, 9, 9, 2)  (predict_horizon > 1)
            - 单帧: (B, 9, 9, 2)

        视觉 GT 支持两种模式:
            - 单帧: future_images[cam] = (B, C, Himg, Wimg) → v_gt_feat: (B, D)
            - 多帧 (multi_frame_vision): future_images[cam] = (B, H, C, Himg, Wimg)
              → v_gt_feat: (B, H, D), 用于 per-frame contrastive

        Returns:
            v_gt_feat: (B, D) 单帧 or (B, H, D) 多帧
            t_gt_feat: marker mode: (B, H, 9, 9, 2) or (B, 9, 9, 2);
                       image mode: (B, D) embedding
        """
        v_feats = []
        t_feat = None
        is_multi_frame_vision = False

        with torch.no_grad():
            for cam_id, cam_name in enumerate(self.camera_names):
                if cam_name == 'gelsight' and self.tactile_mode == 'marker':
                    # marker mode: GT 是 raw marker_offset
                    # 多帧: (B, H, 9, 9, 2); 单帧: (B, 9, 9, 2)
                    t_feat = future_images[cam_id]
                elif cam_name == 'gelsight':
                    # image mode gelsight
                    features, _ = self.backbones[self.cam_backbone_mapping[cam_name]](future_images[cam_id])
                    features = features[0]
                    proj = self.input_proj(features).flatten(2)
                    t_feat = proj.mean(dim=2)  # (B, D)
                else:
                    img = future_images[cam_id]
                    if img.dim() == 5:
                        # 多帧视觉: (B, H, C, Himg, Wimg) → batch encode
                        is_multi_frame_vision = True
                        B, H = img.shape[:2]
                        img_flat = img.view(B * H, *img.shape[2:])  # (B*H, C, Himg, Wimg)
                        features, _ = self.backbones[self.cam_backbone_mapping[cam_name]](img_flat)
                        features = features[0]
                        proj = self.input_proj(features).flatten(2)  # (B*H, D, N)
                        pooled = proj.mean(dim=2)  # (B*H, D)
                        v_feats.append(pooled.view(B, H, -1))  # (B, H, D)
                    else:
                        # 单帧视觉: (B, C, Himg, Wimg)
                        features, _ = self.backbones[self.cam_backbone_mapping[cam_name]](img)
                        features = features[0]
                        proj = self.input_proj(features).flatten(2)  # (B, D, N)
                        pooled = proj.mean(dim=2)  # (B, D)
                        v_feats.append(pooled)

        # vision: average across all vision cameras
        if v_feats:
            if is_multi_frame_vision:
                # 多帧: (num_cams, B, H, D) → mean → (B, H, D)
                v_gt_feat = torch.stack(v_feats, dim=0).mean(dim=0)
            else:
                # 单帧: (num_cams, B, D) → mean → (B, D)
                v_gt_feat = torch.stack(v_feats, dim=0).mean(dim=0)
        else:
            v_gt_feat = torch.zeros(future_images[0].size(0), self.hidden_dim,
                                     device=future_images[0].device)

        if t_feat is None:
            bs = future_images[0].size(0)
            if self.tactile_mode == 'marker':
                if self.predict_horizon > 1:
                    t_feat = torch.zeros(bs, self.predict_horizon, 9, 9, 2,
                                          device=future_images[0].device)
                else:
                    t_feat = torch.zeros(bs, 9, 9, 2,
                                          device=future_images[0].device)
            else:
                t_feat = torch.zeros_like(v_gt_feat if v_gt_feat.dim() == 2 else v_gt_feat[:, 0])

        return v_gt_feat, t_feat

    def forward(self, qpos, images, actions=None, is_pad=None,
                future_images=None, use_predicted_future=False,
                history_images=None):
        """
        Args:
            qpos:    (B, state_dim)
            images:  list of (B, C, H, W), len=num_cameras
            actions: (B, chunk_size, action_dim) or None (inference)
            is_pad:  (B, chunk_size) or None
            future_images: list of (B, C, H, W) at t+h, or None (inference)
            use_predicted_future: bool — 课程学习: True 用预测, False 用 GT
            history_images: list of (B, k, C, H, W) per camera, or None (single-frame)
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

        has_history = history_images is not None and history_images[0].shape[1] > 1
        hist_src_full = None
        if has_history:
            # Temporal mode: encode past frames (skip current, will use src instead)
            hist_src = self._encode_history(history_images, skip_last=True)  # (k-1, N_total, B, D)
            hist_src_full = torch.cat([hist_src, src.unsqueeze(0)], dim=0)  # (k, N_total, B, D)
            v_tokens_hist = hist_src_full[:, :n_vision]  # (k, N_v, B, D)
            t_tokens_hist = hist_src_full[:, n_vision:]  # (k, N_t, B, D)

        # Cache backbone features for sampling loss reuse (avoid duplicate computation)
        self._fwd_cache = {
            'src': src, 'n_vision': n_vision, 'n_tactile': n_tactile,
            'hist_src': hist_src_full,
        }

        if has_history:
            t_hat_raw, v_hat_future, t_embed_future = self.foresight(
                v_tokens_hist, t_tokens_hist, a1_hat.detach(), n_vision,
                proprio=qpos)
        else:
            # Single-frame mode (backward compatible)
            t_hat_raw, v_hat_future, t_embed_future = self.foresight(
                v_tokens, t_tokens, a1_hat.detach(), n_vision,
                proprio=qpos)
        # t_hat_raw: marker mode: (B, H, 162) if H>1 else (B, 162); image mode: (B, D)
        # v_hat_future: (B, D)

        # ---- 6b. marker mode: reshape raw prediction ----
        # P1: fusion/contrastive 用 embed_predictor 直出的 embedding, 不再 roundtrip
        if self.tactile_mode == "marker":
            if self.predict_horizon > 1:
                H = self.predict_horizon
                t_hat_future = t_hat_raw.view(bs, H, 9, 9, 2)  # (B, H, 9, 9, 2)
                t_hat_encoded = t_embed_future[:, -1]  # (B, D) 最后帧 embedding
            else:
                t_hat_future = t_hat_raw.view(bs, 9, 9, 2)   # (B, 9, 9, 2)
                t_hat_encoded = t_embed_future  # (B, D)
        else:
            t_hat_future = t_hat_raw       # (B, D) embedding
            t_hat_encoded = t_embed_future  # (B, D)

        # ---- 7. GT future features (训练时) ----
        v_gt_feat = t_gt_feat = None
        if is_training and future_images is not None:
            v_gt_feat, t_gt_feat = self._compute_gt_future_features(future_images)

        # ---- 8. 课程学习: 选择 future tactile for fusion ----
        # fusion 需要 (B, D) encoded feature, 不是 raw marker_offset
        if is_training:
            if use_predicted_future:
                future_tac_for_decoder = t_hat_encoded
            else:
                # 用 GT: marker mode 需要先 encode GT marker_offset
                if t_gt_feat is not None:
                    if self.tactile_mode == "marker":
                        # 多帧 GT: (B, H, 9, 9, 2) → 取最后一帧 → encode
                        gt_for_fusion = t_gt_feat[:, -1] if t_gt_feat.dim() == 5 else t_gt_feat
                        future_tac_for_decoder = self.marker_encoder(gt_for_fusion)  # (B, D)
                    else:
                        future_tac_for_decoder = t_gt_feat  # (B, D) already embedding
                else:
                    future_tac_for_decoder = t_hat_encoded
        else:
            future_tac_for_decoder = t_hat_encoded

        # ---- 9. Fusion → enriched memory for Decoder₂ ----
        if self.fusion_mode == "gate":
            a1_pooled = self.a1_pool_proj(a1_hat.detach().mean(dim=1))  # (B, D)
            fused_memory = self.gated_fusion(memory, a1_pooled, future_tac_for_decoder)
        elif self.fusion_mode == "ltd":
            # LTD: concat(t_current, t_predicted, diff) → (B, D)
            t_current_pooled = t_tokens.mean(dim=0)  # (B, D)
            ltd_feat = self.ltd_encoder(t_current_pooled, future_tac_for_decoder)  # (B, D)
            # FiLM conditioning: scale and shift memory
            scale = self.ltd_scale(ltd_feat).unsqueeze(0)  # (1, B, D)
            shift = self.ltd_shift(ltd_feat).unsqueeze(0)  # (1, B, D)
            fused_memory = memory * (1 + scale) + shift     # (S, B, D)
        elif self.fusion_mode == "token":
            # Append foresight as extra token, decoder attention decides weight
            # Training: randomly drop foresight token to prevent attention collapse
            drop_foresight = self.training and torch.rand(1).item() < 0.3
            if drop_foresight:
                fused_memory = memory  # no foresight token, force model to use vision
            else:
                foresight_token = future_tac_for_decoder.unsqueeze(0)  # (1, B, D)
                fused_memory = torch.cat([memory, foresight_token], dim=0)
                foresight_pos = self.foresight_pos_embed.expand(-1, bs, -1)  # (1, B, D)
                pos_full = torch.cat([pos_full, foresight_pos], dim=0)

        # ---- 10. Decoder final → A2 ----
        tgt_init = None
        if self.a2_init == "a1_refine":
            # A1 (B,20,7) → (B,20,512) → (20,B,512)
            tgt_init = self.a1_to_hidden(a1_hat.detach()).permute(1, 0, 2)

        a2_hat = self._run_decoder(self.decoder_final, self.query_embed_final,
                                    self.action_head_final, fused_memory, pos_full, bs,
                                    tgt_init=tgt_init)
        # a2_hat: (B, chunk_size, action_dim)

        # 当前触觉 pooled feature (用于加权 foresight loss)
        t_current_feat = t_tokens.mean(dim=0)  # (B, D)

        return (a1_hat, a2_hat, t_hat_future, v_hat_future,
                v_gt_feat, t_gt_feat, t_hat_encoded, t_current_feat, (mu, logvar),
                t_embed_future)

    # ---- Attention hook utilities (for inference visualization) ----

    def enable_attn_hooks(self):
        """注册 hook 捕获 Decoder₂ 最后一层 cross-attention weights。
        调用后每次 forward 会将 attention map 存入 self._attn_weights。
        """
        self._attn_weights = {}
        self._hooks = []
        last_layer = self.decoder_final.layers[-1]

        def hook_fn(module, input, output):
            # nn.MultiheadAttention.forward returns (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
                self._attn_weights['decoder2_cross'] = output[1].detach().cpu()

        self._hooks.append(last_layer.multihead_attn.register_forward_hook(hook_fn))

    def disable_attn_hooks(self):
        """移除所有 attention hooks。"""
        for h in getattr(self, '_hooks', []):
            h.remove()
        self._hooks = []
        self._attn_weights = {}
