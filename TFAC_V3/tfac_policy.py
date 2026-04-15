"""
TFACPolicy: 封装 TFACModel + 损失计算 + 课程学习 + 优化器。
接口与 ACTPolicy 对齐, 方便替换训练脚本。
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from detr.models.transformer import TransformerEncoderLayer, TransformerEncoder
from detr.models.backbone import Backbone, Joiner, PositionEmbeddingSine, PositionEmbeddingLearned
from policy import kl_divergence
from TFAC_V3.tfac_model import TFACModel


class TFACPolicy(nn.Module):
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
                 lambda_draft: float = 0.5,
                 lambda_foresight: float = 1.0,
                 lambda_foresight_vis: float = 0.3,
                 lambda_contrastive: float = 0.1,
                 lambda_contrastive_gt: float = 0.0,
                 lambda_sampling: float = 0.5,
                 num_dec_layers_draft: int = None,
                 foresight_change_weight: bool = False,
                 # V4 modularity switches
                 tactile_mode: str = "image",
                 marker_encoder_type: str = "conv2d",
                 fusion_mode: str = "gate",
                 foresight_tac_decoder: str = "linear",
                 spatial_tac_dec_layers: int = 3,
                 a2_init: str = "zero",
                 max_history: int = 8,
                 predict_horizon: int = 1,
                 sampling_steps: int = 3,
                 ):
        super().__init__()

        self.tactile_mode = tactile_mode
        self.fusion_mode = fusion_mode
        self.predict_horizon = predict_horizon
        self.sampling_steps = sampling_steps

        # --- Build backbone (ImageNet pretrained ResNet18 + FrozenBatchNorm) ---
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

        # --- Build TFACModel ---
        self.model = TFACModel(
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
            # V4 modularity
            tactile_mode=tactile_mode,
            marker_encoder_type=marker_encoder_type,
            fusion_mode=fusion_mode,
            foresight_tac_decoder=foresight_tac_decoder,
            spatial_tac_dec_layers=spatial_tac_dec_layers,
            a2_init=a2_init,
            max_history=max_history,
            predict_horizon=predict_horizon,
        )

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("TFAC number of parameters: %.2fM" % (n_parameters / 1e6,))
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
        self.lambda_foresight = lambda_foresight
        self.lambda_foresight_vis = lambda_foresight_vis
        self.lambda_contrastive = lambda_contrastive
        self.lambda_contrastive_gt = lambda_contrastive_gt
        self.lambda_sampling = lambda_sampling
        self.curriculum_ratio = curriculum_ratio
        self.foresight_change_weight = foresight_change_weight

        print(f'TFAC KL Weight {self.kl_weight}, Curriculum ratio {self.curriculum_ratio}'
              f', Foresight change weight: {self.foresight_change_weight}'
              f', predict_horizon: {predict_horizon}, sampling_steps: {sampling_steps}')

    def forward(self, qpos, images, actions=None, is_pad=None,
                future_images=None, epoch=None, total_epochs=None,
                ignore_latent=False, history_images=None):
        """
        Training: 返回 loss_dict
        Inference: 返回 a2_hat (B, chunk_size, action_dim)

        DataParallel 兼容: images/future_images/history_images 可以是
        stacked tensor (num_cam, B, ...) 或 list of tensors。
        """
        # DataParallel 兼容: tuple/list 均转为 list (DataParallel scatter 递归处理 tuple)
        if isinstance(images, tuple):
            images = list(images)
        if future_images is not None and isinstance(future_images, tuple):
            future_images = list(future_images)
        if history_images is not None and isinstance(history_images, tuple):
            history_images = list(history_images)

        if actions is not None:
            # 课程学习切换
            use_predicted = False
            if epoch is not None and total_epochs is not None:
                use_predicted = (epoch >= total_epochs * self.curriculum_ratio)

            (a1_hat, a2_hat, t_hat, v_hat,
             v_gt, t_gt, t_hat_encoded, t_cur, (mu, logvar),
             t_embed_future) = self.model(
                qpos, images, actions, is_pad, future_images, use_predicted,
                history_images=history_images)

            # --- Losses ---
            loss_dict = {}

            # Action losses (masked by is_pad)
            pad_mask = ~is_pad.unsqueeze(-1)  # (B, chunk, 1)

            l1_draft = (F.l1_loss(a1_hat, actions, reduction='none') * pad_mask).mean()
            l1_final = (F.l1_loss(a2_hat, actions, reduction='none') * pad_mask).mean()
            loss_dict['l1_draft'] = l1_draft
            loss_dict['l1_final'] = l1_final

            # Foresight tactile loss (teacher forcing)
            # t_hat: marker mode (B, H, 9, 9, 2) if H>1 else (B, 9, 9, 2); image mode (B, D)
            # t_gt:  same shape as t_hat
            if t_gt is not None:
                if self.tactile_mode == "marker" and self.predict_horizon > 1:
                    # 多帧: smooth_L1 on each frame, average over H
                    # t_hat: (B, H, 9, 9, 2), t_gt: (B, H, 9, 9, 2)
                    per_frame_loss = F.smooth_l1_loss(
                        t_hat, t_gt, reduction='none').mean(dim=(2, 3, 4))  # (B, H)
                    per_sample_mse_tac = per_frame_loss.mean(dim=1)  # (B,)
                elif self.tactile_mode == "marker":
                    # 单帧: smooth_L1 on (B, 9, 9, 2)
                    per_sample_mse_tac = F.smooth_l1_loss(
                        t_hat, t_gt, reduction='none').mean(dim=(1, 2, 3))  # (B,)
                else:
                    # MSE on embedding (B, D)
                    per_sample_mse_tac = (t_hat - t_gt).pow(2).mean(dim=-1)  # (B,)

                if self.foresight_change_weight and t_cur is not None:
                    if self.tactile_mode == "marker":
                        gt_for_change = t_gt[:, -1] if t_gt.dim() == 5 else t_gt
                        with torch.no_grad():
                            t_gt_enc = self.model.marker_encoder(gt_for_change)
                        change = (t_cur - t_gt_enc).detach().pow(2).mean(dim=-1)
                    else:
                        change = (t_cur - t_gt).detach().pow(2).mean(dim=-1)
                    weight = (change / (change.mean() + 1e-8)).sqrt()
                    weight = weight / (weight.mean() + 1e-8)
                    loss_foresight_tac = (weight * per_sample_mse_tac).mean()
                else:
                    loss_foresight_tac = per_sample_mse_tac.mean()
                loss_dict['foresight_tac'] = loss_foresight_tac
            else:
                loss_foresight_tac = torch.tensor(0.0, device=qpos.device)
                loss_dict['foresight_tac'] = loss_foresight_tac

            if v_gt is not None:
                # v_hat 始终是单帧 (B, D); v_gt 可能是多帧 (B, H, D) 或单帧 (B, D)
                # foresight_vis 只比较最后帧
                v_gt_last = v_gt[:, -1] if v_gt.dim() == 3 else v_gt  # (B, D)
                loss_foresight_vis = F.mse_loss(v_hat, v_gt_last)
                loss_dict['foresight_vis'] = loss_foresight_vis
            else:
                loss_foresight_vis = torch.tensor(0.0, device=qpos.device)
                loss_dict['foresight_vis'] = loss_foresight_vis

            # Contrastive loss — 预测触觉 vs GT视觉
            # P1: 用 embed_predictor 直出的 t_embed_future, 不再 roundtrip
            # 多帧: per-frame contrastive; 单帧: 向后兼容
            if (v_gt is not None and v_gt.dim() == 3
                    and t_embed_future is not None and t_embed_future.dim() == 3):
                H_cont = min(v_gt.shape[1], t_embed_future.shape[1])
                contrastive_losses = []
                for h in range(H_cont):
                    loss_h = self.model.contrastive(
                        v_gt[:, h], t_embed_future[:, h])
                    contrastive_losses.append(loss_h)
                loss_contrastive = torch.stack(contrastive_losses).mean()
            else:
                # 单帧向后兼容
                loss_contrastive = self.model.contrastive(v_gt, t_hat_encoded)
            loss_dict['contrastive'] = loss_contrastive

            # GT contrastive loss — GT触觉 vs GT视觉 (双重对比学习)
            # 多帧: per-frame; 单帧: 最后帧
            if self.lambda_contrastive_gt > 0 and t_gt is not None:
                if v_gt is not None and v_gt.dim() == 3 and t_gt.dim() == 5:
                    # Per-frame GT contrastive
                    H_cont = min(v_gt.shape[1], t_gt.shape[1])
                    gt_contrastive_losses = []
                    for h in range(H_cont):
                        t_gt_h_enc = self.model.marker_encoder(t_gt[:, h])  # (B, D)
                        loss_gt_h = self.model.contrastive(v_gt[:, h], t_gt_h_enc)
                        gt_contrastive_losses.append(loss_gt_h)
                    loss_contrastive_gt = torch.stack(gt_contrastive_losses).mean()
                else:
                    # 单帧向后兼容
                    if self.tactile_mode == "marker":
                        t_gt_last = t_gt[:, -1] if t_gt.dim() == 5 else t_gt
                        t_gt_encoded = self.model.marker_encoder(t_gt_last)
                    else:
                        t_gt_encoded = t_gt
                    loss_contrastive_gt = self.model.contrastive(v_gt, t_gt_encoded)
            else:
                loss_contrastive_gt = torch.tensor(0.0, device=qpos.device)
            loss_dict['contrastive_gt'] = loss_contrastive_gt

            # KL loss
            total_kld, _, _ = kl_divergence(mu, logvar)
            loss_dict['kl'] = total_kld[0]

            # Sampling loss (autoregressive rollout) — skip during validation
            loss_sampling = torch.tensor(0.0, device=qpos.device)
            if (self.training
                    and self.sampling_steps > 0 and self.lambda_sampling > 0
                    and self.tactile_mode == "marker" and self.predict_horizon > 1
                    and t_gt is not None):
                loss_sampling = self._compute_sampling_loss(
                    qpos, images, a1_hat.detach(), t_gt,
                    history_images=history_images)
            loss_dict['sampling'] = loss_sampling

            # Total loss
            loss = (l1_final
                    + self.lambda_draft * l1_draft
                    + self.lambda_foresight * loss_foresight_tac
                    + self.lambda_foresight * self.lambda_foresight_vis * loss_foresight_vis
                    + self.lambda_contrastive * loss_contrastive
                    + self.lambda_contrastive_gt * loss_contrastive_gt
                    + self.lambda_sampling * loss_sampling
                    + self.kl_weight * total_kld[0])
            loss_dict['loss'] = loss

            # Gate weights for logging (only for gate fusion mode)
            if self.fusion_mode == "gate":
                gm, ga, gf = self.model.gated_fusion._last_gate_means
                loss_dict['gate_mem'] = torch.tensor(gm, device=qpos.device)
                loss_dict['gate_a1'] = torch.tensor(ga, device=qpos.device)
                loss_dict['gate_fut'] = torch.tensor(gf, device=qpos.device)

            return loss_dict

        else:
            # Inference: Think → Dream → Act
            a1_hat, a2_hat, _, _, _, _, _, _, _, _ = self.model(
                qpos, images, history_images=history_images)
            return a2_hat

    def _compute_sampling_loss(self, qpos, images, a1_detached, t_gt,
                               history_images=None):
        """
        Sampling loss: 自回归展开 S 步。
        每步用上一步的预测触觉(detach)替换输入触觉, 重新跑 foresight,
        对第 s 帧的预测和 GT 算 loss。

        复用 model._fwd_cache 中的 backbone 特征, 避免重复计算。

        Args:
            qpos: (B, state_dim)
            images: list of (B, C, H, W)
            a1_detached: (B, chunk_size, action_dim) — detached draft action
            t_gt: (B, H, 9, 9, 2) — multi-frame GT
            history_images: list of (B, k, ...) per camera, or None
        Returns:
            loss: scalar — average sampling loss over S steps
        """
        if self.model.tactile_mode != 'marker':
            return torch.tensor(0.0, device=qpos.device)

        S = min(self.sampling_steps, self.predict_horizon)
        bs = qpos.size(0)

        # Reuse cached backbone features from main forward (avoid duplicate computation)
        cache = self.model._fwd_cache
        src = cache['src']
        n_vision = cache['n_vision']
        n_tactile = cache['n_tactile']
        v_tokens = src[:n_vision]   # (N_v, B, D)
        t_tokens = src[n_vision:]   # (N_t, B, D)

        # Reuse cached history features
        hist_src = cache['hist_src']
        if hist_src is not None:
            v_tokens_hist = hist_src[:, :n_vision]
            t_tokens_hist = hist_src[:, n_vision:]
            use_hist = True
        else:
            use_hist = False

        total_loss = 0.0
        # 当前触觉 tokens 用于第一步输入
        current_t_input = t_tokens  # (N_t, B, D)

        # 均匀采样帧索引, 覆盖整个预测范围 (e.g. S=3, H=10 → [0, 4, 9])
        H = self.predict_horizon
        if S >= H:
            frame_indices = list(range(H))
        else:
            frame_indices = [round(i * (H - 1) / (S - 1)) for i in range(S)]

        for step, s in enumerate(frame_indices):
            if use_hist:
                # 替换最后一帧的触觉 tokens
                v_in = v_tokens_hist
                t_in = t_tokens_hist.clone()
                t_in[-1] = current_t_input  # replace last frame tactile
            else:
                v_in = v_tokens
                t_in = current_t_input

            t_hat_raw, _, _ = self.model.foresight(v_in, t_in, a1_detached, n_vision,
                                                     proprio=qpos)
            # t_hat_raw: (B, H, 162)
            t_hat_frames = t_hat_raw.view(bs, H, 9, 9, 2)

            # Loss on frame s: pred[s] vs gt[s]
            step_loss = F.smooth_l1_loss(t_hat_frames[:, s], t_gt[:, s])
            total_loss = total_loss + step_loss

            # 下一步输入: 用预测的第 s 帧 encode 成 tokens (detach)
            with torch.no_grad():
                pred_tac = t_hat_frames[:, s].detach()  # (B, 9, 9, 2)
                # Encode through marker encoder → (B, D), then expand to token format
                pred_encoded = self.model.marker_encoder(pred_tac)  # (B, D)
                # Use as single token repeated N_t times (simplified)
                current_t_input = pred_encoded.unsqueeze(0).expand(n_tactile, -1, -1)

        return total_loss / S

    def configure_optimizers(self):
        return self.optimizer
