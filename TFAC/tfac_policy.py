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
from policy import MyJoiner, kl_divergence
from TFAC.tfac_model import TFACModel


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
                 pretrained_backbones=None,
                 cam_backbone_mapping=None,
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
                 num_dec_layers_draft: int = None,
                 ):
        super().__init__()

        # --- Build backbones ---
        if cam_backbone_mapping is None:
            cam_backbone_mapping = {cam_name: 0 for cam_name in camera_names}
            num_backbones = 1
        else:
            num_backbones = len(set(cam_backbone_mapping.values()))

        if pretrained_backbones is not None:
            num_backbones = len(pretrained_backbones)

        backbones = []
        for i in range(num_backbones):
            N_steps = hidden_dim // 2
            if position_embedding_type in ('v2', 'sine'):
                position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
            elif position_embedding_type in ('v3', 'learned'):
                position_embedding = PositionEmbeddingLearned(N_steps)
            else:
                raise ValueError(f"not supported {position_embedding_type}")

            if pretrained_backbones is None:
                train_backbone = lr_backbone > 0
                backbone = Backbone(name=backbone_type,
                                    train_backbone=train_backbone,
                                    return_interm_layers=masks,
                                    dilation=dilation)
                backbone_model = Joiner(backbone, position_embedding)
                backbone_model.num_channels = backbone.num_channels
            else:
                backbone_model = MyJoiner(pretrained_backbones[i], position_embedding)
                backbone_model.num_channels = 512  # resnet18
            backbones.append(backbone_model)

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
        self.curriculum_ratio = curriculum_ratio

        print(f'TFAC KL Weight {self.kl_weight}, Curriculum ratio {self.curriculum_ratio}')

    def __call__(self, qpos, images, actions=None, is_pad=None,
                 future_images=None, epoch=None, total_epochs=None,
                 ignore_latent=False):
        """
        Training: 返回 loss_dict
        Inference: 返回 a2_hat (B, chunk_size, action_dim)
        """
        if actions is not None:
            # 课程学习切换
            use_predicted = False
            if epoch is not None and total_epochs is not None:
                use_predicted = (epoch >= total_epochs * self.curriculum_ratio)

            a1_hat, a2_hat, t_hat, v_hat, v_gt, t_gt, (mu, logvar) = self.model(
                qpos, images, actions, is_pad, future_images, use_predicted)

            # --- Losses ---
            loss_dict = {}

            # Action losses (masked by is_pad)
            pad_mask = ~is_pad.unsqueeze(-1)  # (B, chunk, 1)

            l1_draft = (F.l1_loss(a1_hat, actions, reduction='none') * pad_mask).mean()
            l1_final = (F.l1_loss(a2_hat, actions, reduction='none') * pad_mask).mean()
            loss_dict['l1_draft'] = l1_draft
            loss_dict['l1_final'] = l1_final

            # Foresight losses
            if t_gt is not None:
                loss_foresight_tac = F.mse_loss(t_hat, t_gt)
                loss_dict['foresight_tac'] = loss_foresight_tac
            else:
                loss_foresight_tac = torch.tensor(0.0, device=qpos.device)
                loss_dict['foresight_tac'] = loss_foresight_tac

            if v_gt is not None:
                loss_foresight_vis = F.mse_loss(v_hat, v_gt)
                loss_dict['foresight_vis'] = loss_foresight_vis
            else:
                loss_foresight_vis = torch.tensor(0.0, device=qpos.device)
                loss_dict['foresight_vis'] = loss_foresight_vis

            # Contrastive loss
            loss_contrastive = self.model.contrastive(v_hat, t_hat)
            loss_dict['contrastive'] = loss_contrastive

            # KL loss
            total_kld, _, _ = kl_divergence(mu, logvar)
            loss_dict['kl'] = total_kld[0]

            # Total loss
            loss = (l1_final
                    + self.lambda_draft * l1_draft
                    + self.lambda_foresight * loss_foresight_tac
                    + self.lambda_foresight * self.lambda_foresight_vis * loss_foresight_vis
                    + self.lambda_contrastive * loss_contrastive
                    + self.kl_weight * total_kld[0])
            loss_dict['loss'] = loss

            # Gate weights for logging (not part of loss)
            gm, ga, gf = self.model.gated_fusion._last_gate_means
            loss_dict['gate_mem'] = torch.tensor(gm)
            loss_dict['gate_a1'] = torch.tensor(ga)
            loss_dict['gate_fut'] = torch.tensor(gf)

            return loss_dict

        else:
            # Inference: Think → Dream → Act
            a1_hat, a2_hat, _, _, _, _, _ = self.model(qpos, images)
            return a2_hat

    def configure_optimizers(self):
        return self.optimizer
