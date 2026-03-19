"""
Tactile Feasibility Score (TFS)

Contrastive learning scorer: given an observation (vision + predicted tactile)
and a candidate action chunk, produces a scalar feasibility score.

Architecture:
    Observation encoder:  [v_emb; t̂_emb] → MLP → Transformer Encoder → L2 norm → o_emb
    Action encoder:       action_chunk (T, D) → 1D Conv stack → MLP → L2 norm → a_emb
    Score: dot_product(o_emb, a_emb)

Training: InfoNCE contrastive loss (positive = same-timestep obs-action pair).
Noisy action training: actions are corrupted with DDPM-schedule noise so that
TFS can score partially-denoised actions during guided inference.

Requires: pre-trained VT alignment (Stage 0), pre-trained TFM (Stage 1).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extractor import DINOv2Extractor
from .vt_alignment import ProjectionHead


class ObservationEncoder(nn.Module):
    """Encodes [vision_emb, tactile_emb] into a fixed-dim representation."""

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        out_dim: int = 128,
        num_layers: int = 2,
        nheads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder_norm = nn.LayerNorm(hidden_dim)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        vision_emb: torch.Tensor,
        tactile_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            vision_emb: (B, embed_dim) aligned vision embedding
            tactile_emb: (B, embed_dim) predicted/real tactile embedding

        Returns:
            (B, out_dim) L2-normalized observation embedding
        """
        v = self.input_proj(vision_emb).unsqueeze(1)  # (B, 1, hidden)
        t = self.input_proj(tactile_emb).unsqueeze(1)  # (B, 1, hidden)
        tokens = torch.cat([v, t], dim=1)  # (B, 2, hidden)

        out = self.encoder(tokens)
        out = self.encoder_norm(out)
        # Mean pool over the 2 tokens
        out = out.mean(dim=1)  # (B, hidden)
        out = self.output_proj(out)  # (B, out_dim)
        out = F.normalize(out, dim=-1)
        return out


class ActionEncoder(nn.Module):
    """Encodes an action chunk into a fixed-dim representation using 1D convolutions."""

    def __init__(
        self,
        action_dim: int = 7,
        chunk_size: int = 20,
        hidden_dim: int = 256,
        out_dim: int = 128,
    ):
        super().__init__()
        # 1D Conv stack: (B, action_dim, T) → (B, hidden_dim, T')
        self.conv_stack = nn.Sequential(
            nn.Conv1d(action_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        # Global average pool + projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: (B, chunk_size, action_dim) action chunk

        Returns:
            (B, out_dim) L2-normalized action embedding
        """
        # (B, T, D) → (B, D, T) for Conv1d
        x = actions.permute(0, 2, 1)
        x = self.conv_stack(x)  # (B, hidden, T')
        x = x.mean(dim=-1)  # global average pool → (B, hidden)
        x = self.output_proj(x)  # (B, out_dim)
        x = F.normalize(x, dim=-1)
        return x


class TactileFeasibilityScore(nn.Module):
    """Full TFS: frozen DINOv2 + frozen alignment + obs encoder + action encoder.

    Training: InfoNCE on (obs, action) pairs with noisy action augmentation.
    Inference: score = dot(obs_emb, action_emb) used as gradient signal for guidance.

    Usage:
        tfs = TactileFeasibilityScore(
            alignment_ckpt="path/to/alignment_best.pth",
            action_dim=7, chunk_size=20,
            device="cuda",
        )

        # Training with real tactile
        loss_dict = tfs.compute_loss(
            vision_imgs, tac_future_imgs, actions,
            noise_levels=noise_k, action_noise=noise,
        )

        # Inference scoring (with predicted tactile embedding)
        score = tfs.score(v_emb_aligned, t_hat_aligned, actions)
    """

    def __init__(
        self,
        alignment_ckpt: Optional[str] = None,
        aligned_dim: int = 256,
        score_dim: int = 128,
        obs_hidden_dim: int = 256,
        obs_num_layers: int = 2,
        obs_nheads: int = 4,
        action_dim: int = 7,
        chunk_size: int = 20,
        action_hidden_dim: int = 256,
        learnable_temperature: bool = True,
        init_temperature: float = 0.07,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()
        self.aligned_dim = aligned_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

        # Frozen DINOv2
        self.dino = DINOv2Extractor(device=device)
        dino_dim = self.dino.embed_dim  # 768

        # Frozen alignment projection heads
        self.vision_align = ProjectionHead(
            in_dim=dino_dim, hidden_dim=512, out_dim=aligned_dim,
        )
        self.tactile_align = ProjectionHead(
            in_dim=dino_dim, hidden_dim=512, out_dim=aligned_dim,
        )

        if alignment_ckpt is not None:
            ckpt = torch.load(alignment_ckpt, map_location="cpu")
            self.vision_align.load_state_dict(ckpt["vision_proj"])
            self.tactile_align.load_state_dict(ckpt["tactile_proj"])
            aligned_dim = ckpt.get("shared_dim", aligned_dim)
            print(f"[TFS] Loaded alignment from {alignment_ckpt}")

        for p in self.vision_align.parameters():
            p.requires_grad = False
        for p in self.tactile_align.parameters():
            p.requires_grad = False
        self.vision_align.eval()
        self.tactile_align.eval()

        # Trainable observation encoder
        self.obs_encoder = ObservationEncoder(
            embed_dim=aligned_dim,
            hidden_dim=obs_hidden_dim,
            out_dim=score_dim,
            num_layers=obs_num_layers,
            nheads=obs_nheads,
            dropout=dropout,
        )

        # Trainable action encoder
        self.action_encoder = ActionEncoder(
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dim=action_hidden_dim,
            out_dim=score_dim,
        )

        # Learnable temperature
        if learnable_temperature:
            self.log_temperature = nn.Parameter(
                torch.tensor(np.log(1.0 / init_temperature))
            )
        else:
            self.register_buffer(
                "log_temperature",
                torch.tensor(np.log(1.0 / init_temperature)),
            )

        self.to(device)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    @torch.no_grad()
    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """DINOv2 → alignment → (B, aligned_dim)."""
        dino_feat = self.dino(images, already_normalized=True)
        return self.vision_align(dino_feat)

    @torch.no_grad()
    def encode_tactile(self, images: torch.Tensor) -> torch.Tensor:
        """DINOv2 → alignment → (B, aligned_dim)."""
        dino_feat = self.dino(images, already_normalized=True)
        return self.tactile_align(dino_feat)

    def score(
        self,
        vision_emb: torch.Tensor,
        tactile_emb: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute feasibility score for (obs, action) pairs.

        Args:
            vision_emb: (B, aligned_dim) aligned vision embedding
            tactile_emb: (B, aligned_dim) aligned tactile embedding (predicted or real)
            actions: (B, chunk_size, action_dim) action chunk

        Returns:
            (B,) scalar feasibility scores
        """
        o_emb = self.obs_encoder(vision_emb, tactile_emb)  # (B, score_dim)
        a_emb = self.action_encoder(actions)  # (B, score_dim)
        # Per-sample dot product
        return (o_emb * a_emb).sum(dim=-1)

    def compute_loss(
        self,
        vision_imgs: torch.Tensor,
        tac_future_imgs: torch.Tensor,
        actions: torch.Tensor,
        noise_levels: Optional[torch.Tensor] = None,
        action_noise: Optional[torch.Tensor] = None,
    ) -> dict:
        """InfoNCE contrastive loss on (observation, action) pairs.

        During training, we use real future tactile (not TFM predictions) and
        optionally corrupt actions with DDPM-schedule noise.

        Args:
            vision_imgs: (B, 3, H, W) current vision images
            tac_future_imgs: (B, 3, H, W) real future tactile images
            actions: (B, chunk_size, action_dim) ground-truth action chunks
            noise_levels: (B,) optional diffusion noise levels (unused by encoder,
                          but actions should be pre-corrupted by caller)
            action_noise: (B, chunk_size, action_dim) optional noise to add to actions

        Returns:
            dict with 'loss', 'accuracy', 'temperature', 'pos_score', 'neg_score'
        """
        B = vision_imgs.shape[0]

        # Encode observations (frozen DINOv2 + frozen alignment)
        v_emb = self.encode_vision(vision_imgs)       # (B, aligned_dim)
        t_emb = self.encode_tactile(tac_future_imgs)  # (B, aligned_dim)

        o_emb = self.obs_encoder(v_emb, t_emb)  # (B, score_dim)

        # Optionally add noise to actions (noisy action training)
        if action_noise is not None:
            actions = actions + action_noise

        a_emb = self.action_encoder(actions)  # (B, score_dim)

        # Similarity matrix
        temp = self.temperature
        logits = temp * o_emb @ a_emb.T  # (B, B)

        # Symmetric InfoNCE
        labels = torch.arange(B, device=logits.device)
        loss_o2a = F.cross_entropy(logits, labels)
        loss_a2o = F.cross_entropy(logits.T, labels)
        loss = (loss_o2a + loss_a2o) / 2.0

        # Metrics
        with torch.no_grad():
            pred_o = logits.argmax(dim=1)
            pred_a = logits.T.argmax(dim=1)
            correct = (pred_o == labels).sum() + (pred_a == labels).sum()
            accuracy = correct.float() / (2 * B)

            diag = torch.diag(logits)
            mask = ~torch.eye(B, dtype=torch.bool, device=logits.device)
            pos_score = diag.mean()
            neg_score = logits[mask].mean()

        return {
            "loss": loss,
            "accuracy": accuracy.item(),
            "temperature": temp.item(),
            "pos_score": pos_score.item(),
            "neg_score": neg_score.item(),
        }

    def trainable_parameters(self):
        """Return only trainable parameters (encoders + temperature)."""
        params = (
            list(self.obs_encoder.parameters())
            + list(self.action_encoder.parameters())
        )
        if self.log_temperature.requires_grad:
            params.append(self.log_temperature)
        return params

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters() if p.requires_grad)

    def save(self, path: str):
        """Save trainable components."""
        torch.save({
            "obs_encoder": self.obs_encoder.state_dict(),
            "action_encoder": self.action_encoder.state_dict(),
            "log_temperature": self.log_temperature.data,
            "aligned_dim": self.aligned_dim,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
        }, path)

    def load(self, path: str):
        """Load trainable components."""
        ckpt = torch.load(path, map_location="cpu")
        self.obs_encoder.load_state_dict(ckpt["obs_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.log_temperature.data = ckpt["log_temperature"]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building TFS on {device}...")

    tfs = TactileFeasibilityScore(
        alignment_ckpt=None, action_dim=7, chunk_size=20, device=device,
    )
    print(f"Trainable params: {tfs.num_trainable_params():,}")
    print(f"Temperature: {tfs.temperature.item():.4f}")

    B = 8
    vis = torch.randn(B, 3, 200, 266, device=device)
    tac = torch.randn(B, 3, 224, 224, device=device)
    actions = torch.randn(B, 20, 7, device=device)

    result = tfs.compute_loss(vis, tac, actions)
    print(f"Loss: {result['loss'].item():.4f}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Pos score: {result['pos_score']:.4f}")
    print(f"Neg score: {result['neg_score']:.4f}")

    # Test score function with pre-computed embeddings
    v_emb = tfs.encode_vision(vis)
    t_emb = tfs.encode_tactile(tac)
    scores = tfs.score(v_emb, t_emb, actions)
    print(f"Scores shape: {scores.shape}")  # (8,)
    print("TFS OK!")
