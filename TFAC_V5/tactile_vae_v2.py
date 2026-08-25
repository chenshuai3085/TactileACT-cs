"""
TactileVAE v2: Contact-Aware Spatio-Temporal VAE for Marker Displacement.

Key architectural differences from v1 (OmniVTA-style):
  1. Cross-Attention Decoder (replaces INR): 9 latent tokens → 81 query positions
     via multi-head cross-attention. Models non-local force propagation.
  2. Temporal Attention Pooling (replaces take-last-frame): learnable query attends
     to all T' temporal frames, auto-focusing on contact-critical moments.
  3. Intensity-Pattern Disentangled Latent: z_intensity (scalar, directly supervised
     with ||m||_2) + z_pattern (15×3×3 spatial map). Enables latent-space scoring
     without decoding.

Input:  (B, T, 9, 9, 2) marker displacement sequence, T=8
Output: (B, 16, 3, 3) latent = [z_intensity(1,3,3) ; z_pattern(15,3,3)]

Formal scoring consumes the complete 16×3×3 latent. The intensity channel is an
auxiliary supervised component, not a standalone scorer input.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Reuse encoder building blocks from v1
# =============================================================================

class CausalConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_h = kernel_size[1] // 2
        self.pad_w = kernel_size[2] // 2
        self.pad_t = kernel_size[0] - 1
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        x = F.pad(x, (self.pad_w, self.pad_w, self.pad_h, self.pad_h, self.pad_t, 0))
        return self.conv(x)


class STResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv1 = CausalConv3D(in_channels, out_channels, kernel_size=3, stride=1)
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.act2 = nn.SiLU(inplace=True)
        self.conv2 = CausalConv3D(out_channels, out_channels, kernel_size=3, stride=1)
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        h = self.act1(self.norm1(x))
        h = self.conv1(h)
        h = self.act2(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


# =============================================================================
# NEW: Temporal Attention Pooling
# =============================================================================

class TemporalAttentionPooling(nn.Module):
    """
    Learnable temporal aggregation via cross-attention.

    Instead of naively taking the last frame (which discards temporal dynamics),
    a learnable query attends to all T' encoded frames and produces a single
    aggregated representation. The attention weights reveal which temporal
    moments are most informative for contact understanding.

    For downstream scoring, contact-critical moments (onset, peak) naturally
    receive higher attention weights.
    """

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: (B, T', D) temporal sequence of flattened spatial features
        Returns:
            out: (B, D) aggregated representation
            attn_weights: (B, 1, T') attention over temporal frames
        """
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)  # (B, 1, D)
        out, attn_weights = self.attn(q, x, x)  # (B, 1, D), (B, 1, T')
        out = self.norm(out.squeeze(1))  # (B, D)
        return out, attn_weights


# =============================================================================
# NEW: Cross-Attention Decoder
# =============================================================================

class CrossAttentionDecoder(nn.Module):
    """
    Decodes latent spatial tokens into marker displacement via cross-attention.

    Unlike INR (which uses local bilinear interpolation), cross-attention allows
    each output position to attend to ALL latent tokens — modeling non-local
    force propagation where pressing one region affects distant markers.

    Architecture:
        - 9 latent tokens (from 3×3 spatial map) as Keys/Values
        - 81 learnable query embeddings (9×9 grid) with 2D sinusoidal positional encoding
        - Multi-head cross-attention → FFN → output displacement (dx, dy)
    """

    def __init__(self, latent_dim=16, hidden_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = 81  # 9×9 output grid

        # Latent token projection (from latent_dim to hidden_dim)
        self.kv_proj = nn.Linear(latent_dim, hidden_dim)

        # Learnable query embeddings + 2D positional encoding
        self.query_embed = nn.Parameter(torch.randn(81, hidden_dim) * 0.02)
        self.register_buffer('pos_encoding', self._build_2d_sinusoidal_pos(9, 9, hidden_dim))

        # KV normalization (shared across layers)
        self.kv_norm = nn.LayerNorm(hidden_dim)

        # Cross-attention layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True),
                'norm1': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                ),
                'norm2': nn.LayerNorm(hidden_dim),
            }))

        # Output head: hidden_dim → 2 (dx, dy)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def _build_2d_sinusoidal_pos(self, H, W, dim):
        """2D sinusoidal positional encoding for the 9×9 output grid."""
        assert dim % 4 == 0
        pe = torch.zeros(H * W, dim)
        pos_h = torch.arange(H).float().unsqueeze(1)  # (H, 1)
        pos_w = torch.arange(W).float().unsqueeze(1)  # (W, 1)
        div_term = torch.exp(torch.arange(0, dim // 4).float() * -(math.log(10000.0) / (dim // 4)))

        # H dimension: sin/cos
        pe_h = torch.zeros(H, dim // 2)
        pe_h[:, 0::2] = torch.sin(pos_h * div_term)
        pe_h[:, 1::2] = torch.cos(pos_h * div_term)

        # W dimension: sin/cos
        pe_w = torch.zeros(W, dim // 2)
        pe_w[:, 0::2] = torch.sin(pos_w * div_term)
        pe_w[:, 1::2] = torch.cos(pos_w * div_term)

        # Combine: (H*W, dim)
        for i in range(H):
            for j in range(W):
                pe[i * W + j] = torch.cat([pe_h[i], pe_w[j]])

        return pe.unsqueeze(0)  # (1, 81, dim)

    def forward(self, z_spatial):
        """
        Args:
            z_spatial: (B, C, 3, 3) latent spatial feature map
        Returns:
            recon: (B, 9, 9, 2) reconstructed marker displacement
        """
        B = z_spatial.shape[0]

        # Flatten latent to tokens: (B, 9, C)
        kv = z_spatial.flatten(2).permute(0, 2, 1)  # (B, 9, C)
        kv = self.kv_proj(kv)  # (B, 9, hidden_dim)

        # Query: learnable + positional
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # (B, 81, hidden_dim)
        queries = queries + self.pos_encoding.expand(B, -1, -1)

        # Cross-attention layers
        kv = self.kv_norm(kv)
        x = queries
        for layer in self.layers:
            # Cross-attention: queries attend to latent tokens
            residual = x
            x = layer['norm1'](x)
            x, _ = layer['cross_attn'](x, kv, kv)
            x = x + residual
            # FFN
            residual = x
            x = layer['norm2'](x)
            x = layer['ffn'](x)
            x = x + residual

        # Output: (B, 81, 2) → (B, 9, 9, 2)
        out = self.output_head(x)  # (B, 81, 2)
        recon = out.view(B, 9, 9, 2)
        return recon


# =============================================================================
# NEW: Intensity-Pattern Disentangled Projection
# =============================================================================

class IntensityPatternProjection(nn.Module):
    """
    Projects encoder features into disentangled intensity + pattern latent space.

    z_intensity (1, 3, 3): scalar field supervised with ||m||_2 that
        represents contact magnitude at each spatial location.

    z_pattern (15, 3, 3): spatial pattern encoding force direction and
        distribution characteristics.

    Total latent: (16, 3, 3) = concat([z_intensity, z_pattern])
    """

    def __init__(self, enc_channels=128, latent_dim=16):
        super().__init__()
        self.pattern_dim = latent_dim - 1  # 15

        # Intensity branch: GAP over spatial → scalar per spatial location
        self.intensity_mu = nn.Sequential(
            nn.Conv3d(enc_channels, 32, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=1),
        )
        self.intensity_logvar = nn.Sequential(
            nn.Conv3d(enc_channels, 32, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=1),
        )

        # Pattern branch: richer spatial structure
        self.pattern_mu = nn.Sequential(
            nn.Conv3d(enc_channels, 64, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(64, self.pattern_dim, kernel_size=1),
        )
        self.pattern_logvar = nn.Sequential(
            nn.Conv3d(enc_channels, 64, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(64, self.pattern_dim, kernel_size=1),
        )

        self._init_logvar()

    def _init_logvar(self):
        for module in [self.intensity_logvar, self.pattern_logvar]:
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)

    def forward(self, h):
        """
        Args:
            h: (B, 128, T', 3, 3) encoder output
        Returns:
            mu: (B, 16, T', 3, 3)
            logvar: (B, 16, T', 3, 3)
        """
        int_mu = self.intensity_mu(h)       # (B, 1, T', 3, 3)
        int_logvar = self.intensity_logvar(h)
        pat_mu = self.pattern_mu(h)         # (B, 15, T', 3, 3)
        pat_logvar = self.pattern_logvar(h)

        mu = torch.cat([int_mu, pat_mu], dim=1)           # (B, 16, T', 3, 3)
        logvar = torch.cat([int_logvar, pat_logvar], dim=1)
        return mu, logvar


# =============================================================================
# TactileVAE v2: Full Model
# =============================================================================

class TactileVAEv2(nn.Module):
    """
    Contact-Aware Spatio-Temporal VAE for marker displacement.

    Architecture:
        Encoder: CausalConv3D + STResBlocks (same as v1, proven effective)
        Temporal Pooling: Attention-based (new, replaces take-last-frame)
        Latent: Intensity-Pattern disentangled (new)
        Decoder: Cross-Attention (new, replaces INR)

    Key advantages for downstream scoring:
        1. z_intensity provides explicit contact-magnitude supervision
        2. Temporal attention auto-focuses on contact-critical moments
        3. Cross-attention decoder models non-local force propagation
    """

    def __init__(self, latent_dim=16, temporal_window=8,
                 decoder_hidden=128, decoder_heads=4, decoder_layers=2,
                 kl_weight=1e-6, direction_weight=0.2,
                 intensity_weight=0.1, rank_weight=0.05):
        super().__init__()
        self.latent_dim = latent_dim
        self.temporal_window = temporal_window
        self.kl_weight = kl_weight
        self.direction_weight = direction_weight
        self.intensity_weight = intensity_weight
        self.rank_weight = rank_weight

        # === Encoder (same proven structure as v1) ===
        self.proj_in = CausalConv3D(2, 32, kernel_size=3, stride=1)
        self.norm_in = nn.GroupNorm(8, 32)
        self.act_in = nn.SiLU(inplace=True)

        self.res_block1 = STResBlock(32, 32, num_groups=8)
        self.down1 = CausalConv3D(32, 64, kernel_size=3, stride=(1, 2, 2))

        self.res_block2 = STResBlock(64, 64, num_groups=8)
        self.down2 = CausalConv3D(64, 128, kernel_size=3, stride=(2, 2, 2))

        # === Temporal Attention Pooling (new) ===
        # Operates on flattened (latent_dim * 3 * 3)-dim per temporal frame
        self.temporal_flatten_dim = latent_dim * 3 * 3  # 16*9=144
        self.temporal_proj = nn.Linear(self.temporal_flatten_dim, 256)
        self.temporal_pool = TemporalAttentionPooling(dim=256, num_heads=4)
        self.temporal_unproj = nn.Linear(256, self.temporal_flatten_dim)

        # === Intensity-Pattern Projection (new) ===
        self.ip_proj = IntensityPatternProjection(enc_channels=128, latent_dim=latent_dim)

        # === Cross-Attention Decoder (new) ===
        self.decoder = CrossAttentionDecoder(
            latent_dim=latent_dim,
            hidden_dim=decoder_hidden,
            num_heads=decoder_heads,
            num_layers=decoder_layers,
        )

        self._init_weights()
        # Re-apply logvar zero init after global init overwrites it
        self.ip_proj._init_logvar()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d,)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x):
        """
        Encode temporal marker sequence to spatio-temporal feature volume.

        Args:
            x: (B, T, 9, 9, 2) marker displacement sequence
        Returns:
            mu: (B, 16, T', 3, 3) where T' = T // 2
            logvar: (B, 16, T', 3, 3)
        """
        # (B, T, 9, 9, 2) → (B, 2, T, 9, 9)
        h = x.permute(0, 4, 1, 2, 3).contiguous()

        h = self.act_in(self.norm_in(self.proj_in(h)))  # (B, 32, T, 9, 9)
        h = self.res_block1(h)                           # (B, 32, T, 9, 9)
        h = self.down1(h)                                # (B, 64, T, 5, 5)
        h = self.res_block2(h)                           # (B, 64, T, 5, 5)
        h = self.down2(h)                                # (B, 128, T/2, 3, 3)

        mu, logvar = self.ip_proj(h)  # (B, 16, T/2, 3, 3)
        return mu, logvar

    def temporal_aggregate(self, z):
        """
        Aggregate temporal dimension using attention pooling.

        Args:
            z: (B, C, T', 3, 3) latent volume
        Returns:
            z_agg: (B, C, 3, 3) single-frame aggregated latent
            attn_weights: (B, 1, T') temporal attention weights
        """
        B, C, T_prime, H, W = z.shape
        # Reshape to (B, T', C*H*W)
        z_flat = z.permute(0, 2, 1, 3, 4).reshape(B, T_prime, -1)  # (B, T', C*3*3)
        z_proj = self.temporal_proj(z_flat)  # (B, T', 256)

        # Attention pooling
        z_agg, attn_weights = self.temporal_pool(z_proj)  # (B, 256), (B, 1, T')

        # Unproject back to spatial shape
        z_agg = self.temporal_unproj(z_agg)  # (B, 1152)
        z_agg = z_agg.view(B, C, H, W)  # (B, 16, 3, 3)
        return z_agg, attn_weights

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu

    def decode(self, z_spatial):
        """
        Decode a single-frame latent to marker displacement.

        Args:
            z_spatial: (B, 16, 3, 3) latent feature map
        Returns:
            recon: (B, 9, 9, 2) reconstructed displacement
        """
        return self.decoder(z_spatial)

    def forward(self, x):
        """
        Full forward: encode → reparameterize → temporal pool → decode.

        Args:
            x: (B, T, 9, 9, 2)
        Returns:
            recon: (B, 9, 9, 2) single-frame reconstruction (of temporally-attended frame)
            mu: (B, 16, T', 3, 3)
            logvar: (B, 16, T', 3, 3)
            z_agg: (B, 16, 3, 3) aggregated latent
            attn_weights: (B, 1, T') temporal attention
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        # Temporal attention pooling
        z_agg, attn_weights = self.temporal_aggregate(z)  # (B, 16, 3, 3)

        # Decode aggregated frame
        recon = self.decode(z_agg)  # (B, 9, 9, 2)

        return recon, mu, logvar, z_agg, attn_weights

    def encode_single_frame(self, x_seq):
        """
        Encode sequence and return aggregated latent (for downstream use).

        Args:
            x_seq: (B, T, 9, 9, 2)
        Returns:
            z_agg: (B, 16, 3, 3) aggregated latent
            z_intensity: (B, 1, 3, 3) intensity component
        """
        mu, logvar = self.encode(x_seq)
        z = self.reparameterize(mu, logvar)
        z_agg, _ = self.temporal_aggregate(z)
        z_intensity = z_agg[:, :1, :, :]  # first channel is intensity
        return z_agg, z_intensity

    def get_intensity(self, z):
        """Extract the auxiliary intensity component for diagnostics or losses."""
        return z[:, :1, :, :]  # (B, 1, 3, 3)

    def get_pattern(self, z):
        """Extract pattern component from latent."""
        return z[:, 1:, :, :]  # (B, 15, 3, 3)

    # =========================================================================
    # Loss functions
    # =========================================================================

    @staticmethod
    def compute_kl_loss(mu, logvar):
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.mean()

    @staticmethod
    def compute_direction_loss(recon, gt):
        recon_flat = recon.reshape(-1, 2)
        gt_flat = gt.reshape(-1, 2)
        gt_mag = gt_flat.norm(dim=1)
        threshold = 0.05 * gt_mag.max().clamp(min=1e-6)
        active_mask = gt_mag > threshold
        if active_mask.sum() < 1:
            return torch.tensor(0.0, device=recon.device)
        cos_sim = F.cosine_similarity(recon_flat[active_mask], gt_flat[active_mask], dim=1)
        return (1.0 - cos_sim).mean()

    @staticmethod
    def compute_intensity_loss(z_intensity, gt_marker):
        """
        Supervise z_intensity with actual contact magnitude.

        Args:
            z_intensity: (B, 1, 3, 3) predicted intensity latent
            gt_marker: (B, 9, 9, 2) ground truth marker displacement
        Returns:
            loss: scalar
        """
        # Compute GT intensity: ||m||_2 at each spatial position, then pool to 3×3
        gt_magnitude = gt_marker.norm(dim=-1)  # (B, 9, 9)
        # Average pool 9×9 → 3×3
        gt_intensity = F.adaptive_avg_pool2d(gt_magnitude.unsqueeze(1), (3, 3))  # (B, 1, 3, 3)
        return F.smooth_l1_loss(z_intensity, gt_intensity)

    @staticmethod
    def compute_rank_loss(z_intensity, gt_marker, margin=0.1):
        """
        Pairwise ranking loss: if sample i has higher contact than j,
        z_intensity_i should be higher than z_intensity_j.

        Ensures latent intensity preserves ordinal relationships for scoring.

        Args:
            z_intensity: (B, 1, 3, 3) predicted
            gt_marker: (B, 9, 9, 2) ground truth
            margin: ranking margin
        Returns:
            loss: scalar
        """
        B = z_intensity.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=z_intensity.device)

        # Global intensity per sample: mean of spatial map
        pred_score = z_intensity.mean(dim=[1, 2, 3])  # (B,)
        gt_magnitude = gt_marker.norm(dim=-1).mean(dim=[1, 2])  # (B,)

        # Sample pairs within batch
        n_pairs = min(B * (B - 1) // 2, 32)  # cap pairs for efficiency
        idx_i = torch.randint(0, B, (n_pairs,), device=z_intensity.device)
        idx_j = torch.randint(0, B, (n_pairs,), device=z_intensity.device)

        # Ensure i != j
        same = idx_i == idx_j
        idx_j[same] = (idx_j[same] + 1) % B

        gt_diff = gt_magnitude[idx_i] - gt_magnitude[idx_j]  # positive means i > j
        pred_diff = pred_score[idx_i] - pred_score[idx_j]

        # Filter out pairs with negligible GT difference
        valid_mask = gt_diff.abs() > 1e-3
        if valid_mask.sum() < 1:
            return torch.tensor(0.0, device=z_intensity.device)

        gt_diff = gt_diff[valid_mask]
        pred_diff = pred_diff[valid_mask]

        # sign: +1 if i should be higher, -1 if j should be higher
        sign = torch.sign(gt_diff)
        # Margin ranking: loss when pred_diff * sign < margin
        loss = F.relu(margin - pred_diff * sign).mean()
        return loss

    def loss(self, recon, gt, mu, logvar, z_agg):
        """
        Total loss = MSE + direction + KL + intensity_supervision + rank.

        Args:
            recon: (B, 9, 9, 2) reconstruction
            gt: (B, 9, 9, 2) ground truth (last frame or attention-weighted)
            mu: (B, 16, T', 3, 3)
            logvar: (B, 16, T', 3, 3)
            z_agg: (B, 16, 3, 3) aggregated latent
        Returns:
            total_loss, dict of component losses
        """
        recon_loss = F.mse_loss(recon, gt)
        kl_loss = self.compute_kl_loss(mu, logvar)
        dir_loss = self.compute_direction_loss(recon, gt)

        z_intensity = self.get_intensity(z_agg)  # (B, 1, 3, 3)
        intensity_loss = self.compute_intensity_loss(z_intensity, gt)
        rank_loss = self.compute_rank_loss(z_intensity, gt)

        total = (recon_loss
                 + self.direction_weight * dir_loss
                 + self.kl_weight * kl_loss
                 + self.intensity_weight * intensity_loss
                 + self.rank_weight * rank_loss)

        losses = {
            'total': total,
            'recon': recon_loss,
            'kl': kl_loss,
            'direction': dir_loss,
            'intensity': intensity_loss,
            'rank': rank_loss,
        }
        return total, losses

    @staticmethod
    def get_last_frame_gt(gt_full, temporal_stride=2):
        """
        Get the last temporally-aligned GT frame for reconstruction target.

        For temporal attention pooling, we use the last frame as primary target
        (attention should learn to focus on most recent / most relevant frame).

        Args:
            gt_full: (B, T, 9, 9, 2)
        Returns:
            gt_last: (B, 9, 9, 2)
        """
        return gt_full[:, -1]


# =============================================================================
# Convenience builder
# =============================================================================

def build_tactile_vae_v2(config=None, **kwargs):
    defaults = dict(
        latent_dim=16,
        temporal_window=8,
        decoder_hidden=128,
        decoder_heads=4,
        decoder_layers=2,
        kl_weight=1e-6,
        direction_weight=0.2,
        intensity_weight=0.1,
        rank_weight=0.05,
    )
    if config is not None:
        defaults.update(config)
    defaults.update(kwargs)
    return TactileVAEv2(**defaults)


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("TactileVAE v2 Smoke Test")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TactileVAEv2(latent_dim=16, temporal_window=8).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for n, p in model.named_parameters()
                if 'decoder' not in n and 'temporal' not in n and 'ip_proj' not in n)
    n_temporal = sum(p.numel() for n, p in model.named_parameters() if 'temporal' in n)
    n_proj = sum(p.numel() for n, p in model.named_parameters() if 'ip_proj' in n)
    n_dec = sum(p.numel() for n, p in model.named_parameters() if 'decoder' in n)

    print(f"Total params:     {n_params:,}")
    print(f"Encoder params:   {n_enc:,}")
    print(f"Temporal pool:    {n_temporal:,}")
    print(f"IP projection:    {n_proj:,}")
    print(f"Decoder params:   {n_dec:,}")

    # Forward pass
    B, T = 4, 8
    x = torch.randn(B, T, 9, 9, 2).to(device)

    model.train()
    recon, mu, logvar, z_agg, attn_w = model(x)

    print(f"\nInput:       {x.shape}")
    print(f"Recon:       {recon.shape}")
    print(f"Mu:          {mu.shape}")
    print(f"Logvar:      {logvar.shape}")
    print(f"z_agg:       {z_agg.shape}")
    print(f"Attn weights:{attn_w.shape}")
    print(f"z_intensity: {model.get_intensity(z_agg).shape}")
    print(f"z_pattern:   {model.get_pattern(z_agg).shape}")

    # Loss
    gt = model.get_last_frame_gt(x)
    total, losses = model.loss(recon, gt, mu, logvar, z_agg)
    print(f"\nLosses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.6f}")

    # Backward
    total.backward()
    print("\nBackward pass OK!")

    # encode_single_frame
    model.eval()
    with torch.no_grad():
        z_out, z_int = model.encode_single_frame(x)
        print(f"\nencode_single_frame: z={z_out.shape}, intensity={z_int.shape}")

    # Latent dims
    print(f"\nTotal latent dims: {z_out.shape[1] * z_out.shape[2] * z_out.shape[3]} (16×3×3=144)")
    print(f"Intensity dims:    {z_int.numel() // B} (1×3×3=9, directly scorable)")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
