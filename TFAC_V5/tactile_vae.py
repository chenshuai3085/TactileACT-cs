"""
TactileVAE: Spatio-Temporal Variational Autoencoder for Marker Displacement.

Encodes a temporal sequence of marker_offset (B, T, 9, 9, 2) into a
low-dimensional spatial latent feature map (B, C, T', 3, 3), and decodes
via an Implicit Neural Representation (INR) decoder.

Architecture (following OmniVTA):
  Encoder: CausalConv3D + ST-ResBlocks + Spatial/Temporal downsampling
  Decoder: INR (bilinear interpolation from latent + Fourier positional encoding → MLP)

Key design choices:
  - Causal 3D convolutions: only look at current and past frames (no future leaking)
  - Spatial feature map: preserves spatial locality of tactile signals
  - INR decoder: models continuous deformation field (physically grounded)
  - Low-dimensional latent (C=8, total 72-dim per frame): easy for downstream prediction

Usage:
    model = TactileVAE(latent_dim=8, temporal_window=8)
    recon, mu, logvar = model(marker_seq)  # (B, T, 9, 9, 2) -> recon (B, T', 9, 9, 2)
    loss = model.loss(recon, gt_aligned, mu, logvar)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CausalConv3D: 因果 3D 卷积 — 时间轴只看过去,不看未来
# =============================================================================

class CausalConv3D(nn.Module):
    """
    Causal 3D Convolution.

    Time axis: left-padded by (kernel_t - 1) so output T equals input T
               when stride_t=1. No future frames are ever accessed.
    Spatial axes: symmetric padding to preserve H, W (when stride=1).

    PyTorch's nn.Conv3d only supports symmetric padding, so we manually
    pad the time axis with F.pad before convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, bias=True):
        super().__init__()
        # Normalize to tuples (T, H, W)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        self.kernel_size = kernel_size
        self.stride = stride

        # Spatial padding: symmetric, preserves H/W when stride=1
        self.pad_h = kernel_size[1] // 2
        self.pad_w = kernel_size[2] // 2

        # Temporal causal padding: only pad left side
        self.pad_t = kernel_size[0] - 1  # left-pad amount

        # Conv with zero padding (all padding handled by F.pad)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            out: (B, C_out, T', H', W')
        """
        # F.pad order: (W_left, W_right, H_left, H_right, T_left, T_right)
        x = F.pad(x, (self.pad_w, self.pad_w,
                       self.pad_h, self.pad_h,
                       self.pad_t, 0))
        return self.conv(x)


# =============================================================================
# ST-ResBlock: Spatio-Temporal Residual Block
# =============================================================================

class STResBlock(nn.Module):
    """
    Spatio-Temporal Residual Block.

    GroupNorm + SiLU + CausalConv3D → GroupNorm + SiLU + CausalConv3D + residual

    If in_channels != out_channels, a 1x1x1 conv is used for the skip connection.
    """

    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv1 = CausalConv3D(in_channels, out_channels,
                                  kernel_size=3, stride=1)

        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.act2 = nn.SiLU(inplace=True)
        self.conv2 = CausalConv3D(out_channels, out_channels,
                                  kernel_size=3, stride=1)

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels,
                                 kernel_size=1, stride=1, padding=0)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            out: (B, C_out, T, H, W)  — same T, H, W
        """
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        return h + self.skip(x)


# =============================================================================
# Fourier Positional Encoding for INR decoder
# =============================================================================

class FourierPositionalEncoding(nn.Module):
    """
    Fourier positional encoding for 2D spatial coordinates.

    Maps (x, y) in [-1, 1] to a higher-dimensional vector using
    sin/cos of multiple frequencies.

    Output dim = 2 * 2 * (num_freqs) = 4 * num_freqs
    (2 input coords × 2 sin/cos × num_freqs levels)
    """

    def __init__(self, num_freqs=4):
        super().__init__()
        self.num_freqs = num_freqs
        # Frequency bands: 2^0, 2^1, ..., 2^(L-1)
        freqs = torch.pow(2.0, torch.arange(num_freqs).float()) * math.pi
        self.register_buffer('freqs', freqs)  # (num_freqs,)

    @property
    def out_dim(self):
        return 2 * 2 * self.num_freqs  # 2 coords × (sin + cos) × num_freqs

    def forward(self, coords):
        """
        Args:
            coords: (..., 2)  spatial coordinates in [-1, 1]
        Returns:
            encoded: (..., out_dim)
        """
        # coords: (..., 2) -> (..., 2, 1) * (num_freqs,) -> (..., 2, num_freqs)
        x = coords.unsqueeze(-1) * self.freqs  # (..., 2, num_freqs)
        # sin and cos -> (..., 2, num_freqs, 2) -> (..., 4 * num_freqs)
        encoded = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # (..., 2, 2*num_freqs)
        encoded = encoded.flatten(-2)  # (..., 4 * num_freqs)
        return encoded


# =============================================================================
# INR Decoder: Implicit Neural Representation
# =============================================================================

class INRDecoder(nn.Module):
    """
    Implicit Neural Representation decoder for tactile deformation field.

    Given a latent spatial feature map z_t (B, C, 3, 3), reconstructs the
    deformation at any query coordinate via:
        d(x) = MLP(concat(γ(x), Φ(z_t, x)))
    where:
        γ(x)       = Fourier positional encoding of coordinate x
        Φ(z_t, x)  = bilinear interpolation of z_t at coordinate x

    This models the deformation as a continuous field, which is physically
    grounded since marker displacement arises from continuous elastomer deformation.
    """

    def __init__(self, latent_dim=8, num_freqs=4, hidden_dim=64, out_dim=2):
        super().__init__()
        self.pos_enc = FourierPositionalEncoding(num_freqs)
        in_dim = latent_dim + self.pos_enc.out_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

        # Pre-compute 9×9 query grid in [-1, 1] (matches marker grid)
        coords_h = torch.linspace(-1, 1, 9)
        coords_w = torch.linspace(-1, 1, 9)
        # grid_sample expects (x, y) = (w, h) ordering
        grid_y, grid_x = torch.meshgrid(coords_h, coords_w, indexing='ij')
        # query_grid: (9, 9, 2)  last dim is (x_w, y_h) for grid_sample
        query_grid = torch.stack([grid_x, grid_y], dim=-1)
        self.register_buffer('query_grid', query_grid)

    def forward(self, z_t):
        """
        Args:
            z_t: (B, C, 3, 3) latent spatial feature map for one time step
        Returns:
            recon: (B, 9, 9, 2) reconstructed marker displacement
        """
        B, C, H_z, W_z = z_t.shape

        # Step 1: Bilinear interpolation — sample local features from z_t
        # grid_sample expects grid: (B, H_out, W_out, 2) with coords in [-1, 1]
        grid = self.query_grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 9, 9, 2)
        local_feat = F.grid_sample(
            z_t, grid, mode='bilinear', padding_mode='border',
            align_corners=True
        )  # (B, C, 9, 9)
        local_feat = local_feat.permute(0, 2, 3, 1)  # (B, 9, 9, C)

        # Step 2: Fourier positional encoding of query coordinates
        # query_grid: (9, 9, 2) → pos_enc: (9, 9, pos_dim)
        pos_feat = self.pos_enc(self.query_grid)  # (9, 9, pos_dim)
        pos_feat = pos_feat.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 9, 9, pos_dim)

        # Step 3: Concat and MLP
        mlp_input = torch.cat([local_feat, pos_feat], dim=-1)  # (B, 9, 9, C + pos_dim)
        recon = self.mlp(mlp_input)  # (B, 9, 9, 2)

        return recon


# =============================================================================
# TactileVAE: Full model
# =============================================================================

class TactileVAE(nn.Module):
    """
    Spatio-Temporal Variational Autoencoder for marker displacement.

    Encoder: CausalConv3D-based spatio-temporal encoder
        (B, T, 9, 9, 2) → (B, C, T', 3, 3)
        where T' = T // 2  (temporal downsampling by 2)

    Decoder: INR-based implicit decoder
        (B, C, 3, 3) per frame → (B, 9, 9, 2)

    The latent z is a spatial feature map, preserving where contact occurs.
    """

    def __init__(self, latent_dim=8, temporal_window=8,
                 num_freqs=4, inr_hidden=64, kl_weight=1e-6,
                 direction_weight=0.2):
        """
        Args:
            latent_dim: number of channels in latent feature map (C)
            temporal_window: expected input sequence length T
            num_freqs: Fourier positional encoding levels for INR
            inr_hidden: hidden dim of INR MLP
            kl_weight: weight for KL divergence loss
            direction_weight: weight for cosine direction loss
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.temporal_window = temporal_window
        self.kl_weight = kl_weight
        self.direction_weight = direction_weight

        # === Encoder ===
        # Projection-in: (B, 2, T, 9, 9) → (B, 32, T, 9, 9)
        self.proj_in = CausalConv3D(2, 32, kernel_size=3, stride=1)
        self.norm_in = nn.GroupNorm(8, 32)
        self.act_in = nn.SiLU(inplace=True)

        # ST-ResBlock 1 + Spatial Downsample
        self.res_block1 = STResBlock(32, 32, num_groups=8)
        # Downsample 1: spatial only (9,9) → (5,5), T unchanged
        self.down1 = CausalConv3D(32, 64, kernel_size=3, stride=(1, 2, 2))

        # ST-ResBlock 2 + Spatio-temporal Downsample
        self.res_block2 = STResBlock(64, 64, num_groups=8)
        # Downsample 2: spatial (5,5)→(3,3) and temporal T→T/2
        self.down2 = CausalConv3D(64, 128, kernel_size=3, stride=(2, 2, 2))

        # Projection-out to mu and logvar
        self.proj_mu = nn.Conv3d(128, latent_dim, kernel_size=1)
        self.proj_logvar = nn.Conv3d(128, latent_dim, kernel_size=1)

        # === Decoder (INR) ===
        self.decoder = INRDecoder(
            latent_dim=latent_dim,
            num_freqs=num_freqs,
            hidden_dim=inr_hidden,
            out_dim=2,  # 2D marker displacement (dx, dy)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Logvar projection initialized to small values for stable KL
        nn.init.zeros_(self.proj_logvar.weight)
        nn.init.zeros_(self.proj_logvar.bias)

    def encode(self, x):
        """
        Encode a temporal sequence of marker displacements.

        Args:
            x: (B, T, 9, 9, 2) marker displacement sequence
        Returns:
            mu:     (B, C, T', 3, 3)
            logvar: (B, C, T', 3, 3)
            where T' = T // 2
        """
        # (B, T, 9, 9, 2) → (B, 2, T, 9, 9)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        # Projection-in
        h = self.proj_in(x)       # (B, 32, T, 9, 9)
        h = self.norm_in(h)
        h = self.act_in(h)

        # Block 1 + Downsample (spatial only)
        h = self.res_block1(h)    # (B, 32, T, 9, 9)
        h = self.down1(h)         # (B, 64, T, 5, 5)

        # Block 2 + Downsample (spatio-temporal)
        h = self.res_block2(h)    # (B, 64, T, 5, 5)
        h = self.down2(h)         # (B, 128, T/2, 3, 3)

        # Project to mu, logvar
        mu = self.proj_mu(h)      # (B, C, T/2, 3, 3)
        logvar = self.proj_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        VAE reparameterization trick.
        At eval time, returns mu directly (no sampling).
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu

    def decode(self, z):
        """
        Decode latent feature maps to marker displacements using INR.

        Args:
            z: (B, C, T', 3, 3) latent spatial feature maps
        Returns:
            recon: (B, T', 9, 9, 2) reconstructed marker displacements
        """
        B, C, T_prime, H_z, W_z = z.shape
        recons = []

        for t in range(T_prime):
            z_t = z[:, :, t, :, :]    # (B, C, 3, 3)
            recon_t = self.decoder(z_t)  # (B, 9, 9, 2)
            recons.append(recon_t)

        # Stack along time: (B, T', 9, 9, 2)
        recon = torch.stack(recons, dim=1)
        return recon

    def forward(self, x):
        """
        Full forward pass: encode → reparameterize → decode.

        Args:
            x: (B, T, 9, 9, 2) marker displacement sequence
        Returns:
            recon:  (B, T', 9, 9, 2) reconstructed displacements
            mu:     (B, C, T', 3, 3)
            logvar: (B, C, T', 3, 3)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def encode_single_frame(self, x_seq):
        """
        Encode a sequence and return the LAST frame's latent feature map.
        Useful at inference: feed recent history, get current latent.

        Args:
            x_seq: (B, T, 9, 9, 2) marker displacement sequence
        Returns:
            z_t: (B, C, 3, 3) latent feature map for the last time step
            mu:  (B, C, 3, 3)
        """
        mu, logvar = self.encode(x_seq)
        z = self.reparameterize(mu, logvar)
        # Last frame in latent sequence
        z_last = z[:, :, -1, :, :]    # (B, C, 3, 3)
        mu_last = mu[:, :, -1, :, :]
        return z_last, mu_last

    @staticmethod
    def compute_kl_loss(mu, logvar):
        """
        KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I).
        Computed per spatial-temporal element, then mean-reduced.
        """
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.mean()

    @staticmethod
    def compute_direction_loss(recon, gt):
        """
        Cosine direction loss: 惩罚重建向量和 GT 方向不一致。

        只在有明显位移的区域计算 (静止区域方向无意义)。
        L_dir = 1 - cosine_similarity(recon, gt)，值域 [0, 2]。

        Args:
            recon: (B, T', 9, 9, 2)
            gt:    (B, T', 9, 9, 2)
        Returns:
            direction_loss: scalar
        """
        # 展平空间维度: (B*T'*9*9, 2)
        recon_flat = recon.reshape(-1, 2)
        gt_flat = gt.reshape(-1, 2)

        # GT 幅值，用于过滤静止区域
        gt_mag = gt_flat.norm(dim=1)  # (N,)
        # 动态阈值: 5% of batch max，避免在接近零的区域算方向
        threshold = 0.05 * gt_mag.max().clamp(min=1e-6)
        active_mask = gt_mag > threshold

        if active_mask.sum() < 1:
            return torch.tensor(0.0, device=recon.device)

        # cosine similarity on active points only
        cos_sim = F.cosine_similarity(
            recon_flat[active_mask], gt_flat[active_mask], dim=1
        )  # (N_active,)

        # 1 - cos_sim: 0=完美对齐, 2=完全反向
        direction_loss = (1.0 - cos_sim).mean()
        return direction_loss

    def loss(self, recon, gt, mu, logvar):
        """
        Total TactileVAE loss = MSE + direction + KL.

        MSE 管幅值准确，direction 管方向准确，KL 管 latent 正则化。

        Args:
            recon:  (B, T', 9, 9, 2) reconstructed displacements
            gt:     (B, T', 9, 9, 2) ground-truth (already time-aligned!)
            mu:     (B, C, T', 3, 3)
            logvar: (B, C, T', 3, 3)
        Returns:
            total_loss, recon_loss, kl_loss, direction_loss
        """
        recon_loss = F.mse_loss(recon, gt)
        kl_loss = self.compute_kl_loss(mu, logvar)
        direction_loss = self.compute_direction_loss(recon, gt)
        total = (recon_loss
                 + self.direction_weight * direction_loss
                 + self.kl_weight * kl_loss)
        return total, recon_loss, kl_loss, direction_loss

    @staticmethod
    def align_gt(gt_full, temporal_stride=2):
        """
        Align ground-truth to match temporally downsampled latent frames.

        Encoder downsamples T→T/2 with stride=2. Each latent frame
        corresponds to input frames at indices 1, 3, 5, 7, ... (0-indexed).

        Args:
            gt_full: (B, T, 9, 9, 2) full input sequence
            temporal_stride: temporal downsampling factor (default 2)
        Returns:
            gt_aligned: (B, T', 9, 9, 2) aligned ground-truth
        """
        return gt_full[:, (temporal_stride - 1)::temporal_stride]


# =============================================================================
# Convenience: build from config
# =============================================================================

def build_tactile_vae(config=None, **kwargs):
    """
    Build TactileVAE from config dict or keyword arguments.

    Default config:
        latent_dim=8, temporal_window=8, num_freqs=4,
        inr_hidden=64, kl_weight=1e-6
    """
    defaults = dict(
        latent_dim=8,
        temporal_window=8,
        num_freqs=4,
        inr_hidden=64,
        kl_weight=1e-6,
        direction_weight=0.2,
    )
    if config is not None:
        defaults.update(config)
    defaults.update(kwargs)
    return TactileVAE(**defaults)


# =============================================================================
# Quick test
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("TactileVAE Smoke Test")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TactileVAE(latent_dim=8, temporal_window=8).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for n, p in model.named_parameters() if 'decoder' not in n)
    n_dec = sum(p.numel() for n, p in model.named_parameters() if 'decoder' in n)
    print(f"Total params:   {n_params:,}")
    print(f"Encoder params: {n_enc:,}")
    print(f"Decoder params: {n_dec:,}")

    # Test forward pass
    B, T = 4, 8
    x = torch.randn(B, T, 9, 9, 2).to(device)

    model.train()
    recon, mu, logvar = model(x)

    print(f"\nInput:   {x.shape}")
    print(f"Recon:   {recon.shape}")
    print(f"Mu:      {mu.shape}")
    print(f"Logvar:  {logvar.shape}")

    # Test loss
    gt_aligned = TactileVAE.align_gt(x, temporal_stride=2)
    print(f"GT aligned: {gt_aligned.shape}")

    total, recon_l, kl_l, dir_l = model.loss(recon, gt_aligned, mu, logvar)
    print(f"\nLoss:  total={total.item():.4f}  recon={recon_l.item():.4f}  kl={kl_l.item():.4f}  dir={dir_l.item():.4f}")

    # Test encode_single_frame
    z_last, mu_last = model.encode_single_frame(x)
    print(f"\nSingle-frame latent: {z_last.shape}")  # (B, 8, 3, 3)
    print(f"Total latent dims per frame: {z_last.shape[1] * z_last.shape[2] * z_last.shape[3]}")

    # Test backward
    total.backward()
    print("\nBackward pass OK!")

    # Test eval mode (no sampling)
    model.eval()
    with torch.no_grad():
        recon_eval, mu_eval, _ = model(x)
        print(f"\nEval recon: {recon_eval.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
