"""
V4 Unit Test: Shape verification for all modules.
Run: python TFAC_V4/test_shapes.py
"""
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_spatial_marker_encoder():
    print("=" * 60)
    print("Test 1: SpatialMarkerEncoder")
    from TFAC_V4.marker_encoder import SpatialMarkerEncoder
    enc = SpatialMarkerEncoder(hidden_dim=512).cuda()
    x = torch.randn(4, 9, 9, 2).cuda()

    tokens = enc(x)
    print(f"  Input:  (4, 9, 9, 2)")
    print(f"  Output: {tuple(tokens.shape)} — expected (9, 4, 512)")
    assert tokens.shape == (9, 4, 512), f"Shape mismatch: {tokens.shape}"

    global_feat = enc.forward_global(x)
    print(f"  Global: {tuple(global_feat.shape)} — expected (4, 512)")
    assert global_feat.shape == (4, 512)
    print("  PASSED")

def test_patch_level_decoder():
    print("=" * 60)
    print("Test 2: PatchLevelDecoder")
    from TFAC_V4.foresight_transformer import PatchLevelDecoder
    dec = PatchLevelDecoder(d_model=512).cuda()
    patch_tokens = torch.randn(4, 9, 512).cuda()

    out = dec(patch_tokens)
    print(f"  Input:  (4, 9, 512)")
    print(f"  Output: {tuple(out.shape)} — expected (4, 9, 9, 2)")
    assert out.shape == (4, 9, 9, 2)
    print("  PASSED")

def test_foresight_transformer_v4_single():
    print("=" * 60)
    print("Test 3: ForesightTransformerV4 (single frame, H=1)")
    from TFAC_V4.foresight_transformer import ForesightTransformerV4
    foresight = ForesightTransformerV4(
        d_model=512, action_dim=7, num_layers=2, nhead=4,
        predict_horizon=1, state_dim=7, n_tac_tokens=9
    ).cuda()

    B = 4
    N_v = 126  # 2 cameras * 63 tokens
    N_t = 9    # spatial encoder
    v_tokens = torch.randn(N_v, B, 512).cuda()
    t_tokens = torch.randn(N_t, B, 512).cuda()
    a1 = torch.randn(B, 10, 7).cuda()
    proprio = torch.randn(B, 7).cuda()

    t_hat_obs, v_hat, t_embed = foresight(v_tokens, t_tokens, a1, N_v, proprio=proprio)
    print(f"  t_hat_obs:    {tuple(t_hat_obs.shape)} — expected (4, 9, 9, 2)")
    print(f"  v_hat_future: {tuple(v_hat.shape)} — expected (4, 512)")
    print(f"  t_embed:      {tuple(t_embed.shape)} — expected (4, 9, 512)")
    assert t_hat_obs.shape == (4, 9, 9, 2)
    assert v_hat.shape == (4, 512)
    assert t_embed.shape == (4, 9, 512)
    print("  PASSED")

def test_foresight_transformer_v4_multi():
    print("=" * 60)
    print("Test 4: ForesightTransformerV4 (multi frame, H=10, k=3)")
    from TFAC_V4.foresight_transformer import ForesightTransformerV4
    H = 10
    foresight = ForesightTransformerV4(
        d_model=512, action_dim=7, num_layers=2, nhead=4,
        predict_horizon=H, state_dim=7, n_tac_tokens=9
    ).cuda()

    B = 4
    k = 3
    N_v = 126
    N_t = 9
    v_tokens = torch.randn(k, N_v, B, 512).cuda()
    t_tokens = torch.randn(k, N_t, B, 512).cuda()
    a1 = torch.randn(B, 10, 7).cuda()
    proprio = torch.randn(B, 7).cuda()

    t_hat_obs, v_hat, t_embed = foresight(v_tokens, t_tokens, a1, N_v, proprio=proprio)
    print(f"  t_hat_obs:    {tuple(t_hat_obs.shape)} — expected (4, 10, 9, 9, 2)")
    print(f"  v_hat_future: {tuple(v_hat.shape)} — expected (4, 512)")
    print(f"  t_embed:      {tuple(t_embed.shape)} — expected (4, 10, 9, 512)")
    assert t_hat_obs.shape == (4, H, 9, 9, 2)
    assert v_hat.shape == (4, 512)
    assert t_embed.shape == (4, H, 9, 512)
    print("  PASSED")

def test_dynamics_model():
    print("=" * 60)
    print("Test 5: TactileDynamicsModel")
    from TFAC_V4.tfac_model import TactileDynamicsModel
    model = TactileDynamicsModel(d_model=512, action_dim=7).cuda()
    z_tac = torch.randn(4, 512).cuda()
    action_summary = torch.randn(4, 7).cuda()
    z_next = model(z_tac, action_summary)
    print(f"  Input:  z_tac=(4, 512), action=(4, 7)")
    print(f"  Output: {tuple(z_next.shape)} — expected (4, 512)")
    assert z_next.shape == (4, 512)
    print("  PASSED")

def test_trajectory_encoder():
    print("=" * 60)
    print("Test 6: TactileTrajectoryEncoder")
    from TFAC_V4.tfac_model import TactileTrajectoryEncoder
    enc = TactileTrajectoryEncoder(d_model=512).cuda()
    traj = torch.randn(4, 10, 512).cuda()
    summary = enc(traj)
    print(f"  Input:  (4, 10, 512)")
    print(f"  Output: {tuple(summary.shape)} — expected (4, 512)")
    assert summary.shape == (4, 512)
    print("  PASSED")

def test_contact_aware_fusion():
    print("=" * 60)
    print("Test 7: ContactAwareFusion")
    from TFAC_V4.tfac_model import ContactAwareFusion
    fusion = ContactAwareFusion(d_model=512, state_dim=7).cuda()
    S, B, D = 140, 4, 512  # 2 + N_total
    memory = torch.randn(S, B, D).cuda()
    a1_feat = torch.randn(B, D).cuda()
    traj_summary = torch.randn(B, D).cuda()
    proprio = torch.randn(B, 7).cuda()

    fused = fusion(memory, a1_feat, traj_summary, proprio=proprio)
    print(f"  Input:  memory=(140, 4, 512), a1=(4, 512), traj=(4, 512)")
    print(f"  Output: {tuple(fused.shape)} — expected (140, 4, 512)")
    assert fused.shape == (S, B, D)
    print(f"  Gates: mem={fusion._last_gate_means[0]:.3f}, "
          f"a1={fusion._last_gate_means[1]:.3f}, "
          f"traj={fusion._last_gate_means[2]:.3f}")
    print("  PASSED")

def test_contrastive_v4():
    print("=" * 60)
    print("Test 8: ForesightContrastiveV4")
    from TFAC_V4.foresight_transformer import ForesightContrastiveV4
    cont = ForesightContrastiveV4(feat_dim=512, proj_dim=128).cuda()

    B = 8
    v_hat = torch.randn(B, 512).cuda()
    t_embed = torch.randn(B, 9, 512).cuda()
    t_traj = torch.randn(B, 10, 512).cuda()

    loss_spatial = cont.forward_spatial(v_hat, t_embed)
    loss_global = cont.forward_global(v_hat, torch.randn(B, 512).cuda())
    loss_temporal = cont.forward_temporal(t_traj)

    print(f"  Spatial loss:  {loss_spatial.item():.4f}")
    print(f"  Global loss:   {loss_global.item():.4f}")
    print(f"  Temporal loss: {loss_temporal.item():.4f}")
    assert loss_spatial.dim() == 0
    assert loss_global.dim() == 0
    assert loss_temporal.dim() == 0
    print("  PASSED")


if __name__ == '__main__':
    print("TFAC V4 Unit Tests — Shape Verification")
    print("=" * 60)

    test_spatial_marker_encoder()
    test_patch_level_decoder()
    test_foresight_transformer_v4_single()
    test_foresight_transformer_v4_multi()
    test_dynamics_model()
    test_trajectory_encoder()
    test_contact_aware_fusion()
    test_contrastive_v4()

    print("\n" + "=" * 60)
    print("ALL UNIT TESTS PASSED!")
    print("=" * 60)
