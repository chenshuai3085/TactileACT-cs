"""
V4 Integration Test: Full forward pass through TFACPolicyV4.
Tests training mode (loss computation) and inference mode.
"""
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_full_forward():
    print("=" * 60)
    print("Integration Test: TFACPolicyV4 full forward pass")
    print("=" * 60)

    from TFAC_V4.tfac_policy import TFACPolicyV4

    B = 4
    chunk_size = 10
    state_dim = 7
    H = 10  # predict_horizon
    camera_names = ['global', 'wrist', 'gelsight']
    hidden_dim = 512

    print("Building policy...")
    policy = TFACPolicyV4(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        position_embedding_type='sine',
        lr_backbone=1e-5,
        masks=False,
        backbone_type='resnet18',
        dilation=False,
        dropout=0.1,
        nheads=8,
        dim_feedforward=2048,
        num_enc_layers=4,
        num_dec_layers=7,
        pre_norm=False,
        num_queries=chunk_size,
        camera_names=camera_names,
        z_dimension=32,
        lr=4e-5,
        weight_decay=1e-4,
        kl_weight=1.0,
        foresight_layers=2,  # reduced for test speed
        foresight_nheads=4,
        foresight_dim_feedforward=1024,
        proj_dim=128,
        contrastive_temperature=0.07,
        curriculum_ratio=0.75,
        lambda_draft=0.3,
        lambda_latent=0.3,
        lambda_obs=0.2,
        lambda_consistency=0.3,
        lambda_contrastive_spatial=0.2,
        lambda_contrastive_temporal=0.1,
        lambda_contrastive_gt=0.1,
        lambda_dynamics=0.1,
        lambda_sampling=0.5,
        marker_encoder_type='spatial',
        fusion_mode='contact_gate',
        a2_init='residual',
        predict_horizon=H,
        sampling_steps=2,  # reduced for test speed
        n_tac_tokens=9,
    )
    policy.cuda()
    policy.train()

    # Create dummy inputs
    print("Creating dummy inputs...")
    qpos = torch.randn(B, state_dim).cuda()
    images = [
        torch.randn(B, 3, 200, 266).cuda(),  # global
        torch.randn(B, 3, 200, 266).cuda(),  # wrist
        torch.randn(B, 9, 9, 2).cuda(),       # gelsight (marker)
    ]
    actions = torch.randn(B, chunk_size, state_dim).cuda()
    is_pad = torch.zeros(B, chunk_size, dtype=torch.bool).cuda()

    # Future images: tactile H frames, vision 2 frames (contrastive indices)
    future_images = [
        torch.randn(B, 2, 3, 200, 266).cuda(),  # global: 2 frames (contrastive)
        torch.randn(B, 2, 3, 200, 266).cuda(),  # wrist: 2 frames
        torch.randn(B, H, 9, 9, 2).cuda(),       # gelsight: H frames
    ]

    # History images: k=3 frames
    k = 3
    history_images = [
        torch.randn(B, k, 3, 200, 266).cuda(),  # global
        torch.randn(B, k, 3, 200, 266).cuda(),  # wrist
        torch.randn(B, k, 9, 9, 2).cuda(),       # gelsight
    ]

    # --- Training forward ---
    print("\nTraining forward pass (epoch 0)...")
    loss_dict = policy(qpos, images, actions, is_pad,
                       future_images=future_images,
                       epoch=0, total_epochs=100,
                       history_images=history_images)

    print(f"\nLoss dict ({len(loss_dict)} entries):")
    for k_name, v in loss_dict.items():
        print(f"  {k_name:30s}: {v.item():.6f}")

    # Verify all losses are present and finite
    expected_keys = [
        'l1_draft', 'l1_final', 'latent', 'obs', 'foresight_vis',
        'consistency', 'dynamics', 'contrastive_spatial',
        'contrastive_temporal', 'contrastive_gt', 'kl', 'sampling', 'loss',
        'gate_mem', 'gate_a1', 'gate_traj', 'residual_alpha',
    ]
    for key in expected_keys:
        assert key in loss_dict, f"Missing loss key: {key}"
        assert torch.isfinite(loss_dict[key]), f"Non-finite loss: {key}={loss_dict[key]}"

    # Verify total loss is positive (for training)
    assert loss_dict['loss'].item() > 0, "Total loss should be positive"
    print("\nAll losses present and finite. PASSED")

    # --- Backward pass ---
    print("\nBackward pass...")
    loss_dict['loss'].backward()
    # Check gradients exist
    n_grads = sum(1 for p in policy.model.parameters() if p.grad is not None)
    n_params = sum(1 for p in policy.model.parameters() if p.requires_grad)
    print(f"  {n_grads}/{n_params} parameters have gradients")
    assert n_grads > 0, "No gradients computed"
    print("  PASSED")

    # --- Gradient flow check ---
    print("\nGradient flow check (consistency loss → decoder_final)...")
    # decoder_final should have gradients
    has_final_grad = False
    for name, p in policy.model.decoder_final.named_parameters():
        if p.grad is not None and p.grad.abs().max() > 0:
            has_final_grad = True
            break
    print(f"  decoder_final has gradients: {has_final_grad}")
    assert has_final_grad, "decoder_final should have gradients"
    print("  PASSED")

    # --- Inference forward ---
    print("\nInference forward pass...")
    policy.eval()
    with torch.no_grad():
        a2_hat = policy(qpos, images, history_images=history_images)
    print(f"  a2_hat shape: {tuple(a2_hat.shape)} — expected ({B}, {chunk_size}, {state_dim})")
    assert a2_hat.shape == (B, chunk_size, state_dim)
    print("  PASSED")

    # --- Memory usage ---
    if torch.cuda.is_available():
        mem_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\nPeak GPU memory: {mem_mb:.0f} MB")

    print("\n" + "=" * 60)
    print("INTEGRATION TEST PASSED!")
    print("=" * 60)


if __name__ == '__main__':
    test_full_forward()
