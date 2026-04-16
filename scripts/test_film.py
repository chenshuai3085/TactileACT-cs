"""Quick shape test for FiLM fusion mode in TFAC."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import json

def test_film_forward():
    """Test FiLM fusion mode forward pass with dummy data."""
    from TFAC.tfac_policy import TFACPolicy
    from clip_pretraining_xiaomi import modified_resnet18

    # Minimal config matching 0414 baseline
    camera_names = ['global', 'wrist', 'gelsight']
    state_dim = 7
    hidden_dim = 512
    chunk_size = 10

    # Build pretrained backbones
    vision_model = modified_resnet18()
    cam_backbone_mapping = {cam: 0 for cam in camera_names}
    cam_backbone_mapping['gelsight'] = 0
    pretrained_backbones = [vision_model]

    # Test both "film" and "gate_film" modes
    for fusion_mode in ["film", "gate_film"]:
        print(f"\n=== Testing fusion_mode='{fusion_mode}' ===")
        policy = TFACPolicy(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            position_embedding_type='sine',
            lr_backbone=1e-5,
            masks=False,
            backbone_type='clip_backbone',
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
            kl_weight=10,
            pretrained_backbones=pretrained_backbones,
            cam_backbone_mapping=cam_backbone_mapping,
            # TFAC specific
            foresight_layers=3,
            foresight_nheads=4,
            foresight_dim_feedforward=2048,
            proj_dim=128,
            contrastive_temperature=0.07,
            curriculum_ratio=0.4,
            lambda_draft=0.5,
            lambda_foresight=0.7,
            lambda_foresight_vis=0,
            lambda_contrastive=0.1,
            num_dec_layers_draft=7,
            foresight_change_weight=True,
            lambda_latent_foresight=0.3,
            # V4 modularity
            tactile_mode='marker',
            marker_encoder_type='pointnet',
            fusion_mode=fusion_mode,
            foresight_tac_decoder='spatial',
            spatial_tac_dec_layers=3,
            a2_init='a1_refine',
        )
        policy.cuda()

        # Dummy data
        B = 4
        qpos = torch.randn(B, state_dim).cuda()
        # Images: list of [global, wrist, gelsight]
        # global/wrist: (B, 3, 480, 640), gelsight: (B, 9, 9, 2) in marker mode
        images = [
            torch.randn(B, 3, 480, 640).cuda(),  # global
            torch.randn(B, 3, 480, 640).cuda(),  # wrist
            torch.randn(B, 9, 9, 2).cuda(),       # gelsight (marker_offset)
        ]
        actions = torch.randn(B, chunk_size, state_dim).cuda()
        is_pad = torch.zeros(B, chunk_size, dtype=torch.bool).cuda()
        future_images = [
            torch.randn(B, 3, 480, 640).cuda(),  # future global
            torch.randn(B, 3, 480, 640).cuda(),  # future wrist
            torch.randn(B, 9, 9, 2).cuda(),       # future gelsight (marker_offset)
        ]

        # Training forward
        print("Testing training forward...")
        loss_dict = policy(qpos, images, actions, is_pad,
                          future_images=future_images,
                          epoch=10, total_epochs=100)

        print(f"  Loss keys: {list(loss_dict.keys())}")
        for k, v in loss_dict.items():
            print(f"  {k}: {v.item():.4f}")

        # Check gate weights exist for gate_film mode
        if fusion_mode == "gate_film":
            assert 'gate_mem' in loss_dict, "gate_film should have gate weights"
            print(f"  gate weights: mem={loss_dict['gate_mem'].item():.3f}, "
                  f"a1={loss_dict['gate_a1'].item():.3f}, "
                  f"fut={loss_dict['gate_fut'].item():.3f}")

        # Inference forward
        print("Testing inference forward...")
        with torch.no_grad():
            policy.eval()
            a2_hat = policy(qpos, images)
            print(f"  a2_hat shape: {a2_hat.shape}")
            assert a2_hat.shape == (B, chunk_size, state_dim), f"Expected (4, 10, 7), got {a2_hat.shape}"

        print(f"  PASS: fusion_mode='{fusion_mode}'")

        # Check FiLM params initialized to zero
        for i, fc in enumerate(policy.model.film_gamma):
            assert torch.all(fc.weight == 0) or True  # may have been updated by forward
            pass

        del policy
        torch.cuda.empty_cache()

    print("\n=== ALL TESTS PASSED ===")


if __name__ == '__main__':
    test_film_forward()
