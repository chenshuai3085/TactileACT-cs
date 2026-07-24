"""Smoke tests for pi0.5 flow-step tactile guidance utilities.

Run:
    python -m pi0_tactile.test_flow_guidance
"""

from __future__ import annotations

import torch

from pi0_tactile.guidance import Pi0ActionAdapter, Pi0FlowGuidanceConfig, Pi0FlowStepGuidance


def test_action_adapter_pad_slice():
    adapter = Pi0ActionAdapter(model_action_dim=32, robot_action_dim=7)
    a7 = torch.arange(2 * 4 * 7, dtype=torch.float32).view(2, 4, 7)
    a32 = adapter.pad_robot_action(a7)
    assert a32.shape == (2, 4, 32)
    assert torch.allclose(a32[..., :7], a7)
    assert torch.allclose(a32[..., 7:], torch.zeros_like(a32[..., 7:]))
    assert torch.allclose(adapter.slice_robot_action(a32), a7)


def test_flow_guidance_updates_only_robot_dims():
    torch.manual_seed(7)
    adapter = Pi0ActionAdapter(model_action_dim=32, robot_action_dim=7)
    guidance = Pi0FlowStepGuidance(
        Pi0FlowGuidanceConfig(
            guidance_steps=2,
            guidance_scale=0.05,
            max_total_delta=0.10,
            accept_only_improved=True,
            detach_velocity=True,
        ),
        adapter,
    )
    x_t = torch.zeros(2, 5, 32)
    v_t = torch.zeros_like(x_t)
    t = torch.ones(2) * 0.2

    target = torch.ones(2, 5, 7) * 0.5

    def score_fn(action7: torch.Tensor) -> torch.Tensor:
        return -((action7 - target.to(action7.device)) ** 2).mean(dim=(1, 2))

    base_clean = guidance.clean_action_estimate(x_t, v_t, t)
    base_score = score_fn(adapter.slice_robot_action(base_clean))
    guided, report = guidance.guide(
        x_t,
        v_t,
        t,
        step_idx=8,
        total_steps=10,
        score_fn=score_fn,
    )
    guided_clean = guidance.clean_action_estimate(guided, v_t, t)
    guided_score = score_fn(adapter.slice_robot_action(guided_clean))

    assert report["applied"] is True
    assert report["accept_rate"] == 1.0
    assert torch.all(guided_score > base_score)
    assert torch.allclose(guided[..., 7:], x_t[..., 7:])
    assert torch.linalg.norm((guided - x_t)[..., :7].flatten(1), dim=1).max() <= 0.100001


def main():
    test_action_adapter_pad_slice()
    test_flow_guidance_updates_only_robot_dims()
    print("pi0_tactile flow guidance smoke tests passed")


if __name__ == "__main__":
    main()
