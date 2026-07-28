"""
Pi0Tactile: Pi0 VLA model extended with tactile perception and foresight guidance.

Architecture:
  Prefix: [SigLIP(images) | lang_tokens | TactileVAE_tokens]
  Suffix: [state | noisy_actions + time_emb]
  Loss:   L_flow_matching + lambda * L_foresight (when t < threshold)
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

for _OPENPI in (
    os.environ.get("OPENPI_SRC", ""),
    os.path.join(os.path.dirname(_ROOT), "openpi", "src"),
    "/home/chenshuai/Project/openpi/src",
):
    if _OPENPI and os.path.exists(_OPENPI) and _OPENPI not in sys.path:
        sys.path.insert(0, _OPENPI)

from openpi.models_pytorch.pi0_pytorch import (
    PI0Pytorch,
    make_att_2d_masks,
)
import openpi.models.gemma as _gemma

from pi0_tactile.tactile_encoder import TactileTokenEncoder
from pi0_tactile.foresight_module import ForesightModule
from pi0_tactile.config import Pi0TactileConfig
from pi0_tactile.guidance import (
    Pi0ActionAdapter,
    Pi0FlowGuidanceConfig,
    Pi0FlowStepGuidance,
)


class Pi0Tactile(nn.Module):
    """
    Pi0 + Tactile tokens in prefix + Foresight auxiliary loss.

    Key modifications from PI0Pytorch:
    1. embed_prefix: append tactile tokens after image/lang tokens
    2. forward: add foresight loss branch when t < threshold
    3. Trainable params: tactile_encoder.proj, foresight_module, action_in/out_proj, LoRA
    """

    def __init__(self, config: Pi0TactileConfig):
        super().__init__()
        self.config = config

        # === Pi0 base model ===
        from openpi.models.pi0_config import Pi0Config
        pi0_config = Pi0Config(
            dtype=config.dtype,
            paligemma_variant=config.paligemma_variant,
            action_expert_variant=config.action_expert_variant,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            max_token_len=config.max_token_len,
            pi05=config.pi05,
            pytorch_compile_mode=config.pytorch_compile_mode,
        )
        self.pi0 = PI0Pytorch(pi0_config)
        self.action_adapter = Pi0ActionAdapter(
            model_action_dim=config.action_dim,
            robot_action_dim=config.robot_action_dim,
        )

        # Get action expert width
        action_expert_cfg = _gemma.get_config(config.action_expert_variant)
        self.expert_width = action_expert_cfg.width

        # === Tactile Encoder (frozen VAE + trainable proj) ===
        self.tactile_encoder = TactileTokenEncoder(
            vae_checkpoint=config.vae_checkpoint,
            vae_latent_dim=config.vae_latent_dim,
            tac_history=config.tac_history,
            width=self.expert_width,
            freeze_vae=config.freeze_vae,
        )

        # === Foresight Module (for auxiliary loss during training) ===
        self.foresight_module = ForesightModule(
            hidden_dim=config.foresight_hidden_dim,
            action_dim=config.robot_action_dim,
            num_layers=config.foresight_layers,
            nheads=config.foresight_nheads,
            dim_feedforward=config.foresight_dim_feedforward,
            vae_latent_dim=config.vae_latent_dim,
            n_tactile_spatial=config.tac_token_num,
            predict_horizon=config.foresight_predict_horizon,
            max_action_len=config.action_horizon,
            checkpoint_path=config.foresight_checkpoint,
        )

        # Foresight training state
        self.global_step = 0

    def build_flow_guidance(self) -> Pi0FlowStepGuidance:
        """Build the configured late-flow-step tactile guidance helper."""
        cfg = Pi0FlowGuidanceConfig(
            guidance_steps=self.config.flow_guidance_steps,
            guidance_scale=self.config.flow_guidance_scale,
            max_total_delta=self.config.flow_guidance_max_total_delta,
            normalize_grad=self.config.flow_guidance_normalize_grad,
            max_grad_norm=self.config.flow_guidance_max_grad_norm,
            accept_only_improved=self.config.flow_guidance_accept_only_improved,
            clamp_min=self.config.flow_guidance_clamp_min,
            clamp_max=self.config.flow_guidance_clamp_max,
            lambda_smooth=self.config.flow_guidance_lambda_smooth,
            detach_velocity=self.config.flow_guidance_detach_velocity,
            recompute_velocity_after_guidance=self.config.flow_guidance_recompute_velocity,
        )
        return Pi0FlowStepGuidance(cfg, self.action_adapter)

    def _pad_model_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.action_adapter.pad_robot_action(action)

    def _slice_robot_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.action_adapter.slice_robot_action(action)

    def freeze_paligemma(self):
        """Freeze PaliGemma (SigLIP + LLM) and only train action expert + new modules."""
        # Freeze vision tower
        self.pi0.paligemma_with_expert.paligemma.vision_tower.requires_grad_(False)
        # Freeze language model
        self.pi0.paligemma_with_expert.paligemma.language_model.requires_grad_(False)
        # Keep action expert trainable
        self.pi0.paligemma_with_expert.gemma_expert.requires_grad_(True)
        # Keep projection layers trainable
        self.pi0.action_in_proj.requires_grad_(True)
        self.pi0.action_out_proj.requires_grad_(True)
        if self.config.pi05:
            self.pi0.time_mlp_in.requires_grad_(True)
            self.pi0.time_mlp_out.requires_grad_(True)
        else:
            self.pi0.state_proj.requires_grad_(True)
            self.pi0.action_time_mlp_in.requires_grad_(True)
            self.pi0.action_time_mlp_out.requires_grad_(True)

        n_frozen = sum(1 for p in self.parameters() if not p.requires_grad)
        n_trainable = sum(1 for p in self.parameters() if p.requires_grad)
        print(f"[Pi0Tactile] Frozen: {n_frozen} params, Trainable: {n_trainable} params")

    def get_trainable_params(self):
        """Return list of trainable parameter groups for optimizer."""
        pi0_trainable = [
            p for p in self.pi0.paligemma_with_expert.gemma_expert.parameters()
            if p.requires_grad
        ]
        pi0_proj = [
            p for name, p in self.pi0.named_parameters()
            if p.requires_grad and "paligemma_with_expert" not in name
        ]
        tac_params = list(self.tactile_encoder.parameters())
        foresight_params = list(self.foresight_module.parameters())

        return [
            {"params": pi0_trainable, "lr_scale": 1.0, "name": "action_expert"},
            {"params": pi0_proj, "lr_scale": 1.0, "name": "pi0_projections"},
            {"params": tac_params, "lr_scale": 1.0, "name": "tactile_encoder"},
            {"params": foresight_params, "lr_scale": 0.5, "name": "foresight"},
        ]

    def embed_prefix_with_tactile(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        marker_offset: torch.Tensor,
    ):
        """
        Embed prefix: images + language + tactile tokens.

        Args:
            images: list of (B, C, H, W) image tensors
            img_masks: list of (B,) boolean masks
            lang_tokens: (B, max_token_len) int token IDs
            lang_masks: (B, max_token_len) boolean masks
            marker_offset: (B, T_hist, 9, 9, 2) tactile sequence

        Returns:
            embs: (B, N_total, width)
            pad_masks: (B, N_total)
            att_masks: (B, N_total)
        """
        # Standard pi0 prefix (images + language)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.pi0.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        # Tactile tokens
        tac_tokens = self.tactile_encoder(marker_offset)  # (B, 9, width)

        # Match dtype
        if prefix_embs.dtype != tac_tokens.dtype:
            tac_tokens = tac_tokens.to(prefix_embs.dtype)

        B = tac_tokens.shape[0]
        n_tac = tac_tokens.shape[1]

        # Tactile pad mask: always valid
        tac_pad_mask = torch.ones(B, n_tac, dtype=torch.bool, device=tac_tokens.device)

        # Tactile attention mask: bidirectional (ar_mask=0), same as images/lang
        tac_att_mask = torch.zeros(B, n_tac, dtype=prefix_att_masks.dtype, device=tac_tokens.device)

        # Concatenate
        embs = torch.cat([prefix_embs, tac_tokens], dim=1)
        pad_masks = torch.cat([prefix_pad_masks, tac_pad_mask], dim=1)
        att_masks = torch.cat([prefix_att_masks, tac_att_mask], dim=1)

        return embs, pad_masks, att_masks

    def forward(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        marker_offset: torch.Tensor,
        actions: torch.Tensor,
        future_marker_offset: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None,
        compute_foresight_loss: Optional[bool] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass.

        Returns:
            dict with keys:
              - 'flow_loss': (B, action_horizon) per-step flow matching loss
              - 'foresight_loss': scalar, auxiliary foresight loss (0 if not applicable)
              - 'total_loss': scalar, combined loss
        """
        actions = self._pad_model_action(actions)
        state_model = self._pad_model_action(state)
        B = actions.shape[0]
        device = actions.device

        # Sample noise and time
        if noise is None:
            noise = self.pi0.sample_noise(actions.shape, device)
        else:
            noise = self._pad_model_action(noise)
        if time is None:
            time = self.pi0.sample_time(B, device)

        # Flow matching: x_t = t * noise + (1-t) * actions
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions  # target velocity

        # === Prefix with tactile ===
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix_with_tactile(
            images, img_masks, lang_tokens, lang_masks, marker_offset
        )

        # === Suffix (state + noisy actions + time) ===
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.pi0.embed_suffix(
            state_model, x_t, time
        )

        # Match dtypes
        if (self.pi0.paligemma_with_expert.paligemma.language_model.layers[0]
                .self_attn.q_proj.weight.dtype == torch.bfloat16):
            prefix_embs = prefix_embs.to(torch.bfloat16)
            suffix_embs = suffix_embs.to(torch.bfloat16)

        # === Full attention ===
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self.pi0._prepare_attention_masks_4d(att_2d_masks)

        # Forward through PaliGemma + Action Expert
        (_, suffix_out), _ = self.pi0.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        # Extract action predictions
        suffix_out = suffix_out[:, -self.config.action_horizon:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.pi0.action_out_proj(suffix_out)

        # === Flow matching loss ===
        flow_loss = F.mse_loss(u_t, v_t, reduction="none")  # (B, H, action_dim)
        flow_loss_per_step = flow_loss.mean(dim=-1)  # (B, H)

        # === Foresight auxiliary loss ===
        foresight_loss = torch.tensor(0.0, device=device)

        enable_foresight_loss = self.training if compute_foresight_loss is None else bool(compute_foresight_loss)

        if (enable_foresight_loss
                and self.global_step >= self.config.foresight_warmup_steps
                and future_marker_offset is not None):

            # Only compute foresight loss for samples with low t (action estimate is clean)
            low_t_mask = time < self.config.foresight_t_threshold  # (B,)

            if low_t_mask.any():
                # Estimate clean actions from flow: x_0 ≈ x_t - t * v_t
                with torch.no_grad():
                    t_exp = time[low_t_mask, None, None]
                    x_0_est = (x_t[low_t_mask] - t_exp * v_t[low_t_mask].detach())
                    x_0_est = self._slice_robot_action(x_0_est)

                # Get current tactile latent for foresight input
                with torch.no_grad():
                    z_current, _ = self.tactile_encoder.vae.encode_single_frame(
                        marker_offset[low_t_mask]
                    )

                # Predict future tactile latent
                z_pred = self.foresight_module(
                    action_pred=x_0_est,
                    tac_latent=z_current,
                    qpos=self._slice_robot_action(state_model[low_t_mask]),
                )
                if z_pred.dim() == 3:
                    z_pred = z_pred[:, -1]

                # GT future tactile latent
                z_gt = self.tactile_encoder.encode_latent_flat(
                    future_marker_offset[low_t_mask]
                )

                foresight_loss = F.l1_loss(z_pred, z_gt)

        # === Total loss ===
        total_loss = flow_loss_per_step.mean() + self.config.lambda_foresight * foresight_loss

        return {
            "flow_loss": flow_loss_per_step,
            "foresight_loss": foresight_loss,
            "total_loss": total_loss,
        }

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        marker_offset: torch.Tensor,
        num_steps: int = 10,
        noise: Optional[torch.Tensor] = None,
        flow_guidance: Optional[Pi0FlowStepGuidance] = None,
        score_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        return_guidance_report: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Inference: generate actions via flow matching ODE with tactile-augmented prefix.

        Args:
            images, img_masks, lang_tokens, lang_masks: standard pi0 inputs
            state: (B, action_dim) robot state
            marker_offset: (B, T_hist, 9, 9, 2) tactile sequence
            num_steps: number of ODE integration steps (default 10)

        Returns:
            actions: (B, action_horizon, action_dim)
        """
        B = state.shape[0]
        device = state.device
        state_model = self._pad_model_action(state)

        if noise is None:
            actions_shape = (B, self.config.action_horizon, self.config.action_dim)
            noise = self.pi0.sample_noise(actions_shape, device)
        else:
            noise = self._pad_model_action(noise)

        if flow_guidance is None and self.config.flow_guidance_steps > 0:
            flow_guidance = self.build_flow_guidance()
        if flow_guidance is not None and flow_guidance.config.enabled and score_fn is None:
            raise ValueError("score_fn is required when flow guidance is enabled")

        # Embed prefix with tactile (cached, computed once)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix_with_tactile(
            images, img_masks, lang_tokens, lang_masks, marker_offset
        )

        # Match dtype
        if (self.pi0.paligemma_with_expert.paligemma.language_model.layers[0]
                .self_attn.q_proj.weight.dtype == torch.bfloat16):
            prefix_embs = prefix_embs.to(torch.bfloat16)

        # Build prefix KV cache
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self.pi0._prepare_attention_masks_4d(prefix_att_2d_masks)

        self.pi0.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, past_key_values = self.pi0.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # ODE integration: t goes from 1 → 0
        dt = -1.0 / num_steps
        x_t = noise
        t_val = 1.0
        guidance_reports = []

        for step_idx in range(num_steps):
            expanded_time = torch.full((B,), t_val, dtype=torch.float32, device=device)

            v_t = self.pi0.denoise_step(
                state_model, prefix_pad_masks, past_key_values, x_t, expanded_time
            )

            if flow_guidance is not None and score_fn is not None:
                with torch.inference_mode(False):
                    with torch.enable_grad():
                        guided_x_t, report = flow_guidance.guide(
                            x_t,
                            v_t,
                            expanded_time,
                            step_idx=step_idx,
                            total_steps=num_steps,
                            score_fn=score_fn,
                        )
                guidance_reports.append(report)
                accepted = bool(report.get("applied")) and float(report.get("accept_rate", 0.0)) > 0.0
                if accepted:
                    x_t = guided_x_t
                    if flow_guidance.config.recompute_velocity_after_guidance:
                        v_t = self.pi0.denoise_step(
                            state_model,
                            prefix_pad_masks,
                            past_key_values,
                            x_t,
                            expanded_time,
                        )

            x_t = x_t + dt * v_t
            t_val += dt

        if return_guidance_report:
            summary = (
                flow_guidance.summarize(guidance_reports)
                if flow_guidance is not None
                else {"enabled": False, "steps": guidance_reports}
            )
            return x_t, {"flow_guidance": summary}
        return x_t  # (B, action_horizon, action_dim) — clean actions

    def load_pi0_weights(self, weight_path: str):
        """Load pretrained pi0 base weights."""
        import safetensors.torch
        model_path = os.path.join(weight_path, "model.safetensors")
        if os.path.exists(model_path):
            safetensors.torch.load_model(self.pi0, model_path, strict=False)
            print(f"[Pi0Tactile] Loaded pi0 weights from {model_path}")
        else:
            print(f"[Pi0Tactile] Warning: {model_path} not found, using random init")

    def set_step(self, step: int):
        """Update global step for foresight warmup scheduling."""
        self.global_step = step
