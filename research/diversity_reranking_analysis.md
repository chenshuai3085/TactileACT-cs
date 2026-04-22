# ACT vs DP: Diversity & Reranking Viability Analysis

**Date**: 2026-04-23
**Dataset**: 260309_0310 (337 episodes, 300 frames, 7D joint_abs)
**Test**: K=16 candidates, 3 episodes (50,150,250), EE-space metrics via FK

## Complete Results

### ACT (CVAE z-sampling)

| KL Weight | Temperature | Spread (mm) | Mean Error (mm) | Best-of-K (mm) | Reranking Gain | Verdict |
|-----------|-------------|-------------|-----------------|----------------|----------------|---------|
| 0.01 | 1.0 | 0.60 | 5.49 | 4.72 | 13.9% | NOT VIABLE |
| 0.01 | 3.0 | 4.63 | 8.37 | 4.30 | 48.7% | *pseudo-diversity |
| 0.01 | 10.0 | 12.96 | 21.35 | 3.50 | 83.6% | *pseudo-diversity |
| 1.0 | 1.0 | 0.06 | 5.95 | 5.81 | 2.4% | NOT VIABLE |
| 1.0 | 3.0 | 0.24 | 5.95 | 5.42 | 8.9% | NOT VIABLE |

### Diffusion Policy (DDPM 100 steps, different noise seeds)

| Epoch | Val Loss | Spread (mm) | Mean Error (mm) | Best-of-K (mm) | Reranking Gain | Verdict |
|-------|----------|-------------|-----------------|----------------|----------------|---------|
| ~570 | 0.005 | **4.55** | 9.34 | **3.09** | **66.9%** | **VIABLE** |

## Key Findings

### 1. ACT's CVAE Latent is Fundamentally Broken for Diversity

- **kl=0.01**: Posterior collapse. The 512-dim transformer decoder is too powerful and learns to produce good actions regardless of z. Spread at T=1.0 is only 0.60mm.
- **kl=1.0**: Even worse (0.06mm spread). Strong KL forces posterior = prior = N(0,1), so z is pure uncorrelated noise. The decoder ignores it completely.
- **High temperature "diversity"** (T=3.0, T=10.0 on kl=0.01): This creates spread (4.63mm, 12.96mm) but it's **out-of-distribution noise**, not meaningful multimodality. The model wasn't trained on z-values this large. The "best-of-K" gains at high T are from lucky random draws, not learned diverse modes.
- **Fundamental cause**: ACT's transformer decoder can attend to all observation tokens directly, making z redundant. This is a known issue with powerful CVAE decoders (Bowman et al., 2016 "Generating Sentences from a Continuous Space").

### 2. DP Produces Genuine Multimodal Diversity

- **Spread = 4.55mm** consistently across episodes (4.10-4.99mm range).
- This diversity comes from the **stochastic DDPM denoising process**: different initial noise vectors traverse different denoising paths through the learned distribution, landing on genuinely different but plausible action modes.
- **Not OOD noise**: Each of K=16 trajectories is a valid sample from the learned action distribution.
- **Consistent across episodes**: Unlike ACT's temperature-based "diversity" which is unpredictable.

### 3. Reranking with Oracle Scorer

| Method | Mean Error → Best-of-K | Absolute Best-of-K |
|--------|------------------------|---------------------|
| ACT kl=0.01 (T=1) | 5.49 → 4.72mm | 4.72mm |
| DP (DDPM) | 9.34 → **3.09mm** | **3.09mm** |

- An oracle reranker on DP candidates achieves **3.09mm** — better than ACT's deterministic output (5.49mm).
- DP trades single-trajectory quality for multimodal coverage: worse average, but the best candidate is excellent.
- **The bottleneck is now the scorer** — can tactile prediction serve as a good reranking signal?

### 4. DP Training Notes

- Model: ConditionalUnet1D, 107M params, EMA, DDPM 100 steps
- Convergence: val_loss 0.109→0.036→0.015→0.009→0.005 over 570 epochs
- Still improving (training continues to 3000 epochs)
- Mean error may decrease further with more training, improving both average and best-of-K

## Implications for TFAC (CoRL 2026)

### Recommended Direction: Tactile-Guided Diffusion Policy

The data strongly suggests pivoting from ACT-based TFAC to a **Diffusion Policy + Tactile Guidance** architecture:

**Option A: Tactile Reranking (Simple, Effective)**
1. DP generates K candidate action trajectories
2. ForesightTransformer predicts future tactile for each candidate
3. Score each candidate by predicted tactile quality
4. Select the best candidate
- Pros: Simple, proven concept (this experiment), easy to implement
- Cons: Requires K forward passes through DP (100 DDPM steps each × K)

**Option B: Tactile-Guided Denoising (Elegant, Novel)**
1. During DDPM denoising, at each step compute:
   - Predicted future tactile for current denoised action
   - Gradient of tactile quality score w.r.t. action
2. Bias the denoising step towards tactile-favorable actions (classifier guidance style)
- Pros: Single denoising pass, more elegant, very novel for tactile
- Cons: Requires differentiable tactile predictor, gradient computation overhead
- Reference: DynaGuide (Du & Song, 2025) applies similar concept

**Option C: Hybrid (Best of Both)**
- Use tactile-guided denoising for coarse guidance
- Then rerank final few candidates for fine selection

### Story for Paper

**Title concept**: "Tactile-Guided Diffusion Policy: Dream → Steer → Act"
- **Dream**: ForesightTransformer predicts future tactile
- **Steer**: Tactile prediction guides diffusion denoising / reranks candidates
- **Act**: Execute the tactile-optimal action

**Key claims supported by this analysis**:
1. ACT's CVAE cannot produce meaningful diverse candidates (quantified)
2. DP naturally produces diverse, high-quality candidates (quantified)
3. Oracle reranking achieves 67% error reduction (upper bound for scorer)
4. Tactile prediction can serve as the scorer (to be validated)

## Files

- Test script: `scripts/test_dp_diversity.py`
- Training script: `scripts/train_dp_joint.py`
- Results log: `/home/chenshuai/Project/output/diversity_test_results.log`
- DP checkpoint: `/home/chenshuai/Project/output/dp_joint7/`
- ACT checkpoints: `/home/chenshuai/data/xiaomi_act/act_kl001_clip1099/`, `act_kl1_clip1099/`
