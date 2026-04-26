# TacDream: 完整方案 — 触觉前瞻评分引导的扩散策略

> 日期: 2026-04-27
> 目标论文: CoRL 2026
> 核心思路: DP生成K个候选动作 → Foresight预测未来触觉(latent) → CQF打分 → 选最优

---

## 一、已验证的实验结论

| 已验证结论 | 意义 |
|---|---|
| DP有足够多样性(spread 3.23mm, gain 67%) | reranking可行 |
| ACT后验坍缩(spread 0.36mm) | **必须用DP**而不是ACT作为base policy |
| Action条件化有效(bounce时高敏感) | Foresight确实依赖action条件，不是死记硬背 |
| Latent空间表示更好 | TactileVAE方向正确 |
| GateFusion是瓶颈(memory被压低67%, foresight rank-1) | 需要新的融合/引导机制 |

### Reranking可行性数据

| 参数 | Spread | Mean error | best-of-16 | K=16 Reranking gain |
|------|--------|------------|------------|---------------------|
| ACT 77M | 0.36mm | 5.46mm | 4.81mm | 不可行 |
| DP 107M | 3.23mm | 7.59mm | 2.5mm | **67%** |

---

## 二、文献调研核心发现

### 触觉预测
- **Touch Dreaming (2026)**: Latent-space tactile prediction比raw prediction**成功率高30%**
- **CLaD (KAIST, 2026)**: JEPA-style EMA target encoder做latent prediction，LIBERO-LONG 94.7%
- **OmniVTA (2026)**: VQ-VAE tokenization + classification loss避免MSE均值回归
- **所有最新方法一致**: decoder-free的latent prediction优于原始空间prediction

### 打分与引导
- **TouchGuide (2026)**: CPM contrastive scoring + noise pretraining (+23% absolute)
- **DynaGuide (2025)**: Dynamics model gradient引导diffusion denoising
- **PPGuide (2026)**: Performance predictor gradient引导
- **EDMP (2023)**: 多cost function ensemble引导

### 触觉引导动作
- **RDP (2025)**: Slow-fast hierarchy，快速触觉残差修正
- **ReTac-ACT (2026)**: State-gated V-T fusion
- **Contact-Grounded Policy (2026)**: 联合预测(action, tactile)确保一致性

### 核心结论
1. **Latent prediction >> raw prediction** (全面共识)
2. **JEPA-style EMA targets 或 VQ-VAE** 都能避免MSE均值回归
3. **Gradient guidance / reranking** 比feature fusion更有效
4. **Phase-aware processing** 一致有帮助
5. **Contrastive scoring** (expert vs non-expert pairs) 是CQF的有效训练方式

---

## 三、三大创新点

### 创新点1: Action-Conditioned Tactile Foresight in Latent Space (ACTF)

不在raw marker空间做MSE，而是:
- 训练TactileVAE将marker_offset编码到compact latent space
- ForesightTransformer预测未来的**tactile latent embedding**
- 预测目标是EMA target encoder的输出（JEPA-style），而非frozen encoder输出

为什么EMA target优于frozen encoder:
```
Frozen: target = f_θ_frozen(T_future)  → target固定，latent空间可能不适合prediction
EMA:    target = f_θ_ema(T_future)     → target随训练evolve，latent空间自适应变得更易预测
JEPA-VLA(2026)证明: predictive embedding >> contrastive embedding (CLIP) for policy learning
```

与OmniVTA的差异化: OmniVTA用CausalConv3D+INR做VQ-VAE tokenization + autoregressive token prediction，是完整world model + planning。我们只用轻量的连续latent prediction + scoring，更高效，且与DP框架天然结合。

### 创新点2: Physics-Decomposed Contact Quality Verifier (PD-CQV)

**详细设计见: [TFAC_CQF_Physics_Decomposed_Design.md](./TFAC_CQF_Physics_Decomposed_Design.md)**

核心: 将接触质量分解为4个物理可解释子维度:
- S_FIC: Forward-Inverse Consistency (动作-触觉因果一致性)
- S_DCS: Distributional Compliance (专家分布合规)
- S_TDS: Temporal Dynamics (时序物理合规)
- S_CMA: Cross-Modal Agreement (跨模态一致性)

### 创新点3: Dual-Mode Guidance (Reranking + Gradient)

同时支持两种模式:

**Mode A — Best-of-K Reranking** (简单可靠):
```
1. DP forward → K个candidate actions {a^1, ..., a^K}
2. 对每个a^k: Foresight(state, a^k) → z_tac_future^k
3. 对每个: score^k = CQF(state, a^k, z_tac_future^k)
4. 选择: a* = argmax_k score^k
```

**Mode B — Gradient Guidance** (更创新，类DynaGuide):
```
在DP去噪过程中，每个去噪步骤 t → t-1:
1. 当前noisy action a_t 经标准去噪得到 a_{t-1}^{base}
2. 计算 gradient: g = ∇_{a_t} CQF(state, a_t, Foresight(state, a_t))
3. 引导: a_{t-1} = a_{t-1}^{base} + w * g  (w是guidance weight)
```

---

## 四、完整架构

```
                        ┌─────────────────────────────┐
                        │   Diffusion Policy (DP)     │
                        │   generate K candidates     │
                        │   {a^1, a^2, ..., a^K}     │
                        └─────────┬───────────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │  For each candidate a^k:    │
                    │                             │
                    │  ┌─────────────────────┐   │
  current obs ───→  │  │ Shared Encoder       │   │
  (V, T, qpos)     │  │ (ResNet18 + PointNet │   │
                    │  │  + TransformerEnc)   │   │
                    │  └──────────┬──────────┘   │
                    │             │ memory        │
                    │  ┌──────────▼──────────┐   │
                    │  │ Foresight Transformer│   │
                    │  │ Cond: a^k            │   │
                    │  │ → z_tac_future^k     │   │
                    │  └──────────┬──────────┘   │
                    │             │               │
                    │  ┌──────────▼──────────┐   │
                    │  │  PD-CQV Scorer      │   │
                    │  │ (S_FIC+S_DCS+S_TDS  │   │
                    │  │  +S_CMA → total)    │   │
                    │  │ → score^k            │   │
                    │  └──────────┬──────────┘   │
                    │             │               │
                    └─────────────┼───────────────┘
                                  │
                        ┌─────────▼─────────┐
                        │  a* = argmax(score) │
                        │  (Reranking)        │
                        └─────────────────────┘
```

---

## 五、训练流程 (3 Stage)

### Stage 1: TactileVAE Pretraining (~5-10 epochs)

- 架构: PointNet spatial encoder + 1D-CNN temporal + Predictor MLP
- 训练方式: **JEPA-style** — online encoder + EMA target encoder, predict future latent from current
- 不用VQ-VAE（避免codebook问题），用连续latent + cosine prediction loss
- Loss: `L = -cos_sim(predictor(z_online_current), z_ema_future.detach()) + λ*variance_reg`
- 输出: pretrained tactile encoder + EMA encoder

为什么JEPA-style而不是VQ-VAE:
1. VQ-VAE有codebook collapse风险（marker数据量不大）
2. JEPA不需要decoder，更轻量
3. JEPA的latent空间自适应evolve，天然适合prediction task
4. CLaD(2026)验证了JEPA-style在机器人预测中的优越性

### Stage 2: Foresight + CQV Pretraining (~20-30 epochs)

- 冻结TactileVAE encoder
- 训练ForesightTransformer: (current_vision, current_tactile_latent, action_chunk) → future_tactile_latent
- 预测目标: EMA target encoder编码的future tactile latent
- Loss: `L_foresight = smooth_L1(z_pred, z_ema_target.detach()) + cosine_loss`
- 同时训练PD-CQV (各子维度 + ranking loss)

### Stage 3: Joint Training (~50-100 epochs)

- Base DP正常训练 (DDPM loss)
- Foresight fine-tune (继续latent prediction)
- CQV fine-tune (ranking loss on expert vs perturbed pairs)
- 可选: DPPO微调 (用CQV score作为reward)

### Loss设计

```python
# Stage 3 Total Loss
L_total = L_ddpm                           # DP主loss
        + λ_foresight * L_foresight_latent  # 触觉latent预测 (cosine + smooth_L1)
        + λ_cqv * L_cqv                    # CQV训练 (ranking + sub-losses)
        + λ_reg * L_tac_regularization     # 防止visual dominance

# 其中:
L_foresight_latent = smooth_L1(z_pred, z_ema.detach()) + (1 - cos_sim(z_pred, z_ema.detach()))
L_cqv = λ_fic*L_cycle + λ_dcs*L_nce + λ_tds*L_dynamics_bce + λ_cma*L_agreement + λ_rank*L_ranking
L_tac_reg = max(0, margin - attention_weight_tactile)
```

---

## 六、推理流程

```python
def inference(observation, K=16):
    # 1. Encode
    memory, vision_feat, z_tac_current = encoder(observation)
    
    # 2. DP batch生成K个候选 (并行)
    candidates = dp.sample(memory, num_samples=K)  # (K, chunk, action_dim)
    
    # 3. Batch foresight (并行)
    z_tac_futures = foresight(memory.expand(K,...), candidates)
    
    # 4. Batch scoring (并行)
    total_scores, sub_scores = pd_cqv(
        memory, vision_feat, z_tac_current, z_tac_futures, candidates
    )
    
    # 5. Select best
    best_action = candidates[total_scores.argmax()]
    return best_action
```

推理效率: Encoder 1次 + DP K=16 batch + Foresight batch + CQV batch = **额外开销<10%**

---

## 七、论文故事线

### Title
"TacDream: Contact-Aware Manipulation via Tactile Foresight Scoring and Diffusion Policy Reranking"

### Abstract方向
> Contact-rich manipulation requires anticipating tactile consequences of candidate actions. We present TacDream, which learns to dream about future contact in a self-supervised latent space, then uses physics-decomposed contact quality scoring to select the best action from a diffusion policy.
>
> Key innovations: (1) JEPA-style tactile foresight predicting in latent space, (2) Physics-Decomposed Contact Quality Verifier with four interpretable sub-dimensions, (3) Efficient best-of-K reranking with <10% inference overhead.

### 消融实验矩阵

| 实验 | 验证什么 |
|---|---|
| DP (no tactile) | 视觉-only baseline |
| DP + tactile (concat) | naive触觉融合 |
| DP + foresight (raw MSE) | 原始空间预测 |
| DP + foresight (latent) | 创新点1: latent预测 |
| DP + foresight + single-score CQF | TouchGuide-style单维度基线 |
| DP + foresight + PD-CQV (各子维度单独) | 各子维度贡献 |
| DP + foresight + PD-CQV reranking | 完整方法 (Mode A) |
| DP + foresight + PD-CQV gradient guidance | 完整方法 (Mode B) |
| K ablation (1,4,8,16,32) | reranking候选数量 |

---

## 八、实现路线图

```
Week 1-2: TactileVAE (JEPA-style)
  - PointNet encoder for marker_offset
  - EMA target encoder setup
  - Cosine prediction + variance regularization
  - 验证: latent空间可视化, 预测质量

Week 3-4: Foresight in Latent Space
  - 修改ForesightTransformer输出latent
  - 预测目标改为EMA encoder output
  - 验证: 对比raw MSE vs latent prediction

Week 5-6: PD-CQV (Contact Quality Function)
  - 实现4个子维度scorer
  - 训练ranking loss
  - 验证: 能否区分好/坏动作（特别在bounce时刻）

Week 7-8: Reranking + Guidance Integration
  - Best-of-K reranking集成到DP
  - 可选: gradient guidance
  - End-to-end ablation experiments

Week 9-10: Paper Writing + Real Robot Experiments
```

---

## 九、关键文献

| 论文 | 年份 | 核心贡献 | 与我们的关系 |
|------|------|---------|-------------|
| Touch Dreaming | 2026 | Latent tac prediction 30% better | 验证latent方向 |
| CLaD | 2026 | JEPA-style EMA for foresight | TactileVAE设计 |
| TouchGuide | 2026 | CPM contrastive + noise pretraining | CQF训练方式 |
| DynaGuide | 2025 | Dynamics gradient guides diffusion | Gradient guidance |
| OmniVTA | 2026 | VQ-VAE + LTD + reflexive controller | 触觉差分编码 |
| PPGuide | 2026 | Performance predictor gradient | 打分替代方案 |
| EDMP | 2023 | Ensemble of costs for diffusion | 多维度打分 |
| JIF | 2025 | Forward-inverse cycle consistency | S_FIC设计 |
| IBC | 2021 | Energy-based action scoring | S_DCS设计 |
| ReTac-ACT | 2026 | State-gated V-T fusion | Phase-aware |
| RDP | 2025 | Slow-fast tactile policy | 备选方案参考 |
| VT-WM | 2026 | RSSM tactile world model | Latent dynamics |
| Contact-Grounded Policy | 2026 | Joint (action, tac) prediction | 一致性保证 |
| VTAM | 2026 | Tactile regularization | 防visual dominance |
| Ranking NCE | 2023 | EBM action ranking | CQF ranking loss |
