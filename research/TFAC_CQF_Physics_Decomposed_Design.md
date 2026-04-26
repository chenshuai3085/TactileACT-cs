# PD-CQV: Physics-Decomposed Contact Quality Verifier — 完整设计方案

> 日期: 2026-04-27
> 目标: 为TacDream框架设计CQF核心模块，用于对DP候选动作进行触觉前瞻评分和筛选

---

## 一、问题背景与动机

### 当前实验已验证的关键前提

| 已验证结论 | 意义 |
|---|---|
| DP有足够多样性(spread 3.23mm, gain 67%) | reranking可行 |
| ACT后验坍缩(spread 0.36mm) | **必须用DP**而不是ACT作为base policy |
| Action条件化有效(bounce时高敏感) | Foresight确实依赖action条件，不是死记硬背 |
| Latent空间表示更好 | TactileVAE方向正确 |
| GateFusion是瓶颈 | 需要新的融合/引导机制 |

### 现有打分方法的局限性

| 方法 | 打分方式 | 局限性 |
|------|---------|--------|
| **TouchGuide CPM** | cosine(obs_emb, action_emb) 一个标量 | 黑盒、不可解释、不区分失败原因 |
| **DynaGuide** | V(predicted_state) 一个value | 只评估最终状态好坏，不关心物理过程 |
| **OmniVTA RLTC** | mean±std统计阈值 | 规则式、不可学习、无法处理复杂场景 |
| **PPGuide** | 二分类(成功/失败) | 太粗粒度，无法区分"稍差"和"碰撞" |

**核心问题**: 这些方法都把Contact Quality当作一个**单一标量**来处理。但接触质量是**多维度的**——一个动作可能力度合适但方向错误，或者方向对但会导致滑动。单一分数无法区分这些情况。

---

## 二、核心创新: Physics-Decomposed Contact Quality Verifier (PD-CQV)

### 设计理念

> 不做黑盒打分，而是将接触质量**分解为多个物理可解释的子维度**，每个维度用专门的验证头评估，最后学习组合。

灵感来源:
- **EDMP (2023)**: 用多个cost function的梯度ensemble来引导diffusion
- **UniForce (2026)**: 用牛顿第三定律作为物理约束
- **JIF (2025)**: 用forward-inverse cycle一致性做自监督验证
- **SGCNet (2025)**: 用HMM强制接触状态转移的物理合理性

关键创新: **把这些分散的思想统一到一个可学习的多头验证框架中**，专门用于触觉接触质量评估。

### 四个子维度概览

```
PD-CQV Score = f_combine(S_FIC, S_DCS, S_TDS, S_CMA)

S_FIC: Forward-Inverse Consistency   — "动作和预测触觉是否因果一致？"
S_DCS: Distributional Compliance     — "预测触觉是否在专家分布内？"
S_TDS: Temporal Dynamics Score        — "触觉变化过程是否物理合规？"
S_CMA: Cross-Modal Agreement          — "触觉预测和视觉状态是否匹配？"
```

---

## 三、各子维度详细设计

### 3.1 Forward-Inverse Consistency Score (S_FIC)

**核心思想**: 如果一个action产生的predicted tactile是"对的"，那从这个predicted tactile应该能**反推出**原始action。如果反推不出来，说明action-tactile pair不一致。

灵感来源: JIF (2025) 的cycle consistency、DynaMo (2024) 的joint forward-inverse

```
Forward:  action → Foresight → z_tac_future (已有)
Inverse:  z_tac_future → InverseHead → action_recovered
Consistency: S_FIC = -||action - action_recovered||₂ / (||action||₂ + ε)
```

**InverseHead架构**:

```python
class InverseDynamicsHead(nn.Module):
    """从(当前状态, 预测未来触觉)反推应该执行的action"""
    def __init__(self, memory_dim, tac_latent_dim, action_dim, chunk_size):
        super().__init__()
        self.state_proj = nn.Linear(memory_dim, 256)
        self.tac_proj = nn.Linear(tac_latent_dim, 256) 
        self.current_tac_proj = nn.Linear(tac_latent_dim, 256)
        # 从(当前状态, 当前触觉, 未来触觉)推断需要什么action
        self.mlp = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, action_dim * chunk_size)
        )
    
    def forward(self, memory_pooled, z_tac_current, z_tac_future):
        s = self.state_proj(memory_pooled)
        t_cur = self.current_tac_proj(z_tac_current)
        t_fut = self.tac_proj(z_tac_future)
        return self.mlp(torch.cat([s, t_cur, t_fut], dim=-1))
```

**训练Loss**:

```python
# 用专家数据训练InverseDynamicsHead
L_inverse = smooth_l1(InverseHead(memory, z_tac_t, z_tac_{t+h}), action_gt)

# Cycle consistency loss (端到端)  
z_pred = Foresight(memory, action)
action_recovered = InverseHead(memory, z_tac_current, z_pred)
L_cycle = smooth_l1(action_recovered, action)
```

**S_FIC的计算**:

```python
def compute_S_FIC(memory, z_tac_current, action_candidate, foresight, inverse_head):
    z_pred = foresight(memory, action_candidate)
    action_recovered = inverse_head(memory, z_tac_current, z_pred)
    # 归一化误差 → 越小越好 → 取负再sigmoid
    relative_error = (action_candidate - action_recovered).norm(dim=-1) / (action_candidate.norm(dim=-1) + 1e-6)
    S_FIC = torch.sigmoid(-relative_error * 5)  # scale to [0,1]
    return S_FIC
```

**为什么有创新性**: TouchGuide只做forward (obs → score)，没有inverse验证。DynaGuide用value function评估predicted state，但不做consistency check。**Forward-Inverse cycle verification用于触觉质量评估，在文献中还没有人做过。**

**物理意义**: 如果预测出的未来触觉确实是这个action会产生的后果，那从后果应该能推回原因。如果推不回来，说明Foresight的预测和action不因果匹配——可能是碰撞、滑动、或其他非预期接触。

---

### 3.2 Distributional Compliance Score (S_DCS)

**核心思想**: 预测的未来触觉latent应该落在"专家演示中正常触觉"的分布内。偏离分布 = 异常接触（碰撞、过度挤压、滑脱）。

灵感来源: OmniVTA的统计阈值 + NEAR (2025) 的能量函数 + IBC的Energy-Based Model

不用简单的mean±std，而是学一个**Conditional Energy Function**:

```python
class DistributionalComplianceScorer(nn.Module):
    """条件能量函数: 在给定状态下,这个触觉latent的'正常度'有多高"""
    def __init__(self, memory_dim, tac_latent_dim, phase_dim=16):
        super().__init__()
        # Phase encoder: 从当前触觉强度推断接触阶段
        self.phase_encoder = nn.Sequential(
            nn.Linear(tac_latent_dim, 64), nn.ReLU(),
            nn.Linear(64, phase_dim)
        )
        # 条件能量函数: E(z_tac | state, phase)
        self.energy_net = nn.Sequential(
            nn.Linear(memory_dim + tac_latent_dim + phase_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # 标量能量值
        )
    
    def forward(self, memory_pooled, z_tac_predicted, z_tac_current):
        phase = self.phase_encoder(z_tac_current)
        energy = self.energy_net(torch.cat([memory_pooled, z_tac_predicted, phase], dim=-1))
        return energy  # 低能量 = 更符合分布
    
    def score(self, memory_pooled, z_tac_predicted, z_tac_current):
        energy = self.forward(memory_pooled, z_tac_predicted, z_tac_current)
        return torch.sigmoid(-energy)  # 转为[0,1]的分数
```

**训练方式**: 用 **Noise Contrastive Estimation (NCE)** 或 **Ranking NCE**

```python
# Positive: 专家数据中的 (state, future_tactile) pair
# Negative: 构造方式 (关键!)

def construct_negatives(z_tac_positive, batch_size):
    negatives = []
    
    # Type 1: 高斯扰动 (模拟小偏差)
    noise = torch.randn_like(z_tac_positive) * 0.5
    negatives.append(z_tac_positive + noise)
    
    # Type 2: 放大版 (模拟碰撞 — 力过大)
    scale = 2.0 + torch.rand(batch_size, 1) * 3.0  # 2x~5x
    negatives.append(z_tac_positive * scale)
    
    # Type 3: 时间错位 (模拟时序不匹配)
    shift = torch.randint(5, 20, (1,)).item()
    negatives.append(torch.roll(z_tac_positive, shift, dims=0))
    
    # Type 4: 其他episode的触觉 (模拟完全不相关的接触)
    perm = torch.randperm(batch_size)
    negatives.append(z_tac_positive[perm])
    
    # Type 5: 零触觉 (模拟脱离接触)
    negatives.append(torch.zeros_like(z_tac_positive))
    
    return negatives

# Ranking NCE Loss (灵感自 Singh et al., 2023)
def ranking_nce_loss(scorer, memory, z_tac_positive, negatives_list):
    pos_energy = scorer(memory, z_tac_positive, z_tac_current)
    
    # positive应该能量最低(最正常)
    loss = 0
    for z_neg in negatives_list:
        neg_energy = scorer(memory, z_neg, z_tac_current)
        # margin ranking: positive能量应该比negative低margin以上
        loss += F.relu(pos_energy - neg_energy + margin).mean()
    
    return loss
```

**Phase-Aware设计**:

```
接近阶段 (phase=approach): 正常触觉 ≈ 零/微小 → 大位移是异常
接触阶段 (phase=contact):  正常触觉 = 适度位移 → 零位移或过大位移是异常  
操作阶段 (phase=manipulate): 正常触觉 = 稳定范围 → 突变是异常
```

phase_encoder自动从当前触觉latent推断阶段，Energy function条件在phase上，所以同一个触觉在不同阶段的"正常度"评分不同。

---

### 3.3 Temporal Dynamics Score (S_TDS)

**核心思想**: 从当前触觉到预测未来触觉的**变化过程**必须物理合理。物理约束包括:
1. **连续性**: 触觉不能"瞬移"，变化必须有界
2. **平滑性**: 正常接触变化是渐进的，突变表示碰撞
3. **方向一致性**: action方向与触觉变化方向应该一致

灵感来源: OmniVTA的LTD (Latent Tactile Differential) + SGCNet的HMM状态转移约束

```python
class TemporalDynamicsScorer(nn.Module):
    def __init__(self, tac_latent_dim, action_dim, chunk_size):
        super().__init__()
        self.delta_proj = nn.Linear(tac_latent_dim, 128)  # 触觉变化
        self.action_proj = nn.Linear(action_dim * chunk_size, 128)  # action
        self.current_proj = nn.Linear(tac_latent_dim, 128)  # 当前触觉
        
        # 子分数1: 变化幅度合理性
        self.magnitude_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        # 子分数2: action-tactile方向一致性
        self.direction_head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        # 子分数3: 相对于当前状态的变化合理性
        self.transition_head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
    
    def forward(self, z_tac_current, z_tac_predicted, action):
        # OmniVTA的核心洞察: differential term最有信息量
        delta = z_tac_predicted - z_tac_current  # 触觉变化量
        
        d = self.delta_proj(delta)
        a = self.action_proj(action.flatten(-2))
        c = self.current_proj(z_tac_current)
        
        # 子分数1: 变化幅度是否在合理范围
        s_magnitude = self.magnitude_head(d)
        
        # 子分数2: action方向和触觉变化方向是否一致
        s_direction = self.direction_head(torch.cat([d, a], dim=-1))
        
        # 子分数3: 从当前状态出发的变化是否合理
        s_transition = self.transition_head(torch.cat([c, d], dim=-1))
        
        return (s_magnitude + s_direction + s_transition) / 3.0
```

**训练方式**:

```python
# 从专家数据中提取正样本:
# (z_tac_t, z_tac_{t+h}, action_t) → label = 1 (合理的时序变化)

# 构造负样本 (物理不合规的变化):

# 碰撞模拟: delta放大 → 物理不合规(变化过快)
delta_gt = z_tac_future - z_tac_current
z_collision = z_tac_current + delta_gt * random(3, 10)  # 3-10倍放大

# 反向变化: delta取反 → 方向不一致
z_reverse = z_tac_current - delta_gt

# 突变: 插入与当前无关的触觉 → 不连续
z_discontinuous = random_sample_from_other_episodes()
```

**物理合规保证**:
- `s_magnitude` 学习了什么是"正常"的变化幅度 → 碰撞(过大)和脱落(过小)都会得低分
- `s_direction` 学习了action和触觉变化的因果方向关系 → 向物体移动但触觉减小→不合理
- `s_transition` 学习了从当前状态合理的转移范围 → 不可能从"无接触"直接跳到"强力接触"

---

### 3.4 Cross-Modal Agreement Score (S_CMA)

**核心思想**: 预测的触觉和视觉观测应该一致。如果视觉显示物体没接触，但触觉预测有大位移，说明预测有问题。

灵感来源: TouchGuide的多模态CPM + ReTac-ACT的state-gated fusion + VTAM的visual dominance regularization

```python
class CrossModalAgreementScorer(nn.Module):
    def __init__(self, vision_dim, tac_latent_dim):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, 128)
        self.tac_proj = nn.Linear(tac_latent_dim, 128)
        
        # 学一个跨模态兼容性函数
        self.compatibility = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    
    def forward(self, vision_features, z_tac_predicted):
        v = self.vision_proj(vision_features)
        t = self.tac_proj(z_tac_predicted)
        return self.compatibility(torch.cat([v, t], dim=-1))
```

**训练**: 正样本 = 同一时刻的真实(vision, tactile) pair；负样本 = 跨样本配对的(vision_i, tactile_j)。

---

## 四、PD-CQV总体架构

```python
class PhysicsDecomposedCQV(nn.Module):
    """Physics-Decomposed Contact Quality Verifier"""
    def __init__(self, config):
        super().__init__()
        self.fic = ForwardInverseConsistency(...)   # S_FIC
        self.dcs = DistributionalComplianceScorer(...)  # S_DCS  
        self.tds = TemporalDynamicsScorer(...)       # S_TDS
        self.cma = CrossModalAgreementScorer(...)     # S_CMA
        
        # 可学习的组合权重 (不是简单平均!)
        self.combiner = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        # 同时输出分解分数 (可解释性)
    
    def forward(self, memory, vision_feat, z_tac_current, z_tac_predicted, action, 
                foresight_model, inverse_head):
        s_fic = self.fic.score(memory, z_tac_current, action, foresight_model, inverse_head)
        s_dcs = self.dcs.score(memory, z_tac_predicted, z_tac_current)
        s_tds = self.tds(z_tac_current, z_tac_predicted, action)
        s_cma = self.cma(vision_feat, z_tac_predicted)
        
        sub_scores = torch.stack([s_fic, s_dcs, s_tds, s_cma], dim=-1)  # (B, 4)
        total_score = self.combiner(sub_scores)  # (B, 1)
        
        return total_score, sub_scores  # 总分 + 分解分数
```

---

## 五、训练策略

### 整体训练流程

```
Stage 1: 预训练各模块 (独立)
  ├── TactileVAE (JEPA-style) 
  ├── InverseDynamicsHead (专家数据)
  └── DistributionalComplianceScorer (NCE on expert distribution)

Stage 2: 联合训练CQV
  ├── Forward-Inverse Cycle loss
  ├── Distributional NCE loss  
  ├── Temporal dynamics BCE loss
  ├── Cross-modal agreement loss
  └── Total CQV ranking loss (端到端)

Stage 3: 与DP联合微调
  ├── DP的DDPM loss
  ├── Foresight的latent prediction loss
  └── CQV作为auxiliary loss或RL reward
```

### CQV总Loss

```python
L_CQV = (
    λ_fic * L_cycle_consistency        # Forward-Inverse cycle
    + λ_dcs * L_distributional_nce     # 分布合规NCE
    + λ_tds * L_temporal_dynamics_bce  # 时序动力学
    + λ_cma * L_cross_modal_agreement  # 跨模态一致性
    + λ_rank * L_ranking               # 总分排序loss (端到端)
)
```

**关键**: `L_ranking` 是端到端的Ranking Loss，保证总分的排序正确:

```python
def ranking_loss(cqv, memory, vision, z_tac_cur, expert_action, foresight, inverse_head):
    # Expert pair
    z_expert = foresight(memory, expert_action)
    score_expert, _ = cqv(memory, vision, z_tac_cur, z_expert, expert_action, ...)
    
    # Perturbed pairs (多种负样本)
    scores_negative = []
    for noise_level in [0.1, 0.3, 0.5, 1.0, 2.0]:
        a_perturbed = expert_action + noise_level * torch.randn_like(expert_action)
        z_perturbed = foresight(memory, a_perturbed) 
        score_neg, _ = cqv(memory, vision, z_tac_cur, z_perturbed, a_perturbed, ...)
        scores_negative.append(score_neg)
    
    # Margin ranking: expert应该比所有negative分数高
    loss = 0
    for s_neg in scores_negative:
        loss += F.relu(margin - score_expert + s_neg).mean()
    
    return loss
```

### TouchGuide的关键教训: Noise Pretraining

TouchGuide论文中最重要的发现: **不做noise pretraining时成功率39%，做了之后62%** (+23% absolute)。

原因: 推理时CQF接收的是DP去噪过程中的**noisy action**，不是clean action。如果CQF只在clean action上训练，推理时遇到noisy action就失效。

应对方案:

```python
# 训练CQV时，对action加与DP一样的noise schedule
def add_diffusion_noise(action, noise_scheduler):
    timestep = sample_geometric_distribution()  # 和TouchGuide一样用几何分布
    noise = torch.randn_like(action)
    noisy_action = noise_scheduler.add_noise(action, noise, timestep)
    return noisy_action, timestep
```

---

## 六、推理时两种使用模式

### Mode A: Best-of-K Reranking (主方法)

```python
def inference_reranking(observation, dp, foresight, cqv, K=16):
    memory, vision_feat, z_tac_current = encode(observation)
    
    # 1. DP batch生成K个候选
    candidates = dp.sample(memory, num_samples=K)  # (K, chunk_size, action_dim)
    
    # 2. Batch foresight
    z_tac_futures = foresight(
        memory.expand(K, -1, -1), 
        candidates
    )  # (K, tac_latent_dim)
    
    # 3. Batch scoring
    total_scores, sub_scores = cqv(
        memory.expand(K, -1, -1),
        vision_feat.expand(K, -1),
        z_tac_current.expand(K, -1),
        z_tac_futures,
        candidates,
        ...
    )  # total: (K, 1), sub: (K, 4)
    
    # 4. 选最优
    best_idx = total_scores.argmax()
    best_action = candidates[best_idx]
    
    # 5. (可选) 输出分解分数用于调试/分析
    # sub_scores[best_idx] → [S_FIC, S_DCS, S_TDS, S_CMA]
    
    return best_action
```

**推理效率分析**:
- Encoder: 1次 (~5ms)
- DP生成K=16: batch并行, 1次forward (~与K=1差不多)
- Foresight K次: batch并行, 1次forward (~2ms)
- CQV K次: batch并行, 极轻量 (~<1ms)
- 总计: 与单次DP推理相当，额外开销<10%

### Mode B: Gradient Guidance (创新加分)

```python
def inference_guidance(observation, dp, foresight, cqv, guidance_scale=5.0):
    memory, vision_feat, z_tac_current = encode(observation)
    
    # 标准DP去噪 + CQV梯度引导
    x_t = torch.randn(chunk_size, action_dim)  # 起始噪声
    
    for t in reversed(range(T)):  # T个去噪步骤
        x_t.requires_grad_(True)
        
        # 标准去噪
        eps_pred = dp.noise_predictor(x_t, t, memory)
        
        # CQV梯度 (只在最后30%步骤引导, 参考TouchGuide)
        if t < T * 0.3:
            z_pred = foresight(memory, x_t)
            score, _ = cqv(memory, vision_feat, z_tac_current, z_pred, x_t, ...)
            grad = torch.autograd.grad(score.sum(), x_t)[0]
            
            # DynaGuide式引导
            eps_pred = eps_pred - guidance_scale * (1 - alpha_bar[t]).sqrt() * grad
        
        x_t = dp.step(eps_pred, t, x_t)
        x_t = x_t.detach()
    
    return x_t
```

---

## 七、物理合规保证机制

### "不能碰撞"
- **S_DCS**: 碰撞产生的触觉远超专家分布 → 能量极高 → 分数极低
- **S_TDS.magnitude**: 碰撞 = 触觉突变 = 变化幅度超界 → 分数极低
- **硬约束** (可选): 如果 `||z_tac_predicted|| > max_threshold`，直接discard该候选，不参与reranking

### "物理合规"
- **S_TDS.direction**: 确保action方向与触觉变化方向因果一致
- **S_TDS.transition**: 确保触觉状态转移连续（不能跳跃）
- **S_FIC**: cycle一致性确保action-tactile pair是物理上因果匹配的

### "模态数据一致性"
- **S_CMA**: 显式检查vision和predicted tactile的兼容性
- 额外: 可以加入**temporal coherence check** — 连续预测帧之间的触觉变化应平滑

---

## 八、与现有方法的创新性对比

| 对比维度 | TouchGuide | DynaGuide | PPGuide | **Ours (PD-CQV)** |
|---------|-----------|-----------|---------|-------------------|
| 打分维度 | 1维(cosine) | 1维(value) | 1维(success prob) | **4维(物理分解)** |
| 可解释性 | 黑盒 | 黑盒 | 黑盒 | **可解释(知道哪个子维度低)** |
| 触觉预测 | 不预测 | 不涉及触觉 | 不涉及触觉 | **Latent触觉预测** |
| Forward-Inverse | 无 | 仅Forward | 无 | **Cycle验证** |
| 物理约束 | 无 | 无 | 无 | **时序动力学+分布合规** |
| 模态一致性 | V-T concat | 仅Vision | 仅Vision | **显式跨模态验证** |
| 阶段感知 | 无 | 无 | 无 | **Phase-conditioned** |

**论文可讲的创新点**:
1. **第一个**将Contact Quality分解为物理可解释子维度的工作
2. **第一个**在触觉打分中引入Forward-Inverse Cycle Consistency验证
3. **第一个**将action-conditioned tactile foresight + multi-dimensional scoring + diffusion reranking三者结合

---

## 九、消融实验设计

| 实验编号 | 配置 | 验证目标 |
|---------|------|---------|
| A1 | DP baseline (no tactile) | 纯视觉基线 |
| A2 | DP + 触觉concat | naive触觉融合 |
| A3 | DP + single-score CQF (TouchGuide-style) | 单维度打分基线 |
| A4 | DP + S_FIC only | 单独Forward-Inverse Consistency |
| A5 | DP + S_DCS only | 单独Distributional Compliance |
| A6 | DP + S_TDS only | 单独Temporal Dynamics |
| A7 | DP + S_CMA only | 单独Cross-Modal Agreement |
| A8 | DP + PD-CQV (equal weights) | 四维等权组合 |
| A9 | DP + PD-CQV (learned combiner) | 四维学习组合 (完整方法) |
| A10 | DP + PD-CQV + gradient guidance | 梯度引导模式 |

额外ablation:
- K的选择: K=1,4,8,16,32
- Noise pretraining: 有 vs 无
- 负样本构造: 只Type1 vs 全部5种 vs 真实失败数据

---

## 十、分阶段实现路线

```
第一步 (1周): 先实现最简版CQV
  - 只用S_DCS (分布合规) + S_TDS (时序动力学)
  - 用MLP而不是能量函数
  - 验证: 能否在离线数据上区分expert vs perturbed action

第二步 (1周): 加入S_FIC (Forward-Inverse)
  - 训练InverseDynamicsHead
  - 验证: cycle consistency是否给正确action更高分

第三步 (1周): 完整PD-CQV + Reranking
  - 加入S_CMA + combiner
  - 集成到DP推理中做best-of-K
  - 验证: reranking后的action质量是否提升

第四步 (可选): Gradient Guidance
  - 在DP去噪中注入CQV梯度
  - 对比reranking vs guidance的效果
```

---

## 十一、文献参考

### 核心参考
- **TouchGuide** (Zhang et al., 2026): CPM contrastive scoring + noise pretraining, arXiv:2601.20239
- **DynaGuide** (Du & Song, 2025): Dynamics model gradient guidance for diffusion, arXiv:2506.13922
- **PPGuide** (Wang et al., 2026): Performance predictor gradient steering, arXiv:2603.10980
- **OmniVTA** (Zheng et al., 2026): LTD encoder + reflexive controller + statistical anomaly detection, arXiv:2603.19201
- **EDMP** (Saha et al., 2023): Ensemble of cost functions for diffusion guidance, arXiv:2309.11414

### Forward-Inverse Consistency
- **JIF** (Khandate et al., 2025): Joint inverse and forward dynamics pre-training, arXiv:2503.12297
- **DynaMo** (Cui et al., 2024): In-domain dynamics pre-training, arXiv:2409.12192
- **UniForce** (Chen et al., 2026): Newton's third law as constraint, arXiv:2602.01153

### Energy-Based / Distribution Scoring
- **IBC** (Florence et al., 2021): Energy-based model for action scoring, arXiv:2109.00137
- **Ranking NCE** (Singh et al., 2023): Action ranking via energy models, arXiv:2309.05803
- **NEAR** (Diwan et al., 2025): Noise-conditioned energy annealed rewards, arXiv:2501.14856

### 触觉接触状态
- **SGCNet** (Wang et al., 2025): HMM state transition for grip phase classification
- **ReTac-ACT** (Ruan et al., 2026): State-gated V-T fusion, arXiv:2603.09565
- **VT-WM** (Higuera et al., 2026): Visuo-tactile world models, arXiv:2602.06001

### 其他重要参考
- **Touch Dreaming** (Niu et al., 2026): Latent tactile prediction 30% better than raw
- **CLaD** (KAIST, 2026): JEPA-style EMA target encoder for latent foresight
- **RDP** (Xue et al., 2025): Slow-fast visual-tactile policy, arXiv:2503.02881
- **VTAM** (Yuan et al., 2026): Tactile regularization prevents visual dominance
- **Diffuser** (Janner et al., 2022): Value-guided diffusion planning, ICML
- **Contact-Grounded Policy** (Xu et al., 2026): Joint (action, tactile) prediction + consistency mapping
