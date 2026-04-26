# 评分与Reranking：从数据到部署的完整深度剖析

> 日期: 2026-04-27
> 前置依赖: TactileVAE (已有) + Foresight in latent space + DP baseline

---

## 一、从数据出发的深度分析

### 1.1 你拥有什么

所有数据集均已分为 success / bounce 两个子目录，总计约 1200+ episodes。

| 类型 | 特征 | 数量(估) | CQF训练价值 |
|------|------|----------|-------------|
| 一次成功插入 | 平滑轨迹，触觉单调递增 | ~600 ep | 正样本来源 |
| 碰到边缘+重试1-2次 | 同一episode含失败和成功 | ~600 ep | **正负样本同时来源** |

### 1.2 bounce episode的时间结构（这是设计CQF的基础）

```
一个典型 bounce episode (300帧, ~250帧有效):

帧号:   0        80       120   140    160   180     220      250
        |--------|--------|-----|------|-----|-------|--------|
        approach  descend  CONTACT  BOUNCE  LIFT   re-descend  SUCCESS
                           edge    spike↑  up↑     插入成功
        
marker magnitude:
        ~0.5     ~1.2     ~3.0   ↑8.0↓  ~1.0    ~2.5    ↑5.0→稳定
```

关键帧分类:
- **帧 0-80 (approach)**: 无接触，触觉≈0，action差异无影响 → 对CQF训练**价值低**
- **帧 80-120 (descend)**: 接近插座，触觉开始微弱 → **中等价值**
- **帧 120-140 (contact edge → BOUNCE)**: 碰到边缘产生spike → **最高价值的负样本!**
  - 这里的action导致了碰撞
  - 同一个state附近，success episode的action能成功插入
- **帧 140-180 (lift + reposition)**: 抬起重新定位 → 恢复阶段，价值低
- **帧 180-250 (re-descend → SUCCESS)**: 修正后成功插入 → **高价值正样本**
  - 和帧120-140的state很相似，但action不同且成功了

### 1.3 核心洞察：bounce episode给了我们**同一state下的因果对比**

这是比人造负样本好100倍的数据。因为:
- 人造负样本（加噪声）: 噪声太大→明显OOD；噪声太小→和正样本无法区分
- 自然bounce数据: 真实世界中"差一点就对了"的action，是最有信息量的hard negative

### 1.4 数据是否足够?

**足够**。理由:
1. ~600个bounce episode × 每个约20帧pre-bounce = **12000个自然负样本**
2. ~600个success episode × 每个约50帧contact phase = **30000个正样本**
3. 对于一个轻量MLP scorer（~500K参数），这个量级绑绑有余
4. TouchGuide论文只用了几十个demonstration episode就训练出了有效的CPM

**可选的额外采集（锦上添花）**:
- 故意错误角度的插入（5-10个episode即可）→ 提供极端负样本
- 不同速度的插入 → 增加多样性
- 但**不是必须的**，现有数据已足够

---

## 二、CQF到底在评什么？——第一性原理分析

### 2.1 Reranking的本质

你有K=16个候选action。假设其中：
- 3-4个会导致碰撞
- 5-6个会成功但不够精准（插入偏了一点）
- 2-3个几乎完美
- 其余介于之间

CQF不需要精确打分，只需要**排序正确**：完美 > 不够精准 > 碰撞。

这意味着CQF的设计目标是**ranking accuracy**，不是**absolute score accuracy**。

### 2.2 CQF能获取的信息（推理时）

```
推理时CQF的输入:
├── memory (from shared encoder): 编码了当前视觉+触觉+qpos的综合表征
├── z_tac_current: 当前触觉latent (from TactileVAE encoder)
├── action_candidate: DP生成的候选action chunk (20, 7)
└── z_tac_predicted: Foresight(memory, action_candidate) 的输出
```

推理时CQF**无法获取**的信息:
- 未来的真实触觉（还没执行action）
- 其他候选的信息（各候选独立评分）
- 历史action执行记录（只有当前观测）

### 2.3 这些输入中哪些信号最有用？

**最强信号: z_tac_predicted（预测的未来触觉）**
- 如果Foresight准确，好action → 正常触觉预测，坏action → 异常触觉预测
- 你的实验已证明：bounce时刻不同action产生截然不同的触觉预测
- 这是CQF最应该依赖的信号

**次强信号: delta = z_tac_predicted - z_tac_current（触觉变化量）**
- OmniVTA的LTD encoder核心发现：变化量比绝对值更有信息量
- 碰撞 = 突然的大delta；成功插入 = 渐进的小delta

**辅助信号: action本身**
- 作为额外的一致性检查：如果action明显OOD（噪声太大），即使Foresight预测正常也应低分
- 防止Foresight出错时CQF失灵

**上下文信号: memory + z_tac_current**
- 告诉CQF当前在什么阶段（approach/contact/insertion）
- 同样的delta在不同阶段意义不同

### 2.4 关键设计决策：CQF应该多复杂？

经过深度思考，我认为**不应该**用之前设计的4-head PD-CQV作为第一版本。理由:

1. **数据效率**: 4个head需要4组loss、4种负样本、4套超参，训练复杂度高
2. **错误传播**: 如果某个head训练差了（比如InverseHead），会拖累整体score
3. **实际瓶颈**: TouchGuide证明一个简单的cosine scorer + 好的训练数据就能+23%成功率
4. **渐进式开发**: 先证明简单scorer有效，再加complexity

**我的结论: 做两层设计**
- **Core Scorer（必须实现）**: 一个精心设计的轻量网络，直接从数据学ranking
- **Physics Heads（论文创新）**: 作为auxiliary sub-scores，提供可解释性和少量额外增益

---

## 三、Core Scorer的具体设计

### 3.1 架构

```python
class ContactQualityScorer(nn.Module):
    """
    核心打分器: 从(state, action, current_tac, predicted_tac)评估接触质量。
    
    设计原则:
    1. 最重要的信号是 delta = z_pred - z_cur (触觉变化量)
    2. action提供因果一致性检查
    3. 当前state提供phase上下文
    4. 最终输出标量score，高=好，低=碰撞/失败
    """
    def __init__(self, 
                 tac_latent_dim=144,      # TactileVAE: 16ch × 3×3 = 144
                 action_dim=7,
                 chunk_size=20,
                 memory_dim=512,          # TransformerEncoder hidden_dim
                 hidden=256):
        super().__init__()
        
        # === Branch 1: Tactile Analysis (最重要的分支) ===
        # 输入: z_current(144) + z_predicted(144) + delta(144) = 432
        self.tac_encoder = nn.Sequential(
            nn.Linear(tac_latent_dim * 3, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        
        # === Branch 2: Action Analysis ===
        # 输入: action chunk展平 (20×7=140)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim * chunk_size, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        
        # === Branch 3: State Context ===
        # 输入: memory pool后 (512) + 当前qpos (7)
        self.state_encoder = nn.Sequential(
            nn.Linear(memory_dim + action_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        
        # === Cross-check: action与tactile delta的一致性 ===
        # 这一层检查"action方向"和"触觉变化方向"是否match
        self.cross_check = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        
        # === Final Scorer ===
        # 4路信息融合 → 标量score
        self.scorer = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
    
    def forward(self, memory_pooled, qpos, z_tac_current, z_tac_predicted, action_chunk):
        """
        Args:
            memory_pooled: (B, 512) — shared encoder memory的pool结果
            qpos: (B, 7) — 当前关节角度
            z_tac_current: (B, 144) — 当前触觉latent (from TactileVAE)
            z_tac_predicted: (B, 144) — Foresight预测的未来触觉latent
            action_chunk: (B, 20, 7) — 候选action chunk
        Returns:
            score: (B, 1) — 接触质量分数 (未经sigmoid, raw logit)
        """
        B = memory_pooled.shape[0]
        
        # 1. 触觉分析: [当前, 预测, 变化量]
        delta = z_tac_predicted - z_tac_current
        tac_input = torch.cat([z_tac_current, z_tac_predicted, delta], dim=-1)
        h_tac = self.tac_encoder(tac_input)  # (B, 256)
        
        # 2. Action分析
        h_act = self.action_encoder(action_chunk.reshape(B, -1))  # (B, 256)
        
        # 3. 状态上下文
        h_state = self.state_encoder(torch.cat([memory_pooled, qpos], dim=-1))  # (B, 256)
        
        # 4. 交叉检查: action-tactile一致性
        h_cross = self.cross_check(torch.cat([h_act, h_tac], dim=-1))  # (B, 256)
        
        # 5. 融合打分
        h_all = torch.cat([h_tac, h_act, h_state, h_cross], dim=-1)  # (B, 1024)
        score = self.scorer(h_all)  # (B, 1)
        
        return score
```

**参数量**: ~500K，非常轻量。

**为什么这个架构合理**:
- `h_tac`捕获触觉预测的质量（最重要）
- `h_act`捕获action本身是否合理（防止Foresight出错时的fallback）
- `h_state`提供上下文（同样的delta在approach和insertion阶段意义不同）
- `h_cross`捕获action和触觉变化之间的因果关系（action向下推→触觉应增大）

### 3.2 为什么不用cosine similarity（对比TouchGuide）？

TouchGuide的CPM: `score = cosine(obs_embedding, action_embedding)`

我不用这个设计的原因:
1. TouchGuide的CPM不接收predicted tactile作为输入——它只看(当前obs, action)
2. 我们多了一路关键信号: Foresight的预测结果
3. cosine similarity是对称的，但action→tactile的关系不是对称的
4. MLP scorer可以学到更复杂的非线性关系

但TouchGuide的一个关键教训必须保留: **noise pretraining**（见下文训练部分）。

---

## 四、训练数据构造（最关键的部分）

### 4.1 数据标注流程

第一步: 自动检测每个episode的关键帧

```python
def annotate_episode(marker_offset):
    """
    输入: marker_offset (T, 9, 9, 2)
    输出: per-frame labels
        0 = approach (无接触)
        1 = contact_good (正常接触/插入)
        2 = pre_bounce (即将碰撞的帧, 核心负样本)
        3 = bounce (碰撞时刻)
        4 = recovery (碰撞后抬起恢复)
    """
    # 每帧的触觉强度
    mag = np.linalg.norm(marker_offset, axis=-1).mean(axis=(1, 2))  # (T,)
    mag_grad = np.gradient(mag)  # 一阶导数
    
    # 检测接触起始: 触觉强度首次超过阈值
    contact_threshold = np.percentile(mag[mag > 0.5], 25) if (mag > 0.5).sum() > 10 else 1.0
    contact_onset = np.argmax(mag > contact_threshold)
    
    # 检测bounce: 触觉强度突然上升后突然下降
    # 定义spike: 梯度 > 正常的3倍std
    grad_std = np.std(mag_grad[mag_grad > 0]) if (mag_grad > 0).sum() > 0 else 1.0
    spike_indices = np.where(mag_grad > 3 * grad_std)[0]
    
    # 检测spike后面是否跟着急剧下降(bounce的标志)
    bounce_frames = []
    for si in spike_indices:
        if si + 10 < len(mag):
            # spike后20帧内触觉下降超过50%
            future_min = mag[si:min(si+20, len(mag))].min()
            if future_min < mag[si] * 0.5:
                bounce_frames.append(si)
    
    # 分配标签
    labels = np.zeros(len(mag), dtype=np.int32)
    labels[:contact_onset] = 0  # approach
    labels[contact_onset:] = 1  # contact_good (默认)
    
    for bf in bounce_frames:
        # bounce前10帧 = pre_bounce (将要碰撞)
        labels[max(0, bf-10):bf] = 2  # pre_bounce
        # bounce时刻
        labels[bf:min(bf+5, len(mag))] = 3  # bounce
        # bounce后到下一次接触 = recovery
        next_contact = bf + 5
        while next_contact < len(mag) and mag[next_contact] < contact_threshold:
            next_contact += 1
        labels[bf+5:next_contact] = 4  # recovery
    
    return labels, mag, bounce_frames
```

对于已标注目录:
- `success/` 目录: 直接用，几乎所有帧都是label=0或1
- `bounce/` 目录: 自动检测bounce帧，标注2/3/4

### 4.2 训练样本构造

每个训练样本是一个五元组:

```
(memory_pooled, qpos, z_tac_current, z_tac_predicted, action_chunk) → label
```

**正样本来源（label = 1）**:

```python
# 来源1: success episode的contact phase (label=1的帧)
# 用GT action + GT future tactile
for ep in success_episodes:
    for t in frames_with_label_1(ep):
        positive = {
            'memory': encoder(obs_t),
            'qpos': qpos_t,
            'z_tac_current': tactile_vae.encode(marker_t),
            'z_tac_predicted': tactile_vae.encode(marker_{t+h}),  # GT future
            'action': action_chunk_t,
            'label': 1.0
        }

# 来源2: bounce episode中成功重试的部分 (recovery后label=1的帧)
# 同上处理，这些帧的action经过修正后成功了
for ep in bounce_episodes:
    for t in frames_after_recovery_with_label_1(ep):
        positive = { ... same structure, label=1.0 }
```

**负样本来源（label = 0）— 按重要性排序**:

```python
# === 最重要: 来源1 — 自然bounce数据 ===
# bounce episode中 pre_bounce 帧 (label=2)
# 这些帧的action真实地导致了碰撞
for ep in bounce_episodes:
    for t in frames_with_label_2(ep):
        natural_negative = {
            'memory': encoder(obs_t),
            'qpos': qpos_t,
            'z_tac_current': tactile_vae.encode(marker_t),
            'z_tac_predicted': tactile_vae.encode(marker_{t+h}),  # GT future (含碰撞)
            'action': action_chunk_t,  # 导致碰撞的真实action
            'label': 0.0
        }

# === 重要: 来源2 — Foresight预测的负样本 ===
# 用已训练的Foresight模型生成
# 对正样本的action加扰动 → 通过Foresight预测 → 得到被扰动的预测触觉
for positive_sample in positives:
    for noise_scale in [0.3, 0.5, 1.0, 2.0]:
        perturbed_action = positive_sample['action'] + noise_scale * torch.randn_like(action)
        z_tac_pred_perturbed = foresight(memory, perturbed_action)
        foresight_negative = {
            'memory': positive_sample['memory'],
            'qpos': positive_sample['qpos'],
            'z_tac_current': positive_sample['z_tac_current'],
            'z_tac_predicted': z_tac_pred_perturbed,
            'action': perturbed_action,
            'label': 0.0 if noise_scale > 0.5 else 0.3  # 小扰动给部分分数
        }

# === 有用: 来源3 — 跨episode替换 ===
# 在相似state下替换为其他episode的action
# (你的实验已证明: bounce时刻替换action影响巨大)
for ep in all_episodes:
    for t in contact_frames(ep):
        other_ep = random_other_episode()
        other_t = find_similar_state_frame(other_ep, state_t)
        cross_negative = {
            'memory': encoder(obs_t),
            'qpos': qpos_t,
            'z_tac_current': tactile_vae.encode(marker_t),
            'z_tac_predicted': foresight(memory_t, other_action),  # 用Foresight预测
            'action': other_ep.action_chunk[other_t],  # 其他episode的action
            'label': 0.0
        }
```

### 4.3 为什么这三种负样本都需要

| 负样本类型 | 模拟的真实场景 | 对CQF的训练效果 |
|-----------|--------------|---------------|
| 自然bounce | 真实的碰撞轨迹 | **最好**——学到真实的碰撞触觉pattern |
| Foresight扰动 | DP采样出的次优action | **关键**——CQF推理时看到的就是Foresight输出 |
| 跨episode替换 | 完全不匹配的action | **基础**——确保CQF不只记住特定trajectory |

**核心: 来源2（Foresight预测的负样本）最重要且最容易被忽视。**

原因: 推理时CQF接收的z_tac_predicted是Foresight的输出，不是GT。如果只用GT训练CQF，它看到的z_tac_predicted分布和推理时不同→分布不匹配→失效。

这就是TouchGuide的noise pretraining的本质——让CQF在训练时也见到"有噪声的输入"。

### 4.4 训练数据混合比例

```
每个batch (batch_size=256):
  - 40% 正样本 (success contact phase)
  - 20% 自然bounce负样本
  - 25% Foresight扰动负样本
  - 15% 跨episode替换负样本
```

用weighted sampler确保bounce帧不会被success帧稀释。

---

## 五、训练流程

### 5.1 Loss函数

**主Loss: Pairwise Ranking Loss**

```python
def ranking_loss(scorer, batch):
    """
    对每个anchor state，正样本应该得分高于负样本。
    """
    # 分离正负样本
    positives = batch[batch['label'] > 0.5]
    negatives = batch[batch['label'] < 0.5]
    
    # 计算所有正负对的分数
    score_pos = scorer(positives['memory'], positives['qpos'],
                       positives['z_cur'], positives['z_pred'], 
                       positives['action'])  # (N_pos, 1)
    score_neg = scorer(negatives['memory'], negatives['qpos'],
                       negatives['z_cur'], negatives['z_pred'],
                       negatives['action'])  # (N_neg, 1)
    
    # Pairwise margin ranking loss
    # 每个positive应该比每个negative高出margin
    margin = 0.5
    loss = 0
    n_pairs = 0
    for sp in score_pos:
        for sn in score_neg:
            loss += F.relu(margin - sp + sn)
            n_pairs += 1
    
    return loss / max(n_pairs, 1)
```

**辅助Loss: Binary Cross-Entropy**

```python
def bce_loss(scorer, batch):
    scores = scorer(batch['memory'], batch['qpos'],
                    batch['z_cur'], batch['z_pred'], batch['action'])
    return F.binary_cross_entropy_with_logits(scores, batch['label'])
```

**总Loss**:
```python
L_total = L_ranking + 0.5 * L_bce
```

为什么两个loss都要:
- Ranking loss保证排序正确（reranking核心需求）
- BCE loss提供绝对值锚定（知道什么是"绝对好"vs"绝对差"）

### 5.2 训练三阶段

```
阶段A (10 epochs): 用GT future tactile训练
  - 正样本: (state, expert_action, gt_future_tac) from success
  - 负样本: (state, bounce_action, gt_future_tac) from bounce
  - 目的: 让CQF先学会区分"正常触觉"vs"碰撞触觉"

阶段B (20 epochs): 混入Foresight预测结果
  - 正样本: 50% GT + 50% Foresight(expert_action) 
  - 负样本: 自然bounce + Foresight(perturbed_action)
  - 目的: 让CQF适应Foresight的输出分布

阶段C (10 epochs): 全部用Foresight预测结果
  - 所有z_tac_predicted都来自Foresight模型，不用GT
  - 目的: 与推理时的分布完全一致
```

这个渐进训练很重要，直接从阶段C开始效果会差，因为Foresight本身有误差，CQF需要先在"干净"数据上建立判断标准。

### 5.3 数据增强

```python
# 1. Action noise (模拟DP的sampling噪声)
action_augmented = action + 0.05 * torch.randn_like(action)

# 2. Tactile noise (模拟TactileVAE的编码噪声)  
z_tac_augmented = z_tac + 0.02 * torch.randn_like(z_tac)

# 3. Temporal jitter (采样时间点±2帧)
t_jittered = t + random.randint(-2, 2)
```

---

## 六、Reranking推理流程

### 6.1 完整推理pipeline

```python
class TacDreamInference:
    def __init__(self, dp_model, encoder, foresight, cqf, tactile_vae, K=16):
        self.dp = dp_model           # DP (EMA版本)
        self.encoder = encoder       # Shared Vision+Tactile encoder
        self.foresight = foresight   # Latent Foresight Transformer
        self.cqf = cqf               # Contact Quality Scorer
        self.tac_vae = tactile_vae   # TactileVAE (frozen)
        self.K = K
    
    @torch.no_grad()
    def select_action(self, observation):
        """
        observation: dict containing:
            'images': {'global': (1,3,H,W), 'wrist': (1,3,H,W)}
            'qpos': (1, 7)
            'marker_offset': (1, T_window, 9, 9, 2) or (1, 9, 9, 2)
        """
        device = next(self.dp.parameters()).device
        
        # ===== Step 1: Encode current observation =====
        # Vision → memory tokens
        memory = self.encoder(observation)  # (N_tokens, 1, 512)
        memory_pooled = memory.mean(dim=0)  # (1, 512)
        
        # Tactile → latent
        marker = observation['marker_offset']  # (1, T, 9, 9, 2)
        z_tac_current = self.tac_vae.encode(marker)  # (1, 144)
        
        qpos = observation['qpos']  # (1, 7)
        
        # ===== Step 2: DP生成K个候选 (batch并行) =====
        # 不需要K次forward——batch noise即可
        noise = torch.randn(self.K, self.dp.action_horizon, 7, device=device)
        
        # DDPM去噪: 100步
        noisy_actions = noise
        for t in self.dp.scheduler.timesteps:
            eps_pred = self.dp.noise_net(
                noisy_actions,
                t.expand(self.K),
                memory_pooled.expand(self.K, -1),
            )
            noisy_actions = self.dp.scheduler.step(eps_pred, t, noisy_actions).prev_sample
        
        candidates = noisy_actions  # (K, chunk_size, 7)
        
        # ===== Step 3: Foresight预测每个候选的未来触觉 (batch并行) =====
        memory_K = memory.expand(-1, self.K, -1)  # (N_tokens, K, 512)
        z_tac_futures = self.foresight(memory_K, candidates)  # (K, 144)
        
        # ===== Step 4: CQF打分 (batch并行) =====
        scores = self.cqf(
            memory_pooled.expand(self.K, -1),     # (K, 512)
            qpos.expand(self.K, -1),              # (K, 7)
            z_tac_current.expand(self.K, -1),     # (K, 144)
            z_tac_futures,                         # (K, 144)
            candidates,                            # (K, chunk_size, 7)
        )  # (K, 1)
        
        # ===== Step 5: 选最优 =====
        best_idx = scores.argmax(dim=0).item()
        best_action = candidates[best_idx]  # (chunk_size, 7)
        
        # 调试信息
        info = {
            'scores': scores.cpu().numpy(),
            'best_score': scores[best_idx].item(),
            'worst_score': scores.min().item(),
            'score_spread': (scores.max() - scores.min()).item(),
            'best_idx': best_idx,
        }
        
        return best_action, info
```

### 6.2 推理效率

```
各模块耗时 (估算, 单GPU):
  Encoder (1次):           ~5ms
  DP去噪 (K=16, batch):   ~50ms (与K=1差不多, batch parallel)
  Foresight (K=16, batch): ~3ms
  CQF (K=16, batch):       ~1ms
  ──────────────────────────
  总计:                     ~59ms

对比不做reranking:
  Encoder + DP:             ~55ms
  额外开销:                 ~4ms (约7%)
```

### 6.3 安全阈值机制

```python
# 可选: 如果所有候选分数都低于阈值, 不执行任何动作
SAFETY_THRESHOLD = 0.3

if scores.max() < SAFETY_THRESHOLD:
    # 所有候选都可能导致碰撞
    # 选项1: 回退到最保守的action(最小位移)
    min_displacement = candidates.norm(dim=-1).sum(dim=-1)
    safest = candidates[min_displacement.argmin()]
    return safest, {'fallback': True}
    
    # 选项2: 暂停执行, 等待human intervention
    # return None, {'emergency_stop': True}
```

---

## 七、完整Pipeline训练顺序

```
  ┌──────────────────────────────────────────────────┐
  │ Stage 0: TactileVAE (已完成/进行中)              │
  │  数据: 所有episode的marker_offset                │
  │  输出: tactile encoder (marker→z, 144dim)        │
  └──────────────┬───────────────────────────────────┘
                 │
  ┌──────────────▼───────────────────────────────────┐
  │ Stage 1: DP Baseline训练                         │
  │  数据: success episodes (vision + qpos + action) │
  │  输出: 训练好的DP model (能生成多样化candidates) │
  └──────────────┬───────────────────────────────────┘
                 │
  ┌──────────────▼───────────────────────────────────┐
  │ Stage 2: Foresight in Latent Space               │
  │  数据: 所有episodes (action + tactile latent)    │
  │  输出: Foresight model (action→z_tac_future)     │
  │  关键: 必须用success和bounce数据都训练           │
  │        Foresight需要在bounce数据上也能预测       │
  └──────────────┬───────────────────────────────────┘
                 │
  ┌──────────────▼───────────────────────────────────┐
  │ Stage 3: CQF训练 (本文档的核心)                  │
  │  数据: success (正) + bounce (负) + perturbed     │
  │  前置: 需要Stage 0,1,2的模型做数据构造           │
  │  输出: ContactQualityScorer                      │
  │                                                  │
  │  训练三阶段:                                      │
  │    A: GT tactile → 学基本判断                     │
  │    B: 混入Foresight预测 → 适应噪声               │
  │    C: 全Foresight预测 → 匹配推理分布             │
  └──────────────┬───────────────────────────────────┘
                 │
  ┌──────────────▼───────────────────────────────────┐
  │ Stage 4: 集成验证                                │
  │  离线: 在held-out数据上验证ranking accuracy      │
  │  在线: 真机实验 DP vs DP+Reranking               │
  └──────────────────────────────────────────────────┘
```

---

## 八、离线验证方法（在上真机之前）

上机之前必须先做离线验证，确认CQF确实有效。

### 8.1 Ranking Accuracy测试

```python
def offline_ranking_test(cqf, foresight, encoder, tac_vae, test_episodes):
    """
    对每个test episode:
    1. 在contact phase取一帧作为query state
    2. expert action = GT action (应该得最高分)
    3. 生成多个perturbed actions
    4. 检查CQF是否把expert排在第一
    """
    correct = 0
    total = 0
    
    for ep in test_episodes:
        for t in contact_frames(ep):
            state = encode(ep, t)
            expert_action = ep.action_chunk[t]
            
            # 生成候选 (expert + 15个perturbed)
            candidates = [expert_action]
            for _ in range(15):
                noise = torch.randn_like(expert_action) * 0.5
                candidates.append(expert_action + noise)
            candidates = torch.stack(candidates)  # (16, chunk, 7)
            
            # Foresight + CQF
            z_preds = foresight(state, candidates)
            scores = cqf(state, z_tac_cur, z_preds, candidates)
            
            # Expert应该排第一
            if scores.argmax() == 0:
                correct += 1
            total += 1
    
    return correct / total  # 应该 > 50% (随机是 6.25% = 1/16)
```

### 8.2 Bounce Detection测试

```python
def bounce_detection_test(cqf, foresight, encoder, tac_vae, bounce_episodes):
    """
    在bounce episode的pre-bounce帧:
    - expert action (导致碰撞的) 应该得低分
    - 如果我们用success episode同一位置的action替代，应该得高分
    """
    for ep in bounce_episodes:
        for t in pre_bounce_frames(ep):
            state = encode(ep, t)
            bounce_action = ep.action_chunk[t]  # 导致碰撞的action
            
            # 找一个success episode在相似state的action
            good_action = find_success_action_for_similar_state(state, success_episodes)
            
            score_bounce = cqf(state, z_cur, foresight(state, bounce_action), bounce_action)
            score_good = cqf(state, z_cur, foresight(state, good_action), good_action)
            
            # good应该 > bounce
            assert score_good > score_bounce, "CQF failed to detect bounce!"
```

### 8.3 合格标准

进行真机实验的前提条件:
1. **Ranking Accuracy > 60%** (随机baseline是6.25%):
   在test set上，expert action在16个候选中排第一的比例
2. **Bounce Detection > 80%**: 
   在bounce episode的pre-bounce帧，CQF给碰撞action的分数低于替代good action
3. **Score Spread > 0.3**:
   K=16个候选的分数最大值-最小值平均超过0.3（说明CQF能区分好坏）

---

## 九、Physics Heads（论文创新加分项）

在Core Scorer训练稳定后，加入Physics Heads作为**auxiliary sub-scores**:

### 9.1 Temporal Smoothness Score

```python
class TemporalSmoothnessHead(nn.Module):
    """检查触觉变化幅度是否在物理合理范围内"""
    def __init__(self, tac_dim=144):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tac_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, delta):
        # delta = z_predicted - z_current
        return self.net(delta)
```

训练: 用success episode的delta作为正样本(label=1), bounce时刻的delta作为负样本(label=0)。

### 9.2 Distributional Compliance Score

```python
class DistributionalHead(nn.Module):
    """检查预测触觉是否在正常分布范围内"""
    def __init__(self, tac_dim=144, state_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tac_dim + state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, z_predicted, state):
        return self.net(torch.cat([z_predicted, state], dim=-1))
```

训练: 正样本=专家分布内的(state, tac); 负样本=放大/缩小/随机替换的(state, tac)。

### 9.3 Extended CQF (论文版本)

```python
class ExtendedCQF(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.core_scorer = ContactQualityScorer(...)
        self.temporal_head = TemporalSmoothnessHead(...)
        self.distributional_head = DistributionalHead(...)
        
        # 可学习融合
        self.combiner = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1)
        )
    
    def forward(self, memory, qpos, z_cur, z_pred, action):
        s_core = self.core_scorer(memory, qpos, z_cur, z_pred, action)
        s_temporal = self.temporal_head(z_pred - z_cur)
        s_distributional = self.distributional_head(z_pred, memory)
        
        sub_scores = torch.cat([s_core, s_temporal, s_distributional], dim=-1)
        total = self.combiner(sub_scores)
        
        return total, sub_scores  # 总分 + 可解释子分数
```

论文消融实验:
- Core only vs Core + Temporal vs Core + Distributional vs Full
- 展示每个head的独立贡献

---

## 十、额外数据采集建议

**结论: 现有数据已足够，但以下额外数据可以进一步提升:**

| 数据类型 | 数量 | 用途 | 优先级 |
|---------|------|------|--------|
| 故意大角度偏移的失败插入 | 10-20 ep | 极端负样本 | 低 |
| 不同速度的成功插入 | 10-20 ep | 增加正样本多样性 | 低 |
| 部分插入后滑脱 | 5-10 ep | 训练"滑动检测" | 中 |
| 力过大导致传感器形变的 | 5 ep | 训练"过力检测" | 低 |

**真正重要的不是采集更多数据，而是充分利用bounce数据中的自然负样本。**

---

## 十一、实现优先级清单

```
第1周: 数据标注pipeline
  ├── 实现 annotate_episode() 对所有bounce episode做帧级标注
  ├── 构造 CQFDataset (正负样本混合，weighted sampling)
  └── 验证: 可视化标注结果，确认bounce帧检测正确

第2周: Core Scorer + 阶段A训练 (GT tactile)
  ├── 实现 ContactQualityScorer
  ├── 用GT future tactile训练
  └── 验证: offline ranking accuracy和bounce detection

第3周: 阶段B+C训练 (混入Foresight)
  ├── 用已有的Foresight模型生成z_tac_predicted
  ├── 混合训练 → 适应Foresight输出分布
  └── 验证: 全Foresight输入的ranking accuracy

第4周: Reranking集成 + 离线验证
  ├── 集成到DP推理pipeline
  ├── 全面离线验证 (ranking accuracy, bounce detection, score spread)
  └── 通过合格标准后→真机实验
```
