# TFAC-LDG: 触觉前瞻与隐空间动力学引导的接触丰富操作策略

**全称**: Tactile Foresight with Latent Dynamics Guidance for Contact-Rich Manipulation  
**副标题**: Think in Tokens, Dream in Latent, Act with Consistency  
**目标会议**: CoRL 2026  
**日期**: 2026-04-14

---

## 目录

- [一、文献调研](#一文献调研)
  - [1.1 指定论文精读](#11-指定论文精读)
  - [1.2 扩展文献检索](#12-扩展文献检索)
  - [1.3 调研总结：三大核心问题的前沿趋势](#13-调研总结三大核心问题的前沿趋势)
- [二、TFAC V3 当前设计的精确诊断](#二tfac-v3-当前设计的精确诊断)
- [三、完整改进方案：TFAC-LDG](#三完整改进方案tfac-ldg)
  - [3.1 核心创新点概览](#31-核心创新点概览)
  - [3.2 改进一：空间Token化触觉编码](#32-改进一空间token化触觉编码)
  - [3.3 改进二：Latent空间触觉预测 + 动态加权Loss](#33-改进二latent空间触觉预测--动态加权loss)
  - [3.4 改进三：触觉一致性Loss（核心创新）](#34-改进三触觉一致性loss核心创新)
  - [3.5 改进四：时序差分残差Action](#35-改进四时序差分残差action)
  - [3.6 改进五：多级时序对比学习](#36-改进五多级时序对比学习)
- [四、完整训练流程](#四完整训练流程)
- [五、论文故事线](#五论文故事线)
- [六、实施路线图](#六实施路线图)

---

## 一、文献调研

### 1.1 指定论文精读

#### Paper 1: Reactive Diffusion Policy (RDP)

**论文**: Xue et al., 2025 — *Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation*

**核心贡献**: 提出Slow-Fast双速策略架构：
- **Slow Policy (1-2Hz)**: Latent Diffusion Policy (LDP)，从低频视觉生成隐空间action chunks
- **Fast Policy (>20Hz)**: Asymmetric Tokenizer (AT)，用高频触觉/力反馈做闭环反应式控制

**触觉处理方式**:
- **不做触觉预测**，而是用实时触觉做反应式修正
- 触觉表示：对marker deformation field做**PCA降维**（9×9×2 → ~15个主成分），前几个成分直接对应切向力、扭矩、法向力，物理可解释
- Asymmetric Tokenizer：编码器只看action（无触觉），解码器只看触觉（不看观测）。这迫使解码器利用触觉做细粒度修正，同时保持隐空间平滑

**动作生成机制**:
- Latent Diffusion在隐action空间（dim=4或8）操作，不是原始action空间
- AT解码器在chunk内逐步自回归，每一步融入最新触觉读数
- **相对轨迹表示**：action转为相对当前frame，压缩隐空间
- 训练两阶段：(1) 训练AT（L1重建+KL），(2) 训练LDP

**对你的启发**:
- **PCA on marker deformation**：降维、抗噪、物理可解释
- **Slow-Fast层级**：避免显式预测未来触觉，用实时触觉高频修正
- **Asymmetric Tokenizer**：分离"做什么"（隐策略）和"怎么调"（触觉修正）

---

#### Paper 2: OmniVTA

**论文**: Zheng et al., 2026 — *OmniVTA: Visuo-Tactile World Modeling for Contact-Rich Robotic Manipulation*

**核心贡献**: 完整的视触觉世界模型框架，四个耦合模块：
1. **TactileVAE**: 自监督触觉表征（时空编码器+隐式解码器）
2. **VTWM**: 双流时空扩散Transformer预测视觉+触觉未来
3. **AVTFP**: 自适应视触融合策略（接触概率门控）
4. **RLTC**: 60Hz反射性触觉控制器

**触觉预测方法**:
- **TactileVAE**: 将marker阵列视为空间分布的token，先空间self-attention（帧内），再时序attention（跨帧）
- **隐式解码器**: 类似神经隐式场，给定空间坐标(x,y)预测该点的3D变形——连续函数而非离散网格
- **VTWM**: 双流架构，视觉用SD-VAE编码，触觉用TactileVAE编码，在共享投影空间做预测
- **Factorized Attention**: 空间attention + 时序attention + action cross-attention（与你的V2/V3相同！）
- **Dynamic-Aware Weighted Loss**: 按每个空间位置的时序变化量构造权重图——变化大的区域（强接触）获得更高loss权重

**动作生成机制**:
- **LTD Encoder**: 计算当前触觉和预测触觉的**差分特征**（differential），而非直接用预测值
- **接触概率预测**: MLP从LTD编码输出预测接触概率，作为门控信号
- **门控网络**: 视觉+触觉特征拼接 → 两层MLP → 逐通道模态权重 W_v + W_t = 1
- **RLTC**: 60Hz闭环，用单帧触觉+预测触觉+机器人状态做修正action

**关键发现**:
- Local feature maps > global token representations（对你的单token瓶颈是直接批评）
- 触觉预测精度直接决定策略性能（Fig. 16验证了foresight的重要性）
- 差分信号比绝对预测更有信息量

**对你的启发**:
- **Dynamic-Aware Weighted Loss**: 直接适用于你的marker预测
- **LTD Encoder（差分计算）**: 用当前-预测的差异替代绝对预测值
- **接触概率门控**: 比softmax gate更有物理意义
- **隐式解码器**: 可替代你的直接MSE回归

---

#### Paper 3: Visuo-Tactile World Models (VT-WM)

**论文**: Higuera et al., 2026 (Meta FAIR) — *Visuo-Tactile World Models*

**核心贡献**: 多任务视触觉世界模型，证明触觉让世界模型**物理上grounded**：
- 35%更高的成功率（vs 纯视觉世界模型）
- 防止视觉幻觉（物体消失/瞬移）
- 满足牛顿定律（未接触物体不动）

**架构**:
- **视觉编码**: Cosmos tokenizer（预训练，冻结）
- **触觉编码**: Sparsh-X tokenizer（预训练，自监督）
- **预测器**: 12层Transformer，factorized attention
  - Spatial self-attention: 帧内所有token交互
  - Temporal self-attention: 跨时间步同位置token交互
  - Action cross-attention: 每个self-attention后与action交叉注意
- **RoPE位置编码**: 相对位置编码

**训练**:
- **Teacher Forcing Loss**: L1预测下一步
- **Sampling Loss**: 自回归展开H=3-5步，**stop gradient on sampled states**（防止训练不稳定）
- 最终loss = L_teacher + L_sampling（等权重）

**动作生成机制**:
- **CEM规划**: 在世界模型中rollout多条action轨迹，用cost function评估
- **关键洞见**: 触觉**不出现在cost function中**。它的作用是让视觉预测更物理合理，间接提升规划质量
- 触觉让世界模型"grounded in contact physics"

**对你的启发**:
- **Latent空间预测**: 最有力的证据证明 latent >> observation space
- **Sampling Loss（stop gradient）**: 确认你的V3方向正确，但他们的实现更clean
- **预训练tokenizer**: 冻结的预训练编码器非常有效
- **触觉作为physics grounding**: 预测触觉的价值在于让"想象"更合理

---

#### Paper 4: ViTacFormer

**论文**: Heng et al., 2025 — *ViTacFormer: Learning Cross-Modal Representation for Visuo-Tactile Dexterous Manipulation*

**核心贡献**: 跨模态Transformer，双向cross-attention让视觉和触觉在表征空间互相增强

**方法**:
- Visual tokens作为Q，Tactile tokens作为K/V做cross-attention，反向也做一遍（双向）
- Token级交互（保留空间结构），而非embedding级对齐
- 融合后token经MLP head输出action

**对你的启发**:
- **双向Cross-Modal Attention**: 比单向InfoNCE更深度地对齐视觉和触觉
- **Token级交互**: 保留空间结构的交互优于全局池化后的对比

---

#### Paper 5: DynaGuide

**论文**: Du & Song, 2025 — *DynaGuide: Steering Diffusion Policies with Active Dynamic Guidance*

**核心贡献**: 用学好的dynamics model在diffusion推理时**梯度引导**去噪过程

**核心机制**:
- Dynamics model: (s_t, a_t) → s_{t+1}
- 在每一步去噪时，计算 ∇_a L(dynamics_model(s_t, a_t), s_target)
- 用这个梯度nudge去噪方向，使action满足动力学约束
- **不需要改变训练流程**，纯推理时引导

**对你的启发（极其重要）**:
- **梯度引导的思路**: 触觉预测模型可以不只是"给decoder额外信息"，而是直接"引导action生成"
- 我们的方案将其转化为**training-time的可微loss**（Tactile Consistency Loss），比inference-time guidance更通用

---

#### Paper 6: ReTac-ACT

**论文**: Ruan et al., 2026 — *ReTac-ACT: A State-Gated Vision-Tactile Fusion Transformer for Precision Assembly*

**核心贡献**: 在ACT上增加State-Gated Fusion，根据任务状态动态调整视觉/触觉权重

**方法**:
- **State Gate**: 小网络根据proprio state输出gate值(0-1)，控制V/T混合比例
- f = g(state) * f_vis + (1 - g(state)) * f_tac
- 核心洞察：接近阶段视觉主导，接触阶段触觉主导
- **双向cross-attention**: 视觉和触觉互相增强
- **触觉重建目标**: 确保学到的触觉特征与操作相关

**对你的启发**:
- **State-dependent gating**: 比你的三路softmax更有物理意义
- **本体感觉条件门控**: proprio直接指示接触状态
- **触觉重建辅助loss**: 确保表征有意义

---

#### Paper 7: ACT for Spacecraft (Posadas-Nava et al.)

**论文**: 将ACT框架适配航天器交会对接

**参考价值较低**，主要确认：
- ACT框架的通用性
- CVAE KL权重需仔细调节（你从10降到1的方向正确）
- chunk_size消融实验可参考

---

### 1.2 扩展文献检索

通过系统性网络检索（8个方向，25+篇论文），按主题整理如下：

#### 1.2.1 触觉预测方法

| 论文 | 作者/年份 | 核心方法 | 关键创新 |
|------|----------|----------|----------|
| **OmniVTA** | Zheng 2026 | 双流时空扩散Transformer | Dynamic-Aware Weighted Loss |
| **VT-WM** | Higuera 2026 (Meta) | 12层Transformer + 预训练tokenizer | Latent空间预测 + Sampling Loss |
| **Phy-Tac** | Lyu 2025 | 物理条件隐扩散模型 (Phy-LDM) | **隐空间扩散**预测触觉，避免MSE均值回归 |
| **CGP** | Xu 2026 (Meta) | 条件扩散模型 | **接触一致性映射**：state和触觉必须物理一致 |
| **DreamTacVLA** | Ye 2025 | 层级空间对齐 | **HSA Loss**：触觉token与视觉视角空间对齐 |
| **exUMI TPP** | Xu 2025 | Action-aware时序预测 | 预测作为预训练目标，解决触觉稀疏性 |
| **TaSA** | Ponnivalavan 2026 | 预测性感觉衰减 | 用**预测误差**（非预测本身）作为信号 |
| **Imagine2touch** | Ayad 2024 | 视觉→触觉跨模态预测 | 低维触觉预测可行且有效 |

#### 1.2.2 预测如何影响动作

| 论文 | 作者/年份 | 机制 | 关键洞见 |
|------|----------|------|----------|
| **TouchGuide** | Zhang 2026 | 接触物理模型引导扩散 | 对比学习训练的触觉可行性评分引导去噪 |
| **DynaGuide** | Du 2025 | 动力学模型梯度引导 | 预测模型梯度修正action采样 |
| **RDP** | Xue 2025 | Slow-Fast层级 | 视觉慢规划 + 触觉残差修正 |
| **Adaptive VT Fusion** | Li 2025 | 力预测注意力 | **未来力预测**作为self-supervised辅助任务 |
| **STORM** | Lin 2025 | 扩散VLA + MCTS | **树搜索over预测结果**选最优action |
| **SC-VLA** | Liu 2026 | 预测头 + 在线精化 | 预测**任务进度** + 迭代修正 |
| **DreamVLA** | Zhang 2025 | 动态区域引导 | **只预测变化区域**而非全场景 |
| **HapticVLA** | Gubernatorov 2026 | 触觉蒸馏 | 训练时用触觉，部署时预测替代 |

#### 1.2.3 对比学习与融合

| 论文 | 作者/年份 | 方法 | 关键创新 |
|------|----------|------|----------|
| **ConViTac** | Wu 2025 | 对比嵌入条件化 (CEC) | 对比表征作为**cross-attention的条件信号** |
| **SARL** | Khurana 2025 | 空间感知自监督 | **保留空间结构**的对比目标 |
| **VTAO-BiManip** | Sun 2025 | 掩码多模态预训练 | 跨模态掩码预测 |
| **ReTac-ACT** | Ruan 2026 | 双向cross-attention + 重建 | 本体感觉条件门控 + 重建目标 |
| **TacVLA** | Zhang 2026 | 接触感知门控 | **只在接触时激活**触觉token |
| **CMT** | Lee 2026 | 物理信息正则化 | 双侧力平衡约束 |
| **MS-Bot** | Feng 2024 | 阶段引导融合 | 任务**子阶段**动态调整模态优先级 |

#### 1.2.4 动力学模型与世界模型

| 论文 | 作者/年份 | 方法 | 关键创新 |
|------|----------|------|----------|
| **RoboPack** | Ai 2024 | 循环图神经网络 | GNN建模触觉+粒子动力学 |
| **MoWM** | Yu 2025 | 混合世界模型 | **双空间预测**：latent（鲁棒）+ pixel（细节）|

#### 1.2.5 Marker Displacement / GelSight 相关

| 论文 | 作者/年份 | 方法 | 关键发现 |
|------|----------|------|----------|
| **MagicSkin** | Tijani 2025 | 半透明标志点 | marker displacement是强触觉模态 |
| **LVTG** | Liu 2026 | CLIP式对比预训练 | 独立验证CLIP预训练对视触觉有效 |
| **TransForce** | 2024 | 序列图像翻译 | GelSight力预测可跨传感器迁移 |

---

### 1.3 调研总结：三大核心问题的前沿趋势

#### 问题一：如何提升触觉预测质量？

**2025-2026的主流趋势是离开原始空间MSE**：

1. **Latent空间预测** (VT-WM, CGP, Phy-Tac) — 避免高维原始空间的均值回归
2. **扩散式预测** (Phy-LDM, CGP) — 捕获多模态未来分布
3. **空间感知目标** (SARL, DreamTacVLA HSA) — 保留MSE会破坏的空间结构
4. **动态加权Loss** (OmniVTA) — 按变化量加权，聚焦接触区域
5. **预测误差作为信号** (TaSA) — 用偏差而非预测本身

#### 问题二：预测如何提升action质量？

**Top方法**：

1. **推理时梯度引导** (TouchGuide, DynaGuide) — 预测的触觉可行性引导diffusion去噪
2. **Slow-Fast层级 + 残差** (RDP) — 预测做规划，实时触觉做修正
3. **预测性注意力门控** (Adaptive VT, MS-Bot) — 未来预测控制融合权重
4. **接触一致性映射** (CGP) — 预测的state和触觉必须物理一致
5. **差分信号** (OmniVTA LTD) — 当前vs预测的差异比绝对预测更有效

#### 问题三：如何改进对比学习？

**关键进展**：

1. **对比embedding作为条件化信号** (ConViTac CEC) — 不只是loss，是fusion的主动引导
2. **空间感知对比** (SARL) — 保留空间结构，全局池化破坏信息
3. **双向cross-attention + 重建** (ReTac-ACT) — 重建目标比纯对比更强
4. **掩码多模态预测** (VTAO-BiManip) — 跨模态掩码预训练
5. **接触感知门控** (TacVLA) — 只在接触阶段做对比/对齐

---

## 二、TFAC V3 当前设计的精确诊断

基于对全部代码的深度阅读，识别出以下**关键设计缺陷**：

### 缺陷1：单Token触觉瓶颈（根本问题）

```
MarkerEncoder(PointNet): marker_offset(B, 9, 9, 2) → 单个(1, B, 512) token
```

整个9×9空间场被压缩成**1个token**。ForesightTransformer的spatial attention只能看到这1个触觉token和N_v个视觉token，触觉的空间结构**完全丢失**。SpatialTactileDecoder要从这个单一全局向量反推回9×9的空间分布——本质上在做**信息不可逆的重建**，必然退化为均值预测。

OmniVTA的核心发现——"local feature maps > global token"——直接批评了这个设计。

### 缺陷2：只用最后一帧预测融合

```python
t_embed_future = embed_predictor(future_query_outputs)  # (B, H=10, D)
t_hat_encoded = t_embed_future[:, -1]  # 只取最后一帧！
```

预测了10帧未来触觉，但融合时**只用第10帧的embedding**。前9帧的时序变化轨迹完全丢弃——而这个变化轨迹才是最有价值的信息（"触觉会怎么变化"比"最终会是什么"更能指导action）。

### 缺陷3：视觉预测和GT对比完全关闭

```json
"lambda_foresight_vis": 0,    // 视觉预测loss关闭
"lambda_contrastive_gt": 0     // GT对比loss关闭
```

视觉预测头完全没有梯度信号，等于废弃。GT对比关闭意味着对比学习**只依赖于噪声很大的预测触觉**，信号质量差。

### 缺陷4：梯度隔离阻断因果链

```python
a1_hat_detached = a1_hat.detach()  # A1梯度完全截断
foresight_out = self.foresight(..., a1_hat_detached, ...)
```

A1被detach后，foresight预测的好坏**完全无法回传到Decoder1**。Decoder1没有任何动力去生成"更容易预测未来触觉"的action——而理想情况下，好的action应该导致可预测的触觉反馈。

### 缺陷5：对比学习信号太弱

- batch_size=64，只有63个负样本（InfoNCE需要大量负样本）
- 相邻帧高度相似但被当作负样本，没有困难负样本挖掘
- 对比权重仅0.1，在总loss中占比极小
- GT对比被关闭，只有噪声大的预测触觉参与对比

### 缺陷6：SpatialTactileDecoder的信息瓶颈

```python
# 从单个全局向量(B, D)重建9×9空间场
x = self.fc(x)  # (B, D) → (B, D)
x = x.view(B, D, 1, 1)
x = self.deconv(x)  # ConvTranspose: (B, D, 1, 1) → (B, 2, 9, 9)
```

从一个全局pool后的向量通过反卷积恢复空间结构，这个操作**本质上不可能恢复已丢失的空间信息**，只能产生"平均的"空间分布。

### 缺陷7：7个竞争Loss的平衡问题

```python
L = l1_final(1.0) + l1_draft(0.5) + foresight_tac(0.7) + foresight_vis(0!)
  + contrastive(0.1) + contrastive_gt(0!) + kl(1.0) + sampling(0.5)
```

foresight_tac权重(0.7)接近action loss(1.0)，可能导致模型优先优化触觉重建而非action质量。两个loss被完全关闭。

---

## 三、完整改进方案：TFAC-LDG

### 3.1 核心创新点概览

| # | 创新点 | 解决的问题 | 文献支撑 | 创新性分析 |
|---|--------|-----------|----------|-----------|
| **C1** | 空间Token化触觉编码 | 单token瓶颈 | OmniVTA的local>global发现 | 在marker displacement预测中首创 |
| **C2** | Latent空间预测 + 动态加权Loss | MSE均值回归 | VT-WM + OmniVTA | 双空间监督在ACT框架中首创 |
| **C3** | 触觉一致性Loss | 预测和action脱节 | DynaGuide思路 | 首次提出training-time触觉一致性约束 |
| **C4** | 时序差分残差Action | 融合只用最后帧 | RDP残差 + OmniVTA的LTD差分 | 组合创新 |
| **C5** | 多级时序对比学习 | 对比学习弱 | ConViTac条件化 + 时序对比 | 多级结合创新 |

---

### 3.2 改进一：空间Token化触觉编码

**替代当前的单Token瓶颈**

#### 现状
```
PointNet(9×9×2) → 1个(B, 512) token
```

#### 改进方案

```
marker_offset (B, 9, 9, 2)
    ↓ reshape to (B, 81, 2)           # 81个标志点，每个2维位移
    ↓ 逐点MLP: Linear(2, D) → GELU → Linear(D, D)
    ↓ + 2D可学习位置编码 (9×9位置embedding)
    ↓ 2层SelfAttention (标志点间交互)
    → (B, 81, D)

    # 3×3 patch聚合，减少token数量
    ↓ reshape to (B, 9, 9, D) → 3×3 non-overlapping patches
    ↓ 每个patch内mean pool: (B, 3, 3, D) → (B, 9, D)
    → 转置为 (9, B, D) = 9个空间tactile token
```

#### 设计要点

1. **每个空间token保留了对应3×3区域的位移信息和位置信息**
2. ForesightTransformer的spatial attention可以在vision tokens和9个tactile tokens之间做**细粒度空间交互**
3. 预测时也输出9个空间token，每个token只负责预测对应区域——**分治策略**

#### 预测解码方式

```
future_query_output (B, H, D)
    ↓ cross-attention: query=future_queries, key/value=tactile_spatial_tokens
    → (B, H, 9, D)              # 对每个patch区域做条件化预测
    ↓ per-patch MLP: Linear(D, 3×3×2=18)
    → (B, H, 9, 18) → reshape to (B, H, 9, 9, 2)
```

#### 伪代码

```python
class SpatialMarkerEncoder(nn.Module):
    """空间Token化触觉编码器"""
    def __init__(self, hidden_dim=512, n_patches=9):
        super().__init__()
        # 逐点编码
        self.point_embed = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # 2D可学习位置编码 (9×9)
        self.pos_embed = nn.Parameter(torch.randn(81, hidden_dim))
        # 标志点间交互
        self.spatial_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=2
        )
        self.n_patches = n_patches  # 3×3 = 9个patch

    def forward(self, marker_offset):
        """
        marker_offset: (B, 9, 9, 2)
        return: (9, B, D) = 9个空间token
        """
        B = marker_offset.shape[0]
        x = marker_offset.reshape(B, 81, 2)          # (B, 81, 2)
        x = self.point_embed(x)                        # (B, 81, D)
        x = x + self.pos_embed.unsqueeze(0)            # + 位置编码
        x = self.spatial_attn(x)                       # 标志点交互 (B, 81, D)

        # 3×3 patch聚合
        x = x.reshape(B, 9, 9, -1)                    # (B, 9, 9, D)
        x = x.reshape(B, 3, 3, 3, 3, -1)             # (B, 3, 3, 3, 3, D)
        x = x.mean(dim=(2, 4))                        # patch内mean → (B, 3, 3, D)
        x = x.reshape(B, 9, -1)                       # (B, 9, D)

        return x.permute(1, 0, 2)                     # (9, B, D)
```

---

### 3.3 改进二：Latent空间触觉预测 + 动态加权Loss

**替代当前的原始空间MSE/SmoothL1**

#### 现状
```python
loss = smooth_l1(marker_hat(B,H,9,9,2), marker_gt(B,H,9,9,2))  # 原始空间
```

#### 改进方案：双空间监督 + 动态加权

```python
# ========== Latent空间预测（主要Loss）==========
# 当前触觉编码
z_t = spatial_marker_encoder(marker_offset_t)        # (9, B, D) 空间tokens
# 未来触觉GT编码 (stop gradient，防止表征坍塌)
z_gt = spatial_marker_encoder(marker_offset_future).detach()  # (H, 9, B, D)
# Foresight预测的latent
z_hat = foresight_transformer(z_t, v_tokens, a1)     # (H, 9, B, D)

# Latent Loss: 在平滑的编码器输出空间做预测
L_latent = MSE(z_hat, z_gt)

# ========== 观察空间Loss（辅助，带动态加权）==========
marker_hat = patch_decoder(z_hat)  # (B, H, 9, 9, 2)

# 动态加权：按每个标志点的时序变化量加权
temporal_change = torch.norm(
    marker_gt_future - marker_current.unsqueeze(1),
    dim=-1  # 2维位移的范数
)  # (B, H, 9, 9)

# 归一化为权重（变化大的区域权重高）
dynamic_weight = temporal_change / (temporal_change.mean(dim=(-2,-1), keepdim=True) + 1e-6)
dynamic_weight = dynamic_weight.clamp(0.1, 5.0)  # 限制范围

# 加权SmoothL1
L_obs = (dynamic_weight.unsqueeze(-1) * F.smooth_l1_loss(
    marker_hat, marker_gt, reduction='none'
)).mean()

# ========== 总触觉预测Loss ==========
L_foresight = 0.5 * L_latent + 0.5 * L_obs
```

#### 动态加权的直觉

对于GelSight marker displacement:
- **大部分标志点在非接触区域几乎不动**（位移≈0）
- **只有接触区域的标志点有显著位移**
- 标准MSE/SmoothL1对所有点一视同仁，模型会倾向于预测全零（最小化整体loss）
- 动态加权让接触区域获得5-10倍的权重，强迫模型关注真正重要的变化

---

### 3.4 改进三：触觉一致性Loss（核心创新）

**这是本方案最核心的创新点——建立action→tactile的因果监督链。**

#### 动机

当前TFAC的信息流是**单向**的：

```
观测 → 预测未来触觉 → 融合 → 生成action
```

缺失的链条是**反向验证**：

```
生成的action → 该action会导致什么触觉？ → 与预期一致吗？
```

人类操作时，会预想动作的触觉后果。如果预想到"插入时应该感受到均匀阻力"，但实际感到了偏移力，就会修正。我们用Tactile Consistency Loss模拟这个过程。

#### 方法

```python
class TactileDynamicsModel(nn.Module):
    """轻量触觉动力学模型：(当前触觉latent, action) → 下一步触觉latent"""
    def __init__(self, latent_dim=512, action_dim=7, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z_tactile, action):
        """
        z_tactile: (B, D) 当前触觉latent (全局pool后)
        action: (B, 7) 单步action 或 (B, chunk*7) 整个chunk
        return: (B, D) 预测的下一步触觉latent
        """
        x = torch.cat([z_tactile, action], dim=-1)
        return self.net(x)
```

#### Consistency Loss计算

```python
# 当前触觉latent (stop gradient)
z_current = spatial_marker_encoder(marker_t).mean(dim=0).detach()  # (B, D)
# GT未来触觉latent (stop gradient)
z_gt_future = spatial_marker_encoder(marker_future).mean(dim=0).detach()  # (B, D)

# 方式A：Margin Ranking Loss
# A2应该比A1产生更接近GT的未来触觉
z_from_A1 = dynamics_model(z_current, A1.detach().mean(dim=1))
z_from_A2 = dynamics_model(z_current, A2.mean(dim=1))  # A2有梯度！

d_A1 = torch.norm(z_from_A1 - z_gt_future, dim=-1)  # (B,)
d_A2 = torch.norm(z_from_A2 - z_gt_future, dim=-1)  # (B,)

L_consistency = F.relu(d_A2 - d_A1 + margin).mean()  # margin=0.1

# 方式B（更简单）：直接让A2导致的触觉接近GT
L_consistency = F.mse_loss(z_from_A2, z_gt_future)
```

#### 为什么这是创新

1. **建立了action→tactile因果链**: 之前所有工作中，foresight和action是单向的。我们增加反向约束
2. **DynaGuide是inference-time的gradient guidance**（需要diffusion policy）；我们变成**training-time可微loss**，适用于任何policy架构
3. **自然给A2创建超越A1的动力**: 如果A2不比A1好（触觉意义上），consistency loss会惩罚它
4. **Dynamics model可以在Stage 1预训练**，有良好的初始化

#### Dynamics Model训练

```python
# 在Stage 1和Stage 2同时训练
# Stage 1 (预训练): 用GT action
L_dynamics = MSE(dynamics(z_t, a_gt), z_{t+1}.detach())

# Stage 2 (联合训练): 用A1和A2
L_dynamics += MSE(dynamics(z_t, A1.detach()), z_{t+1}.detach())
L_consistency = 上述公式
```

---

### 3.5 改进四：时序差分残差Action

**替代当前的"只用最后帧融合"和"Decoder2从头生成"**

#### 现状
```python
# 只用第10帧embedding
t_hat_encoded = t_embed_future[:, -1]
# Decoder2有7层自由度，init from A1但可以完全重写
```

#### 改进方案

##### A. 时序差分融合

```python
# 所有H帧的embedding
z_hat_trajectory = foresight_output  # (B, H, D)
# 当前触觉
z_current = tactile_encoder(current_marker).mean(dim=0)  # (B, D)

# 差分计算: 每帧相对于当前的变化
z_diff = z_hat_trajectory - z_current.unsqueeze(1)  # (B, H, D)

# 用轻量Transformer编码变化轨迹
class TactileTrajectoryEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=2):
        super().__init__()
        self.temporal_pos = nn.Parameter(torch.randn(10, 1, d_model))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers=num_layers
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, z_diff):
        """z_diff: (B, H, D) → (B, D) 变化轨迹摘要"""
        x = z_diff.permute(1, 0, 2)  # (H, B, D)
        x = x + self.temporal_pos[:x.size(0)]
        x = self.encoder(x)  # (H, B, D)
        x = x.permute(1, 2, 0)  # (B, D, H)
        x = self.pool(x).squeeze(-1)  # (B, D)
        return x

# 融合时使用变化轨迹摘要而非单帧
tactile_change_summary = trajectory_encoder(z_diff)  # (B, D)
```

**为什么时序差分更好**:
- OmniVTA的LTD编码器证明**差分信号比绝对预测更有信息量**
- 10帧的变化轨迹包含"触觉会怎么变"的完整时序信息
- 差分自然消除了静态背景bias

##### B. 残差Action

```python
# Decoder_final输出残差
delta_A = decoder_final(memory_fused, tgt=zero_init)  # (B, chunk, 7)

# Contact-aware缩放
contact_prob = torch.sigmoid(contact_mlp(proprio_state))  # (B, 1)
alpha = 0.1 + 0.4 * contact_prob  # 无接触0.1, 有接触0.5

# 最终action
A2 = A1.detach() + alpha * delta_A
```

**残差结构的优势**:
- A2围绕A1做修正，不会从头生成离谱的action
- Decoder_final只需要学习"修正量"，任务更简单
- contact-aware的alpha确保无接触时修正量小（触觉无意义时不做大调整）
- 可以减少Decoder_final的层数（从7层减到3-4层），减少参数

---

### 3.6 改进五：多级时序对比学习

**替代当前的弱InfoNCE**

#### 现状
```python
# 只有predicted tactile vs GT vision的InfoNCE，权重0.1
# batch_size=64，只有63个负样本
# GT对比关闭
```

#### 改进方案：三级对比学习

```python
# ============ Level 1: Cross-Modal (改进现有) ============
# 从全局池化改为token级对比
# 正样本: 同一帧、空间邻近的(V_patch, T_patch)
# 负样本: batch内其他帧 + 同帧远距离patch
def spatial_infonce(v_tokens, t_tokens, temperature=0.07):
    """
    v_tokens: (B, N_v, D) 视觉spatial tokens
    t_tokens: (B, 9, D) 触觉spatial tokens
    """
    # 对每个tactile token，找最近的visual token作为正样本
    sim = torch.einsum('bid,bjd->bij', t_tokens, v_tokens)  # (B, 9, N_v)
    # 正样本：每个tactile token对应的最高相似度vision token
    pos_sim = sim.max(dim=-1).values  # (B, 9)
    # 负样本：其他batch样本的所有token
    # ... (InfoNCE计算)

L_cross = spatial_infonce(v_tokens, t_tokens)

# ============ Level 2: Temporal Predictive (新增，最重要) ============
# 让模型在时间维度上有区分度
# 正样本: (z_hat_future[h], z_gt_future[h]) 同一时刻
# 负样本: (z_hat_future[h], z_gt_future[h±k]) 不同时刻
# 困难负样本: 相邻帧(h±1, h±2)
def temporal_infonce(z_hat, z_gt, temperature=0.1):
    """
    z_hat: (B, H, D) 预测的未来触觉latent
    z_gt: (B, H, D) GT未来触觉latent
    """
    B, H, D = z_hat.shape
    # 对每个时间步h，正样本是z_gt[h]
    z_hat_flat = F.normalize(z_hat.reshape(B*H, D), dim=-1)
    z_gt_flat = F.normalize(z_gt.reshape(B*H, D), dim=-1)
    
    # 相似度矩阵 (B*H, B*H)
    logits = z_hat_flat @ z_gt_flat.T / temperature
    labels = torch.arange(B*H, device=logits.device)
    
    loss = (F.cross_entropy(logits, labels) + 
            F.cross_entropy(logits.T, labels)) / 2
    return loss

L_temporal = temporal_infonce(z_hat_projected, z_gt_projected)

# ============ Level 3: GT Cross-Modal (重新启用) ============
# 用GT触觉和GT视觉做对比（无噪声）
L_gt = infonce(project_tac(z_gt_tactile), project_vis(z_gt_vision))

# ============ 总对比Loss ============
L_contrastive = 0.3 * L_cross + 0.5 * L_temporal + 0.2 * L_gt
```

#### 关键改进说明

1. **启用GT对比** (`lambda_contrastive_gt` 设回0.2)：提供无噪声的对齐信号
2. **时序对比** (Level 2)：最重要的新增——让预测在时间维度有区分度，"第3帧和第5帧的触觉应该不同"
3. **困难负样本**: 相邻帧(h±1)是最有效的负样本（相似但不相同），InfoNCE中引入更大的batch有效负样本数
4. **空间token级对比** (Level 1)：从全局池化改为token级，保留空间信息

---

## 四、完整训练流程

```
┌─────────────────────────────────────────────────────┐
│                 Stage 0: 编码器预训练                  │
│  ├── 空间Token化MarkerEncoder + ResNet18 视觉backbone  │
│  ├── Cross-modal对比 (Level 1) + 触觉空间重建Loss      │
│  ├── Epoch: 3-5                                       │
│  └── 输出: 预训练好的encoder权重                        │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│            Stage 1: Dynamics + Foresight预训练        │
│  ├── 冻结Stage 0的encoder                             │
│  ├── 训练ForesightTransformer (GT action条件化)        │
│  ├── 训练TactileDynamicsModel: z_{t+1} = f(z_t, a_gt) │
│  ├── Loss = L_latent + L_obs_dynamic + L_dynamics     │
│  ├── Epoch: 5-10                                       │
│  └── 输出: 预训练好的dynamics + foresight权重           │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│                Stage 2: 联合端到端训练                  │
│  ├── 加载Stage 0 + Stage 1权重                         │
│  ├── 完整pipeline训练                                  │
│  ├── 课程学习: 前40%用GT触觉, 后60%用预测触觉           │
│  ├── 学习率: cosine decay                              │
│  ├── Epoch: 2000                                       │
│  └── Loss组成见下                                      │
└─────────────────────────────────────────────────────┘
```

### Stage 2 Loss组成

```python
L_total = 1.0 * L1(A2, GT)              # 最终action质量（最重要）
        + 0.3 * L1(A1, GT)              # 草稿action
        + 0.3 * L_latent                 # latent空间触觉预测
        + 0.2 * L_obs_dynamic            # 动态加权观测空间触觉预测
        + 0.3 * L_consistency            # 触觉一致性约束（核心创新）
        + 0.1 * L_contrastive            # 多级对比学习
        + 1.0 * L_KL                     # CVAE
        + 0.3 * L_sampling               # 自回归展开

# 可选
        + 0.1 * L_dynamics               # dynamics model辅助监督
```

### 超参数建议

| 参数 | 当前V3值 | 建议值 | 理由 |
|------|---------|--------|------|
| batch_size | 64 | 64-128 | 增大有利于对比学习 |
| lr | 4e-5 | 4e-5 | 保持 |
| lr_schedule | 无 | cosine decay | 稳定训练后期 |
| curriculum_ratio | 0.4 | 0.4 | 保持 |
| chunk_size | 10 | 10-20 | 可实验20 |
| foresight_horizon | 10 | 10 | 保持 |
| n_tactile_tokens | 1 | 9 | 核心改动 |
| consistency_margin | - | 0.1 | 新增 |
| contact_alpha_range | - | [0.1, 0.5] | 新增 |
| gradient_clip | 无 | 1.0 | 增加训练稳定性 |

---

## 五、论文故事线

### Title
**TFAC-LDG: Tactile Foresight with Latent Dynamics Guidance for Contact-Rich Manipulation**

### Abstract 故事

> 人类在操控物体时，不仅依靠视觉规划，还会预想未来的触觉感受来校验和调整动作——如果预想到插入时会感受到均匀阻力，但实际感到了偏移力，就会修正动作方向。受此启发，我们提出TFAC-LDG，一个"想-梦-验-行"(Think-Dream-Verify-Act)的四阶段框架：
>
> 1. **Think**: 从视觉和当前触觉理解场景，生成草稿动作
> 2. **Dream**: 在latent空间预测未来触觉变化轨迹（而非原始信号空间，避免均值回归）
> 3. **Verify**: 通过触觉一致性验证——检查动作是否会导致预期的触觉反馈
> 4. **Act**: 基于验证结果，用时序差分残差修正最终动作
>
> 核心贡献有三: (1) 空间Token化的latent触觉预测，以保留标志点的空间结构；(2) Tactile Consistency Loss，首次建立action→tactile的因果监督——好的动作应导致预期的触觉反馈；(3) 时序差分残差引导，用预测的触觉变化轨迹来修正动作。在XXX benchmark上达到SOTA。

### 创新点（for reviewers）

1. **Tactile Consistency Loss（主要创新）**: 首次提出用触觉动力学一致性来监督action生成。此前所有触觉预测工作都是单向的（obs→prediction），我们首次建立反向因果约束（action→predicted tactile→verification）。这是DynaGuide梯度引导思想在policy learning中的泛化，但作为training-time可微loss，不依赖diffusion policy

2. **空间Token化Marker编码**: 在GelSight marker displacement场景首次采用空间token化编码（9个patch token替代单个全局token），结合patch级解码，使触觉预测保留空间结构

3. **双空间动态加权监督**: 同时在latent空间（鲁棒性）和观测空间（可解释性）做触觉预测监督，并用Dynamic-Aware Weighted Loss聚焦接触区域

4. **时序差分残差动作生成**: 利用预测的触觉变化轨迹（而非单帧绝对预测）的差分信号，通过残差结构(A2=A1+αΔA)修正动作，α由接触概率自适应调节

### 消融实验设计

| 实验 | 目的 | 比较 |
|------|------|------|
| Exp1 | 空间tokenization | 1 token vs 9 tokens vs 81 tokens |
| Exp2 | 预测空间 | 原始空间MSE vs latent vs 双空间 |
| Exp3 | 动态加权 | 无加权 vs 均匀 vs dynamic-aware |
| Exp4 | Consistency Loss | 无 vs MSE版 vs margin ranking版 |
| Exp5 | 残差vs完整生成 | A2全生成 vs A2=A1+ΔA |
| Exp6 | 时序融合 | 只用最后帧 vs 全部H帧差分 |
| Exp7 | 对比学习级别 | 无 vs L1 vs L1+L2 vs L1+L2+L3 |
| Exp8 | Contact-aware alpha | 固定alpha vs 学习alpha vs contact-aware |

### 基线对比

| 方法 | 类型 |
|------|------|
| ACT (Zhao et al.) | 无触觉baseline |
| ACT + Tactile (concat) | 简单触觉融合 |
| RDP (Xue et al.) | Slow-Fast触觉反应式 |
| ReTac-ACT (Ruan et al.) | State-gated ACT |
| TFAC V3 (ours, current) | 当前方法 |
| **TFAC-LDG (ours, proposed)** | 完整改进方案 |

---

## 六、实施路线图

### 优先级排序

| 优先级 | 改进 | 预期收益 | 工作量 | 依赖 |
|--------|------|----------|--------|------|
| **P0** | 空间Token化触觉编码 | 高（解决根本瓶颈）| 2-3天 | 无 |
| **P1** | Latent空间预测 + 动态加权Loss | 高（解决均值回归）| 2天 | P0 |
| **P2** | 残差Action (A2=A1+ΔA) | 中高（让触觉影响action）| 1天 | 无 |
| **P3** | 触觉一致性Loss | 高（核心创新）| 2天 | P1 |
| **P4** | 时序差分融合（用全部H帧）| 中（信息利用率）| 1天 | P1 |
| **P5** | 多级对比学习 | 中（表征质量）| 2天 | P0 |
| **P6** | Contact-aware alpha | 低中（物理先验）| 0.5天 | P2 |

### 建议分步验证

**Round 1 (P0+P1+P2)**: 基础架构改进
- 空间Token化 + Latent预测 + 残差action
- 跑消融实验验证单独效果
- 预计1-2周

**Round 2 (P3)**: 核心创新
- 加入Tactile Consistency Loss
- 仔细调节margin和权重
- 这是论文最核心的创新，需要充分验证
- 预计1周

**Round 3 (P4+P5)**: 完善方案
- 时序差分融合 + 多级对比学习
- 预计1周

**Round 4 (P6 + 消融实验)**: 收尾
- Contact-aware alpha
- 完整消融实验表格
- 预计1周

---

## 附录A：完整参考文献

### 指定论文

1. Xue et al., 2025. "Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation"
2. Zheng et al., 2026. "OmniVTA: Visuo-Tactile World Modeling for Contact-Rich Robotic Manipulation"
3. Higuera et al., 2026. "Visuo-Tactile World Models" (Meta FAIR)
4. Heng et al., 2025. "ViTacFormer: Learning Cross-Modal Representation for Visuo-Tactile Dexterous Manipulation"
5. Du & Song, 2025. "DynaGuide: Steering Diffusion Policies with Active Dynamic Guidance"
6. Ruan et al., 2026. "ReTac-ACT: A State-Gated Vision-Tactile Fusion Transformer for Precision Assembly"
7. Posadas-Nava et al. "Action Chunking with Transformers for Image-Based Spacecraft Guidance and Control"

### 扩展检索论文

8. Lyu et al., 2025. "Phy-Tac: Physics-Conditioned Tactile Goals via Latent Diffusion"
9. Xu et al., 2026. "Contact-Grounded Policy (CGP)" (Meta)
10. Ye et al., 2025. "DreamTacVLA: Learning to Feel the Future"
11. Ponnivalavan et al., 2026. "TaSA: Two-Phased Deep Predictive Learning of Tactile Sensory Attenuation"
12. Xu et al., 2025. "exUMI: Tactile Prediction Pretraining (TPP)"
13. Ayad et al., 2024. "Imagine2touch: Predictive Tactile Sensing using Low-Dimensional Signals"
14. Zhang et al., 2026. "TouchGuide: Inference-Time Steering via Touch Guidance"
15. Li et al., 2025. "Adaptive Visuo-Tactile Fusion with Predictive Force Attention"
16. Lin et al., 2025. "STORM: Search-Guided Generative World Models"
17. Liu et al., 2026. "Self-Correcting VLA (SC-VLA)"
18. Zhang et al., 2025. "DreamVLA: Dynamic-Region-Guided World Knowledge Prediction"
19. Gubernatorov et al., 2026. "HapticVLA: Tactile without Inference-Time Sensing"
20. Wu et al., 2025. "ConViTac: Contrastive Embedding Conditioning for Visual-Tactile Fusion"
21. Khurana et al., 2025. "SARL: Spatially-Aware Self-Supervised Representation Learning"
22. Sun et al., 2025. "VTAO-BiManip: Masked Visual-Tactile-Action Pre-training"
23. Zhang et al., 2026. "TacVLA: Contact-Aware Tactile Fusion for VLA"
24. Lee et al., 2026. "CMT: Symmetry-Aware Fusion via Bilateral Force Priors"
25. Feng et al., 2024. "Play to the Score (MS-Bot): Stage-Guided Dynamic Multi-Sensory Fusion"
26. Lei et al., 2026. "Learning When to See and When to Feel"
27. Ai et al., 2024. "RoboPack: Learning Tactile-Informed Dynamics Models"
28. Yu et al., 2025. "MoWM: Mixture-of-World-Models"
29. Tijani et al., 2025. "MagicSkin: Balancing Marker and Markerless Modes"
30. Liu et al., 2026. "LVTG: Low-Cost Vision-Based Tactile Gripper with Pretraining"

---

## 附录B：待讨论的关键设计决策

1. **空间token的粒度**: 81个token（逐点）vs 9个token（3×3 patch）vs 其他？
   - 建议9个（3×3 patch），平衡信息量和计算开销
   
2. **Latent Dynamics Model的结构**: 简单MLP vs 1-2层Transformer？
   - 建议先用3层MLP（简单有效），效果不佳再换Transformer

3. **Consistency Loss形式**: Margin ranking vs 直接MSE？
   - 建议先用Margin ranking（更符合"A2应优于A1"的语义）

4. **Decoder_final层数**: 是否因残差模式减少？
   - 建议从7层减到3-4层，减少参数和过拟合风险

5. **是否保留视觉预测Loss**: 当前完全关闭
   - 建议启用，权重设为0.1-0.2

---

*文档生成时间: 2026-04-14*  
*基于7篇指定论文 + 25篇扩展文献的深度调研*
