# CQF (Contact Quality Function) 完整设计方案

> 日期: 2026-04-27
> 状态: 设计中，待讨论确认
> 前置依赖: TactileVAE + Foresight (latent) + DP Baseline
> 数据依赖: annotations.pkl (已生成，Z轴bounce检测)

---

## 一、CQF在系统中的位置

### 1.1 推理时的完整pipeline

```
观测(视觉+触觉+qpos)
  │
  ├─→ SharedEncoder → memory (已有，给DP和Foresight用)
  ├─→ TactileVAE.encode(marker_t) → z_tac_current (144维)
  │
  ├─→ DP → K=16个candidate actions {a¹, ..., a^K}
  │
  │   对每个candidate a^k:
  │     Foresight(memory, a^k) → z_tac_predicted^k
  │     CQF(state_t, z_tac_current, z_tac_predicted^k, [a^k]) → score^k
  │
  └─→ 选 a* = argmax_k score^k → 执行
```

### 1.2 CQF的输入输出

```
输入:
  - state: (qpos, eef)           # 当前机器人状态，13维
  - z_tac_current:               # 当前触觉latent，144维 (TactileVAE)
  - z_tac_predicted:             # Foresight预测的未来触觉latent，144维
  - action_chunk (可选):         # 候选action，(chunk_size, 7)

输出:
  - score: 标量，越高越好
```

### 1.3 依赖链

```
TactileVAE (Stage 0, frozen) ──┐
Foresight latent (Stage 1, frozen) ──┤── CQF Training (Stage 2)
DP Baseline (并行训练, frozen) ──────┘
                                       │
                                       ▼
                               Reranking集成 (Stage 3)
```

---

## 二、标注数据统计 (已完成)

### 2.1 数据总览

通过 `scripts/annotate_episodes.py` 已对13个数据集完成帧级标注:

| 指标 | 数值 |
|------|------|
| 总episodes | 2759 |
| success episodes | 1371 |
| bounce episodes | 1388 |
| label=0 approach帧 | 380,027 (51.6%) |
| label=1 insertion帧 | 256,169 (34.8%) |
| label=2 pre_bounce帧 | 42,500 (5.8%) |
| label=3 lift帧 | 46,214 (6.3%) |
| label=4 reposition帧 | 11,312 (1.5%) |

### 2.2 CQF训练数据

| 类型 | 来源label | 帧数 | 用途 |
|------|-----------|------|------|
| 正样本 | label=1 (insertion) | 256,169 | CQF应输出高分 |
| 负样本 | label=2 (pre_bounce) | 42,500 | CQF应输出低分 |
| 正负比 | — | 6:1 | 需weighted sampler平衡 |

### 2.3 Bounce episode的lift分布

```
1次lift: 24 episodes (1.7%)
2次lift: 1269 episodes (91.4%)  ← 绝大多数
3次lift: 93 episodes (6.7%)
4次lift: 2 episodes (0.1%)
```

每个bounce episode平均30.6帧pre_bounce (约2×15帧/lift)。

### 2.4 标注Label定义

```
0 = approach:    z在下降，接近插孔
1 = insertion:   最后一次下降段（成功插入）→ CQF正样本
2 = pre_bounce:  lift前15帧（导致碰撞的action）→ CQF核心负样本
3 = lift:        z在上升，碰撞后抬起
4 = reposition:  lift结束后到下一次下降前的平稳段
```

---

## 三、信号分析——CQF应该依赖什么信号

### 3.1 各信号的重要性

| 信号 | 来源 | 物理意义 | 重要性 |
|------|------|---------|--------|
| **delta = z_pred - z_cur** | 触觉变化量 | 碰撞=突变大delta，好插入=渐进小delta | **最高** |
| **z_tac_predicted** | Foresight输出 | 预测的未来触觉是否在正常范围 | 高 |
| **action_chunk** | DP候选 | action本身是否合理（兜底信号） | 中（有风险，见问题3） |
| **state (qpos+eef)** | 当前观测 | 提供阶段上下文（approach vs contact时delta含义不同） | 中 |

### 3.2 数据验证：pre_bounce和insertion帧的触觉对比

以260402/bounce/episode_100为例:

```
Marker magnitude:
  pre_bounce (t=86~100): ~2.4  ← 和insertion几乎一样
  lift start (t=101):    ~4.0  ← 明显增大
  insertion (t=200):     ~2.4
  episode end (t=316):   ~4.4

Force-Torque:
  pre_bounce (t=90):     Fz = -0.95
  lift start (t=101):    Fz = -2.86  ← 碰撞力突增
  insertion (t=200):     Fz = 14.18  ← 正常插入力

结论: 当前帧的触觉强度在pre_bounce和insertion时几乎相同(~2.4)，
      无法靠当前触觉区分。关键区分信号是"未来触觉会怎样变化"。
```

---

## 四、架构设计

### 4.1 推荐架构: 3-Branch MLP

```python
class ContactQualityScorer(nn.Module):
    """
    3-branch特征提取 + 融合打分
    
    Branch 1 (触觉分析, 最重要):
      输入: [z_tac_current(144), z_tac_predicted(144), delta(144)] = 432维
      输出: h_tac (256维)
      学习: "预测的触觉变化是否正常"
    
    Branch 2 (Action分析):
      输入: action_chunk展平 (20×7=140维)
      输出: h_act (256维)
      学习: "action本身是否在合理范围"
      注意: 需要50% dropout mask防止走捷径 (见问题3)
    
    Branch 3 (状态上下文):
      输入: [qpos(7), eef(6)] = 13维
      输出: h_state (256维)
      学习: "当前在什么阶段，为delta提供上下文"
    
    融合:
      [h_tac(256), h_act(256), h_state(256)] = 768维
      → MLP → score (标量)
    """
```

### 4.2 参数量估算

```
Branch 1: Linear(432→256) + Linear(256→256) ≈ 176K
Branch 2: Linear(140→256) + Linear(256→256) ≈ 102K  
Branch 3: Linear(13→256) + Linear(256→256) ≈ 69K
融合层: Linear(768→256) + Linear(256→128) + Linear(128→1) ≈ 230K
────────────────────────────────────────────────────────
总计: ~577K 参数
```

### 4.3 为什么选MLP而不是Transformer/更复杂的架构

1. **数据量匹配**: 42K负样本+256K正样本，对~500K参数的MLP足够
2. **推理速度**: CQF要跑K=16次，MLP比Transformer快一个量级
3. **信号直接**: delta大小、action范围这些信号不需要注意力机制
4. **避免过拟合**: 更大模型在这个数据量上容易过拟合到训练集分布

### 4.4 为什么不用cosine similarity（对比TouchGuide CPM）

TouchGuide的CPM: `score = cosine(obs_embedding, action_embedding)`

不采用的原因:
1. TouchGuide CPM不接收predicted tactile——只看(当前obs, action)
2. 我们多了Foresight预测这路关键信号
3. cosine是对称的，但action→tactile的因果关系不对称
4. MLP可以学到更复杂的非线性判断边界

### 4.5 是否需要PD-CQV的Physics Heads

之前设计的PD-CQV包含4个物理分解头 (S_FIC, S_DCS, S_TDS, S_CMA)。

**建议: 先不加，用Core Scorer验证可行性后再考虑。**

理由:
1. 4个head需要4组loss、4种负样本构造，训练复杂度高
2. 如果Core Scorer已经work，Physics Heads是锦上添花
3. TouchGuide证明简单scorer + 好数据就能+23%成功率
4. Physics Heads可以作为论文消融实验后加

如果要加，优先加的是:
- S_TDS (Temporal Dynamics): 最简单，只需要delta → score，且有OmniVTA LTD的直接支撑
- S_FIC (Forward-Inverse Consistency): 创新性最强，但需要额外训练InverseHead

---

## 五、Loss函数设计

### 5.1 主Loss: Margin Ranking Loss

```python
L_rank = max(0, margin - score_pos + score_neg)
```

- 直接优化排序——reranking只需要排序正确
- margin=0.5，正负样本分数差距至少0.5
- 使用hard pair mining: 每个正样本只和batch内得分最高的K=5个负样本配对

### 5.2 辅助Loss: BCE Loss

```python
L_bce = BCE(sigmoid(score), label)
```

- 提供绝对分数校准
- 推理时需要safety threshold（所有候选分数都低→不执行），需要绝对值有意义

### 5.3 总Loss

```python
L_total = L_rank + 0.5 * L_bce
```

### 5.4 为什么两个loss都需要

- **只用ranking**: 排序对了但分数绝对值没意义，无法设safety threshold
- **只用BCE**: 只优化0/1分类准确度，不直接优化正样本之间的精细排序
- **两者结合**: ranking保证排序正确，BCE提供绝对值锚定

---

## 六、训练数据构造

### 6.1 每个训练样本的格式

```python
sample = {
    'qpos':           (7,),      # 当前关节角度
    'eef':            (6,),      # 当前末端位姿
    'z_tac_current':  (144,),    # TactileVAE.encode(marker_t)
    'z_tac_predicted':(144,),    # Phase A: TactileVAE.encode(marker_{t+h})
                                 # Phase B/C: Foresight(memory, action)
    'action_chunk':   (20, 7),   # action[t:t+chunk_size]
    'label':          float,     # 1.0 = 好, 0.0 = 坏
}
```

### 6.2 正样本构造 (label=1, insertion帧)

```
来源: annotations.pkl中label=1的帧
条件: t + chunk_size <= T 且 t + h <= T (不越界)

sample.qpos = proprio_joint[t]
sample.eef = proprio_eef[t]
sample.z_tac_current = TactileVAE.encode(marker_offset[t])
sample.z_tac_predicted = TactileVAE.encode(marker_offset[t+h])  # Phase A
sample.action_chunk = joint_abs[t:t+chunk_size]
sample.label = 1.0
```

来源细分:
- success episode的insertion帧: 平稳插入过程
- bounce episode的insertion帧: 碰撞修正后的成功插入（label=1在最后一段lift之后）

### 6.3 负样本构造 (label=2, pre_bounce帧)

```
来源: annotations.pkl中label=2的帧
条件: t + chunk_size <= T 且 t + h <= T
      且 t + h >= lift_start (确保future触觉包含碰撞信号，见问题1)

sample.qpos = proprio_joint[t]
sample.eef = proprio_eef[t]
sample.z_tac_current = TactileVAE.encode(marker_offset[t])
sample.z_tac_predicted = TactileVAE.encode(marker_offset[t+h])  # Phase A
sample.action_chunk = joint_abs[t:t+chunk_size]
sample.label = 0.0
```

### 6.4 Phase B额外负样本: Action扰动

```
对每个正样本:
  perturbed_action = action_chunk + noise_scale * randn
  z_tac_predicted = Foresight(memory, perturbed_action)  # 用Foresight预测
  
  noise_scale = [0.3σ, 0.5σ, 1.0σ, 2.0σ]
  小扰动(0.3σ) → label=0.3 (soft label，可能还行)
  大扰动(2.0σ) → label=0.0 (肯定不行)
```

### 6.5 正负样本平衡策略

```
原始比例: 正256K : 负42.5K = 6:1
处理方式:
  1. Weighted Sampler: 每个batch中正:负 = 1:1
  2. Hard Negative Mining: 每N个epoch，对所有负样本打分，
     下一轮优先采样"分数最高的负样本"（最容易被误判为正的）
  3. Phase B的扰动负样本额外提供~50K样本，进一步缓解不平衡
```

---

## 七、三阶段训练

### 7.1 阶段A: GT触觉 (10 epochs)

```
目的: 在干净数据上建立"碰撞触觉 vs 正常触觉"的判断标准
数据:
  - 正: insertion帧 + GT future tactile (TactileVAE.encode(marker_{t+h}))
  - 负: pre_bounce帧 + GT future tactile (含碰撞信号)
z_tac_predicted来源: 100% GT
```

### 7.2 阶段B: 混合 (30 epochs, GT比例线性90%→10%)

```
目的: 让CQF从GT pattern过渡到Foresight的输出分布
数据:
  - 正/负样本的z_tac_predicted:
    epoch 0: 90% GT + 10% Foresight
    epoch 15: 50% GT + 50% Foresight  
    epoch 30: 10% GT + 90% Foresight
  - 额外: action扰动负样本 (全部用Foresight预测)
GT比例: linear decay from 90% to 10%
```

### 7.3 阶段C: 纯Foresight (10 epochs)

```
目的: 与推理时分布完全一致
数据:
  - 所有z_tac_predicted都来自Foresight(memory, action)
  - 不使用GT
```

### 7.4 为什么需要3阶段

核心原因是**分布偏移**: Foresight的预测有误差，和GT触觉分布不同。

- 直接从C开始: Foresight预测的碰撞信号可能比GT弱（被平滑了），
  CQF在noisy数据上难以建立判断标准
- 先A再C: A学到的GT pattern在C不一定有效（GT碰撞特征 ≠ Foresight碰撞特征）
- A→B→C: A建立基准 → B渐进迁移 → C最终适配

阶段B最重要，分配最多epoch。GT比例线性衰减而非hard切换，让CQF平滑过渡。

---

## 八、关键超参数

| 参数 | 推荐值 | 理由 |
|------|--------|------|
| hidden_dim | 256 | 3个branch，总参数~500K，匹配数据量 |
| batch_size | 256 | 正负各128 (weighted sampler) |
| lr | 1e-4 | Adam, cosine decay to 1e-6 |
| margin | 0.5 | ranking loss margin |
| h (foresight horizon) | 10 | 和现有Foresight config一致，过滤pre_bounce使t+h>=lift_start |
| chunk_size | 20 | 和ACT/DP一致 |
| action dropout | 0.5 | Branch 2 mask概率，防止走捷径 |
| hard pair K | 5 | ranking loss中每个样本配top-5难样本 |
| Phase A epochs | 10 | 快速建立基准 |
| Phase B epochs | 30 | 最关键的过渡阶段 |
| Phase C epochs | 10 | 最终适配 |
| safety threshold | 0.3 | 推理时所有候选低于此值→不执行 |

---

## 九、离线验证指标

训好后必须通过以下测试才能上真机:

### 9.1 Ranking Accuracy (目标 > 60%)

```
对test set中每个contact-phase帧:
  1. expert action = GT action (应排第1)
  2. 生成15个perturbed actions (加不同量级噪声)
  3. 共16个候选，通过Foresight + CQF打分
  4. 检查expert是否排第一
  
随机baseline = 1/16 = 6.25%
目标 > 60%
```

### 9.2 Bounce Detection (目标 > 80%)

```
对bounce episode:
  1. 取pre_bounce帧的GT action (导致碰撞的)
  2. 取同episode insertion帧的GT action (成功的)
  3. CQF给insertion action的分数应 > pre_bounce action的分数
  
目标 > 80%
```

### 9.3 Score Spread (目标 > 0.3)

```
用DP生成K=16个真实候选:
  score_max - score_min 的均值应 > 0.3
  
说明CQF能有效区分候选质量，而不是给所有候选差不多的分
```

---

## 十、推理效率估算

```
各模块耗时 (估算, 单GPU):
  SharedEncoder (1次):       ~5ms
  DP去噪 (K=16, batch):     ~50ms (batch parallel, 与K=1差不多)
  Foresight (K=16, batch):   ~3ms
  CQF (K=16, batch):         ~1ms (MLP非常快)
  ─────────────────────────────
  总计:                       ~59ms

不做reranking时:
  SharedEncoder + DP:         ~55ms
  额外开销:                   ~4ms (约7%)
```

### Safety Threshold机制

```python
if scores.max() < SAFETY_THRESHOLD:
    # 所有候选都可能导致碰撞
    # 方案A: 选位移最小的候选(最保守)
    # 方案B: 暂停执行
```

---

## 十一、已识别的设计问题

### 问题1（严重）：Foresight horizon h 与 pre_bounce标注的对齐

**描述**:
pre_bounce定义为lift前15帧。如果h < 15，部分pre_bounce帧的"未来触觉"(t+h)还没到碰撞时刻，看起来和正样本一样。

**具体例子**:
```
lift starts at frame 101
pre_bounce frames: 86, 87, ..., 100

h=10时:
  t=86 → future at t+10=96  → 还在approach，触觉正常 ← 无效负样本!
  t=90 → future at t+10=100 → lift边界，信号模糊
  t=96 → future at t+10=106 → lift期间，触觉异常 ✓

h=15时:
  t=86 → future at t+15=101 → 正好lift start，有碰撞信号 ✓
  t=96 → future at t+15=111 → lift期间 ✓
  所有pre_bounce帧都有效
```

**影响**: h太小 → 部分负样本的future触觉看起来正常 → CQF学到噪声

**可能的解决方案**:
- **方案A**: 设 h=15，保证所有pre_bounce帧有效。但Foresight预测15步精度可能下降。
- **方案B**: 设 h=10，但只用pre_bounce帧中 t+h >= lift_start 的帧作负样本。
  即只用最后10帧(共~28K负样本，仍然足够)。
- **方案C**: 使用多个h值(5, 10, 15)，CQF同时看多个horizon的预测。更多信号但复杂度增加。
- **方案D**: h设为动态值——每个样本选取使t+h刚好落在lift开始处的h。
  但推理时h是固定的，训练和推理分布不一致。

**已决定**: 采用方案B — h=10（和现有Foresight config一致），只用pre_bounce中t+h>=lift_start的帧作为负样本。有效负样本约28K帧，足够。

---

### 问题2（中等）：State输入的定义不一致

**描述**:
之前设计文档中，state一会儿是 `memory_pooled (512维)` (SharedEncoder输出)，一会儿是 `qpos+eef (13维)`。这是两个完全不同的东西。

**分析**:
- memory_pooled (512维):
  - 包含视觉+触觉+qpos的综合编码
  - 训练CQF时需要对每个样本跑SharedEncoder → 成本高、引入额外依赖
  - 如果SharedEncoder更新了，CQF可能需要重新训练
  
- qpos+eef (13维):
  - 从HDF5直接读取，零额外成本
  - 对插插座任务: eef的(x,y,z,rx,ry,rz)已编码"离插孔多近、什么姿态"
  - z_tac_current已提供触觉信息，不需要memory重复编码

**推荐**: CQF训练和推理统一用 qpos(7) + eef(6) = 13维作为state。
不依赖SharedEncoder，训练更独立，且信息足够。

---

### 问题3（中等）：CQF是否应该接收action作为输入

**描述**:
如果CQF看到action，可能走捷径：直接从action数值判断好坏，绕过触觉预测信号。

**走捷径的原因**:
训练数据中，positive的action (insertion阶段) 和 negative的action (pre_bounce阶段)
在数值分布上可能有差异（比如insertion时z方向位移更大）。CQF可以不看触觉，
只靠action分布就达到不错的训练accuracy，但推理时面对DP生成的同分布候选就失效了。

**保留action的理由**:
- 提供冗余信号：如果Foresight对某个action预测错误，CQF可以从action本身检测异常
- 推理时K=16个候选是同分布的（都来自DP），action分布差异小，CQF被迫使用触觉信号

**不保留action的理由**:
- 更干净：强迫CQF100%依赖触觉预测做判断
- 更robust：不会学到训练集特有的action分布偏差
- 更符合CQF的设计意图："评估触觉前瞻质量"

**可能的折中方案**:
- **方案A**: 去掉action输入。CQF只看 (state, z_tac_cur, z_tac_pred)。
  参数量减少~100K，更简洁。
- **方案B**: 保留action但加50% dropout mask。训练时随机丢弃action branch，
  强迫CQF在没有action时也能判断。
- **方案C**: 保留action但加gradient penalty。如果CQF对action的梯度太大(意味着
  过度依赖action)，加惩罚项。

**待决定**: 保留还是去掉action输入？推荐方案B (保留+dropout)或方案A (去掉)。

---

### 问题4（小）：Ranking Loss实现效率

**描述**:
朴素的pairwise ranking loss对所有正负对计算，batch=256时有128×128=16384对，
大部分是easy pair（分数差距已经很大），贡献梯度很小。

**影响**: 训练效率低，梯度被easy pair稀释。

**解决方案**: Hard Pair Mining
```python
# 对每个正样本，只和batch内得分最高的K个负样本配对
# 对每个负样本，只和batch内得分最低的K个正样本配对
# K=5~10
```

这是标准做法，FaceNet、ReID等领域广泛使用。

---

### 问题5（小但重要）：Phase A学到的pattern在Phase C可能无效

**描述**:
Phase A用GT碰撞触觉训练CQF。但Foresight预测碰撞时的输出可能和GT不同——
Foresight可能把碰撞触觉预测得偏平滑（MSE loss的均值回归效应）。

Phase A学到"GT碰撞触觉的特征"，到Phase C面对"Foresight预测的碰撞触觉"时，
这些特征可能不match。

**影响**: Phase A的训练不白费，但需要Phase B足够长来完成迁移。

**解决方案**:
1. Phase B设为30 epochs（比A和C都长），给CQF足够时间从GT pattern迁移到Foresight pattern
2. GT比例线性衰减（90%→10%），而非hard切换
3. 监控Phase B期间的验证集指标，确认迁移在发生

---

### 问题6（待确认）：TactileVAE的latent维度

**描述**:
之前假设TactileVAE latent = 144维 (16ch × 3×3)。但TactileVAE实际训练后的
latent维度可能不同。

**影响**: 直接影响CQF的输入维度和Branch 1的设计。

**解决**: 等TactileVAE训好后确认latent维度，再固化CQF架构。

---

### 问题7（待确认）：chunk边界处理

**描述**:
对label=2的pre_bounce帧t，action_chunk = action[t:t+20]。但如果t+20超过episode
长度T，chunk不完整。

同样，t+h可能超出episode范围。

**解决方案**:
```python
# 过滤条件
valid = (t + chunk_size <= T) and (t + h <= T)
# 对于pre_bounce帧还需要额外条件 (见问题1):
valid = valid and (t + h >= lift_start)  # 确保future包含碰撞
```

不做padding或截断，直接丢弃不完整的样本。数据量足够，不会有问题。

---

## 十二、与现有方法的对比

| 方法 | 打分方式 | CQF的优势 |
|------|---------|-----------|
| TouchGuide CPM | cosine(obs_emb, action_emb) | CQF额外利用了Foresight预测的未来触觉 |
| DynaGuide | V(predicted_state) | CQF专门为触觉接触设计，有delta信号 |
| OmniVTA RLTC | mean±std统计阈值 | CQF是可学习的，非规则式 |
| PPGuide | 二分类(成功/失败) | CQF用ranking loss，更精细的排序 |

### CQF的核心创新点

1. **触觉前瞻评分**: 不是直接评估action好坏，而是通过预测未来触觉来间接评估
2. **自然负样本**: 利用真实碰撞数据（不是人造噪声）作为负样本训练
3. **三阶段分布对齐**: 解决GT→Foresight的分布偏移问题
4. **和DP的天然结合**: DP生成多样化候选 → Foresight预测 → CQF筛选

---

## 十三、实现文件规划

```
TFAC_V5/
  cqf_model.py          # ContactQualityScorer网络定义
  cqf_dataset.py         # CQFDataset: annotations.pkl → 训练样本
  train_cqf.py           # 三阶段训练脚本
  eval_cqf.py            # 离线验证 (ranking acc, bounce detection, score spread)
  config_cqf.json        # 超参数配置
```

---

## 十四、完整实现顺序

```
Step 1: CQFDataset
  ├── 读annotations.pkl + HDF5
  ├── 正样本(label=1) + 负样本(label=2) 构造
  ├── 边界处理 (chunk越界, h越界, pre_bounce帧有效性过滤)
  ├── Weighted Sampler (1:1 正负比)
  └── 验证: 打印样本分布, 可视化几个正负样本

Step 2: ContactQualityScorer
  ├── 3-branch架构实现
  ├── Action branch的dropout mask
  └── 验证: forward pass维度检查

Step 3: Phase A训练 (GT tactile)
  ├── Ranking Loss + BCE Loss
  ├── Hard Pair Mining
  └── 验证: 训练loss下降, 验证集ranking accuracy

Step 4: Phase B训练 (混合)
  ├── 需要Foresight模型
  ├── GT比例线性衰减
  ├── 扰动负样本生成
  └── 验证: ranking accuracy在Foresight输入下不大幅下降

Step 5: Phase C训练 (纯Foresight)
  └── 验证: 全部离线指标通过

Step 6: Reranking集成
  ├── 集成到DP推理pipeline
  ├── 离线全面验证
  └── 通过后→真机实验
```

注意: Step 1-2不依赖其他模型，可以立即开始。
Step 3的Phase A可以在raw marker空间做 (不需要TactileVAE)，用于快速验证思路。
Step 4-5需要TactileVAE和Foresight。

---

## 十五、待讨论确认的决策

| # | 问题 | 选项 | 推荐 |
|---|------|------|------|
| 1 | Foresight horizon h | ~~A: h=15 / B: h=10+过滤 / C: 多h~~ | **已定: B (h=10+过滤)** |
| 2 | Action是否作为CQF输入 | A: 去掉 / B: 保留+50%dropout | B |
| 3 | 先做Core Scorer还是PD-CQV | 先Core，验证后加Physics Heads | Core first |
| 4 | Phase A是否可在raw space做 | 可以，快速验证不需要等TactileVAE | 是 |
| 5 | TactileVAE latent维度 | 等训好确认 | — |
