# TFAC V1 vs V3 技术差异报告

> 日期: 2026-04-15 | 用途: 组会汇报

---

## 一、整体架构对比

两个版本共享 **Think → Dream → Act** 的核心范式，但 V3 在多个维度进行了升级：

| 维度 | V1 (TFAC/) | V3 (TFAC_V3/) |
|------|-----------|---------------|
| 训练阶段 | **1 阶段** 端到端 | **1 阶段** 端到端 (代码保留3阶段能力, 当前未启用) |
| ForesightTransformer | SelfAttn + CrossAttn | **Factorized**: Spatial → Temporal → CrossAttn |
| 历史帧输入 | 仅当前帧 | 支持历史 k 帧 (max_history=8) |
| 预测帧数 | 单帧 (t+h) | **多帧** (t+1, ..., t+H), predict_horizon 可配 |
| Sampling Loss | 无 | **自回归展开** S 步, 模拟推理误差累积 |
| Embedding 预测 | roundtrip (预测→MarkerEncoder→embedding) | **embed_predictor** MLP 直出 embedding |
| 对比学习 | 单帧, 仅 pred vs GT | **per-frame 对比** + 可选 GT 对比 (lambda_contrastive_gt) |
| 多 GPU | 不支持 | **DataParallel** 支持 |
| Backbone 构建 | 外部传入 CLIP pretrained backbone | 内建 ImageNet ResNet18 + FrozenBatchNorm |

---

## 二、训练流程

### 当前实际使用: 1 阶段端到端训练

V1 和 V3 **当前都是 1 阶段端到端训练**。V3 的 backbone 使用 ImageNet pretrained ResNet18, 不经过 CLIP/Foresight 预训练。

**V1**: backbone 通过外部传入的 CLIP pretrained 权重初始化 (MyJoiner)。
**V3**: backbone 使用 `detr.models.backbone.Backbone` 内建的 ImageNet ResNet18 + FrozenBatchNorm。

### 代码保留的 3 阶段能力 (当前未启用)

V3 代码中保留了 `pretrain_clip.py` 和 `pretrain_foresight.py`, 以及 `train.py` 中的预训练权重加载逻辑, 但当前配置中 **未设置 `pretrain_foresight_ckpt`**, 所以实际不走预训练流程。

```
[代码保留, 当前未启用]
Stage 0: Vision-Tactile CLIP 预训练 (pretrain_clip.py)
Stage 1: Foresight 预训练 (pretrain_foresight.py)
Stage 2: 联合训练 — 加载预训练权重 + foresight lr 缩放 0.1x
```

---

## 三、ForesightTransformer 结构差异

### V1: 简单两阶段注意力

```
[V; T] tokens → SelfAttn([V;T]) → CrossAttn(Q=[V;T], K/V=A1) → FFN
                 ↑ 所有 tokens 一起做 self-attn, 无时序概念
```

每层 `ForesightLayer` 包含:
1. **SelfAttn**: V 和 T tokens 全局交互
2. **CrossAttn**: 用 A1 (draft action) 作为条件
3. **FFN**

### V3: Factorized Attention (借鉴 VT-WM)

```
[V; T] tokens (k帧) → SpatialSelfAttn → TemporalSelfAttn → CrossAttn(A1) → FFN
                        ↑ 同帧内交互       ↑ 跨帧同位置交互    ↑ action 条件
```

每层 `ForesightLayer` 包含:
1. **Spatial SelfAttn**: 同一时刻内 V/T tokens 交互 → reshape 把 k 并入 batch
2. **Temporal SelfAttn**: 同一空间位置跨 k 个时间步交互 → reshape 把 N_vt 并入 batch
3. **CrossAttn**: Q=所有 tokens, K/V=A1
4. **FFN**

**自动退化**: k=1 时 V3 跳过 Temporal SelfAttn, 退化为 V1 行为。

### V3 多帧预测机制 (predict_horizon > 1)

```
Foresight 输出 → future_queries (H, B, D)
                     ↓
              CrossAttn(Q=queries, K/V=vt_features)
                     ↓
              Causal SelfAttn (t+k 只看 t+1..t+k-1)
                     ↓
              tactile_out → raw marker (B, H, 162)
              embed_predictor → embedding (B, H, D)
```

- 使用 H 个可学习 future query tokens
- Cross-attention 从 foresight 特征中提取信息
- **Causal self-attention**: 保证时序因果性, t+k 不能看到 t+k+1
- 双路输出: raw marker prediction (给像素级 loss) + embedding (给 contrastive/fusion)

---

## 四、Sampling Loss (V3 独有)

**目的**: 训练时 foresight 用 GT 触觉作为输入, 推理时只能用自己的预测结果 → 存在 train-test gap. Sampling loss 模拟推理时的误差累积。

```python
# 自回归展开 S 步:
步骤 0: 当前触觉 → foresight → 预测 t+s0 → 与 GT[s0] 算 loss
步骤 1: 用预测的 t+s0 替换输入 → foresight → 预测 t+s1 → 与 GT[s1] 算 loss
步骤 2: 用预测的 t+s1 替换输入 → foresight → 预测 t+s2 → 与 GT[s2] 算 loss
...
最终 loss = 各步 loss 平均
```

**关键优化**:
- 复用主 forward 中缓存的 backbone 特征 (`model._fwd_cache`), 避免重复计算
- 均匀采样帧索引覆盖整个预测范围 (e.g. S=3, H=10 → [0, 4, 9])
- 历史编码跳过当前帧, 仅替换触觉 tokens

---

## 五、Embedding Predictor (V3 独有)

### V1: roundtrip 路径
```
foresight → tactile_out → raw marker (B,9,9,2)
                                ↓ MarkerEncoder
                          t_hat_encoded (B, D)  → 给 fusion + contrastive
```
问题: 需要经过 raw → encode 的 roundtrip, 梯度路径长, 且 raw prediction 的误差会传导到 embedding。

### V3: embed_predictor 直出
```
foresight → embed_predictor (MLP) → t_embed_future (B, D)  → 给 fusion + contrastive
          → tactile_out → raw marker (B, H, 162)            → 给像素级 loss
```
优势: fusion 和 contrastive 使用专门的 MLP 直出 embedding, 不依赖 raw prediction 质量。

---

## 六、对比学习差异

### V1
```python
# 单帧: 预测触觉 vs GT 视觉
loss_contrastive = contrastive(v_gt, t_hat_encoded)
```

### V3
```python
# per-frame 对比: 每帧分别做 InfoNCE, 然后取平均
for h in range(H):
    loss_h = contrastive(v_gt[:, h], t_embed_future[:, h])
contrastive = mean(losses)

# 可选: GT 对比 (lambda_contrastive_gt)
# GT 触觉 vs GT 视觉, 加强跨模态对齐
for h in range(H):
    t_gt_h_enc = marker_encoder(t_gt[:, h])
    loss_gt_h = contrastive(v_gt[:, h], t_gt_h_enc)
```

---

## 七、Loss 函数对比

### V1 总 Loss
```
loss = l1_final
     + 0.5  * l1_draft
     + 1.0  * foresight_tac      (smooth_L1 / MSE)
     + 0.3  * foresight_vis      (MSE)
     + 0.1  * contrastive        (InfoNCE)
     + 10   * kl                 (KL divergence)
```

### V3 总 Loss
```
loss = l1_final
     + 0.5  * l1_draft
     + 1.0  * foresight_tac      (多帧 smooth_L1 平均)
     + 0.3  * foresight_vis      (MSE, 比较最后帧)
     + 0.1  * contrastive        (per-frame InfoNCE)
     + λ_gt * contrastive_gt     (可选, GT 对比)
     + 0.5  * sampling           (自回归展开 loss)     ← V3 新增
     + 10   * kl
```

---

## 八、Backbone 与训练基础设施

| 方面 | V1 | V3 |
|------|----|----|
| Backbone 来源 | 外部传入 (MyJoiner + CLIP pretrained) | 内建 ImageNet ResNet18 + FrozenBatchNorm |
| Backbone 数量 | 可能 2 个 (vision + gelsight 分别) | 统一 1 个 (marker mode 不需要 gelsight backbone) |
| DataLoader | 8 workers, 无 persistent_workers | 自适应 (cache=2, 否则=8) + persistent_workers |
| 多 GPU | 不支持 | DataParallel (tuple 递归 scatter) |
| 预训练权重加载 | 手动 load_state_dict | 自动 key mapping + foresight lr_scale |
| Optimizer | 2 组: backbone + 其他 | 3 组: backbone + foresight (0.1x lr) + 其他 |

---

## 九、Model forward 返回值对比

### V1: 9 个值
```python
(a1_hat, a2_hat, t_hat_future, v_hat_future,
 v_gt_feat, t_gt_feat, t_hat_encoded, t_current_feat, (mu, logvar))
```

### V3: 10 个值
```python
(a1_hat, a2_hat, t_hat_future, v_hat_future,
 v_gt_feat, t_gt_feat, t_hat_encoded, t_current_feat, (mu, logvar),
 t_embed_future)     ← V3 新增: embed_predictor 直出的 embedding
```

---

## 十、不变的部分

以下模块在 V1 和 V3 中**完全一致**:
- GatedFusion (三路 softmax 门控)
- CVAE encoder (训练时推断 z, 推理时 z=0)
- Dual decoder 架构 (Decoder_draft A1 + Decoder_final A2)
- Token fusion / LTD fusion / Gate fusion 三种融合模式
- a2_init: "zero" / "a1_refine"
- ForesightContrastive (InfoNCE projection head)
- SpatialTactileDecoder (ConvTranspose2d 上采样)
- 课程学习 (curriculum_ratio 控制 GT/预测切换)
- MarkerEncoder (Conv2D / PointNet)

---

## 十一、关键代码文件映射

| 功能 | V1 文件 | V3 文件 |
|------|--------|--------|
| 核心模型 | `TFAC/tfac_model.py` | `TFAC_V3/tfac_model.py` |
| Foresight | `TFAC/foresight_transformer.py` | `TFAC_V3/foresight_transformer.py` |
| Policy | `TFAC/tfac_policy.py` | `TFAC_V3/tfac_policy.py` |
| 训练 | `TFAC/train.py` | `TFAC_V3/train.py` |
| 数据集 | `TFAC/dataset.py` | `TFAC_V3/dataset.py` |
| CLIP 预训练 | — | `TFAC_V3/pretrain_clip.py` |
| Foresight 预训练 | — | `TFAC_V3/pretrain_foresight.py` |

---

## 十二、一句话总结

**V1** 是最小可行的 Think→Dream→Act: 单帧输入 → 简单 attention 预测未来 → 门控融合 → 精化动作。

**V3** 在此基础上引入 **时序建模** (Factorized Attention + 历史帧)、**多帧预测** (future queries + causal attention)、**Sampling Loss** (缩小 train-test gap)、以及 **embed_predictor** (避免 roundtrip), 是当前最完整的版本。两者当前都是 1 阶段端到端训练。
