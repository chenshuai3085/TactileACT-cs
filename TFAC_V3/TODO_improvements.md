# TFAC V3 实验与改进计划

> 更新时间: 2026-04-13

---

## 一、已完成的改动

### 1. Per-Frame Contrastive Learning (多帧对比学习)

**问题**: V3 predict_horizon=10 预测 10 帧未来触觉，但对比学习只对齐最后一帧 (t+H)，导致梯度失衡:
- 像素级监督 (foresight_tac + sampling): 覆盖 13 帧
- 语义级监督 (contrastive): 仅 1 帧
- 比例 13:1，前 9 个 future_query 完全没有语义梯度

**改动**:
- `dataset.py`: 视觉相机也返回多帧 future `(H, C, Himg, Wimg)`, 由 `multi_frame_vision=True` 控制
- `tfac_model.py:_compute_gt_future_features`: 支持多帧 GT 视觉编码, 返回 `(B, H, D)`
- `tfac_policy.py`: contrastive 和 contrastive_gt 都改为 per-frame, 遍历 H 帧独立 InfoNCE 再平均
- 三组 config 都已加 `"multi_frame_vision": true`

**改后梯度分布**: 像素:语义 = 13:10 (远优于原来的 13:1)

### 2. Bug 修复

- **foresight_vis shape mismatch**: `v_gt` 从单帧 `(B,D)` 变成多帧 `(B,H,D)` 后, `F.mse_loss(v_hat, v_gt)` 会报错。修复: 取最后帧 `v_gt[:, -1]`
- **GT contrastive 多帧兼容**: `marker_encoder(t_gt)` 在 `t_gt` 为 `(B,H,9,9,2)` 时会出错。修复: per-frame loop

---

## 二、当前待跑实验: 三组对比学习消融

验证 per-frame contrastive 对 action 质量 (l1_final) 的影响。

### 实验配置

| 实验组 | config | lambda_contrastive | lambda_contrastive_gt | 说明 |
|--------|--------|-------------------|----------------------|------|
| no_contrastive | config_v3_no_contrastive.json | 0 | 0 | 无对比学习 (baseline) |
| yes_contrastive | config_v3_yes_contrastive.json | 0.1 | 0 | 仅 Pred_T ↔ GT_V |
| dual_contrastive | config_v3_dual_contrastive.json | 0.1 | 0.05 | Pred_T ↔ GT_V + GT_T ↔ GT_V |

### 共同参数

```
predict_horizon=10, history_len=3, sampling_steps=3
lambda_foresight=0.7, lambda_draft=0.5, lambda_sampling=0.5
kl_weight=1, curriculum_ratio=0.4
tactile_mode=marker, marker_encoder=pointnet, fusion=gate
a2_init=a1_refine, spatial_tac_dec_layers=3
multi_frame_vision=true (新增)
batch_size=64, num_epochs=2000
```

### 训练命令

```bash
conda run -n TactileACT python TFAC_V3/train.py --config TFAC_V3/config_v3_no_contrastive.json
conda run -n TactileACT python TFAC_V3/train.py --config TFAC_V3/config_v3_yes_contrastive.json
conda run -n TactileACT python TFAC_V3/train.py --config TFAC_V3/config_v3_dual_contrastive.json
```

### 关注指标

- **l1_final** (主指标): best ckpt 按 l1_final 选取
- contrastive loss 下降趋势: per-frame 应比原单帧更平稳
- gate 权重分布: memory/a1/future 的比例
- 最终物理评估成功率

---

## 三、架构分析: 已识别的设计问题

基于代码 review + VT-WM / ViTacFormer 论文对比，按严重程度排序:

### 问题 1 (Critical): Future Query 之间无通信

**现状** (`foresight_transformer.py`):
```python
queries = self.future_queries.expand(H, B, D)  # H=10 个独立 query
q_out = self.future_cross_attn(queries, vt, vt)[0]  # 每个独立 cross-attend
# 没有 self-attention，frame t+3 不知道 frame t+1 预测了什么
```

**后果**: 10 帧预测完全独立，无法建模帧间时序依赖 (如 "触觉变化是渐进的")。VT-WM 用 GRU 自回归天然有序；我们的 queries 无序。

### 问题 2 (Significant): Predict → Re-encode Roundtrip

**现状**: Foresight 预测 raw 9×9×2 marker → 再用 MarkerEncoder(PointNet) 编码回 embedding → 用于 contrastive/fusion。

**后果**: 梯度路径长 (decoder → encoder 双重变换)，信息瓶颈 (latent → 162维raw → 512维latent)。VT-WM 全程在潜在空间预测，无此问题。

### 问题 3 (Medium): Foresight 用 Backbone 特征而非 Encoder Memory

**现状** (`tfac_model.py`):
```python
v_tokens = src[:n_vision]   # backbone 输出, 未经 Transformer encoder
t_tokens = src[n_vision:]   # 同上
```

**后果**: Foresight 拿到的是单模态特征，缺少 encoder 中的跨模态交互信息。

### 问题 4 (Minor): GatedFusion Softmax 零和竞争

三路 softmax 门控意味着 future_tac 权重上升必然挤压 memory/a1。可能限制 fusion 表达力。

---

## 四、后续改进计划 (按优先级)

### P0: Future Query Causal Self-Attention (最小改动, 最大收益)

在 foresight_transformer.py 的 `ForesightTransformer.forward` 中, future_cross_attn 之后加一步 causal self-attention:

```python
# 新增: future queries 之间的因果自注意力
self.future_self_attn = nn.MultiheadAttention(hidden_dim, nhead)
causal_mask = torch.triu(torch.ones(H, H, device=q_out.device), diagonal=1).bool()
q_out2 = self.future_self_attn(q_out, q_out, q_out, attn_mask=causal_mask)[0]
q_out = self.future_norm2(q_out + q_out2)
```

**预期效果**: frame t+k 能看到 t+1..t+k-1 的预测，建模渐进变化。

**改动量**: ~15 行代码

### P1: Embedding-Space 预测头 (避免 roundtrip)

并行两条路径:
- 原始路径: SpatialTactileDecoder → raw 9×9×2 (用于 foresight_tac 像素级 loss)
- 新增路径: Linear(D, D) → embedding (用于 contrastive + fusion, 无需 re-encode)

```python
self.embed_predictor = nn.Linear(hidden_dim, hidden_dim)
t_hat_embed = self.embed_predictor(q_out)    # 直接 embedding, 用于 contrastive/fusion
t_hat_raw = self.spatial_tac_decoder(q_out)  # raw marker, 用于 foresight_tac loss
```

**预期效果**: contrastive/fusion 的梯度路径缩短一半, 避免 decoder→encoder roundtrip。

**改动量**: ~30 行 (foresight_transformer.py + tfac_model.py + tfac_policy.py)

### P2: Future Query 时序位置编码

```python
self.future_temporal_pe = nn.Parameter(torch.randn(H, 1, D))
queries = self.future_queries + self.future_temporal_pe  # 加上时序信息
```

**改动量**: ~5 行, 可和 P0 一起做。

### P3 (探索性): Token-Level 对比学习

受 ViTacFormer 启发, 将 global InfoNCE 改为 spatial token 级别密集对齐。需要更多调研, 暂缓。

---

## 五、论文调研笔记

### VT-WM (Higuera et al., 2026) — Visuo-Tactile World Models

- **方法**: RSSM (GRU + 离散随机变量) 在潜在空间建模视觉-触觉动态
- **关键**: 全程在 latent space 预测, 不回原始空间; GRU 自回归天然有序
- **对我们的启示**: 避免 predict→reencode roundtrip; 多帧预测需要帧间通信机制
- **区别**: model-based RL (需要 planning), 我们是 imitation learning

### ViTacFormer (Heng et al., 2025) — Cross-Modal Visuo-Tactile Representation

- **方法**: 双流编码器 + 交替式 Cross-Attention (V→T, T→V 交替堆叠)
- **对比学习**: token-level 密集对齐, 不只是 global CLS token
- **对我们的启示**: token-level contrastive 可提供更丰富的语义信号
- **区别**: 只做当前时刻表示学习, 不做未来预测

---

## 六、实验时间线

| 阶段 | 内容 | 状态 |
|------|------|------|
| 消融 Round 1 | 三组 contrastive 消融 (no/yes/dual) | **待启动** |
| 分析 Round 1 | 对比 l1_final + 物理评估 | 等 Round 1 完成 |
| 改进 P0+P2 | causal self-attn + temporal PE | 等消融结论, 需讨论确认 |
| 改进 P1 | embed predictor (避免 roundtrip) | 等 P0 验证, 需讨论确认 |
| 消融 Round 2 | P0/P1 改进后的消融实验 | 待定 |
