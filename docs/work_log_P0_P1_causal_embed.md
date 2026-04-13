# Work Log: P0 + P1 架构改进

**日期**: 2026-04-13  
**分支**: tacfore  
**Commit**: `V3: P0 future query causal self-attention + P1 embed_predictor 消除roundtrip`

---

## 改动背景

V3 多帧预测 (predict_horizon > 1) 存在两个结构性问题:

### P0 — Future Queries 无通信

H 个 future query 各自独立 cross-attend to vt features, 互相不知道对方预测了什么。触觉变化是连续渐进的, t+3 应该参考 t+1/t+2 的信息。

**问题**: 每帧独立预测, 无时序连贯性。

### P1 — Predict -> Re-encode Roundtrip

Foresight 内部 query 已经是 512 维语义向量, 但硬要先 decode 成 162 维 raw marker (SpatialTactileDecoder), 再 encode 回 512 维 (MarkerEncoder/PointNet), 才能用于 contrastive 和 fusion。

**问题**: 信息瓶颈 (512->162->512) + 梯度路径过长。

---

## 修改方案

### P0: Causal Self-Attention

在 future_cross_attn 之后, 加一层 causal self-attention:

```
future_queries (H, B, D)
    |
    v
cross_attn(Q=queries, K/V=vt_context)   -- 已有
    |
    v
causal_self_attn(Q/K/V=q_out)           -- 新增: 上三角 mask
    |                                       t+k 能看到 t+1..t+k-1
    v
q_out (H, B, D)                          -- 时序连贯的 future queries
```

Causal mask: `torch.triu(ones(H, H), diagonal=1).bool()` — 上三角为 True (被 mask), 即每个 query 只能 attend 到自身和之前的 query。

### P1: Embed Predictor

新增 2 层 MLP, 从 foresight query 直接输出 512 维 embedding:

```
foresight query (H, B, D=512)
    |
    +---> tactile_out (SpatialTactileDecoder) --> raw 9x9x2  (保留, 给 foresight_tac loss)
    |
    +---> embed_predictor (MLP: D->D->D)     --> (H, B, D)   (新增, 给 contrastive + fusion)
```

两条路径并行:
- **tactile_out**: 像素级监督 (foresight_tac loss) 不变
- **embed_predictor**: contrastive loss 和 GatedFusion 改用此路径, 消除 roundtrip

---

## 修改文件清单

### 1. `TFAC_V3/foresight_transformer.py` (核心)

**`__init__`** (predict_horizon > 1 分支内新增):
```python
# P0: Causal self-attention
self.future_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
self.future_self_norm = nn.LayerNorm(d_model)

# P1: Embed predictor (bypass decode->re-encode)
self.embed_predictor = nn.Sequential(
    nn.Linear(d_model, d_model),
    nn.GELU(),
    nn.Linear(d_model, d_model),
)
```

**`forward` 多帧分支**:
```python
# Cross-attention (已有)
q_out = self.future_cross_attn(queries, vt, vt)[0]
q_out = self.future_norm(queries + q_out)

# P0: Causal self-attention (新增)
causal_mask = torch.triu(torch.ones(H, H, device=q_out.device), diagonal=1).bool()
q_out2 = self.future_self_attn(q_out, q_out, q_out, attn_mask=causal_mask)[0]
q_out = self.future_self_norm(q_out + q_out2)

# P1: Embed predictor 并行输出 (新增)
t_embed_future = self.embed_predictor(q_out).permute(1, 0, 2)  # (B, H, D)
```

**返回值**: 从 2 个 -> 3 个: `t_hat_future, v_hat_future, t_embed_future`

**单帧分支**: 也加了 embed_predictor, 返回 `(B, D)` 的 t_embed_future。

### 2. `TFAC_V3/tfac_model.py`

- Foresight 调用解包 3 值: `t_hat_raw, v_hat_future, t_embed_future`
- **关键改动**: `t_hat_encoded` 从 `marker_encoder(t_hat_last)` (roundtrip) 改为 `t_embed_future[:, -1]` (直接 embed)
- 返回元组从 9 个扩为 10 个, 新增 `t_embed_future`

### 3. `TFAC_V3/tfac_policy.py`

- Training 路径解包 10 值, 新增 `t_embed_future`
- **Contrastive loss**: per-frame contrastive 改用 `t_embed_future[:, h]`, 不再每帧 `marker_encoder(t_hat_h)` roundtrip
- GT contrastive (`contrastive_gt`) **不变** — 它对齐 GT_T <-> GT_V, 仍需 marker_encoder encode GT
- Inference 路径解包 10 值 (多一个 `_`)
- Sampling loss 中 foresight 调用解包 3 值

### 4. `TFAC_V3/pretrain_foresight.py`

- Foresight 调用解包 3 值: `t_hat_raw, v_hat_future, _`

### 5-7. `TFAC_V3/eval_*.py` (eval_foresight, eval_contrastive, eval_foresight_sequence)

- `policy.model()` 调用解包加 `_t_embed`

---

## 不变的部分

| 模块 | 说明 |
|------|------|
| foresight_tac loss | 仍用 raw 9x9x2 vs GT marker, 像素级监督 |
| GT contrastive loss | 仍用 marker_encoder(t_gt), 与 foresight 路径无关 |
| Sampling loss | 仍用 roundtrip (detach 自回归展开, 梯度不回传) |
| ForesightContrastive | 接口不变, 输入仍是 (B, D) |
| GatedFusion | 接口不变, future_feat 来源从 roundtrip 变为 embed_predictor |
| Config | 无新参数, predict_horizon>1 时自动生效 |

---

## 参数量变化

| 新增模块 | 参数量 |
|----------|--------|
| future_self_attn (MHA, 512d, 4head) | ~1.05M |
| future_self_norm (LayerNorm) | ~1K |
| embed_predictor (512->512->512 MLP) | ~0.53M |
| **总计新增** | **~1.6M** |
| 模型总参数 (之前 ~118M -> 现在 ~120M) | |

---

## 向后兼容

- `predict_horizon=1`: 不创建 future_self_attn, embed_predictor 走单帧路径, 行为与修改前一致
- `tactile_mode="image"`: 原来就没有 roundtrip, embed_predictor 路径同样生效
- 旧 checkpoint: 加载时 missing keys 会包含 `future_self_attn.*`, `future_self_norm.*`, `embed_predictor.*`, 这些模块随机初始化, 需要 fine-tune

---

## 验证

- ForesightTransformer 单元测试: 输入输出 shape 正确, 3 返回值
- predict_horizon=1 向后兼容测试: 通过
- TFACPolicy 完整 forward + backward + inference 集成测试: 通过
- 无 shape error, loss 正常下降

---

## 预期效果

1. **P0 causal self-attn**: 多帧预测更连贯, 后帧可参考前帧信息, 减少帧间跳变
2. **P1 embed_predictor**: 消除 512->162->512 信息瓶颈, 梯度路径更短, contrastive 对齐更准确
