# TFAC V1 vs V3 简要对比

> 2026-04-15 组会用

共同点：Think → Dream → Act 范式、双 Decoder (A1 草稿 + A2 精化)、GatedFusion、CVAE、课程学习、MarkerEncoder 均一致。都是 1 阶段端到端训练。

---

## V3 相比 V1 的核心改动

**1. Foresight 从单帧 → 时序多帧**

- V1: 只看当前帧，SelfAttn + CrossAttn(A1)，预测单帧 t+h 触觉
- V3: 输入历史 k 帧 (当前 k=3)，Factorized Attention (Spatial → Temporal → CrossAttn)，用 future queries + causal self-attention 预测 t+1 到 t+H 共 H 帧触觉 (当前 H=10)

**2. Sampling Loss (缩小 train-test gap)**

- V1: 无
- V3: 自回归展开 S 步 — 用上一步的预测触觉替换输入，重新跑 foresight，对 GT 算 loss。模拟推理时只能用自己预测结果的场景，防止误差累积。复用 backbone 缓存避免重复计算。

**3. Embed Predictor (解耦 raw 预测和 embedding)**

- V1: 预测 raw marker → 再过 MarkerEncoder 得 embedding，给 fusion/contrastive 用 (roundtrip)
- V3: 新增 embed_predictor MLP 直接输出 embedding，raw 预测只给像素级 loss，fusion/contrastive 不再依赖 raw 质量

**4. Per-frame 对比学习**

- V1: 单帧 InfoNCE (预测触觉 vs GT 视觉)
- V3: 每帧分别做 InfoNCE 再平均，可选额外 GT 对比 (GT 触觉 vs GT 视觉)

**5. 工程改进**

- DataParallel 多 GPU 支持
- Backbone 改为内建 ImageNet ResNet18 (V1 用外部传入的 CLIP backbone)
- DataLoader 自适应 worker + persistent_workers

---

## 当前 V3 配置要点

| 参数 | 值 |
|------|-----|
| history_len | 3 |
| predict_horizon | 10 |
| sampling_steps | 3 |
| lambda_sampling | 0.5 |
| curriculum_ratio | 0.4 (前40% 用 GT，后60% 用预测) |
| foresight_layers | 3 |
| fusion_mode | gate |
| a2_init | a1_refine |
