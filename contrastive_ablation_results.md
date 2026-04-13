# TFAC 对比学习消融实验 — 完整结果汇总

> 三组模型: no_contrastive / yes_contrastive / dual_contrastive
> Best ckpt 均按 l1_final (动作质量) 选取, 共 1000 epochs
> 验证集: 68 episodes, 物理评估采样 3400 samples (68 episodes × 50 timesteps/episode)

---

## 1. 训练配置

| 超参数 | no_contrastive | yes_contrastive | dual_contrastive |
|--------|:-:|:-:|:-:|
| λ_contrastive (PredT↔GTV) | 0 | 0.1 | 0.1 |
| λ_contrastive_gt (GTV↔GTT) | — | — | 0.1 |
| λ_draft | 0.5 | 0.5 | 0.5 |
| λ_foresight | 0.7 | 0.7 | 0.7 |
| λ_foresight_vis | 0 | 0 | 0 |
| kl_weight | 10 | 10 | 10 |
| curriculum_ratio | 0.4 | 0.4 | 0.4 |
| foresight_change_weight | True | True | True |
| chunk_size | 10 | 10 | 10 |
| enc_layers / dec_layers | 4 / 7 | 4 / 7 | 4 / 7 |
| dec_layers_draft | 7 | 7 | 7 |
| foresight_layers | 3 | 3 | 3 |
| foresight_horizon | 8 | 8 | 8 |
| tactile_mode | marker | marker | marker |
| fusion_mode | gate | gate | gate |
| foresight_tac_decoder | spatial | spatial | spatial |
| a2_init | a1_refine | a1_refine | a1_refine |

---

## 2. 训练验证指标 (Best Epoch)

| 指标 | no_contrastive | yes_contrastive | dual_contrastive | 说明 |
|------|:-:|:-:|:-:|------|
| l1_final ↓ | 0.0846 | **0.0720** | 0.0840 | 最终动作误差 |
| l1_draft ↓ | 0.0843 | **0.0770** | 0.0826 | 草稿动作误差 |
| foresight_tac ↓ | **0.0687** | 0.1346 | 0.1258 | 触觉预测误差 |
| foresight_vis | 0.955 | 1.018 | 1.049 | 视觉预测误差 |
| contrastive | 2.968 | 0.469 | 0.423 | 对比学习 loss |
| contrastive_gt | — | — | 1.393 | GT对比学习 loss |
| kl | 0.0073 | 0.0079 | 0.0078 | KL散度 |
| total loss | **0.248** | 0.331 | 0.473 | 总损失 |
| gate_mem | 0.318 | 0.307 | 0.299 | 门控: memory权重 |
| gate_a1 | 0.347 | 0.347 | 0.336 | 门控: draft action权重 |
| gate_fut | 0.335 | 0.345 | **0.366** | 门控: future tactile权重 |

---

## 3. 跨模态检索 (语义对齐, 68 samples)

| 指标 | no_contrastive | yes_contrastive | dual_contrastive |
|------|:-:|:-:|:-:|
| V→T R@1 ↑ | 1.5% | **76.5%** | 70.6% |
| V→T R@5 ↑ | 7.4% | **100.0%** | 97.1% |
| T→V R@1 ↑ | 1.5% | **77.9%** | 67.6% |
| T→V R@5 ↑ | 11.8% | **100.0%** | 97.1% |
| 对角线相似度 | 0.023 | **0.857** | 0.712 |
| 非对角线相似度 | 0.021 | 0.152 | 0.016 |
| MedRank (V→T) ↓ | 33 | **0** | **0** |
| MedRank (T→V) ↓ | 28 | **0** | **0** |

---

## 4. t-SNE 投影空间成对距离 (cosine distance, 68 samples)

| 成对距离 | no_contrastive | yes_contrastive | dual_contrastive | 含义 |
|----------|:-:|:-:|:-:|------|
| PredT ↔ GTV ↓ | 0.975 | **0.146** | 0.259 | 预测触觉与GT视觉的距离 |
| PredT ↔ GTT | **0.019** | 0.880 | 0.725 | 预测触觉与GT触觉的距离 |
| GTV ↔ GTT | 0.978 | 0.867 | **0.752** | GT视觉与GT触觉的距离 |

> no: PredT 紧贴 GTT (像素级复现)
> yes: PredT 紧贴 GTV (语义对齐成功)
> dual: PredT 在 GTV 和 GTT 之间 (两个loss拉扯)

---

## 5. foresight_tac 与 l1_final 逐样本相关性 (68 samples)

| 指标 | no_contrastive | yes_contrastive | dual_contrastive |
|------|:-:|:-:|:-:|
| Pearson r | 弱/不显著 | 弱/不显著 | 弱/不显著 |
| 显著性 p>0.05 | 是 | 是 | 是 |

> 结论: 触觉预测精度不直接决定动作质量

---

## 6. 综合物理评估 (3400 samples)

### 6.1 方向准确性

| 指标 | no_contrastive | yes_contrastive | dual_contrastive |
|------|:-:|:-:|:-:|
| 平均角度误差 ↓ | **15.3°** | 25.9° | 24.3° |
| 中位数角度误差 ↓ | **7.2°** | 12.5° | 11.7° |
| 准确率@30° ↑ | **87.9%** | 75.4% | 76.8% |
| 准确率@45° ↑ | **92.8%** | 83.8% | 85.0% |

### 6.2 接触区域定位

| 指标 | no_contrastive | yes_contrastive | dual_contrastive |
|------|:-:|:-:|:-:|
| IoU ↑ | **0.678** | 0.518 | 0.531 |
| Precision ↑ | **0.832** | 0.756 | 0.765 |
| Recall ↑ | **0.784** | 0.611 | 0.625 |

### 6.3 幅值相关性

| 指标 | no_contrastive | yes_contrastive | dual_contrastive |
|------|:-:|:-:|:-:|
| 平均 Pearson r ↑ | **0.821** | 0.544 | 0.571 |
| 中位数 r ↑ | **0.881** | 0.620 | 0.639 |
| r > 0.7 占比 ↑ | **82.5%** | 42.2% | 42.7% |
| r > 0.5 占比 ↑ | **94.1%** | 59.4% | 63.1% |

### 6.4 空间 RMSE

| 指标 | no_contrastive | yes_contrastive | dual_contrastive |
|------|:-:|:-:|:-:|
| 整体均值 RMSE ↓ | **1.172** | 1.982 | 1.828 |

---

## 7. 分阶段物理评估 (按接触强度, 3400 samples)

### 低接触阶段 (GT magnitude: 0.1-3.3)

| 指标 | no_contrastive | yes_contrastive | dual_contrastive |
|------|:-:|:-:|:-:|
| 角度误差 ↓ | **20.9°** | 36.9° | 34.9° |
| IoU ↑ | **0.643** | 0.421 | 0.451 |
| 幅值相关 ↑ | **0.744** | 0.308 | 0.367 |
| RMSE ↓ | **1.042** | 1.894 | 1.750 |

### 中接触阶段 (GT magnitude: 3.3-4.3)

| 指标 | no_contrastive | yes_contrastive | dual_contrastive |
|------|:-:|:-:|:-:|
| 角度误差 ↓ | **13.5°** | 24.2° | 22.6° |
| IoU ↑ | **0.638** | 0.454 | 0.469 |
| 幅值相关 ↑ | **0.814** | 0.520 | 0.527 |
| RMSE ↓ | **1.077** | 1.976 | 1.827 |

### 高接触阶段 (GT magnitude: 4.3-14.0)

| 指标 | no_contrastive | yes_contrastive | dual_contrastive |
|------|:-:|:-:|:-:|
| 角度误差 ↓ | **11.7°** | 17.2° | 15.8° |
| IoU ↑ | **0.751** | 0.679 | 0.672 |
| 幅值相关 ↑ | **0.905** | 0.803 | 0.819 |
| RMSE ↓ | **1.396** | 2.077 | 1.906 |

---

## 8. 核心发现总结

| 评估维度 | 最佳模型 | 关键数据 |
|---------|---------|---------|
| 动作质量 (l1_final) | **yes_contrastive** | 0.072 vs 0.085 (改善 17.4%) |
| 触觉物理精度 (全部物理指标) | **no_contrastive** | 角度误差 15.3° vs 25.9°, RMSE 1.17 vs 1.98 |
| 语义对齐 (跨模态检索) | **yes_contrastive** | R@1=76.5% vs 1.5% |
| 投影空间分布 (t-SNE) | **yes_contrastive** | PredT↔GTV 距离 0.146 vs 0.975 |
| 触觉-动作相关性 | 三者均弱 | 不显著 (p>0.05) |

### 核心悖论

> 物理精度最高的模型 (no_contrastive) 动作最差, 动作最好的模型 (yes_contrastive) 物理精度最低。

### 解释

> 对比学习的作用不是让触觉预测"更像真实物理", 而是引导 foresight 学习**对动作决策有用的语义特征** — 从"忠实重建触觉"转向"提取行动相关的触觉语义"。

### dual_contrastive 分析

> dual 加入 GTV↔GTT 对齐后, 物理精度略优于 yes 但动作质量与 no 相当 (0.084 vs 0.085)。额外的 GT 对比 loss 占 total loss ~29.4%, 挤占了动作优化的梯度预算, 且将 PredT 拉向 GTV 和 GTT 之间的折中位置, 削弱了语义对齐效果。
