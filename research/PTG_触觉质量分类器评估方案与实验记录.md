# PTG 触觉质量分类器评估方案与实验记录

日期：2026-06-09

## 目标

为 PTG 去噪过程中的梯度引导选择一个可靠的触觉质量评分/分类器。核心问题不是“训练集能不能分开”，而是：

1. GT TactileVAE latent 是否包含可区分好/坏接触的信号；
2. 分类器对未见过的 episode 是否仍然泛化；
3. 分类器是否能输出可微分 score，用于 diffusion denoising 的梯度引导；
4. 对未标注擦黑板数据，是否能先用可解释弱标签建立质量类别。

## 当前结论

之前的 99.24% 是 frame-level shuffled 5-fold CV 的测试折准确率，不是独立外部测试集准确率。它说明同分布帧在 latent 空间里高度可分，但因为相邻帧和同一 episode 可能同时出现在训练折和测试折，所以它偏乐观。

更科学的主指标应使用 episode-level split：

- 训练集和测试集按 episode 分开；
- 测试 episode 在训练中完全不可见；
- 训练折可以做类别平衡，但测试折保留自然类别比例；
- 主指标看 balanced accuracy、pre-bounce F1、ROC-AUC 和混淆矩阵。

## 已完成的快速分类结果

数据：`/home/chenshuai/Project/output/ptg_verify_level1/latent_data.npz`

任务：Success insert vs Bounce pre-bounce。

协议：平衡下采样 Insert=3000，Pre-bounce=1620，然后做 frame-level 5-fold CV。

| 排名 | 方法 | Accuracy |
|---:|---|---:|
| 1 | MLP (128,64) | 0.9924 ± 0.0017 |
| 2 | MLP (256,128) | 0.9924 ± 0.0052 |
| 3 | MLP (256,128,64) | 0.9900 ± 0.0013 |
| 4 | KNN (k=3) | 0.9894 ± 0.0029 |
| 5 | GBM (300,d=7) | 0.9840 ± 0.0047 |
| 6 | GBM (200,d=5) | 0.9825 ± 0.0068 |
| 7 | RandomForest | 0.9799 ± 0.0058 |
| 8 | KNN (k=7) | 0.9729 ± 0.0031 |
| 9 | LogReg (C=10) | 0.9680 ± 0.0037 |
| 10 | LDA | 0.9610 ± 0.0033 |
| 11 | LogReg (C=1) | 0.9582 ± 0.0039 |
| 12 | KNN (k=15) | 0.9539 ± 0.0071 |

解释：MLP 最好，但这个结果只能作为候选筛选，不能作为最终泛化结论。

## MLP 最佳方案的实现方式

输入：TactileVAE latent，形状为 144 维，来自 `mu_last.flatten()`。

模型：

```text
StandardScaler
MLPClassifier(
  hidden_layer_sizes=(128, 64),
  activation="relu",
  solver="adam",
  alpha=1e-4,
  learning_rate_init=1e-3,
  batch_size=256,
  max_iter=500,
  early_stopping=True,
  validation_fraction=0.15,
  n_iter_no_change=20
)
```

如果用于 PTG 梯度引导，sklearn MLP 只适合验证，不适合最终可微引导。最终应在 PyTorch 里实现同结构 scorer：

```text
latent z: [B, 144]
score = MLP(z)
loss_guidance = -log P(good | z_pred)
gradient = d loss_guidance / d action
```

其中 `z_pred = Foresight(obs, action)`，梯度路径是 `action -> predicted tactile latent -> scorer -> score`。

## 新增严格评估脚本

脚本：`TFAC_V5/evaluate_tactile_quality_models.py`

输出：

- `/home/chenshuai/Project/output/ptg_quality_eval/insert_prebounce_episode_cache.npz`
- `/home/chenshuai/Project/output/ptg_quality_eval/tactile_quality_model_eval.json`

评估协议：

1. `frame_cv`：帧级 shuffled 5-fold CV，偏乐观，只做上限参考。
2. `group_cv`：按 episode 的 GroupKFold，主结果。
3. `group_holdout`：20% episode holdout，最终 sanity check。
4. 训练折对 Insert 下采样，使训练类别平衡；测试折不平衡处理，只报告真实测试分布上的指标。

主排序指标：`group_cv.balanced_accuracy.mean`。

## 严格 episode-level 评估结果

数据规模：

- Insert frames：9194
- Pre-bounce frames：1620
- 总帧数：10814
- Episode groups：162

GroupKFold by episode，主指标 balanced accuracy：

| 方法 | Group-CV Balanced Acc |
|---|---:|
| LDA | 0.9039 ± 0.0167 |
| MLP (128,64) | 0.8995 ± 0.0252 |
| RandomForest-300 | 0.8954 ± 0.0230 |
| LogReg C=10 | 0.8779 ± 0.0372 |
| GBM (200,d=5) | 0.8697 ± 0.0370 |
| KNN (k=3) | 0.8402 ± 0.0471 |

20% unseen episode holdout：

| 方法 | Accuracy | Balanced Acc | Pre-bounce F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| MLP (128,64) | 0.9641 | 0.9525 | 0.8174 | 0.9885 |
| GBM (200,d=5) | 0.9450 | 0.9211 | 0.7353 | 0.9762 |
| RandomForest-300 | 0.9242 | 0.9080 | 0.6676 | 0.9704 |
| LogReg C=10 | 0.9338 | 0.8958 | 0.6874 | 0.9393 |
| LDA | 0.9387 | 0.8915 | 0.7000 | 0.9645 |
| KNN (k=3) | 0.9143 | 0.8765 | 0.6243 | 0.9054 |

解释：

1. 这些是按 episode 泛化的测试结果，比 frame-level 99.24% 更可信。
2. LDA 在 Group-CV balanced accuracy 上最稳，说明主要分界方向仍然接近线性，而且泛化方差最小。
3. MLP 在 Group-CV 的 pre-bounce F1 和 ROC-AUC 更好，并且在一次 holdout 上明显最好，说明非线性 scorer 有潜力，但对 episode 分布更敏感。
4. 当前最科学的推荐不是“直接认定 MLP 最好”，而是：
   - 若追求稳健解释和快速上线：先用 LDA/LogReg scorer；
   - 若追求 PTG 可微引导上限：PyTorch MLP scorer + episode-level validation + calibration；
   - 最终 scorer 选择以 episode-level balanced accuracy、pre-bounce recall/F1、ROC-AUC 为准。

## 擦黑板数据评估方案

数据：`/home/chenshuai/data/dataset/260522_v8l_caheiban`

当前情况：只有 `success/*.hdf5`，没有人工好/坏标注。因此不能报告真实监督分类准确率。

已采用弱标签定义：

- `too_light`：窗口平均力低于全数据 20% 分位，表示接触不足；
- `too_heavy`：窗口平均力高于 85% 分位，或 95% 峰值力高于 90% 分位；
- `rough`：力变化、action 变化或 marker 变化的 robust z-score 大于 1.0；
- `good`：力大小处于合理区间，且变化平稳。

脚本：`TFAC_V5/assess_caheiban_quality.py`

输出：

- `/home/chenshuai/Project/output/caheiban_quality_eval/caheiban_quality_eval.json`
- `/home/chenshuai/Project/output/caheiban_quality_eval/caheiban_window_quality_labels.csv`

注意：这个评估是“模型能否复现物理规则弱标签”，不是“模型是否匹配人工质量标注”。下一步最好人工抽样标注 50-100 个窗口校准阈值。

## 擦黑板弱标签实验结果

数据：80 个 success episode，窗口长度 32，stride 16，共 3505 个窗口。

弱标签分布：

| 类别 | 含义 | 窗口数 |
|---|---|---:|
| too_light | 力过小/接触不足 | 701 |
| too_heavy | 力过大/峰值过大 | 595 |
| rough | 力/action/marker 变化不平滑 | 971 |
| good | 力合适且变化平稳 | 1238 |

按 episode GroupKFold 复现弱标签：

| 方法 | Balanced Acc |
|---|---:|
| RandomForest | 0.9958 ± 0.0022 |
| MLP (64,32) | 0.8363 ± 0.0202 |

解释：RandomForest 几乎完美复现弱标签，是因为弱标签本身由阈值规则构造，树模型天然擅长拟合阈值边界。MLP 更接近未来可微 scorer 的形式，但对这种硬阈值伪标签拟合较弱。擦黑板下一步应做人工小样本标注来判断这些弱标签是否符合真实“好动作”。

## 推荐路线

1. 插入任务先以 LDA 作为稳健 baseline scorer，以 PyTorch MLP 作为可微上限 scorer。
2. MLP scorer 不应只用 frame-level CV 选择，必须用 episode-level split 和 held-out episodes 调参。
3. 擦黑板先使用弱标签建立可解释质量分类器，同时导出 CSV 让人工快速检查。
4. 通用模型不建议直接混合插入任务和擦黑板任务的离散标签空间；更合理的是共享 tactile encoder + task-specific quality head，或者统一连续质量 score：力适中项 + 平滑项 + 异常接触项。

## 统一插座 + 擦黑板 taxonomy 实验

日期：2026-06-09

脚本：

- `TFAC_V5/evaluate_unified_quality_taxonomy.py`
- `TFAC_V5/quick_taxonomy_probe.py`
- `TFAC_V5/run_unified_quality_fast_eval.py`

输出目录：

- `/home/chenshuai/Project/output/unified_quality_taxonomy/`

统一特征：

```text
[z_cur(144), z_future(144), z_delta(144), ||z_cur||, ||z_future||, ||delta||]
```

其中 `z` 来自 TactileVAE，目的是贴近未来 DP guidance：推理时 Foresight 能预测未来 tactile latent，scorer 应尽量基于 latent 工作。

统一 T4 标签：

| id | 类别 | 插座来源 | 擦黑板来源 |
|---:|---|---|---|
| 0 | weak_no_contact | approach | too_light |
| 1 | good_stable | success insert | good |
| 2 | excessive_or_risk | bounce episode 的 pre-bounce/risk | too_heavy |
| 3 | rough_or_impact | bounce/recovery | rough |

约束：插座坏数据只从 bounce episode 中取，success episode 不生成坏标签。

擦黑板弱标签阈值：

```text
force_low_q20 = 8.6813
force_high_q85 = 14.6634
force_peak_q90 = 15.2145
force/action/marker delta robust-z > 1.0 -> rough
```

全量样本：

| 任务 | weak | good | risk/heavy | rough/impact |
|---|---:|---:|---:|---:|
| 插座 | 13406 | 19581 | 2430 | 3182 |
| 擦黑板 | 701 | 1238 | 595 | 971 |

快速 probe 使用每任务每类最多 800 个样本，SGD/logistic-style 线性分类器。

taxonomy 对比：

| taxonomy | mixed macro-F1 | mixed balanced acc | good/bad AUC | cross-task macro-F1 |
|---|---:|---:|---:|---:|
| binary good/bad | 0.6739 | 0.6743 | 0.7602 | 0.4523 |
| T3 weak/good/bad | 0.5743 | 0.5781 | 0.7149 | 0.2809 |
| T4 weak/good/risk/rough | 0.5724 | 0.5745 | 0.7679 | 0.1964 |

单任务结果：

| taxonomy | 插座 macro-F1 | 擦黑板 macro-F1 |
|---|---:|---:|
| binary | 0.8117 | 0.5680 |
| T3 | 0.6085 | 0.6371 |
| T4 | 0.6725 | 0.5139 |

关键 sanity check：

用擦黑板真实力/平滑特征 `[force_mean, force_p95, force_delta_mean, action_delta_mean, marker_delta_mean]` 直接预测擦黑板弱标签：

| 标签 | SGD macro-F1 | RF macro-F1 |
|---|---:|---:|
| board T4 | 0.8492 | 0.9972 |
| board binary | 0.8086 | 0.9967 |

结论：

1. 当前 latent-only 统一 scorer 还不够好，尤其跨任务泛化弱。
2. 当前最好的统一 taxonomy 是 `binary good/bad`，因为它在 mixed 和 cross-task 指标上都最好。
3. T4 更有解释性，但跨任务宏 F1 很低，说明四类语义虽然合理，但插座 latent 和擦黑板 latent 的几何结构没有天然对齐。
4. 擦黑板弱标签本身不是问题；用力/平滑物理特征可以很好复现。真正瓶颈是：只用当前 TactileVAE latent 学擦黑板“力大小合适 + 力变化柔顺”不充分。
5. 下一步不应直接把 latent-only T4 scorer 接入 DP guidance。更合理的推进方式：
   - 先用 binary good/bad 做最小可用 scorer；
   - 对擦黑板加入 force/marker 物理统计作为辅助监督或 auxiliary head；
   - 训练 PyTorch scorer 时使用多任务结构：shared tactile encoder + task head + scalar score head；
   - DP guidance 先 offline rerank 验证，再做 denoising gradient。

## Marker Proxy Scorer 迭代

日期：2026-06-09

脚本：`TFAC_V5/evaluate_marker_proxy_scorer.py`

输出目录：`/home/chenshuai/Project/output/marker_proxy_scorer/`

动机：上一轮 latent-only 统一 scorer 对擦黑板力大小/平滑质量表达不足。为了更接近可部署的 DP guidance，本轮不直接使用未来真实力，而是从 marker_offset 触觉形变中提取可微/可预测的物理代理特征：

```text
mag_mean, mag_std, mag_last,
mag_max_mean, mag_max_last, mag_p90_mean,
area_mean, area_last,
centroid_x, centroid_y,
spread_x, spread_y,
marker_delta_mean, marker_delta_p90,
centroid_delta_mean, mag_delta_mean,
mag_half_change, marker_first_last_l2
```

这些特征未来可以由 `Foresight -> predicted marker/decoded tactile` 计算，因此比直接使用未来 force 更适合 classifier guidance。

结果：

| taxonomy | model | mixed macro-F1 | mixed balanced acc | good/bad AUC | score corr |
|---|---|---:|---:|---:|---:|
| T4 | SGD | 0.4940 | 0.4978 | 0.7412 | 0.3426 |
| T4 | GBM | 0.6332 | 0.6328 | 0.8309 | 0.4782 |
| binary | SGD | 0.6380 | 0.6385 | 0.7010 | 0.3654 |
| binary | GBM | 0.7458 | 0.7473 | 0.8319 | 0.5745 |

跨任务 binary：

| direction | model | macro-F1 | balanced acc | AUC |
|---|---|---:|---:|---:|
| insertion -> board | GBM | 0.5858 | 0.5858 | 0.6173 |
| board -> insertion | GBM | 0.6664 | 0.6679 | 0.7378 |

与 latent-only 对比：

| feature | taxonomy | mixed macro-F1 | good/bad AUC | cross macro-F1 |
|---|---|---:|---:|---:|
| latent-only | binary | 0.6739 | 0.7602 | 0.4523 |
| marker proxy | binary | 0.7458 | 0.8319 | 0.6261 |
| latent-only | T4 | 0.5724 | 0.7679 | 0.1964 |
| marker proxy | T4 | 0.6332 | 0.8309 | 0.2155 |

结论：

1. Marker proxy 明显优于 latent-only，尤其 binary good/bad：mixed macro-F1 提升约 0.07，AUC 提升约 0.07，cross-task macro-F1 从 0.45 提升到约 0.63。
2. T4 四类仍然不适合直接跨任务作为主分类目标，尽管 mixed 指标比 latent-only 更好；跨任务仍低。
3. 当前最有希望的 scorer 设计是：
   - 主头：binary good/bad，用于 classifier guidance；
   - 辅助头：T4，提供可解释原因但不直接主导梯度；
   - score head：连续质量分数，学习 marker proxy 的适中强度和平滑性；
   - task-conditioned calibration：不同任务允许不同的接触强度中心，但共享“弱/过强/粗糙”的物理结构。
4. 创新点可以定义为 **Foresight-conditioned Marker Proxy Guidance**：
   - 不直接用黑箱 latent 分类；
   - 从预测触觉中抽取强度、面积、空间中心、平滑度等物理代理；
   - 用二分类概率作为主 guidance score；
   - 用 T4 作为辅助解释和安全诊断。

## 可微 Multi-head Marker Proxy Scorer

日期：2026-06-09

脚本：`TFAC_V5/train_marker_proxy_multitask_scorer.py`

输出目录：`/home/chenshuai/Project/output/marker_proxy_multitask_scorer/`

模型结构：

```text
input: marker_proxy_features + task_id(one-hot)
shared MLP encoder
  -> binary head: good / bad                 # 主 guidance head
  -> T4 head: weak / good / risk / rough     # 辅助解释 head
  -> score head: continuous quality score    # 排序/校准 head
```

loss：

```text
L = 1.0 * CE(binary)
  + 0.35 * CE(T4)
  + 0.5 * SmoothL1(sigmoid(score), quality_score)
```

Group-CV 泛化结果：

| metric | mean | std |
|---|---:|---:|
| binary balanced acc | 0.7410 | 0.0132 |
| binary macro-F1 | 0.7480 | 0.0149 |
| binary AUC | 0.8603 | 0.0112 |
| T4 balanced acc | 0.7195 | 0.0139 |
| T4 macro-F1 | 0.7186 | 0.0137 |
| score corr | 0.6211 | 0.0168 |

跨任务：

| direction | binary macro-F1 | binary AUC | T4 macro-F1 | score corr |
|---|---:|---:|---:|---:|
| insertion -> board | 0.5036 | 0.5593 | 0.3045 | 0.0995 |
| board -> insertion | 0.6673 | 0.7978 | 0.3681 | 0.4878 |

最终 mixed 训练 checkpoint：

- `/home/chenshuai/Project/output/marker_proxy_multitask_scorer/marker_proxy_multitask_final.pt`
- `/home/chenshuai/Project/output/marker_proxy_multitask_scorer/marker_proxy_multitask_final_summary.json`

最终 mixed train 指标只用于确认模型容量，不作为泛化结论：

| metric | train |
|---|---:|
| binary balanced acc | 0.8341 |
| binary macro-F1 | 0.8464 |
| binary AUC | 0.9474 |
| T4 balanced acc | 0.8219 |
| T4 macro-F1 | 0.8188 |
| score corr | 0.7865 |

结论：

1. 可微 multi-head MLP 已经超过 marker proxy GBM 的关键泛化指标：binary AUC 从 0.8319 到 0.8603，score corr 从 0.5745 到 0.6211，T4 macro-F1 从 0.6332 到 0.7186。
2. 这是目前最适合接入 DP classifier guidance 的 scorer 版本，因为它可微、输出 `P(good)`、解释类别和连续质量分。
3. 仍然存在跨任务不对称：`board -> insertion` 明显强于 `insertion -> board`。后续要做 task-conditioned calibration 或者用 mixed training，不应依赖单任务训练直接迁移。
4. 下一步建议做 offline DP candidate reranking：
   - 对每个 observation 采样 K 个 DP action；
   - 用 Foresight 预测未来 marker/latent；
   - 计算 marker proxy；
   - scorer 排序；
   - 验证 top action 是否有更高真实质量，再进入 denoising guidance。

## Action-aware Marker Field Scorer：当前最佳方案

日期：2026-06-09

脚本：

- `TFAC_V5/train_marker_field_scorer.py`
- `TFAC_V5/train_action_aware_marker_scorer.py`
- `TFAC_V5/action_aware_scorer_runtime.py`
- `TFAC_V5/summarize_scorer_experiments.py`

输出目录：

- `/home/chenshuai/Project/output/marker_field_scorer/`
- `/home/chenshuai/Project/output/action_aware_marker_scorer/`
- `/home/chenshuai/Project/output/tactile_scorer_comparison/`

### 设计动机

普通 classifier guidance 的核心是用分类器梯度修正扩散采样方向：

```text
epsilon_guided = epsilon_theta - eta_t * grad_x log p(good | x_t)
```

对本项目，`x_t` 不是图像，而是 DP denoising 中的候选 action chunk。因此 scorer 必须对 action 可微。只对 GT 触觉 latent 做分类不能直接保证 action 梯度有效；更合理的链路是：

```text
noisy/current action
  -> Foresight predicts future tactile marker/latent
  -> action-aware tactile quality scorer
  -> score = log P(good) 或 continuous quality
  -> grad(score) wrt action
  -> guide DP denoising / reranking
```

因此新增 action-aware scorer：

```text
input:
  marker window field: (T=8, 9, 9, 2)
  marker proxy: intensity / area / centroid / spread / smoothness
  action sequence: (T=8, 6)
  action proxy: speed / acceleration / first-last displacement / delta
  task_id: insertion or board

shared encoder:
  marker 3D-CNN encoder
  action 1D-CNN encoder
  proxy MLP fusion

heads:
  binary head: good / bad, main DP guidance objective
  T4 head: weak / good / risk_or_heavy / rough_or_impact, interpretability
  score head: continuous quality, ranking/calibration
```

### 标签标准修正

这版实验修正了一个重要标准：

1. 插座任务：
   - `success insert -> good`
   - `bounce pre-bounce / bounce / recovery -> bad`
   - `approach / weak_no_contact -> binary neutral`，不参与二分类 loss，但参与 T4 loss
2. 擦黑板任务：
   - `good -> good`
   - `too_light / too_heavy / rough -> bad`
   - 这符合“力过小或过大都是差动作”的定义

这样二分类更贴近最终“动作好坏”，T4仍保留多个失败原因。

### 模型对比

Group-CV by episode，插座 + 擦黑板 mixed 训练测试：

| 模型 | Binary Bal Acc | Binary AUC | T4 Macro-F1 | Score Corr |
|---|---:|---:|---:|---:|
| marker_proxy MLP | 0.7410 | 0.8603 | 0.7186 | 0.6211 |
| marker_field CNN | 0.7497 | 0.8544 | 0.7394 | 0.5982 |
| action_aware marker scorer | **0.8763** | **0.9562** | **0.7773** | **0.7372** |

当前最佳是 `action_aware marker scorer`。它说明 action 信息不是冗余的：同样的触觉后果分类，如果加入候选 action 的运动平滑性/幅度信息，质量判断明显更稳。

### 跨任务结果

| 模型 | insertion -> board AUC | board -> insertion AUC |
|---|---:|---:|
| marker_proxy MLP | 0.5593 | 0.7978 |
| marker_field CNN | 0.4711 | 0.7268 |
| action_aware marker scorer | **0.7169** | **0.8001** |

解释：

1. action-aware 后，跨任务排序 AUC 明显提升，尤其 insertion -> board 从 0.5593 到 0.7169。
2. 但跨任务固定阈值分类仍不稳定，因此不能使用统一 `P(good)>0.5` 作为所有任务的硬判定。
3. 对 DP guidance，更合理的是用连续 score 或 `log P(good)` 的梯度，并做 task-conditioned calibration。

### Runtime 可微接口

新增 `TFAC_V5/action_aware_scorer_runtime.py`，提供部署接口：

```python
runtime = ActionAwareScorerRuntime(
    "/home/chenshuai/Project/output/action_aware_marker_scorer/action_aware_marker_scorer_final.pt"
)
score = runtime.score(marker_pred, action_chunk, task_id, mode="hybrid")
```

其中 `marker_pred` 和 `action_chunk` 均可带梯度。自测结果：

```text
grad_marker_norm = 0.1298
grad_action_norm = 1.1525
grad_marker_finite = true
grad_action_finite = true
usable_for_guidance = true
```

保存路径：

- `/home/chenshuai/Project/output/action_aware_marker_scorer/runtime_gradient_sanity.json`

### 当前推荐实现方案

短期上线：

1. DP 采样 K 个 action candidate；
2. Foresight 对每个 candidate 预测未来 tactile marker；
3. `ActionAwareScorerRuntime.score(..., mode="hybrid")` 打分；
4. 选择分数最高的 candidate，先做 reranking，不直接改 denoising；
5. 记录 score 与真实力/触觉质量、人工复核标签的相关性。

中期 classifier guidance：

```text
for denoising step t in late steps:
    noise_pred = DP(noisy_action_t, obs, t)
    marker_pred = Foresight(obs, noisy_action_t)
    score = scorer(marker_pred, noisy_action_t, task_id)
    grad = d score / d noisy_action_t
    noise_pred_guided = noise_pred - eta_t * sqrt(1 - alpha_bar_t) * normalize_or_clip(grad)
    noisy_action_{t-1} = scheduler.step(noise_pred_guided)
```

建议：

- 先只在后 30%-50% denoising steps 加 guidance；
- `eta_t` 从小值开始，如 0.02 / 0.05 / 0.1；
- 对 `grad` 做 norm clipping；
- 黑板任务使用 task-specific score calibration，不用跨任务统一阈值；
- 继续收集人工小样本标签校准擦黑板弱标签。

## DP Action Space 对齐与离线重排验证

日期：2026-06-09

### 为什么要训练 joint_abs 版本

前面的 action-aware scorer 默认使用 `actions/eef_abs` 6维动作。这对分析动作平滑性有价值，但当前插座 DP checkpoint 的 action space 是：

```text
variant = tactile_vae_frozen
action_dim = 7
action source = actions/joint_abs
pred_horizon = 20
```

因此如果直接把 DP 的 7维 joint action 喂给 6维 EEF scorer，会发生动作空间错配。已将训练脚本和 runtime 改为支持：

```bash
--action_key joint_abs --action_dim 7
```

输出目录：

- `/home/chenshuai/Project/output/action_aware_marker_scorer_joint_abs/`

joint_abs 版 mixed group-CV：

| metric | mean | std |
|---|---:|---:|
| binary balanced acc | 0.8597 | 0.0071 |
| binary AUC | 0.9453 | 0.0074 |
| T4 macro-F1 | 0.7552 | 0.0188 |
| score corr | 0.7063 | 0.0185 |

runtime 梯度自测：

```text
grad_marker_norm = 0.8156
grad_action_norm = 0.3696
grad_marker_finite = true
grad_action_finite = true
usable_for_guidance = true
```

说明 7维 DP action 上也存在有效可微梯度。

### 离线重排脚本

新增：

- `TFAC_V5/eval_action_aware_reranking.py`

评估链路：

```text
candidate joint action
  -> LatentForesight predicts z_pred
  -> TactileVAE decoder predicts future marker
  -> denormalize marker to raw marker_offset
  -> ActionAwareScorerRuntime scores candidate
  -> rank candidates
```

注意：Foresight decoder 输出是 normalized marker，必须用 TactileVAE checkpoint 的 `norm_stats` 反归一化：

```text
mean = [0.2102, -0.6422]
std  = [1.6805, 3.6717]
```

### 包含 expert candidate 的上限实验

设置：

```text
K = 16
N = 80 insertion frames
candidates = expert action + noisy perturbations
score_mode = hybrid 或 log_p_good
```

结果：

| score_mode | expert rank-1 | expert top-3 | scorer best L1 | random L1 |
|---|---:|---:|---:|---:|
| hybrid | 0.9875 | 1.0000 | 0.0176 | 7.3147 |
| log_p_good | 0.9875 | 1.0000 | 0.0171 | 7.2632 |

解释：这个实验偏容易，因为 expert 在候选里；但它证明 scorer 能明显识别专家动作和大扰动动作。

### 不包含 expert candidate 的更严格实验

设置：

```text
K = 16
N = 80 insertion frames
candidates = all noisy perturbations around expert, no exact expert
score_mode = hybrid
noise_scales = 0.05,0.1,0.2,0.4,0.8
```

结果：

| selection | L1 to expert |
|---|---:|
| oracle best candidate | 1.2628 |
| action-aware selected | 1.8504 |
| random selected | 6.6734 |

其他指标：

- action-aware 选中动作比随机更近 expert：82.5% frames；
- score 与 `-L1` 的 frame 内相关：0.2536；
- action-aware selected 距 oracle 还有差距，但已显著优于随机。

结论：

1. `ActionAware + Foresight` 已经能在局部扰动候选中筛出更接近专家的动作。
2. 这比单纯分类准确率更接近 DP guidance 的目标：评分器确实能影响 action 选择。
3. 当前评估仍不是最终真机质量验证，因为 noisy perturbation 的“真实好坏”用 L1-to-expert 近似；下一步需要真实 DP sampled candidates 和/或 rollout 后触觉质量指标。
4. 下一阶段应做：
   - DP sampled candidate reranking；
   - 对 selected action 的 Foresight 预测 marker 进行质量分布分析；
   - 接入 denoising loop，小 guidance scale 做离线 ablation。

### 真实 DP sampled candidates 初步结果

已将 `TFAC_V5/eval_action_aware_reranking.py` 扩展为：

```bash
--candidate_mode dp_sampling
```

设置：

```text
K = 16
N = 40 insertion frames
candidates = DP DDPM sampled actions
score_mode = hybrid
```

结果：

| selection | L1 to expert |
|---|---:|
| oracle best DP candidate | 0.2676 |
| action-aware selected | 0.7876 |
| random selected | 0.8774 |

其他指标：

- action-aware 选中动作比随机更近 expert：57.5% frames；
- score 与 `-L1` 的 frame 内相关：0.1487；
- score range 中位数只有 0.2595，说明真实 DP 候选之间的 scorer 分数差异远小于扰动候选实验。

解释：

1. 对真实 DP sampled candidates，action-aware scorer 有轻微正向效果，但远弱于专家扰动候选实验。
2. 这说明当前 scorer 可以识别“明显坏的扰动动作”，但真实 DP 候选本身较集中，Foresight 预测的触觉差异不够大，导致排序信号弱。
3. 不能直接用大 guidance scale 强推，否则可能放大未校准的score噪声。
4. 下一步需要：
   - 用 predicted marker sequence 而不是单帧重复，增强触觉后果差异；
   - 做 task/phase-specific score normalization；
   - 对 DP candidates 训练 pairwise/ranking calibration head；
   - 先以 reranking 小步上线，再进入 denoising guidance。

### DP sampled score mode 对比

为了判断是评分器本身不行，还是 score mode 选择不合适，进一步测试：

```text
candidate_mode = dp_sampling
K = 32
N = 40
score_mode in {log_p_good, p_good, quality}
```

结果：

| score_mode | selected L1 | random L1 | oracle L1 | beats random | corr(score,-L1) |
|---|---:|---:|---:|---:|---:|
| log_p_good | 0.8639 | 0.7480 | 0.2271 | 0.450 | 0.1195 |
| p_good | 0.8830 | 0.8734 | 0.2590 | 0.550 | 0.1751 |
| quality | **0.7725** | 0.9962 | 0.2445 | **0.625** | **0.2704** |

结论：

1. 对真实 DP sampled candidates，binary probability/log-probability 不适合直接排序。
2. continuous `quality` head 明显更稳定，但仍离 oracle 很远。
3. 这说明最终 guidance score 不应只定义为 `log P(good)`；更合理的是：
   - `quality` 作为主排序/引导分数；
   - binary head 作为安全约束或低分截断；
   - T4 head 作为解释和失败原因；
   - 针对 DP sampled candidates 再训练 pairwise/ranking calibration head。

### DP candidate ranking calibration 负结果

新增：

- `TFAC_V5/train_dp_candidate_ranker.py`

目的：

```text
DP candidates -> Foresight predicted marker
              -> action-aware scorer/proxy features
              -> small ranker predicts -L1(candidate, expert)
```

数据与协议：

```text
K = 32
N = 120 frames
features = [p_good, log_p_good, quality, hybrid, T4 probs, marker proxy, action proxy]
target = -L1(candidate action, expert action)
split = GroupKFold by frame
```

结果：

| model | selected L1 | random L1 | oracle L1 | beats random | corr(score,-L1) |
|---|---:|---:|---:|---:|---:|
| base_quality | 0.6186 | 0.6000 | 0.2032 | 0.4917 | 0.1789 |
| base_hybrid | 0.6185 | 0.6000 | 0.2032 | 0.4917 | 0.1563 |
| Ridge | 0.6563 | 0.6000 | 0.2032 | 0.4500 | 0.1707 |
| GBR | 0.6327 | 0.6000 | 0.2032 | 0.4667 | 0.1816 |
| RF | 0.6221 | 0.6000 | 0.2032 | 0.5000 | 0.2311 |
| MLP | 0.6792 | 0.6000 | 0.2032 | 0.4250 | 0.1699 |

结论：

1. 用 L1-to-expert 作为 DP candidate ranking calibration 目标没有成功，所有模型都没有稳定超过 random。
2. 这不是说明触觉 scorer 没有价值，而是说明 `expert L1` 不是合适的触觉质量监督：离专家近不一定等价于触觉更好，尤其 DP candidates 本身很集中时。
3. 下一步不应继续拟合 L1 ranking，而应回到触觉后果监督：
   - predicted marker 的强度/面积/平滑质量 proxy；
   - 插座 bounce/risk 标签；
   - 黑板力大小和平滑弱标签；
   - 小规模人工或真机 rollout 质量标签。
4. 对 DP guidance 的当前实用建议：
   - 使用 `quality` head 作为主分数；
   - binary/T4 作为安全约束和解释；
   - task/phase 内做score normalization；
   - guidance scale 保守，只在 late denoising steps 加小梯度。

### 可解释 score formula 搜索

新增：

- `TFAC_V5/eval_dp_candidate_score_formulas.py`

目的：不重新采样DP，而是在缓存的 DP candidates 上搜索简单可解释公式：

```text
score = z(base_score)
        - a * z(action_speed/accel/smoothness penalty)
        - b * z(marker_delta penalty)
        + 0.25 * z(T4_good)
```

其中 `z(.)` 是每个frame候选内部的z-score，减少不同frame分数尺度不一致的问题。

数据仍为：

```text
K = 32
N = 120 frames
split = GroupKFold by frame
```

结果：

| method | selected L1 | random L1 | oracle L1 | beats random | corr(score,-L1) |
|---|---:|---:|---:|---:|---:|
| base_quality | 0.6186 | 0.6000 | 0.2032 | 0.4917 | 0.1789 |
| formula search | **0.5865** | 0.6000 | 0.2032 | **0.5583** | 0.1506 |

常见最优公式：

```text
score = z(p_good) - 1.0 * z(action_accel_p90) + 0.25 * z(T4_good)
```

结论：

1. 加入动作加速度平滑惩罚有轻微帮助，符合“动作/触觉后果要柔顺”的目标。
2. 但提升很小，仍远离 oracle，不能作为最终强评分器。
3. 这进一步说明：当前DP候选排序缺少真正触觉质量监督；只靠 expert L1 或简单平滑公式无法解决。
4. 最值得继续的是收集或构造更直接的触觉质量 target：bounce风险、黑板力大小/平滑、人工小样本标签、或真机rollout结果。

## 2026-06-09 擦黑板弱标签方案与可预测性评估

### 调研后形成的评分器原则

参考 classifier guidance、classifier-free guidance、TouchGuide、AdaVTF 后，当前 PTG 评分器应满足：

1. 评分目标必须显式表达“好/坏触觉后果”，不能只学动作分布或 expert L1。
2. 对 DP guidance 来说，最终分数要能对 action 求梯度；不可导模型可以作为 teacher/验证器，但不能直接作为 denoising 内部引导器。
3. 触觉/力信号应接触阶段加权，非接触阶段只做弱约束，避免把 approach 的无触觉误判成坏触觉。
4. 最合理结构不是单一二分类，而是：
   - binary good/bad：安全门控；
   - 多类 reason：解释坏在哪里；
   - continuous quality：用于排序和梯度引导。

### 新增实验脚本

新增：

- `TFAC_V5/eval_board_quality_label_schemes.py`

输出目录：

- `/home/chenshuai/Project/output/board_quality_label_schemes/`
- `/home/chenshuai/Project/output/board_quality_label_schemes/w32_s16/board_quality_scheme_eval.json`
- `/home/chenshuai/Project/output/board_quality_label_schemes/w32_s16/board_quality_scheme_results.csv`
- `/home/chenshuai/Project/output/board_quality_label_schemes/w32_s16/figures/`

实验设置：

```text
dataset = /home/chenshuai/data/dataset/260522_v8l_caheiban
episodes = 80 success hdf5
window = 32
stride = 16
split = GroupKFold by episode
force only used to create weak labels; model input excludes force
```

比较的弱标签：

1. `t4_quantile`：too_light / good_smooth / too_heavy / rough。
2. `t5_reason`：too_light / good_smooth / too_heavy / rough_force / rough_motion。
3. `t5_scoreband`：先构造连续 quality，再把非好样本分到 too_light / too_heavy / rough_force / rough_motion。

比较的 force label source：

1. `ft`
2. `left_force`
3. `right_force`

比较的输入：

1. `right_marker`
2. `both_marker`
3. `both_marker_eef`
4. `both_marker_actions`

### 快速筛选结果

命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_board_quality_label_schemes.py \
  --feature_sets right_marker,both_marker,both_marker_eef,both_marker_actions \
  --class_models logreg \
  --reg_models ridge \
  --max_per_class 600 \
  --viz_max 1200
```

最优分类：

| force | scheme | feature | model | balanced acc | macro-F1 | good AUC | score corr |
|---|---|---|---|---:|---:|---:|---:|
| left_force | t5_scoreband | both_marker_actions | LogReg | 0.8816 | 0.8745 | 0.9598 | 0.8391 |

最优回归：

| force | scheme | feature | model | quality corr | R2 |
|---|---|---|---|---:|---:|
| left_force | t4_quantile | both_marker_actions | Ridge | 0.9149 | 0.8259 |

结论：

1. 擦黑板质量不是不可分；只要用力大小+平滑构造合理 weak label，marker/action proxy 能很好预测。
2. `both_marker_actions` 明显优于只用单侧 marker，说明两侧触觉+动作柔顺性都对质量判断有用。
3. `left_force` 在当前数据上作为弱标签来源最可学习；这可能说明左侧传感器更贴近擦拭质量，或标定/接触状态更稳定，需要后续结合硬件安装确认。

### 最优组合精跑

命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_board_quality_label_schemes.py \
  --force_sources left_force \
  --schemes t5_scoreband \
  --feature_sets both_marker_actions \
  --class_models logreg,rf,gbm,mlp \
  --reg_models ridge,rf,mlp \
  --max_per_class 900 \
  --viz_max 1800
```

分类结果：

| model | balanced acc | macro-F1 | good AUC | score corr |
|---|---:|---:|---:|---:|
| RF | 0.9035 | 0.9093 | **0.9779** | **0.8630** |
| GBM | **0.9066** | **0.9142** | 0.9767 | 0.8356 |
| MLP | 0.8523 | 0.8605 | 0.9668 | 0.8488 |
| LogReg | 0.8825 | 0.8710 | 0.9622 | 0.8469 |

连续质量回归：

| model | quality corr | R2 |
|---|---:|---:|
| RF | **0.9850** | **0.9697** |
| MLP | 0.9427 | 0.8833 |
| Ridge | 0.9229 | 0.8478 |

最优弱标签阈值：

```text
target_force_q55 = 7.8553
force_iqr_sigma = 4.6805
force_q25 = 4.3862
force_q80 = 9.7589
force_p95_q90 = 13.6153
score_good_q65 = 0.5978
score_bad_q35 = 0.2612
```

重要结论：

1. RF/GBM 是当前擦黑板弱标签上的最强 teacher：5类 macro-F1 约 0.91，good/bad AUC 约 0.978。
2. 可微 MLP 也已经有可用信号：5类 macro-F1 0.861，good/bad AUC 0.967，连续质量 corr 0.943。
3. 因为 RF/GBM 不可对 action 求梯度，不能直接作为 DP denoising guidance；它们适合做 teacher 或离线验证器。
4. 下一步应训练一个可微 PTG scorer v2：
   - 输入：predicted marker proxy + action proxy + task/phase embedding；
   - 监督：插座 bounce/risk 标注 + 擦黑板 `t5_scoreband` weak label + continuous quality；
   - 可选蒸馏：用 RF/GBM teacher 的 soft score 作为辅助 target；
   - 评估：GroupKFold、跨任务、DP candidate reranking、gradient sanity。

### 当前对最终方案的判断

目前最合理路线是：

```text
不可导强 teacher:
  插座: 标注/MLP/RF 检验 good-vs-bounce risk
  擦黑板: RF/GBM 检验 force-band + smoothness weak quality

可导部署 scorer:
  PyTorch multi-head scorer
  heads = binary good/bad + reason class + continuous quality + optional teacher distillation

DP 使用方式:
  score = z(quality) + alpha*z(log_p_good) + beta*z(reason_good) - gamma*z(action_accel)
  only contact/late denoising steps
  small guidance scale + grad clipping
```

这比“只做二分类”更适合论文和真实部署：既有明确好坏标准，又有连续梯度，还能解释坏的原因。

## 2026-06-09 PTG Proxy Scorer v2：统一可微评分器

### 新增实验脚本

新增：

- `TFAC_V5/train_ptg_proxy_scorer_v2.py`

输出：

- `/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_eval.json`
- `/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_final.pt`
- `/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_features.npz`
- `/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_samples.csv`

模型目标：把插座和擦黑板统一进一个可微 MLP scorer。

输入：

```text
left marker proxy 18
right marker proxy 18
left/right abs diff 18
eef action proxy 10
joint action proxy 10
task one-hot 2
```

总 feature dim = 74，不使用 force 作为输入。

统一 reason taxonomy：

| id | meaning |
|---:|---|
| 0 | weak/no-contact or too-light |
| 1 | good stable/smooth |
| 2 | excessive/risk or too-heavy |
| 3 | impact or rough-force |
| 4 | rough-motion |

多头输出：

1. binary good/bad；
2. 5-way reason；
3. continuous quality。

### 正式训练命令

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/train_ptg_proxy_scorer_v2.py \
  --epochs 80 \
  --final_epochs 90 \
  --max_per_task_class 1200 \
  --max_insertion_per_episode 220 \
  --device cuda:0 \
  --force_rebuild
```

数据：

```text
n = 8305
insertion = 4800
board = 3505
reason counts = {0:2252, 1:2106, 2:1911, 3:1781, 4:255}
binary counts = {-1 neutral:1200, good:2106, bad:4999}
split = GroupKFold by task::episode
```

### 结果

整体 mixed group-CV：

| metric | mean | std |
|---|---:|---:|
| binary balanced acc | 0.9082 | 0.0281 |
| binary macro-F1 | 0.8930 | 0.0321 |
| binary AUC | **0.9701** | 0.0143 |
| reason balanced acc | 0.7901 | 0.0293 |
| reason macro-F1 | 0.7678 | 0.0434 |
| quality corr | 0.7562 | 0.0391 |
| quality R2 | 0.5678 | 0.0616 |

分任务观察：

| task | binary AUC range | reason macro-F1 range | quality corr range |
|---|---:|---:|---:|
| insertion | 0.9355 - 0.9748 | 0.6973 - 0.7493 | 0.6140 - 0.7231 |
| board | 0.9547 - 0.9935 | 0.7666 - 0.8807 | 0.9104 - 0.9533 |

梯度 sanity：

```text
input_grad_norm = 0.1029
score_value = 1.0057
usable_for_feature_guidance = true
```

### 结论

1. `PTG Proxy Scorer v2` 是当前最合理的可微统一评分器雏形：它不是最高分 teacher，但能输出对输入 feature 的梯度。
2. 黑板任务连续质量学得很好，quality corr 在 0.91-0.95，适合用 continuous quality 做 DP guidance。
3. 插座任务 continuous quality 较弱，但 binary AUC 高，说明插座更适合用 `P(good)` / risk reason 做安全引导，而不是强依赖连续质量。
4. 当前 v2 还不是最终可直接接 action 的 scorer，因为它对 proxy feature 可导，但 marker/action proxy 的 torch runtime 还需要扩展成 left/right/both 版本，才能完整传回 action/Foresight。
5. 下一步应做：
   - 实现 `PTGProxyScorerV2Runtime`，用 torch 计算 left/right marker proxy 和 action proxy；
   - 在 DP sampled candidates 上比较 v2 score、旧 action-aware score、RF teacher proxy 的排序表现；
   - 加 teacher distillation，让可微 MLP 更接近 RF/GBM teacher；
   - 对插座单独增强 risk/bounce head，避免 continuous quality 过弱。

## 2026-06-09 PTG Proxy Scorer v2 Runtime

新增：

- `TFAC_V5/ptg_proxy_scorer_v2_runtime.py`

功能：

1. 加载 `/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_final.pt`。
2. 在 torch 中计算：
   - left marker proxy；
   - right marker proxy；
   - left/right abs-diff proxy；
   - eef action proxy；
   - joint action proxy。
3. 输出：
   - `quality_score`
   - `p_good`
   - `log_p_good`
   - `reason_prob`
   - `guidance_score`
4. 支持对 left/right marker、eef action、joint action 求梯度。

sanity 命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/ptg_proxy_scorer_v2_runtime.py --device cuda:0
```

结果：

```text
score = -3.0721
grad_left_norm = 0.0162
grad_right_norm = 0.0296
grad_eef_norm = 0.0948
grad_joint_norm = 0.0142
usable_for_guidance = true
```

结论：

1. v2 runtime 已经具备可导 scorer 的工程接口。
2. 它目前可对 action chunk 产生梯度；对真实 DP denoising 的完整梯度还需要经过 Foresight 的 predicted marker 链接。
3. 与旧 action-aware scorer 相比，v2 的优点是：
   - 同时支持 left/right tactile；
   - 支持擦黑板和插座统一 reason taxonomy；
   - 有明确的黑板 force-band + smoothness weak quality；
   - guidance score 可以组合 quality、log_p_good、reason_good 和 action smoothness。
4. 下一步最关键实验：在真实 DP sampled candidates 上比较 v2 runtime 的 reranking 效果，不能只看离线分类指标。

## 2026-06-09 PTG v2 在真实 DP Candidates 上的排序评估

新增：

- `TFAC_V5/eval_ptg_v2_reranking.py`
- `TFAC_V5/eval_ptg_v2_score_formulas.py`

输出：

- `/home/chenshuai/Project/output/ptg_v2_reranking/dp_sampling_quality_K32_N40.json`
- `/home/chenshuai/Project/output/ptg_v2_reranking/dp_sampling_guidance_K32_N40.json`
- `/home/chenshuai/Project/output/ptg_v2_reranking/dp_sampling_p_good_K32_N40.json`
- `/home/chenshuai/Project/output/ptg_v2_reranking/dp_sampling_reason_good_K32_N40.json`
- `/home/chenshuai/Project/output/ptg_v2_reranking/ptg_v2_candidates_K32_N60_seed42.npz`
- `/home/chenshuai/Project/output/ptg_v2_reranking/ptg_v2_formula_eval_ptg_v2_candidates_K32_N60_seed42.json`

### 单 mode DP sampled reranking

设置：

```text
task = insertion
candidate_mode = dp_sampling
K = 32
N = 40 frames
Foresight predicts one marker; current integration feeds same predicted marker to left/right v2 inputs
target metric = L1-to-expert, only as offline proxy
```

结果：

| mode | selected L1 | random L1 | oracle L1 | beats random | corr(score,-L1) | score range |
|---|---:|---:|---:|---:|---:|---:|
| quality | 0.9350 | 0.8831 | 0.2501 | 0.525 | 0.1793 | 0.0108 |
| guidance | **0.7552** | 0.8838 | 0.2441 | 0.525 | 0.1174 | 0.0401 |
| p_good | 0.9409 | 0.7886 | 0.2466 | 0.600 | 0.2175 | 0.00012 |
| reason_good | 0.8121 | 0.7959 | 0.2605 | 0.475 | 0.0958 | 0.00022 |

对比旧 action-aware scorer 的已有结果：

```text
old action-aware quality:
selected L1 = 0.7725
random L1 = 0.9962
beats random = 0.625
corr = 0.2704
```

解释：

1. v2 `guidance` 的 selected L1 接近甚至略好于旧 action-aware quality，但 beats random 和 corr 更弱，稳定性不足。
2. `p_good` 和 `reason_good` 头严重饱和，score range 只有 1e-4 量级，不适合直接做 guidance。
3. 单独 `quality` 在插座 DP candidates 上失败，selected L1 比 random 更差。

### 同一候选池公式搜索

为了避免不同 score mode 使用不同 DP samples，新增同池候选缓存：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_ptg_v2_score_formulas.py \
  --K 32 \
  --n_eval 60 \
  --device cuda:0 \
  --force_rebuild
```

设置：

```text
K = 32
N = 60 frames
score features = quality, p_good, log_p_good, reason_good, risk/impact probs, action smoothness proxies
normalization = per-frame candidate z-score
```

同池结果：

| method | selected L1 | random L1 | oracle L1 | beats random | corr(score,-L1) |
|---|---:|---:|---:|---:|---:|
| best formula | 0.8099 | 0.7644 | 0.2333 | 0.600 | 0.1114 |
| p_good | 0.9197 | 0.7644 | 0.2333 | 0.533 | 0.1317 |
| log_p_good | 0.9197 | 0.7644 | 0.2333 | 0.533 | 0.1317 |
| reason_good | 0.9522 | 0.7644 | 0.2333 | 0.533 | 0.1358 |
| quality | 1.0009 | 0.7644 | 0.2333 | 0.433 | 0.1047 |

最佳公式：

```text
score = z(quality) - 1.0*z(action_abs_delta_max) + 0.25*z(reason_good)
```

结论：

1. 同池公式搜索仍然没有超过 random L1，说明不是简单 score mode 没调好。
2. v2 离线分类/黑板质量很强，但当前插座 DP candidate 排序不可靠。
3. 插座 guidance 当前应继续使用旧 action-aware scorer 的 `quality` 或专门 risk/bounce scorer，而不是直接切到 v2。
4. v2 更适合黑板任务，因为它在黑板 continuous quality 上 corr 达到 0.91-0.95，并且黑板质量定义就是 force-band + smoothness，和 v2 监督一致。
5. 下一步要提升插座 DP guidance，关键不是再调公式，而是：
   - 训练插座专门 risk scorer，目标直接用 bounce/pre-bounce；
   - 让 Foresight 输出更长的 predicted marker sequence，而不是单帧重复；
   - 用同一批 DP candidates 做真实 tactile-quality proxy 或 rollout 标签，而不是 L1-to-expert；
   - 增加 score temperature/calibration，避免 binary/reason head 饱和。

## 2026-06-09 插座专用 Insertion Risk Scorer

新增：

- `TFAC_V5/train_insertion_risk_scorer.py`

动机：统一 v2 在黑板上有效，但插座 DP candidates 排序不可靠；插座任务更需要直接识别 `good_insert` vs `pre_bounce/bounce risk`。

输入：

```text
left marker window (8,9,9,2)
left marker proxy 18
joint_abs action window (8,7)
joint action proxy 10
```

标签：

| id | meaning | binary | quality |
|---:|---|---:|---:|
| 0 | weak/approach | neutral(-1) | 0.30 |
| 1 | good_insert | 1 | 1.00 |
| 2 | pre_bounce_risk | 0 | 0.05 |
| 3 | impact_or_recovery | 0 | 0.00 |

正式命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/train_insertion_risk_scorer.py \
  --epochs 80 \
  --final_epochs 90 \
  --max_per_class 1400 \
  --max_per_episode 260 \
  --device cuda:0 \
  --force_rebuild
```

输出：

- `/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_eval.json`
- `/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt`
- `/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_features.npz`
- `/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_samples.csv`

数据：

```text
n = 5600
reason counts = {0:1400, 1:1400, 2:1400, 3:1400}
binary counts = {-1 neutral:1400, good:1400, bad:2800}
split = GroupKFold by episode
```

结果：

| metric | mean | std |
|---|---:|---:|
| binary balanced acc | 0.9437 | 0.0163 |
| binary macro-F1 | 0.9364 | 0.0187 |
| binary AUC | **0.9877** | 0.0050 |
| reason balanced acc | 0.7880 | 0.0166 |
| reason macro-F1 | 0.7894 | 0.0154 |
| quality corr | 0.7656 | 0.0385 |
| quality R2 | 0.5682 | 0.0641 |

梯度 sanity：

```text
grad_marker_norm = 0.0314
grad_action_norm = 1.2441
grad_marker_proxy_norm = 0.0662
grad_action_proxy_norm = 0.1107
usable_for_guidance = true
```

结论：

1. 插座专用 risk scorer 离线指标明显强于统一 v2 的插座分任务 reason/quality，且 binary AUC 接近 0.99。
2. 该模型和当前 DP/Foresight 链路更对齐：left tactile + joint_abs action。
3. action 梯度很强，具备成为插座 DP guidance 主评分器的潜力。
4. 仍需真实 DP sampled candidates reranking 验证；如果候选排序仍失败，瓶颈大概率在 Foresight 单帧预测或 L1-to-expert 评估目标，而不是离线分类器本身。

## 2026-06-09 Insertion Risk Scorer Runtime 与 DP Candidate Reranking

新增：

- `TFAC_V5/insertion_risk_scorer_runtime.py`
- `TFAC_V5/eval_insertion_risk_reranking.py`

runtime sanity：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/insertion_risk_scorer_runtime.py --device cuda:0
```

结果：

```text
score = -5.0529
grad_marker_norm = 0.0286
grad_action_norm = 0.0178
usable_for_guidance = true
```

DP sampled reranking 设置：

```text
task = insertion
candidate_mode = dp_sampling
K = 32
N = 40 frames
Foresight predicts one marker; repeated to 8-frame scorer window
target metric = L1-to-expert, only offline proxy
```

结果：

| mode | selected L1 | random L1 | oracle L1 | beats random | corr(score,-L1) | score range |
|---|---:|---:|---:|---:|---:|---:|
| quality | **0.7176** | 0.9048 | 0.2501 | 0.425 | **0.2077** | 0.5876 |
| risk_guidance | **0.7176** | 0.9048 | 0.2501 | 0.425 | 0.1872 | 1.6549 |
| p_good | 0.8023 | 0.9048 | 0.2501 | **0.575** | 0.1414 | 0.3213 |
| neg_risk | 1.0402 | 0.9048 | 0.2501 | 0.300 | 0.1074 | 0.2707 |

对比当前主要 baselines：

| scorer | selected L1 | random L1 | beats random | corr |
|---|---:|---:|---:|---:|
| old action-aware quality | 0.7725 | 0.9962 | 0.625 | 0.2704 |
| PTG v2 guidance | 0.7552 | 0.8838 | 0.525 | 0.1174 |
| insertion risk quality | **0.7176** | 0.9048 | 0.425 | 0.2077 |

结论：

1. 插座 risk scorer 在离线分类上很强，在 DP sampled candidate 上也能把 selected L1 降到 0.7176，是目前 selected L1 最好的结果之一。
2. 但 beats-random 只有 42.5%，说明它不是稳定地每帧都选更好候选，而是少数帧选得很好、少数帧选错很重。
3. `neg_risk` 单独使用失败，说明“避免风险”不等价于“选择接近专家/更好动作”；需要 quality 或 combined score。
4. `p_good` 稳定性略好但 selected L1 较差，且分类头仍可能饱和。
5. 当前最合理插座策略不是大 scale 梯度直接推，而是：
   - candidate reranking/late-step guidance；
   - score = risk_scorer quality 为主；
   - binary/risk 只作安全约束或 clipping；
   - 小 scale + grad clipping；
   - 后续必须用更合理的 tactile-quality candidate target 替代 L1-to-expert。

## 2026-06-09 Guidance Suitability 评估：准确分类 + 可导势函数

用户强调：评分/分类器必须同时满足两个目标：

1. 能准确评估和分类触觉质量；
2. 最终能作为 DP classifier guidance 的梯度来源。

因此新增专门评估脚本，不再把 reranking 当主目标：

- `TFAC_V5/eval_scorer_guidance_suitability.py`

评估定义：

```text
准确性：
  binary AUC / macro-F1 / reason macro-F1 / quality correlation

梯度引导适用性：
  score saturation：概率是否大量贴近 0/1；
  score range：是否有足够动态范围；
  gradient norm：对 marker/action 是否有非零有限梯度；
  local gradient ascent：沿 action 梯度小步上升后 score 是否提高。
```

命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_scorer_guidance_suitability.py --device cuda:0
```

输出：

- `/home/chenshuai/Project/output/scorer_guidance_suitability/guidance_suitability_eval.json`

### 关键设计修正：概率用于分类，logit energy 用于梯度引导

评估发现 `p_good` 虽然分类 AUC 很高，但严重饱和：

| scorer | p_good 饱和比例 | quality 饱和比例 |
|---|---:|---:|
| insertion risk scorer | 87.43% | 34.30% |
| PTG v2 mixed scorer | 72.75% | 6.88% |

这说明：

1. `p_good` 适合报告“好/坏分类置信度”；
2. `p_good` 不适合直接作为 DP 梯度引导主 score，因为 sigmoid/softmax 饱和后梯度容易变小或变得不稳定；
3. DP guidance 更适合使用未压缩 logit 形式的 energy。

因此新增 runtime score：

```text
insertion energy =
  quality_logit
  + 0.25 * (good_logit - bad_logit)
  + 0.25 * (good_insert_logit - logsumexp(pre_bounce_logit, impact_logit))

unified PTG v2 energy =
  quality_logit
  + 0.25 * (good_logit - bad_logit)
  + 0.25 * (good_stable_smooth_logit - logsumexp(other_bad_reason_logits))

energy_clipped = 4 * tanh(energy / 4)
```

实现位置：

- `TFAC_V5/insertion_risk_scorer_runtime.py`
- `TFAC_V5/ptg_proxy_scorer_v2_runtime.py`

解释：

- `quality_logit` 提供连续质量排序；
- `good_logit - bad_logit` 保留二分类安全边界；
- `reason_logit_margin` 明确惩罚 pre-bounce / impact / rough / too-heavy / too-light 等坏原因；
- `energy` 用于训练/分析时看完整动态范围；
- `energy_clipped` 用于真实 DP guidance，避免过大的梯度把 denoising 推崩。

### Insertion Risk Scorer：当前插座主推荐

准确性：

| metric | value |
|---|---:|
| binary AUC | 0.9975 |
| reason macro-F1 | 0.8666 |
| quality corr | 0.8479 |

不同 score 的诊断：

| score | corr with quality | binary AUC | good-bad margin | range |
|---|---:|---:|---:|---:|
| quality | **0.8479** | 0.9931 | 0.7921 | 0.9999 |
| p_good | 0.6205 | **0.9975** | 0.9464 | 1.0000 |
| log_p_good | 0.6084 | **0.9975** | 14.5445 | 18.4207 |
| risk_guidance | 0.6780 | 0.9958 | 6.3505 | 7.9472 |
| energy | 0.7925 | 0.9968 | **23.9310** | **41.4722** |
| energy_clipped | 0.7294 | 0.9968 | 7.3407 | 7.9994 |

梯度局部上升测试，沿 action 梯度走一步后 score 提升：

| score | weak/approach | good_insert | pre_bounce | impact/recovery |
|---|---:|---:|---:|---:|
| quality | 100.0% | 98.4% | 100.0% | 100.0% |
| log_p_good | 92.2% | 48.4% | 87.5% | 87.5% |
| risk_guidance | 100.0% | 98.4% | 100.0% | 100.0% |
| energy | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| energy_clipped | **100.0%** | **100.0%** | **100.0%** | **100.0%** |

结论：

1. 插座任务当前最佳方案是 **Insertion Risk Scorer + energy/energy_clipped guidance score**。
2. 分类/解释时用 `p_good`、`reason_prob`、`quality_score`；
3. DP 梯度引导时用 `energy_clipped` 作为默认势函数；
4. 若离线分析或小 scale 引导，可使用完整 `energy`；
5. `log_p_good` 不适合作主 score，因为对已经 good 的样本改善率只有 48.4%，说明分类头饱和后局部梯度不稳定。

### PTG Proxy Scorer v2：当前跨任务/黑板主推荐

mixed 准确性：

| metric | value |
|---|---:|
| binary AUC | 0.9898 |
| reason macro-F1 | 0.8109 |
| quality corr | 0.7045 |

分任务：

| task | binary AUC | reason macro-F1 | quality corr | p_good 饱和 | quality 饱和 |
|---|---:|---:|---:|---:|---:|
| board | **0.9996** | **0.9716** | **0.9626** | 92.41% | 1.03% |
| insertion | 0.9862 | 0.7533 | 0.6860 | 70.68% | 7.49% |

结论：

1. `PTG Proxy Scorer v2` 对擦黑板非常强，已经能很好复现“力大小合适 + 力变化柔顺”的弱质量标准；
2. 对插座也可用，但不如插座专用 risk scorer；
3. 跨任务默认 guidance score 应使用 `energy_clipped`，而不是 `p_good`；
4. 更合理的最终系统是 task-conditioned multi-head scorer：
   - 插座：专用 insertion risk head；
   - 黑板：PTG v2 / board quality head；
   - 共享接口：`score(mode="energy_clipped")`。

### 当前最佳评分/分类器定义

推荐命名：`TacQualityEnergy Scorer`

核心思想：

```text
Foresight(action, state) -> predicted tactile consequence
predicted tactile + action -> multi-head scorer

分类输出：
  p_good
  reason_prob
  quality_score

梯度引导输出：
  energy_clipped = 4 * tanh((quality_logit + class_margin + reason_margin) / 4)
```

这比单纯 binary classifier 更合理，因为：

1. binary 分类只告诉“好/坏”，容易饱和；
2. continuous quality 提供可排序的质量坡度；
3. reason margin 告诉模型坏在哪里，避免只学一个不可解释黑箱；
4. clipped energy 保留梯度动态范围，同时控制 guidance 强度。

DP 接入建议：

```python
score = scorer.score(pred_marker_seq, action_seq, mode="energy_clipped")
grad = torch.autograd.grad(score.sum(), noisy_action)[0]
noisy_action = noisy_action + guidance_scale * normalize_or_clip(grad)
```

保守上线策略：

1. 只在 denoising 后 20%-40% steps 使用；
2. `guidance_scale` 从很小开始；
3. 对 grad 做 norm clipping；
4. 插座优先用 insertion risk scorer；
5. 黑板优先用 PTG v2 / board quality scorer；
6. 每次引导后仍要检查动作平滑约束。

## 2026-06-09 Energy 系数搜索

动机：

上一节的 `0.25/0.25` logit margin 权重是人工设定。为了让最终 score 更科学，新增系数搜索：

- `TFAC_V5/search_energy_score_coeffs.py`

搜索形式：

```text
energy = wq * quality_logit + wb * binary_margin + wr * reason_margin
```

目标函数：

```text
objective =
  0.45 * quality_corr
  + 0.35 * binary_auc
  + 0.10 * tanh(good_bad_margin / 4)
  + 0.10 * range_score
```

其中 `range_score` 偏好 p01-p99 range 接近 8，避免 score 过小没梯度或过大导致 guidance 不稳。

命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/search_energy_score_coeffs.py --device cuda:0
```

输出：

- `/home/chenshuai/Project/output/scorer_guidance_suitability/energy_coeff_search.json`

结果：

| target | wq | wb | wr | objective | quality corr | binary AUC | margin | p01-p99 range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| insertion | 0.50 | 0.10 | 0.00 | 0.8576 | 0.7894 | 0.9960 | 7.8265 | 13.8735 |
| PTG v2 mixed | 0.50 | 0.00 | 0.20 | 0.8241 | 0.7006 | 0.9825 | 4.8589 | 9.8610 |
| PTG v2 board | 0.75 | 0.10 | 0.00 | **0.9508** | **0.9353** | **0.9980** | 4.5469 | 8.0589 |
| PTG v2 insertion | 0.75 | 0.00 | 0.10 | 0.8226 | 0.6842 | 0.9821 | 4.9291 | 9.2310 |

重要结论：

1. 插座最佳 energy 不需要 reason margin，说明 reason head 对解释有用，但主梯度由 `quality_logit + binary safety margin` 更稳定。
2. 黑板最佳 energy 也是 `quality_logit + binary margin`，这符合黑板任务的定义：力大小合适和柔顺本质是连续质量。
3. mixed scorer 需要少量 reason margin，因为两个任务的坏原因不同，reason head 有助于统一语义。
4. 默认推荐：
   - 插座：`energy = 0.5 * quality_logit + 0.1 * binary_margin`；
   - 黑板：`energy = 0.75 * quality_logit + 0.1 * binary_margin`；
   - 跨任务 mixed：`energy = 0.5 * quality_logit + 0.2 * reason_margin`；
   - 实际 DP guidance 使用 clipped 版本。

代码更新：

1. `InsertionRiskScorerRuntime.score(mode="energy")` 默认使用搜索得到的插座权重；
2. `PTGProxyScorerV2Runtime.weighted_energy_score(...)` 支持显式传入任务权重；
3. 黑板 guidance 推荐调用：

```python
score = scorer.weighted_energy_score(
    left_marker_seq,
    right_marker_seq,
    eef_action_seq=eef_action_seq,
    joint_action_seq=joint_action_seq,
    task_id=board_task_id,
    quality_weight=0.75,
    binary_weight=0.1,
    reason_weight=0.0,
    clip=True,
)
```

## 2026-06-09 Full-Chain Guidance Gradient 测试

动机：

仅证明 scorer 对 marker/action 有梯度还不够。真正的 DP classifier guidance 需要完整链路：

```text
action_seq
  -> Foresight(action, state, current tactile)
  -> predicted tactile latent
  -> TactileVAE decoder
  -> predicted marker
  -> TacQualityEnergy scorer
  -> d score / d action_seq
```

因此新增完整链路梯度测试：

- `TFAC_V5/eval_full_chain_guidance_gradient.py`

该脚本不做 reranking，不以 L1-to-expert 为主指标，只验证 `score.backward()` 是否能穿过 Foresight 回到 action，以及沿梯度小步更新 action 后 score 是否提高。

命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_full_chain_guidance_gradient.py \
  --device cuda:0 --K 8 --n_eval 16 --score_mode energy_clipped
```

输出：

- `/home/chenshuai/Project/output/full_chain_guidance_gradient/insertion_full_chain_energy_clipped_K8_N16.json`

设置：

```text
task = insertion
K = 8 action samples per frame
N = 16 frames
score = InsertionRiskScorerRuntime.score(mode="energy_clipped")
gradient path = action -> Foresight -> TactileVAE decoder -> marker -> scorer
step_size = 0.02 along normalized action gradient
```

结果：

| metric | value |
|---|---:|
| finite grad rate | **1.0000** |
| score improved rate after gradient step | **0.9766** |
| score delta mean | 0.0388 |
| score delta median | 0.0099 |
| grad norm mean | 2.9588 |
| grad relative norm mean | 0.0060 |
| marker delta norm mean | 1.1688 |
| latent z delta norm mean | 0.7720 |

解释：

1. `finite_grad_rate=1.0` 说明没有断图，梯度能从 scorer 穿过 decoded marker 和 Foresight 回到 action。
2. `score_improved_rate=97.66%` 说明沿 action 梯度方向小步更新，绝大多数样本的 TacQualityEnergy 确实提高。
3. 这比单独的 scorer gradient sanity 更强，因为它验证的是完整 guidance 链路。
4. 当前 full-chain 只对插座任务成立，因为现有 DP/Foresight checkpoint 是插座任务的。擦黑板 scorer 已经离线很强，但要验证黑板 full-chain guidance，需要先训练或接入黑板任务自己的 Foresight/DP。

当前状态：

- 插座：分类/评分准确性、非饱和 energy、full-chain action gradient 三个条件均已初步满足；
- 擦黑板：弱标签质量标准和 scorer 准确性已较强，但 full-chain guidance 尚缺对应 Foresight/DP。

## 2026-06-09 DP Denoising Guidance Dry-Run

动机：

full-chain gradient 证明了 `d score / d action` 存在，但还没有证明把这个梯度注入 DP denoising loop 后是稳定的。因此新增独立 dry-run，不改主推理代码：

- `TFAC_V5/eval_tac_energy_guided_denoising.py`

该脚本用同一初始噪声对比：

```text
baseline DDPM denoising
guided DDPM denoising with late-step TacQualityEnergy gradients
```

guidance 方式：

1. DP 每步先正常 denoising；
2. 后段 steps 才启用 TacQualityEnergy；
3. 将 normalized noisy_action 映射到 raw action；
4. 走 Foresight + scorer 得到 `energy_clipped`；
5. 对 normalized noisy_action 求梯度；
6. 小步更新并 clamp 到 `[-1, 1]`。

第一组 naive guidance：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_tac_energy_guided_denoising.py \
  --device cuda:0 --K 4 --n_eval 8 \
  --guidance_scale 0.015 --guide_start_frac 0.75 --guide_every 2
```

输出：

- `/home/chenshuai/Project/output/tac_energy_guided_denoising/insertion_guided_denoising_energy_clipped_K4_N8.json`

结果：

| metric | value |
|---|---:|
| score delta mean | **+0.6695** |
| guided beats baseline | 0.6875 |
| range violation max | 0.0 |
| smoothness delta mean | -0.0329 |
| norm action delta mean | 3.2863 |

小规模参数 sweep：

| guidance_scale | start_frac | every | smooth_weight | score delta mean | beats baseline | range violation | smoothness delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.015 | 0.75 | 2 | 0.0 | **+0.6695** | **0.6875** | 0.0 | -0.0329 |
| 0.005 | 0.85 | 2 | 0.0 | +0.4935 | 0.5938 | 0.0 | -0.0587 |
| 0.005 | 0.80 | 4 | 0.0 | +0.2037 | 0.4688 | 0.0 | -0.0719 |
| 0.005 | 0.85 | 4 | 0.0 | +0.2106 | 0.4375 | 0.0 | -0.0693 |
| 0.008 | 0.85 | 4 | 0.0 | +0.2126 | 0.4375 | 0.0 | -0.0698 |
| 0.005 | 0.85 | 4 | 0.02 | +0.2105 | 0.4375 | 0.0 | -0.0695 |

第二组 trust-region / acceptance guidance：

新增机制：

```text
--accept_only_improved
--max_norm_delta_per_step 0.02
```

每次 guidance 后重新计算 score，只接受当前 step 上 score 提高的样本；同时限制每步 normalized action 改变量。

命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_tac_energy_guided_denoising.py \
  --device cuda:0 --K 4 --n_eval 8 \
  --guidance_scale 0.02 --guide_start_frac 0.75 --guide_every 2 \
  --max_norm_delta_per_step 0.02 --accept_only_improved \
  --output /home/chenshuai/Project/output/tac_energy_guided_denoising/insertion_guided_accept_trust_K4_N8.json
```

结果：

| metric | value |
|---|---:|
| score delta mean | **+0.7291** |
| guided beats baseline | 0.6875 |
| accept rate mean | 0.9303 |
| range violation max | 0.0 |
| smoothness delta mean | -0.0164 |
| norm action delta mean | 3.2889 |

结论：

1. TacQualityEnergy guidance 在 denoising loop 中平均能显著提高 scorer energy，且没有 normalized action 越界。
2. 动作平滑性没有变差，平均 smoothness delta 反而略降。
3. 但逐样本稳定性不足，best setting 也只有 68.75% samples 高于 baseline，未达到稳定上线阈值。
4. trust-region acceptance 提高了平均 score 和安全性，但没有解决所有样本稳定提升的问题。
5. 因此当前结论必须分开：
   - **scorer 合理**：分类准确、score 非饱和、full-chain gradient 成立；
   - **naive denoising injection 尚不稳定**：需要更好的注入策略。

下一步建议：

1. 只对低 score 或高风险样本启用 guidance，避免扰动本来已经好的动作；
2. 使用 uncertainty / score margin 自适应 guidance scale；
3. 在 predicted clean action 或 x0 estimate 上引导，而不是每个 noisy state 后直接改；
4. 把 action smoothness / joint limit 作为显式 barrier energy；
5. 在真机前只使用小 scale + late steps + trust-region acceptance。

## 2026-06-09 Low-Score Gated Guidance 迭代

动机：

上一轮 naive / trust-region guidance 说明平均 score 能提升，但逐样本稳定性不足。一个自然假设是：不应该引导所有样本，只应该修正低 score / 高风险样本。

因此在 `TFAC_V5/eval_tac_energy_guided_denoising.py` 中加入：

```text
--guide_gate {all, below_mean, below_median, below_quantile, below_threshold}
--gate_quantile
--gate_threshold
```

并记录：

```text
guide_gate_rate_per_step
guide_accept_rate_per_step
```

测试设置：

```text
K = 4
N = 8
score_mode = energy_clipped
accept_only_improved = true
max_norm_delta_per_step = 0.02
```

结果：

| setting | score delta mean | beats baseline | gate rate | accept rate | smoothness delta |
|---|---:|---:|---:|---:|---:|
| all + trust-region | **+0.7291** | **0.6875** | - | 0.9303 | -0.0164 |
| below_quantile q=0.35 | +0.6039 | 0.5938 | 0.5000 | 0.4447 | -0.0396 |
| below_mean | +0.5696 | 0.5938 | 0.4784 | 0.4231 | -0.0456 |
| below_median | +0.4574 | 0.5000 | 0.2500 | 0.2163 | -0.0534 |
| below_median, scale=0.015 | +0.4442 | 0.5000 | 0.2500 | 0.2091 | -0.0575 |
| below_quantile q=0.5, start=0.80 | +0.2036 | 0.4375 | 0.5000 | 0.4500 | -0.0710 |

结论：

1. low-score gate 可以减少实际接受的 guidance 更新，但没有提升最终逐样本稳定性。
2. 当前最佳 dry-run 仍是全量 trust-region acceptance。
3. 这说明主要瓶颈不是“是否只推低分样本”，而是 denoising 注入变量/位置：直接修改 noisy action 容易被后续 DDPM dynamics 抵消或放大。
4. 下一步更应该尝试：
   - 在 predicted clean action / x0 estimate 上引导；
   - 或只在最后输出 action 上做 one-shot energy refinement；
   - 或将 energy scorer 作为训练时 auxiliary guidance / consistency loss，而不是纯推理时硬推。

当前推荐不变：

- scorer 选择：`TacQualityEnergy` 是合理的；
- 推理注入：不要直接上线 naive noisy-action guidance；
- 工程策略：先用 trust-region / one-shot clean-action refinement 做安全版本。

## 2026-06-09 Clean-Action Trust-Region Refinement

动机：

noisy-step denoising guidance 平均能提高 score，但逐样本不稳定。更安全的方式是：

```text
DP 正常生成 clean action
  -> 对最终 raw action 做少量 trust-region energy refinement
  -> 只有 score 提高才接受
  -> 总 action 改变量受限
```

新增脚本：

- `TFAC_V5/eval_clean_action_energy_refinement.py`

这不是 reranking，也不是改训练；它是一个可控的后处理 guidance 原型。

命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_clean_action_energy_refinement.py \
  --device cuda:0 --K 4 --n_eval 8 \
  --refine_steps 4 --refine_step_size 0.02 --max_total_delta 0.08
```

输出：

- `/home/chenshuai/Project/output/clean_action_energy_refinement/insertion_clean_refine_energy_clipped_K4_N8.json`

设置：

```text
score = energy_clipped
refine_steps = 4
refine_step_size = 0.02
max_total_delta = 0.08
accept rule = per-sample accept only if score improves
```

结果：

| metric | value |
|---|---:|
| score delta mean | **+0.3024** |
| score delta median | +0.1563 |
| refined beats baseline | **1.0000** |
| accept rate per step | 0.9766 |
| action delta norm mean | 0.0748 |
| action delta norm max | 0.0800 |
| smoothness delta mean | +0.0089 |
| passes clean refinement sanity | **true** |

和 noisy-step denoising guidance 对比：

| method | score delta mean | beats baseline | action control | conclusion |
|---|---:|---:|---|---|
| noisy-step guidance | +0.6695 | 0.6875 | clamp only | 平均提高但不稳定 |
| noisy-step trust-region | +0.7291 | 0.6875 | per-step accept | 平均提高但不稳定 |
| low-score gated noisy-step | +0.2036 ~ +0.6039 | 0.4375 ~ 0.5938 | gated accept | 没解决稳定性 |
| clean-action trust-region | +0.3024 | **1.0000** | total delta <= 0.08 | 当前最稳 |

结论：

1. TacQualityEnergy scorer 作为评分/分类器是可用的；
2. 直接 noisy-step classifier guidance 还不稳定；
3. **clean-action trust-region refinement 是当前最合理的落地方式**：
   - 不改 DP 主采样；
   - 只微调最终动作；
   - 每步只接受 score 提升；
   - 限制总动作改变量；
   - 对所有样本都提升了 predicted tactile quality energy。

推荐当前工程路线：

```text
DP output action
  -> Foresight predicts tactile consequence
  -> TacQualityEnergy evaluates quality
  -> 4-step trust-region action refinement
  -> output refined action
```

上线前仍需：

1. 加 joint limit barrier；
2. 加 action smoothness barrier；
3. 扩大 N 做更稳统计；
4. 真机前只使用很小 `max_total_delta`；
5. 黑板任务需要对应 Foresight/DP 后才能做 full-chain refinement。

## 2026-06-09 Constrained Clean-Action TacQualityEnergy Guidance

前一版 clean-action refinement 已经比 noisy-step guidance 稳定，但还存在一个关键问题：如果只最大化 TacQualityEnergy，梯度可能把 action 推向训练数据分布边界。一个能用于 DP guidance 的 scorer，不能只是分类准确，还必须保证 score 对 action 可微、梯度方向能提高预测触觉质量、action 改变量受 trust region 限制、refined action 不越过训练 action 范围，并且不显著破坏动作平滑性。

因此将优化目标改成 constrained objective：

```text
objective(a)
  = TacQualityEnergy(Foresight(a))
    - lambda_smooth * action_smoothness(a)
    - lambda_limit * action_limit_barrier(a)
```

其中：

```text
TacQualityEnergy = 0.5 * quality_logit + 0.1 * binary_margin
energy_clipped = 4 * tanh(TacQualityEnergy / 4)
```

本轮实验使用：

```text
score_mode = energy_clipped
smooth_weight = 0.02
joint_limit_weight = 10.0
joint_margin_frac = 0.03
refine_steps = 4
refine_step_size = 0.02
max_total_delta = 0.08
accept rule = accept only if constrained objective improves
```

代码改动：

- `TFAC_V5/eval_clean_action_energy_refinement.py`
- 从 DP config 读取 `norm_stats.action_min/action_max`；
- 新增 action limit soft barrier；
- 新增 hard range violation 统计；
- refined action 每步只接受 constrained objective 提升；
- 输出中记录 `base_limit_barrier`、`refined_limit_barrier`、`limit_barrier_delta`、`base_hard_range_violation`、`refined_hard_range_violation`。

Sanity check 命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_clean_action_energy_refinement.py \
  --device cuda:0 --K 4 --n_eval 4 \
  --refine_steps 4 --refine_step_size 0.02 --max_total_delta 0.08 \
  --smooth_weight 0.02 --joint_limit_weight 10.0 --joint_margin_frac 0.03 \
  --output /home/chenshuai/Project/output/clean_action_energy_refinement/insertion_clean_refine_constrained_K4_N4_sanity.json
```

更大样本命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_clean_action_energy_refinement.py \
  --device cuda:0 --K 4 --n_eval 40 \
  --refine_steps 4 --refine_step_size 0.02 --max_total_delta 0.08 \
  --smooth_weight 0.02 --joint_limit_weight 10.0 --joint_margin_frac 0.03 \
  --output /home/chenshuai/Project/output/clean_action_energy_refinement/insertion_clean_refine_constrained_K4_N40.json
```

Sanity check 结果：

| metric | value |
|---|---:|
| n action samples | 16 |
| constrained score delta mean | +0.2515 |
| refined beats baseline | 1.0000 |
| action delta norm max | 0.0800 |
| refined hard range violation max | 0.0 |
| passes sanity | true |

N=40 结果：

| metric | value |
|---|---:|
| n frames | 40 |
| n action samples | 160 |
| constrained score delta mean | **+0.3056** |
| constrained score delta median | +0.1953 |
| refined beats baseline | **0.9750** |
| action delta norm mean | 0.0737 |
| action delta norm max | 0.0800 |
| smoothness delta mean | +0.0025 |
| refined hard range violation mean | 0.0 |
| refined hard range violation max | 0.0 |
| accept rate per step mean | 0.9547 |
| passes sanity | true |

当前最合理的插插座任务 guidance 方案：

```text
DP 正常输出 clean action
  -> Foresight 预测未来触觉 latent/marker
  -> TacQualityEnergy 计算可微质量 energy
  -> 加 action smoothness + action range barrier
  -> 对 clean action 做 4 步 trust-region 梯度 refinement
  -> 每步只接受 objective 提升的 proposal
```

这个方案不是 reranking。它直接使用 `d objective / d action`，因此满足“评分/分类器最终用于 DP 梯度引导”的目标。

和 classifier guidance 的关系：

经典 classifier guidance 在 diffusion 里使用：

```text
grad_x log p(class | x_t)
```

这里对应为：

```text
grad_action TacQualityEnergy(Foresight(action))
```

区别是 classifier/scorer 不直接看 noisy action，而是看 Foresight 预测出的触觉后果。因此梯度链路是：

```text
action -> Foresight -> predicted tactile -> scorer -> objective
```

这正是 PTG 的核心：不是问 action 本身像不像专家，而是问这个 action 导致的未来触觉后果好不好。

当前结论边界：

1. 插插座任务上，TacQualityEnergy 作为 scorer/gradient objective 是成立的；
2. full-chain gradient 已经验证可通；
3. constrained clean-action guidance 比 noisy-step guidance 更稳定；
4. 加边界约束后，N=40 / 160 actions 仍有 97.5% 提升且 0 越界；
5. 黑板任务还没有 board-specific Foresight/DP full-chain 验证；
6. 当前评估仍是 offline predicted tactile，不等于真机闭环成功率；
7. clean-action refinement 是更安全的 classifier/scorer guidance 注入位置，不是标准每步 DDPM noisy-state guidance。

下一步最关键：

1. 黑板任务训练/接入对应 Foresight 后，跑同样的 full-chain gradient + clean-action refinement；
2. 将 constrained objective 接入推理服务，作为可开关的小步 action refinement；
3. 在真机上从更小 `max_total_delta` 开始，例如 0.02 / 0.04 / 0.08 分级测试。

## 2026-06-09 Unified Taxonomy Fast Eval 补充

### 要回答的问题

用户提出的关键点是：评分/分类器不能只是“一个好坏类别”，也不能只做 reranking；最终要作为 DP classifier guidance 的梯度源。因此这里补齐一个统一 taxonomy 的快速评估，判断：

1. binary / t3 / t4 哪种分类定义更合理；
2. 是否能用一个任务无关模型同时覆盖插座和擦黑板；
3. 是否必须使用 task-conditioned scorer；
4. 黑板任务中“力大小合适 + 力变化柔顺”的弱标签是否能被模型学习。

### 黑板弱标签定义

数据：

```text
/home/chenshuai/data/dataset/260522_v8l_caheiban
```

窗口：

```text
window = 32
stride = 16
```

黑板 good 的定义：

```text
force 不过小
force 不过大
force_delta / action_delta / marker_delta 不粗糙
```

黑板 bad 的细分类：

```text
too_light: 力过小
too_heavy: 平均力或峰值力过大
rough: 力变化、动作变化或 marker 变化不柔顺
```

重要约束：

```text
force 只用于生成弱标签；
模型输入仍是 marker/action proxy；
因此后续可以接 Foresight predicted tactile/action，不依赖未来真实力传感器。
```

### 实验命令

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/run_unified_quality_fast_eval.py \
  --include-rf --max-per-task-class 1200
```

输出：

```text
/home/chenshuai/Project/output/unified_quality_taxonomy/unified_quality_eval_fast.json
```

### Unified Taxonomy 结果

best candidate 是 `binary + RandomForest`：

| taxonomy/model | mixed macro-F1 | mixed AUC | score Spearman | objective |
|---|---:|---:|---:|---:|
| binary + RF | **0.7773** | **0.8700** | 0.4299 | **0.7376** |
| binary + LogReg | 0.7189 | 0.8176 | 0.3951 | 0.6862 |
| t3 + RF | 0.6689 | 0.8474 | 0.4161 | 0.6503 |
| t4 + RF | 0.6521 | 0.8618 | 0.4304 | 0.6384 |

解释：

1. 如果只看传统 ML 的统一特征，binary 最容易、效果最好；
2. t3/t4 解释性更强，但 macro-F1 下降；
3. 这说明最终系统不应该只保留 binary，也不应该只依赖 t4 分类概率；
4. 更合理的是：
   - binary head：好/坏安全判断；
   - reason head：解释坏在哪里；
   - quality/energy head：连续梯度。

### 跨任务零样本迁移

| setting | macro-F1 | AUC |
|---|---:|---:|
| binary RF, insertion -> board | 0.4102 | 0.4582 |
| binary RF, board -> insertion | 0.5126 | 0.5261 |
| t4 RF, insertion -> board | 0.1913 | 0.4668 |
| t4 RF, board -> insertion | 0.2100 | 0.4802 |

结论：

1. 任务无关的单头模型不能可靠跨任务迁移；
2. 插座的坏触觉是 bounce/risk，黑板的坏触觉是 too_light/too_heavy/rough，它们的语义不同；
3. 因此正确路线不是“一个类别/一个无条件分类器”，而是：

```text
shared tactile/action representation
  + task id
  + binary head
  + reason head
  + continuous quality/energy head
```

### 和已有 stronger scorer 对比

| scorer | mixed binary AUC | reason/t4 macro-F1 | quality corr |
|---|---:|---:|---:|
| Unified RF fast eval | 0.8700 | 0.6521 t4 | 0.4299 Spearman |
| Action-aware marker scorer joint_abs | 0.9453 | 0.7552 t4 | 0.7063 |
| PTG Proxy Scorer v2 | **0.9701** | **0.7678 reason** | **0.7562** |

PTG v2 分任务：

| task | binary AUC | reason macro-F1 | quality corr |
|---|---:|---:|---:|
| insertion | 0.9562 | 0.7206 | 0.6530 |
| board | **0.9790** | **0.8363** | **0.9364** |

这说明：

1. PTG v2 在黑板任务上已经很好地学习了“力大小合适 + 力变化柔顺”的弱质量标准；
2. PTG v2 比传统 unified RF/LogReg 更适合作为跨任务 scorer；
3. 插座任务最好仍保留专门的 insertion risk scorer，因为插座坏数据定义更清晰、风险结构更特殊。

### 当前推荐分类/评分设计

不要只用二分类概率做 guidance。推荐使用：

```text
TacQualityEnergy
  = wq * quality_logit
    + wb * binary_margin
    + wr * reason_margin
```

其中：

```text
binary_margin = logit_good - logit_bad
reason_margin = logit_good_reason - logsumexp(bad_reason_logits)
energy_clipped = 4 * tanh(energy / 4)
```

当前最佳权重：

| target | wq | wb | wr | reason |
|---|---:|---:|---:|---|
| insertion | 0.50 | 0.10 | 0.00 | risk scorer 最稳 |
| board | 0.75 | 0.10 | 0.00 | quality corr 最高 |
| mixed | 0.50 | 0.00 | 0.20 | reason head 帮助统一坏原因 |

### 当前设计结论

最终方案不是“只分好坏”，而是：

1. 多头分类/评分器：
   - binary：好/坏；
   - reason：too_light / good / too_heavy-risk / rough-impact / rough_motion；
   - quality：连续质量分；
2. guidance 不直接用 `p_good`，因为概率容易饱和；
3. guidance 用 logit energy，保证梯度更连续；
4. 插座和黑板共享框架，但使用 task id 和不同 energy 权重；
5. full-chain guidance 目前插座已验证，黑板还需要对应 Foresight/DP。

## 2026-06-09 Board Guidance Readiness

### 为什么做这个实验

当前没有发现明确指向擦黑板数据集：

```text
/home/chenshuai/data/dataset/260522_v8l_caheiban
```

的 board-specific DP/Foresight checkpoint。因此不能声称黑板任务已经完成 full-chain guidance。

但仍然可以验证一个必要条件：

```text
PTG v2 board scorer 是否对 marker/action 有可用梯度？
```

如果 scorer 自身没有稳定梯度，那么后续接 Foresight/DP 也没有意义。因此新增一个 board guidance readiness gate。

### 新增脚本

```text
TFAC_V5/eval_board_guidance_readiness.py
```

它测试：

```text
真实 board marker/action window
  -> PTGProxyScorerV2 board energy
  -> d energy / d(left_marker, right_marker, eef_action, joint_action)
```

然后做一次单位梯度小步上升，检查：

1. 梯度是否有限；
2. 四类输入梯度是否非零；
3. 小步后 energy 是否提升；
4. eef/joint action smoothness 是否明显变差。

### 重要失败结果：默认大步长会失败

Sanity 命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_board_guidance_readiness.py \
  --device cuda:0 --n_eval 32 --batch_size 16 \
  --output /home/chenshuai/Project/output/board_guidance_readiness/board_ptg_v2_energy_readiness_N32_sanity.json
```

默认：

```text
marker_step = 0.02
action_step = 0.02
```

结果：

| metric | value |
|---|---:|
| finite grad rate | 1.0000 |
| positive grad rate | 1.0000 |
| score delta mean | -0.7179 |
| score improved rate | 0.2500 |

解释：

1. 梯度是存在的；
2. 但步长太大，尤其 action step 太大；
3. board scorer 的 action 梯度尺度较敏感，不能直接用粗暴大步长。

这是一个重要安全结论：黑板 guidance 必须使用 very small action step / trust-region / accept-only-improved / backtracking。

### 步长扫描

N=64 扫描结果说明 action step 是主敏感项：

| marker_step | action_step | score delta mean | improve rate |
|---:|---:|---:|---:|
| 0.0002 | 0.0002 | +0.0118 | 1.000 |
| 0.0005 | 0.0002 | +0.0125 | 1.000 |
| 0.0010 | 0.0002 | +0.0132 | 1.000 |
| 0.0020 | 0.0002 | +0.0146 | 1.000 |
| 0.0050 | 0.0002 | +0.0171 | 1.000 |
| 0.0050 | 0.0050 | -0.0896 | 0.312 |

因此正式实验选择：

```text
marker_step = 0.005
action_step = 0.0002
```

### 正式实验

命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_board_guidance_readiness.py \
  --device cuda:0 --n_eval 240 --batch_size 32 \
  --marker_step 0.005 --action_step 0.0002 \
  --quality_weight 0.75 --binary_weight 0.1 --reason_weight 0.0 \
  --output /home/chenshuai/Project/output/board_guidance_readiness/board_ptg_v2_energy_readiness_N240_safe_step.json
```

输出：

```text
/home/chenshuai/Project/output/board_guidance_readiness/board_ptg_v2_energy_readiness_N240_safe_step.json
```

结果：

| metric | value |
|---|---:|
| n windows | 240 |
| score delta mean | **+0.01937** |
| score delta median | +0.01330 |
| score improved rate | **0.9917** |
| finite grad rate all inputs | **1.0000** |
| positive grad rate all inputs | **1.0000** |
| left grad norm mean | 1.3430 |
| right grad norm mean | 1.4027 |
| eef grad norm mean | 63.4560 |
| joint grad norm mean | 1.0972 |
| eef smooth delta mean | -6.8e-7 |
| joint smooth delta mean | -1.64e-5 |
| passes board guidance readiness | **true** |

### 结论

可以确认：

1. PTG v2 board scorer 具备 scorer-level guidance potential；
2. 真实黑板窗口上，board energy 对 tactile marker 和 action 都有有限非零梯度；
3. 小步 guidance 能稳定提高 board energy；
4. 动作平滑度基本没有变差。

但必须保留边界：

1. 这不是 board full-chain；
2. 还没有验证：

```text
action -> board Foresight -> predicted tactile -> PTG v2 board energy -> dscore/daction
```

3. 黑板 full-chain 的必要条件是训练或接入 board-specific Foresight/DP checkpoint。

### 对最终方案的影响

黑板 guidance 推荐配置：

```text
energy = 0.75 * quality_logit + 0.10 * binary_margin
energy_clipped = 4 * tanh(energy / 4)
```

推理时必须：

```text
very small action step
trust-region
accept-only-improved
optional backtracking line search
```

这和插座的 clean-action refinement 思路一致，但黑板 action step 要更小。

## 2026-06-09 TacQuality Guidance Profile 固化

### 动机

前面的实验已经得到一组比较明确的结论：

1. 插座适合用 insertion risk scorer；
2. 黑板适合用 PTG v2 board scorer；
3. 两者都不应该直接用饱和的 `p_good`；
4. guidance 应使用 logit energy；
5. 插座和黑板需要不同的 energy 权重和 action step。

如果这些配置散落在不同脚本里，后续接 DP 推理时很容易出错。因此新增统一配置模块：

```text
TFAC_V5/tac_quality_guidance_config.py
```

### 统一 energy 形式

```text
energy = wq * quality_logit
       + wb * binary_margin
       + wr * reason_margin

energy_clipped = clip_scale * tanh(energy / clip_scale)
```

其中：

```text
binary_margin = logit_good - logit_bad
reason_margin = logit_good_reason - logsumexp(bad_reason_logits)
clip_scale = 4.0
```

### Guidance Profiles

#### insertion

```text
scorer = InsertionRiskScorerRuntime
energy = 0.50 * quality_logit + 0.10 * binary_margin
refine_steps = 4
action_step = 0.02
max_total_delta = 0.08
smooth_weight = 0.02
joint_limit_weight = 10.0
joint_margin_frac = 0.03
```

证据：

```text
full-chain gradient passed
constrained clean-action refinement passed
```

#### board

```text
scorer = PTGProxyScorerV2Runtime
energy = 0.75 * quality_logit + 0.10 * binary_margin
marker_step = 0.005
action_step = 0.0002
max_total_delta = 0.02
smooth_weight = 0.02
```

证据：

```text
scorer-level board guidance readiness passed on real board windows
```

限制：

```text
仍需要 board-specific Foresight/DP 做 full-chain guidance。
```

#### mixed

```text
scorer = PTGProxyScorerV2Runtime
energy = 0.50 * quality_logit + 0.20 * reason_margin
```

用途：

```text
跨任务分析或 fallback；部署优先使用 task-specific profile。
```

### 代码接入

1. `PTGProxyScorerV2Runtime.weighted_energy_score()` 现在调用统一的 `weighted_logit_energy()`；
2. `eval_board_guidance_readiness.py` 默认参数来自 `get_guidance_profile("board")`；
3. 命令行仍可覆盖参数，方便继续做 ablation。

### 验证

命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python -m py_compile \
  TFAC_V5/tac_quality_guidance_config.py \
  TFAC_V5/ptg_proxy_scorer_v2_runtime.py \
  TFAC_V5/eval_board_guidance_readiness.py
```

通过。

公式一致性：

```text
weighted_logit_energy max_abs_diff = 0.0
```

使用 profile 默认值重跑黑板 N=32：

```text
output = /home/chenshuai/Project/output/board_guidance_readiness/board_ptg_v2_energy_readiness_N32_profile_default.json
score_delta_mean = +0.02522
score_improved_rate = 0.96875
finite_grad_rate_all_inputs = 1.0
passes_board_guidance_readiness = true
```

### 当前工程结论

后续 DP 推理/实验不应该再手写权重，而应使用：

```python
from TFAC_V5.tac_quality_guidance_config import get_guidance_profile

profile = get_guidance_profile(task)
```

这一步把“实验中找到的最佳评分/分类器方案”固化为可复用工程接口，是后续真正接入 DP classifier guidance 的基础。

## 2026-06-09 Evidence Summary / Completion Audit

为了避免实验结果分散在多个 JSON 中，新增自动汇总脚本：

```text
TFAC_V5/summarize_ptg_guidance_evidence.py
```

命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/summarize_ptg_guidance_evidence.py
```

输出：

```text
/home/chenshuai/Project/output/ptg_guidance_evidence/ptg_guidance_evidence_summary.json
/home/chenshuai/Project/output/ptg_guidance_evidence/ptg_guidance_evidence_summary.md
```

### 当前审计结果

| item | status | evidence |
|---|---|---|
| insertion scorer GroupKFold | PASS | binary AUC = 0.9877 |
| insertion full-chain gradient | PASS | score improved rate = 0.9766 |
| insertion constrained clean refinement | PASS | beats = 0.975, hard violation = 0 |
| board PTG v2 scorer quality | PASS | mixed AUC = 0.9701, quality corr = 0.7562 |
| board scorer-level readiness | PASS | improved rate = 0.9917, finite grad = 1.0 |
| board full-chain DP/Foresight | FAIL | missing board-specific Foresight/DP |
| unified taxonomy baseline | PASS | binary RF best baseline exists |
| task-conditioned scorer preference | PASS | task-agnostic cross-task transfer weak |

### Completion Assessment

```text
objective_complete = false
```

原因：

```text
All scorer and insertion full-chain checks pass, but board full-chain guidance is still missing.
```

下一步：

```text
Train or locate board-specific Foresight/DP, then run board full-chain guidance/refinement.
```

这个 audit 明确区分了三类证据：

1. scorer 离线准确性；
2. scorer-level gradient readiness；
3. full-chain DP/Foresight guidance。

当前插座已经覆盖 1/2/3；黑板覆盖了 1/2，但还缺 3。

## 2026-06-09 Board Tactile Surrogate Full-Chain Probe

### 动机

黑板当前缺正式的 board-specific Foresight/DP checkpoint。为了继续推进，而不是停在 scorer-level readiness，这里训练一个轻量 tactile consequence surrogate：

```text
current left/right marker window
future eef/joint action chunk
  -> future left/right marker window
```

然后验证：

```text
action -> surrogate predicted tactile -> PTG v2 board energy -> dscore/daction
```

这不是最终 production Foresight，但它直接验证 PTG 的核心机制是否成立。

### 新增脚本

```text
TFAC_V5/train_board_tactile_surrogate.py
```

关键设计：

1. 使用 `260522_v8l_caheiban`；
2. episode-level train/val split；
3. 不使用未来真实 force 作为输入；
4. surrogate 只预测 future marker；
5. scorer 使用 PTG v2 board profile：

```text
energy = 0.75 * quality_logit + 0.10 * binary_margin
```

### Sanity Run

命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/train_board_tactile_surrogate.py \
  --device cuda:0 --epochs 3 --batch_size 512 --n_grad_eval 64 \
  --output /home/chenshuai/Project/output/board_tactile_surrogate/board_tactile_surrogate_sanity.json
```

结果：

| metric | value |
|---|---:|
| n samples | 7112 |
| n train | 5707 |
| n val | 1405 |
| val marker MAE mean | 0.6723 |
| guidance score delta mean | +0.01870 |
| score improved rate | 1.0000 |
| finite action grad rate | 1.0000 |
| positive action grad rate | 1.0000 |
| passes surrogate full-chain | true |

### Formal Run

命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/train_board_tactile_surrogate.py \
  --device cuda:0 --epochs 35 --batch_size 512 --n_grad_eval 256 \
  --output /home/chenshuai/Project/output/board_tactile_surrogate/board_tactile_surrogate_eval.json
```

输出：

```text
/home/chenshuai/Project/output/board_tactile_surrogate/board_tactile_surrogate_eval.json
/home/chenshuai/Project/output/board_tactile_surrogate/board_tactile_surrogate_final.pt
```

结果：

| metric | value |
|---|---:|
| val marker MSE mean | 0.2503 |
| val marker MAE mean | **0.2978** |
| val marker MAE p95 | 0.8435 |
| guidance n | 256 |
| guidance score delta mean | **+0.01273** |
| score improved rate | **0.9805** |
| finite action grad rate | **1.0000** |
| positive action grad rate | **1.0000** |
| eef grad norm mean | 65.9267 |
| joint grad norm mean | 1.3846 |
| passes board surrogate full-chain | **true** |

### 结论

可以新增一条证据：

```text
黑板 surrogate full-chain guidance: PASS
```

它证明：

1. 只要有可微 tactile consequence model，PTG v2 board energy 可以回传到 action；
2. 小步 action gradient 能提高 predicted board tactile quality；
3. 黑板任务的 classifier/scorer guidance 方向在机制上成立。

但仍不能过度宣称：

```text
surrogate full-chain ≠ production board Foresight/DP full-chain
```

原因：

1. surrogate 输入是真实当前 marker 和未来 action chunk；
2. production 系统还需要 DP 生成 action；
3. production Foresight 还需要从视觉/触觉/history/action 预测未来 tactile；
4. 仍需正式 board Foresight/DP checkpoint。

### Audit 更新

`summarize_ptg_guidance_evidence.py` 已加入：

```text
Board surrogate full-chain guidance
```

当前黑板状态：

| item | status |
|---|---|
| scorer quality | PASS |
| scorer-level readiness | PASS |
| surrogate full-chain | PASS |
| production Foresight/DP full-chain | FAIL |

这使黑板从“只有 scorer 证据”推进到“有 learned consequence model 的 full-chain 证据”，但最终完成仍需要 production Foresight/DP。

## 2026-06-09 Scorer Space Visualization and Current Best Selection

### 问题

需要回答三个关键问题：

1. 分类空间是否可以可视化；
2. 是否应该只做 good/bad，还是应该做多个类别；
3. 插座和擦黑板两个任务之间，当前最合理的通用评分/分类器是哪一个。

### 新增脚本

```text
TFAC_V5/visualize_ptg_scorer_space.py
```

该脚本只读取已有实验输出，不重新训练模型，也不重新生成标签。

输入：

```text
/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_features.npz
/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_eval.json
/home/chenshuai/Project/output/marker_field_scorer/marker_field_scorer_eval.json
/home/chenshuai/Project/output/unified_quality_taxonomy/unified_quality_eval_fast.json
/home/chenshuai/Project/output/ptg_guidance_evidence/ptg_guidance_evidence_summary.json
```

输出：

```text
/home/chenshuai/Project/output/ptg_scorer_space/ptg_scorer_space_summary.json
/home/chenshuai/Project/output/ptg_scorer_space/ptg_scorer_space_summary.md
/home/chenshuai/Project/output/ptg_scorer_space/ptg_proxy_reason_task_space.png
/home/chenshuai/Project/output/ptg_scorer_space/ptg_proxy_task_facets.png
/home/chenshuai/Project/output/ptg_scorer_space/ptg_proxy_quality_distributions.png
```

### 可视化设计

使用 PTG proxy scorer v2 的 74 维输入特征：

```text
left marker proxy
right marker proxy
left-right abs difference
eef action proxy
joint action proxy
task id
```

为了避免样本量偏置，空间图按 `task + reason` 平衡抽样：

```text
max_per_task_reason = 700
```

可视化内容：

1. PCA reason 多类空间；
2. t-SNE reason 多类空间；
3. t-SNE task 空间；
4. PCA good/bad/neutral 空间；
5. continuous quality target 颜色图；
6. 按 task 分面的 reason/quality 空间。

### 类别定义

最终不建议只做一个二分类头。推荐保留多头：

```text
binary head: good / bad
reason head: weak_or_no_contact_too_light / good_stable_smooth / excessive_or_risk_too_heavy / impact_or_rough_force / rough_motion
quality head: continuous quality score
```

原因：

1. 插座任务中，坏数据主要来自 bounce episode 的 pre-bounce / bounce / recovery；
2. 黑板任务中，坏数据至少有三类机制：
   - 力过小；
   - 力过大；
   - 力变化不柔顺/动作粗糙；
3. DP 梯度引导时不能直接优化 hard class，需要连续可微 energy；
4. reason head 可以让 energy 更稳定，也能解释模型为什么认为某个 action 差。

### 实验结果

| model | GroupKFold binary AUC | balanced acc | reason/T4 F1 | quality corr |
|---|---:|---:|---:|---:|
| PTG proxy scorer v2 | **0.9701** | **0.9082** | **0.7678** | **0.7562** |
| marker-field NN | 0.8544 | 0.7497 | 0.7394 | 0.5982 |
| traditional RF baseline | 0.8700 | 0.7910 | - | 0.4299 |

当前最合理方案：

```text
TacQualityEnergy / PTGProxyScorerV2Runtime
```

理由：

1. episode-level GroupKFold 分类效果最好；
2. 有二分类、多类原因、连续评分三个输出；
3. runtime 是 PyTorch 可微实现；
4. 黑板 scorer-level gradient readiness 已通过；
5. 黑板 surrogate full-chain action-gradient 已通过；
6. 插座 production full-chain guidance 已通过。

### 为什么 marker-field NN 不是当前主方案

marker-field NN 直接输入 raw marker window，理论上更端到端，但当前结果不如 proxy v2：

```text
binary AUC: 0.8544 vs 0.9701
quality corr: 0.5982 vs 0.7562
```

可能原因：

1. 黑板伪标签由力传感器和力变化定义，而 raw marker 到真实力之间存在尺度/接触区域差异；
2. 插座和黑板的 marker 分布差异很大，直接混训容易学到 task/domain 差异；
3. proxy 特征把“力大小、接触面积、接触中心、变化平滑度”显式提出来，更符合当前人工定义的好坏标准。

因此 marker-field NN 保留为 ablation 或未来增强分支；主方案暂时不替换。

### 与 Diffusion Guidance 文献对齐

本项目方向不是 reranking，而是 classifier/scorer guidance：

```text
action/noisy_action -> tactile consequence model -> tactile quality energy -> dE/daction
```

文献对应关系：

1. Classifier guidance：外部分类器对 diffusion sample 提供梯度；
2. CFG：用条件/无条件 score 差值放大条件引导；
3. TouchGuide：task-specific tactile/contact feasibility model 影响 diffusion/flow policy 推理；
4. PPGuide：performance predictor 给 diffusion policy 提供 inference-time gradient。

我们的创新点应该放在：

```text
task-conditioned tactile quality energy
+ tactile foresight consequence model
+ multi-head reason/quality calibration
+ trust-region action refinement
```

而不是只做候选 reranking。

## 2026-06-09 Board Production Chain Audit

### 动机

当前黑板任务已有两类强证据：

1. scorer-level guidance readiness；
2. learned surrogate full-chain guidance。

但这仍不等于 production DP/Foresight full-chain。为了防止过度宣称，新增一个机器可读审计脚本，专门判断是否存在覆盖黑板数据集的正式 DP/Foresight checkpoint。

### 新增脚本

```text
TFAC_V5/audit_board_production_chain.py
```

审计对象：

```text
/home/chenshuai/data/dataset/260522_v8l_caheiban
```

审计逻辑：

1. 扫描 `/home/chenshuai/Project/output` 和仓库内的 config；
2. 查找 `args.json/config.json/dp_config.json`；
3. 判断 config 是否引用黑板数据集；
4. 判断同目录是否有 `.pt/.pth/.ckpt` checkpoint；
5. 只把 production DP/Foresight 算作候选；
6. 明确不把 board surrogate checkpoint 算作 production Foresight。

### 运行命令

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/audit_board_production_chain.py
```

输出：

```text
/home/chenshuai/Project/output/board_production_chain_audit/board_production_chain_audit.json
/home/chenshuai/Project/output/board_production_chain_audit/board_production_chain_audit.md
```

### 审计结果

| item | value |
|---|---:|
| reported configs | 19 |
| production board candidates | 0 |
| board DP candidates | 0 |
| board Foresight candidates | 0 |
| passes production full-chain prereq | false |

已有黑板证据：

| item | value |
|---|---:|
| board scorer AUC | 0.9701 |
| board quality corr | 0.7562 |
| scorer readiness pass | true |
| scorer readiness improved rate | 0.9917 |
| surrogate full-chain pass | true |
| surrogate improved rate | 0.9805 |

### 结论

当前不能把整体目标标成 complete。

准确说法应该是：

```text
Scorer design: ready as current best.
Insertion production full-chain: verified.
Board scorer-level guidance: verified.
Board surrogate full-chain: verified.
Board production DP/Foresight full-chain: missing.
```

因此后续真正需要补的是：

```text
Train or locate board-specific production DP/Foresight
  -> run action -> production Foresight -> PTG board energy -> dscore/daction
  -> run clean-action trust-region refinement
  -> verify score improves without action/force safety violations
```

这一步完成之前，surrogate 只能作为机制证明，不能作为最终 production 证明。

## 2026-06-09 Board Surrogate Clean-Action Refinement

### 动机

前面的 board surrogate full-chain probe 只验证了单步梯度：

```text
action -> surrogate predicted tactile -> PTG score -> one gradient step
```

这还不够接近最终 DP guidance。实际部署更合理的形式是：

```text
DP produces clean action
  -> small trust-region refinement
  -> accept only if PTG score improves
```

因此新增一个多步 constrained refinement 实验，验证 PTG board energy 是否可以稳定地优化 action。

### 新增脚本

```text
TFAC_V5/eval_board_surrogate_action_refinement.py
```

链路：

```text
current marker + action
  -> board tactile surrogate
  -> predicted future left/right marker
  -> PTGProxyScorerV2 board energy
  -> dscore/daction
  -> trust-region action update
```

配置：

```text
refine_steps = 4
action_step = 0.0002
max_total_delta = 0.02
accept_only_improved = true
energy = 0.75 * quality_logit + 0.10 * binary_margin
```

### Sanity Run

命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_board_surrogate_action_refinement.py \
  --device cuda:0 --n_eval 32 --batch_size 32 \
  --output /home/chenshuai/Project/output/board_surrogate_action_refinement/board_surrogate_refine_sanity_N32.json
```

结果：

| metric | value |
|---|---:|
| n windows | 32 |
| score delta mean | +0.03656 |
| score improved rate | 1.0000 |
| eef delta norm mean | 0.00070 |
| joint delta norm mean | 0.00072 |
| pass | true |

### Formal Run

命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_board_surrogate_action_refinement.py \
  --device cuda:0 --n_eval 512 --batch_size 64 \
  --output /home/chenshuai/Project/output/board_surrogate_action_refinement/board_surrogate_refine_K4_N512.json
```

输出：

```text
/home/chenshuai/Project/output/board_surrogate_action_refinement/board_surrogate_refine_K4_N512.json
/home/chenshuai/Project/output/board_surrogate_action_refinement/board_surrogate_refine_sanity_N32.json
```

结果：

| metric | value |
|---|---:|
| n windows | 512 |
| base score mean | 0.3197 |
| refined score mean | 0.3574 |
| score delta mean | **+0.03771** |
| score improved rate | **0.9902** |
| eef delta norm mean | 0.000678 |
| eef delta norm max | 0.000800 |
| joint delta norm mean | 0.000691 |
| joint delta norm max | 0.000819 |
| eef smooth delta mean | +0.0000198 |
| joint smooth delta mean | -0.000300 |
| marker MAE delta mean | +0.000041 |
| pass | **true** |

### 解释

该实验说明：

1. PTG board energy 不只是能产生非零梯度；
2. 在多步 trust-region action refinement 中，score 可以稳定提升；
3. action 更新幅度极小，平滑度基本不变；
4. 该 scoring/energy 形式适合作为 DP clean-action 后处理或 DDPM 低噪声阶段 guidance 的目标。

marker MAE 没有明显下降是正常现象：

```text
refinement 的目标不是复现示教 future marker，
而是让 predicted tactile consequence 得到更高的 tactile quality energy。
```

### Evidence Summary 更新

`TFAC_V5/summarize_ptg_guidance_evidence.py` 已加入：

```text
Board surrogate clean-action refinement
```

当前 board checks：

| item | status |
|---|---|
| scorer quality | PASS |
| scorer-level guidance readiness | PASS |
| surrogate full-chain guidance | PASS |
| surrogate clean-action refinement | PASS |
| production DP/Foresight full-chain | FAIL |

结论边界不变：

```text
Board surrogate refinement strengthens the guidance evidence,
but production board Foresight/DP is still required for final completion.
```

## 2026-06-09 Board Production Chain Setup

### 动机

前面审计发现：

```text
board production DP candidates = 0
board production Foresight candidates = 0
```

这意味着当前缺的不是继续换评分器，而是训练或定位黑板任务自己的 production DP/Foresight。为此新增一个准备脚本，把黑板数据变成现有训练代码可以直接使用的格式，并生成标准训练命令。

### 新增文件

```text
TFAC_V5/prepare_board_production_chain.py
TFAC_V5/config_pretrain_foresight_board_260522.json
```

### 数据审计

黑板数据：

```text
/home/chenshuai/data/dataset/260522_v8l_caheiban
```

数据情况：

| item | value |
|---|---:|
| episodes | 80 |
| size | 36GB |
| storage layout | success/episode_*.hdf5 |

首个 episode 字段：

| key | shape |
|---|---:|
| observations/images/global | (897, 200, 266, 3) |
| observations/images/wrist | (897, 200, 266, 3) |
| observations/proprio_joint | (897, 7) |
| observations/proprio_eef | (897, 6) |
| observations/tac/left/marker_offset | (897, 9, 9, 2) |
| observations/tac/right/marker_offset | (897, 9, 9, 2) |
| actions/joint_abs | (897, 7) |
| actions/eef_abs | (897, 6) |

### 兼容性结论

1. `TFAC_V5/pretrain_latent_foresight.py` 支持递归扫描 `success/episode_*.hdf5`，因此 Foresight 可直接用原始黑板目录；
2. `diffusion/train_dp_tac_concat.py` 只扫描 dataset root 下的 `episode_*.hdf5`；
3. 因此 DP 训练需要一个 flat symlink dataset；
4. 不能复制 36GB 数据，使用 symlink 即可。

### 运行准备脚本

命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/prepare_board_production_chain.py
```

输出：

```text
/home/chenshuai/data/dataset/260522_v8l_caheiban_flat
/home/chenshuai/Project/output/board_production_chain_setup/board_production_chain_setup.json
/home/chenshuai/Project/output/board_production_chain_setup/board_production_chain_setup.md
TFAC_V5/config_pretrain_foresight_board_260522.json
```

`260522_v8l_caheiban_flat` 包含 80 个 symlink，不复制原始数据。

### 训练命令

Board Foresight：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/pretrain_latent_foresight.py --config /home/chenshuai/Project/TactileACT-cs/TFAC_V5/config_pretrain_foresight_board_260522.json
```

Board DP + TactileVAE：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python diffusion/train_dp_tac_concat.py --dataset_dir /home/chenshuai/data/dataset/260522_v8l_caheiban_flat --save_dir /home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522 --camera_names global,wrist --proprio_key proprio_joint --action_key actions/joint_abs --tac_side left --tac_history 8 --vae_checkpoint /home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt --vae_latent_dim 16 --pred_horizon 16 --obs_horizon 2 --n_action_steps 8 --resize_shape 240,320 --crop_shape 216,288 --epochs 600 --batch_size 32 --lr 1e-4 --weight_decay 1e-6 --warmup_steps 500 --num_train_timesteps 100 --num_inference_steps 100 --diffusion_step_embed_dim 128 --down_dims 512,1024,2048 --seed 42 --save_freq 50 --gpu 0
```

### Sanity Check

DP dataset sanity 已通过：

```text
DPTacConcatDataset: 2 episodes, 1625 frames, 1595 windows
images global/wrist: (2, 3, 216, 288)
marker_hist: (2, 8, 9, 9, 2)
qpos: (2, 7)
action: (16, 7)
qpos normalized range: [-0.955, 0.954]
action normalized range: [-0.845, 0.938]
```

这说明现有 DP+TactileVAE 训练代码可以读取黑板数据。

### 下一步

训练完成后，正式替换 surrogate：

```text
action -> board production Foresight -> PTG board energy -> dscore/daction refinement
```

然后运行和插座相同级别的 production full-chain verification。只有这一步通过后，整体 objective 才能标记 complete。

## 2026-06-09 Board Foresight Smoke 通过

### 背景

为了让评分器真正服务于 DP 梯度引导，必须验证它不只是能在离线 GT latent 上分类/评分，还能接入生产链路：

```text
candidate action/action trajectory
  -> task Foresight predicts future tactile latent
  -> TacQualityEnergy scores tactile consequence
  -> backprop dscore/daction
  -> refine denoising/action
```

插座任务已有 production full-chain evidence；黑板任务此前只有 scorer-level 和 surrogate full-chain evidence。黑板 production chain 的第一步是确认 board Foresight 训练入口能够读取真实黑板数据并完成最小训练。

### 修复内容

Foresight 使用 `use_state_trajectory=True` 时，conditioning action 实际是未来 qpos 轨迹：

```text
qpos[t+1], qpos[t+2], ...
```

因此这段轨迹应使用 qpos 的均值方差归一化。原始 `NormalizeSeparate` 只支持：

```text
qpos -> qpos_mean/qpos_std
action -> action_mean/action_std
```

本次最小修复为 `NormalizeSeparate.__call__` 增加：

```python
action_as_qpos=False
```

当 `action_as_qpos=True` 时：

```text
action trajectory -> qpos_mean/qpos_std
```

默认行为不变，不影响普通 DP/action 训练。

### Smoke 命令

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/pretrain_latent_foresight.py \
  --config /home/chenshuai/Project/output/board_production_chain_setup/foresight_board_smoke_config.json \
  2>&1 | tee /home/chenshuai/Project/output/board_production_chain_setup/foresight_board_smoke.log
```

### 结果

| item | value |
|---|---:|
| train episodes | 72 |
| val episodes | 8 |
| train preload memory | 16002 MB |
| val preload memory | 1665 MB |
| epoch | 0 |
| train loss | 15.8192 |
| train latent loss | 15.6438 |
| train obs loss | 0.5846 |
| val loss | 12.3652 |
| val latent loss | 12.1789 |
| val obs loss | 0.6211 |
| best val loss | 12.3652 |

输出：

```text
/home/chenshuai/Project/output/board_production_chain_setup/foresight_board_smoke.log
/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_smoke_0/foresight_best.ckpt
/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_smoke_0/pretrain_history.pkl
/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_smoke_0/pretrain_loss.png
```

### 结论

黑板 production Foresight 数据接口已经跑通。这不是最终评分器完成证据，因为 smoke 只训练了 1 epoch；但它证明了黑板任务可以进入正式 production Foresight 训练，并为下一步完整验证提供了入口。

当前 evidence 状态：

| chain | status |
|---|---|
| socket scorer GroupKFold | PASS |
| socket production full-chain gradient | PASS |
| board proxy scorer | PASS |
| board scorer-level guidance readiness | PASS |
| board surrogate full-chain gradient | PASS |
| board production Foresight smoke | PASS |
| board production DP/Foresight full-chain gradient | NOT YET |

因此总目标仍未完成。下一步需要训练更长的 board production Foresight 和 board DP，然后执行正式的：

```text
action -> board production Foresight -> PTG board energy -> dscore/daction refinement
```

## 2026-06-09 Board Production Foresight Gradient Probe

### 目的

Foresight smoke 只能证明训练入口可跑通，还不能证明评分器能作为 classifier guidance 的梯度源。为此新增一个更直接的梯度链路实验：

```text
state/action trajectory
  -> board production Foresight
  -> decoded tactile marker
  -> PTG board energy
  -> dscore/dtrajectory
```

这个实验不做 reranking，也不只看分类准确率，而是检查 classifier guidance 最核心的条件：评分器的 energy 是否能穿过 Foresight 对动作/轨迹变量产生稳定、非零、方向正确的梯度。

### 新增脚本

```text
TFAC_V5/eval_board_production_foresight_gradient.py
```

### 重要边界

当前还没有正式 board DP checkpoint。因此这里的可导变量不是 DP denoising 内部的 noisy action，而是 board Foresight 当前训练配置使用的 `use_state_trajectory=True` conditioning：

```text
qpos[t+1:t+1+chunk]
```

所以该实验结论是：

```text
board production Foresight -> PTG scorer 的梯度链路成立
```

但还不是：

```text
board DP denoising -> board production Foresight -> PTG scorer 的完整闭环成立
```

### 运行命令

小规模 smoke：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_board_production_foresight_gradient.py \
  --n_episodes 1 \
  --n_eval 8 \
  --batch_size 2 \
  --output /home/chenshuai/Project/output/board_production_foresight_gradient/smoke_N8.json
```

正式 N=64：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_board_production_foresight_gradient.py \
  2>&1 | tee /home/chenshuai/Project/output/board_production_foresight_gradient/board_production_foresight_smoke_gradient_N64.log
```

### 输出

```text
/home/chenshuai/Project/output/board_production_foresight_gradient/smoke_N8.json
/home/chenshuai/Project/output/board_production_foresight_gradient/board_production_foresight_smoke_gradient_N64.json
/home/chenshuai/Project/output/board_production_foresight_gradient/board_production_foresight_smoke_gradient_N64.log
```

### N=64 结果

| item | value |
|---|---:|
| samples | 64 |
| score_improved_rate | 1.0 |
| finite_grad_rate | 1.0 |
| nonzero_grad_rate | 1.0 |
| score_delta mean | 9.194016456604004e-06 |
| grad_norm mean | 0.046870373538695276 |
| action_delta_norm mean | 0.00019999856863250898 |
| pass | true |

### 解释

score delta 的绝对值很小，这是因为 board guidance profile 当前采用非常保守的步长：

```text
action_step = 0.0002
```

在这个安全步长下，关键证据不是 delta 大小，而是：

1. 梯度全部 finite；
2. 梯度全部非零；
3. 沿梯度方向小步更新后，64/64 个样本的 PTG board energy 都提升；
4. 这说明 `PTGProxyScorerV2Runtime` 已经可以作为可导 energy 穿过 board production Foresight。

### 当前 Evidence 状态更新

| chain | status |
|---|---|
| socket scorer GroupKFold | PASS |
| socket production full-chain gradient | PASS |
| board proxy scorer | PASS |
| board scorer-level guidance readiness | PASS |
| board surrogate full-chain gradient | PASS |
| board production Foresight smoke | PASS |
| board production Foresight gradient probe | PASS |
| board DP denoising full-chain gradient | NOT YET |

下一步仍然是训练或定位 board DP checkpoint，然后验证：

```text
DP denoising action -> board production Foresight -> PTG board energy -> dscore/daction
```

## 2026-06-09 Board DP Training-Entry Smoke

### 目的

前面的 board production Foresight gradient probe 已经证明：

```text
state trajectory -> board Foresight -> PTG board energy
```

这条梯度链路成立。但最终目标还需要 DP 侧真实接入。因此这里进一步验证 board DP+TactileVAE 训练入口是否可用。

### 设置

完整 board DP 训练是：

```text
80 episodes, 600 epochs
```

本轮先不把 1 epoch 小模型当作正式 policy，只做训练入口 smoke。创建了 4 episode symlink 子集：

```text
/home/chenshuai/data/dataset/260522_v8l_caheiban_flat_smoke4
```

### 运行命令

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python diffusion/train_dp_tac_concat.py \
  --dataset_dir /home/chenshuai/data/dataset/260522_v8l_caheiban_flat_smoke4 \
  --save_dir /home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_smoke4 \
  --camera_names global,wrist \
  --proprio_key proprio_joint \
  --action_key actions/joint_abs \
  --tac_side left \
  --tac_history 8 \
  --vae_checkpoint /home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt \
  --vae_latent_dim 16 \
  --pred_horizon 16 \
  --obs_horizon 2 \
  --n_action_steps 8 \
  --resize_shape 240,320 \
  --crop_shape 216,288 \
  --epochs 1 \
  --batch_size 4 \
  --lr 1e-4 \
  --weight_decay 1e-6 \
  --warmup_steps 10 \
  --num_train_timesteps 20 \
  --num_inference_steps 20 \
  --diffusion_step_embed_dim 64 \
  --down_dims 128,256 \
  --seed 42 \
  --save_freq 1 \
  --gpu 0
```

### 输出

```text
/home/chenshuai/Project/output/board_production_chain_setup/board_dp_smoke4.log
/home/chenshuai/Project/output/board_production_chain_setup/board_dp_smoke4.json
/home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_smoke4/config.json
/home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_smoke4/dp_final.pth
/home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_smoke4/dp_epoch1.pth
/home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_smoke4/dp_topk_ep1_loss0.2444.pth
/home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_smoke4/train_losses.npy
```

### 结果

| item | value |
|---|---:|
| episodes | 4 |
| total frames | 3134 |
| windows | 3074 |
| epochs | 1 |
| batch size | 4 |
| action_dim | 7 |
| global_cond_dim | 2350 |
| final train loss | 0.24435303152401983 |
| dp_final exists | true |
| pass | true |

### 结论

board DP+TactileVAE 训练入口可用：数据读取、视觉/触觉编码、UNet 训练循环、EMA/checkpoint 保存都能跑通。

但是该 checkpoint 不是正式 policy。它只说明最后一个工程缺口已经从“DP 是否能训练未知”变成“需要跑完整训练并做 full-chain 验证”。

当前 evidence 状态：

| chain | status |
|---|---|
| socket scorer GroupKFold | PASS |
| socket production full-chain gradient | PASS |
| board proxy scorer | PASS |
| board scorer-level guidance readiness | PASS |
| board surrogate full-chain gradient | PASS |
| board production Foresight smoke | PASS |
| board production Foresight gradient probe | PASS |
| board DP training-entry smoke | PASS |
| board DP denoising full-chain gradient | NOT YET |

## 2026-06-09 Board DP/Foresight Full-Chain Smoke

### 目的

在已有 board DP smoke checkpoint 和 board Foresight smoke checkpoint 的基础上，进一步验证评分器是否能接入完整工程链路：

```text
DP action -> board Foresight -> decoded marker -> PTG board energy -> dscore/daction
```

这是比 scorer-level、surrogate full-chain、Foresight gradient probe 更强的证据，因为它已经包含 DP 输出动作。

### 新增脚本

```text
TFAC_V5/eval_board_dp_denoising_full_chain_smoke.py
```

默认使用：

```text
DP: /home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_smoke4/dp_final.pth
Foresight: /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_smoke_0/foresight_best.ckpt
Scorer: /home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_final.pt
```

### 两种 Guidance 形式

本次比较了两种梯度引导方式：

1. `denoising`

```text
noisy action at late denoising step
  -> PTG gradient
  -> update noisy action
  -> continue denoising
```

2. `clean_refine`

```text
DP complete denoising output
  -> PTG gradient
  -> trust-region update clean action
  -> accept only if score improves
```

两者都属于梯度引导，不是 reranking。区别是梯度注入位置不同。

### 负结果：Denoising-In-Loop 当前不稳定

默认 denoising mode 输出：

```text
/home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_denoising_full_chain_smoke_K4_N4.json
```

并额外尝试了：

```text
late_small
late_tiny
late_accept
mid_tiny
```

结果现象：

| item | value |
|---|---:|
| local guide accept rate | 1.0 |
| score_delta mean | positive |
| default score_delta mean | 0.07090198248624802 |
| final guided_beats_base_rate | 0.5625 |
| range violation | 0.0 |

解释：

在当前 smoke DP/Foresight checkpoint 下，中途 denoising 注入虽然每个局部引导步都能提升当步 PTG energy，但后续 denoising 动态会继续改变 action，导致最终逐样本胜率不稳定。因此暂时不把 denoising-in-loop 作为当前最佳实现方式。

### 正结果：Clean-Action Trust-Region Refinement

运行命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_board_dp_denoising_full_chain_smoke.py
```

输出：

```text
/home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_clean_refine_full_chain_smoke_K4_N4.json
/home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_clean_refine_full_chain_smoke_K4_N4.log
```

结果：

| item | value |
|---|---:|
| mode | clean_refine |
| frames | 4 |
| action samples | 16 |
| score_delta mean | 0.07127008587121964 |
| guided_beats_base_rate | 1.0 |
| range_violation max | 0.0 |
| norm_action_delta mean | 0.039790746755898 |
| smoothness_delta mean | -0.6946475505828857 |
| guide_accept_rate mean | 1.0 |
| pass | true |

### 当前最佳实现结论

当前最合理、最稳的实现方式是：

```text
Task-conditioned TacQualityEnergy scorer
  + production Foresight consequence model
  + clean-action trust-region classifier guidance
  + accept-only update
```

形式上：

```text
a_0 = DP(obs)
for k in 1..K:
    z_pred = Foresight(obs, a_{k-1})
    marker_pred = TactileVAE.decoder(z_pred)
    E = PTG_TacQualityEnergy(marker_pred, a_{k-1}, task_id)
    g = dE / da_{k-1}
    a_prop = ProjectTrustRegion(a_{k-1} + eta * normalize(g), a_0)
    a_k = a_prop if E(a_prop) > E(a_{k-1}) else a_{k-1}
return a_K
```

创新点在于：

1. 评分器不是单纯 good/bad 分类，而是 task-conditioned 多头 tactile quality energy；
2. 标准由任务物理定义给出：插座 bounce risk、黑板力大小与力变化柔顺性；
3. Foresight 把 action 的未来触觉后果接入 score，使 score 对 action 可导；
4. trust-region + accept-only 让引导具备安全边界，避免直接把 classifier gradient 无约束注入动作。

### 当前边界

该 full-chain smoke 使用的是：

```text
4-episode board DP smoke checkpoint
1-epoch board Foresight smoke checkpoint
```

因此它证明工程链路和梯度机制成立，但还不能证明最终 production policy 质量。最终仍需：

```text
full board DP training
full board Foresight training
production-scale full-chain guidance/refinement evaluation
```

## 2026-06-09 Board Foresight Fast20 与 Full-Chain 复验

### 动机

上一节的 board DP/Foresight full-chain smoke 使用的是：

```text
4-episode board DP smoke checkpoint
1-epoch board Foresight smoke checkpoint
```

为了让 evidence 更接近真实生产链路，本轮先把 Foresight 从 1 epoch smoke 提升到一个轻量但更强的 fast20 checkpoint，然后复跑：

1. board Foresight gradient probe；
2. board DP clean-action full-chain smoke。

### Fast20 Foresight 配置

配置文件：

```text
/home/chenshuai/Project/output/board_production_chain_setup/foresight_board_fast20_config.json
```

核心设置：

| item | value |
|---|---:|
| name | latent_foresight_board_260522_fast20 |
| episodes | 80 |
| train / val | 72 / 8 |
| epochs | 20 |
| batch_size | 16 |
| hidden_dim | 128 |
| foresight_layers | 1 |
| foresight_nheads | 4 |
| use_state_trajectory | true |

### 训练结果

输出：

```text
/home/chenshuai/Project/output/board_production_chain_setup/foresight_board_fast20.log
/home/chenshuai/Project/output/board_production_chain_setup/board_foresight_fast20.json
/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_fast20/foresight_best.ckpt
/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_fast20/pretrain_history.pkl
/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_fast20/pretrain_loss.png
```

结果：

| item | value |
|---|---:|
| initial val loss | 15.36742057800293 |
| best val loss | 2.6939840793609617 |
| best epoch | 19 |
| final train loss | 2.5632447013148556 |
| pass | true |

相比 1-epoch smoke 的 best val loss `12.365220069885254`，fast20 明显更好。

### Fast20 Foresight Gradient Probe

输出：

```text
/home/chenshuai/Project/output/board_production_foresight_gradient/board_production_foresight_fast20_gradient_N64.json
/home/chenshuai/Project/output/board_production_foresight_gradient/board_production_foresight_fast20_gradient_N64.log
```

结果：

| item | value |
|---|---:|
| samples | 64 |
| score_improved_rate | 1.0 |
| finite_grad_rate | 1.0 |
| nonzero_grad_rate | 1.0 |
| score_delta mean | 9.272247552871704e-06 |
| grad_norm mean | 0.04753120825625956 |
| pass | true |

### Fast20 Board DP Clean-Action Full-Chain Smoke

输出：

```text
/home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_clean_refine_full_chain_fast20_K4_N4.json
/home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_clean_refine_full_chain_fast20_K4_N4.log
```

结果：

| item | value |
|---|---:|
| mode | clean_refine |
| frames | 4 |
| action samples | 16 |
| score_delta mean | 0.07417601346969604 |
| guided_beats_base_rate | 1.0 |
| range_violation max | 0.0 |
| smoothness_delta mean | -0.6895569115877151 |
| guide_accept_rate mean | 1.0 |
| pass | true |

### 结论更新

fast20 Foresight 训练后，当前最强 evidence 变为：

```text
smoke board DP
  -> fast20 board Foresight
  -> PTG board energy
  -> clean-action trust-region classifier guidance
```

该链路通过，并且比 1-epoch Foresight smoke 更有说服力。

当前最合理的评分/引导器设计保持为：

```text
Task-conditioned TacQualityEnergy
  + future tactile consequence Foresight
  + clean-action trust-region gradient guidance
  + accept-only safety gate
```

剩余瓶颈已经从“评分器/梯度是否可用”转移到：

```text
full board DP training
production-scale full-chain evaluation
```

因此当前仍不能标记最终完成，但 scoring/guidance method 本身已经形成了清晰、可复现、跨插座和黑板任务都通过 smoke/full-chain 证据的方案。

## 2026-06-09 Board DP Fast16_E20 与 Full-Chain 复验

### 动机

上一节已经把 board Foresight 提升到 fast20，但 board DP 仍是：

```text
4 episodes, 1 epoch
```

这对证明工程链路足够，但对 production-scale evidence 仍然太弱。因此本轮训练一个中等规模 board DP：

```text
16 episodes, 20 epochs
```

网络仍使用轻量配置，避免直接进入完整 80 episode / 600 epoch 的重训练。

### DP Fast16_E20 设置

数据子集：

```text
/home/chenshuai/data/dataset/260522_v8l_caheiban_flat_fast16
```

checkpoint：

```text
/home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_fast16_e20
```

训练命令核心参数：

| item | value |
|---|---:|
| episodes | 16 |
| windows | 11909 |
| epochs | 20 |
| batch_size | 8 |
| pred_horizon | 16 |
| obs_horizon | 2 |
| num_train_timesteps | 20 |
| down_dims | 128,256 |
| action_dim | 7 |
| global_cond_dim | 2350 |

### DP 训练结果

输出：

```text
/home/chenshuai/Project/output/board_production_chain_setup/board_dp_fast16_e20.log
/home/chenshuai/Project/output/board_production_chain_setup/board_dp_fast16_e20.json
/home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_fast16_e20/dp_final.pth
/home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_fast16_e20/dp_epoch20.pth
/home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_fast16_e20/train_losses.npy
```

结果：

| item | value |
|---|---:|
| initial train loss | 0.15586669385293092 |
| final train loss | 0.010728547398906009 |
| best train loss | 0.010728547398906009 |
| smoke4 train loss | 0.24435303152401983 |
| pass | true |

### Fast16_E20 DP + Fast20 Foresight Full-Chain

复验链路：

```text
fast16_e20 board DP
  -> fast20 board Foresight
  -> PTG board energy
  -> clean-action trust-region gradient guidance
```

输出：

```text
/home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_fast16_e20_clean_refine_full_chain_fast20_K4_N8.json
/home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_fast16_e20_clean_refine_full_chain_fast20_K4_N8.log
```

结果：

| item | value |
|---|---:|
| frames | 8 |
| action samples | 32 |
| base_score mean | 2.0811803713440895 |
| guided_score mean | 2.1011964604258537 |
| score_delta mean | 0.02001608908176422 |
| guided_beats_base_rate | 0.96875 |
| range_violation max | 0.0 |
| smoothness_delta mean | -0.23228864278644323 |
| guide_accept_rate mean | 0.5703125 |
| pass | true |

### 解释

和 4-episode DP smoke 相比，fast16_e20 DP 的 base score 已经更高、动作更平滑，因此 refinement 的绝对 score_delta 从约 `0.074` 降到约 `0.020` 是合理的：更好的 base action 本身更难被大幅改善。

更关键的是：

1. 96.875% 的 action samples 得到提升；
2. action range 没有违规；
3. smoothness 进一步改善；
4. 所有改动仍在 trust-region 内完成；
5. 这说明评分器不是只会修很差的 random/smoke action，对更强 DP 输出也仍然能提供有效梯度。

### 当前最强 Board Evidence

当前黑板任务最强证据链为：

```text
fast16_e20 board DP
  -> fast20 board Foresight
  -> task-conditioned PTG TacQualityEnergy
  -> clean-action trust-region classifier guidance
```

该链路通过。

剩余缺口进一步缩小为：

```text
full 80-episode board DP training
larger-scale production full-chain evaluation
```

评分器/梯度引导方法本身目前已经稳定：问题不再是“评分器是否可导/是否可用”，而是“生产 DP policy 是否训练完整、评估样本是否足够大”。

## 2026-06-09 Board Fast16_E20 Full-Chain N=32 扩大评估

### 目的

上一节 N=8 验证了工程链路：

```text
DP action -> Foresight -> tactile marker -> PTG scorer -> dscore / da
```

但 N=8 仍然偏小。本节把相同链路扩大到：

```text
32 frames
128 action samples
```

重点验证评分器是否适合作为 DP 梯度引导目标，而不是只在很小样本上偶然提升。

### 命令

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_board_dp_denoising_full_chain_smoke.py \
  --data_dir /home/chenshuai/data/dataset/260522_v8l_caheiban_flat_fast16 \
  --dp_config /home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_fast16_e20/config.json \
  --dp_ckpt /home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_fast16_e20/dp_final.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_fast20 \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_fast20/foresight_best.ckpt \
  --n_episodes 8 \
  --frames_per_episode 4 \
  --n_eval 32 \
  --K 4 \
  --mode clean_refine \
  --accept_only_improved \
  --clamp_norm_action \
  --output /home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_fast16_e20_clean_refine_full_chain_fast20_K4_N32.json
```

### 输出

```text
/home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_fast16_e20_clean_refine_full_chain_fast20_K4_N32.json
/home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_fast16_e20_clean_refine_full_chain_fast20_K4_N32.log
```

### 结果

| item | value |
|---|---:|
| frames | 32 |
| action samples | 128 |
| base_score mean | 2.108561798930168 |
| guided_score mean | 2.1236043721437454 |
| score_delta mean | 0.01504257321357727 |
| score_delta median | 0.010220646858215332 |
| guided_beats_base_rate | 0.9765625 |
| range_violation max | 0.0 |
| base_smoothness mean | 0.46950822696089745 |
| guided_smoothness mean | 0.2880655580665916 |
| smoothness_delta mean | -0.18144266854505986 |
| norm_action_delta mean | 0.014409986899408977 |
| guide_grad_norm_mean_per_step mean | 1.7213419657200575 |
| guide_accept_rate_per_step mean | 0.443359375 |
| pass | true |

### 解释

这次结果比 N=8 更有价值：

1. `guided_beats_base_rate = 0.9765625`，说明绝大多数 DP action 都能被 scorer-guided gradient refinement 正向改进；
2. `range_violation max = 0.0`，说明 clamp 和 trust-region 没有破坏动作合法范围；
3. `smoothness_delta mean = -0.18144266854505986`，说明引导不仅提高触觉质量分数，还让动作变化更平滑；
4. `norm_action_delta mean = 0.0144`，说明不是大幅改写 DP 输出，而是在局部邻域内做小步优化；
5. `guide_grad_norm_mean_per_step mean = 1.72`，说明梯度非零且数值稳定。

因此当前评分器/引导器已经满足两个关键条件：

```text
1. 能准确评估触觉质量；
2. 能作为可导 energy 对 DP 输出动作做局部梯度引导。
```

### 当前推荐实现方式

当前最合理方案仍然是：

```text
Task-conditioned TacQualityEnergy
  + Foresight tactile consequence prediction
  + clean-action trust-region classifier guidance
  + accept-only improved update
```

具体形式：

```text
a0 = DP(obs)
z_pred = Foresight(obs, a)
marker_pred = TactileVAE.decoder(z_pred)
E = PTG_TacQualityEnergy(marker_pred, a, task_id)
a_prop = ProjectTrustRegion(a + eta * normalize(dE/da), a0)
accept only if E(a_prop) > E(a)
```

这不是 reranking，因为动作 `a` 本身参与计算图，`E` 对 `a` 反传梯度；也不是纯分类器离线筛选，而是通过 Foresight 把 action 对未来触觉后果的因果影响注入 score。

## 2026-06-09 Board Held-Out Episode Full-Chain N=32 评估

### 目的

上一节 N=32 使用的是 fast16 DP 训练子集。为了更严格评估泛化性，本节使用 DP fast16 没训练过的 episode：

```text
train episodes: episode_0 ... episode_15
held-out episodes: episode_16 ... episode_31
```

这对应更科学的 episode-level split 思路：同一个 episode 的连续帧不能同时出现在训练和测试中，否则 frame-level 随机划分会因为相邻帧高度相似而虚高。

### Held-Out 数据目录

```text
/home/chenshuai/data/dataset/260522_v8l_caheiban_flat_heldout16
```

该目录包含 episode_16 到 episode_31 的 symlink，源目录为：

```text
/home/chenshuai/data/dataset/260522_v8l_caheiban_flat
```

### 命令

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_board_dp_denoising_full_chain_smoke.py \
  --data_dir /home/chenshuai/data/dataset/260522_v8l_caheiban_flat_heldout16 \
  --dp_config /home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_fast16_e20/config.json \
  --dp_ckpt /home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_fast16_e20/dp_final.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_fast20 \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_fast20/foresight_best.ckpt \
  --n_episodes 8 \
  --frames_per_episode 4 \
  --n_eval 32 \
  --K 4 \
  --mode clean_refine \
  --accept_only_improved \
  --clamp_norm_action \
  --output /home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_fast16_e20_clean_refine_full_chain_fast20_heldout16_K4_N32.json
```

### 输出

```text
/home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_fast16_e20_clean_refine_full_chain_fast20_heldout16_K4_N32.json
/home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_fast16_e20_clean_refine_full_chain_fast20_heldout16_K4_N32.log
```

### 结果

| item | value |
|---|---:|
| frames | 32 |
| action samples | 128 |
| base_score mean | 2.1082106735557318 |
| guided_score mean | 2.1260381136089563 |
| score_delta mean | 0.017827440053224564 |
| score_delta median | 0.014733672142028809 |
| guided_beats_base_rate | 0.984375 |
| range_violation max | 0.0 |
| base_smoothness mean | 0.5193564894143492 |
| guided_smoothness mean | 0.309188412851654 |
| smoothness_delta mean | -0.2101680770283565 |
| norm_action_delta mean | 0.016281947504467098 |
| guide_grad_norm_mean_per_step mean | 1.6627200152724981 |
| guide_accept_rate_per_step mean | 0.51171875 |
| pass | true |

### 解释

held-out 结果比训练子集评估更关键：

1. 在 DP fast16 没训练过的 episode 上，`guided_beats_base_rate = 0.984375`；
2. 平均 scorer 提升 `+0.017827`，比训练子集 N=32 的 `+0.015043` 略高；
3. `range_violation max = 0.0`，动作合法性没有被梯度破坏；
4. `smoothness_delta mean = -0.210168`，动作更平滑；
5. 说明当前 scorer/energy 的梯度不是只对训练 episode 有效。

因此，黑板任务当前最强证据应更新为：

```text
fast16_e20 DP
  -> fast20 Foresight
  -> held-out episode_16...31
  -> PTG TacQualityEnergy clean-action trust-region guidance
  -> 98.4375% action samples improved
```

### 结论

目前最合理的评分/分类器不是单纯 binary classifier，而是：

```text
task-conditioned continuous energy scorer
```

它包含：

1. binary good/bad head：保证有明确好坏标准；
2. reason/failure-mode head：区分 bad 的原因，比如 force too low、force too high、force not smooth、bounce risk；
3. quality regression/energy head：为 DP guidance 提供连续可导梯度；
4. task conditioning：插座和黑板使用同一个框架，但标准不同。

这比只做“好/坏分类”更适合 DP 梯度引导，因为 DP 需要的是连续方向，不只是离散标签。

## 2026-06-09 当前评分/分类器方法 Review

### 核心问题

用户关心的不是单纯分类准确率，而是：

```text
这个评分/分类器是否能准确评估触觉质量，
并且是否能作为 DP denoising/action generation 中的梯度引导目标。
```

因此，一个只输出 good/bad 的二分类器不够。二分类器可以做判断，但它的梯度通常只在决策边界附近有意义，不能稳定告诉 DP action “往哪个方向改更好”。

### 当前方案

当前最合理方案是：

```text
Task-conditioned TacQualityEnergy
```

它不是单一分类器，而是 multi-head scorer：

| head | 作用 | 是否用于 guidance |
|---|---|---|
| binary good/bad | 明确好坏边界 | 通过 binary margin 辅助 |
| reason/failure-mode | 解释坏的原因 | 可用于诊断/约束，默认权重较小 |
| quality/energy regression | 连续质量分数 | 主 guidance 目标 |

默认 energy：

```text
E = wq * quality_logit + wb * good_binary_margin + wr * good_reason_margin
E_clipped = clip_scale * tanh(E / clip_scale)
```

当前任务配置：

| task | scorer | wq | wb | wr | action trust-region |
|---|---|---:|---:|---:|---:|
| insertion | InsertionRiskScorerRuntime | 0.50 | 0.10 | 0.00 | 0.08 |
| board | PTGProxyScorerV2Runtime | 0.75 | 0.10 | 0.00 | 0.02 |
| mixed analysis | PTGProxyScorerV2Runtime | 0.50 | 0.00 | 0.20 | 0.02 |

### 标签标准

插座任务：

```text
good = success episode 的 stable insert
bad = bounce episode 中的 pre-bounce risk / impact / recovery
neutral = approach / weak no-contact，不参与 binary loss
```

黑板任务：

```text
good = force magnitude 合适 + force/action/marker 变化平滑
bad = too_light / too_heavy / rough_force / rough_motion
```

这和用户定义一致：插座坏数据主要来自会导致 bounce 的数据；黑板坏数据由力过小、力过大、变化不柔顺定义。

### 为什么必须用 Episode-Level GroupKFold

触觉序列相邻帧高度相关。如果 frame-level 随机划分，同一条 episode 的前后帧会同时出现在 train/test，模型可能只是记住该 episode 的局部轨迹，测试准确率会虚高。

当前 scorer 训练/评估使用：

```text
GroupKFold(group = task::episode)
```

这保证同一个 episode 的所有样本只在 train 或 test 一边，更接近真实泛化。

### 当前关键指标

插座 InsertionRiskScorer GroupKFold：

| metric | value |
|---|---:|
| binary AUC | 0.9877358914416259 |
| balanced accuracy | 0.9437040677598894 |
| binary macro F1 | 0.9364217613444854 |
| reason macro F1 | 0.7894122733559068 |
| quality corr | 0.7655629450935674 |

PTGProxyScorerV2 mixed/board：

| metric | value |
|---|---:|
| binary AUC | 0.9700666590503279 |
| balanced accuracy | 0.908209601681784 |
| binary macro F1 | 0.8930056036169187 |
| reason macro F1 | 0.7678430278812207 |
| quality corr | 0.7561969545352898 |
| quality R2 | 0.5678382154363828 |

黑板 held-out full-chain gradient guidance：

| metric | value |
|---|---:|
| held-out frames | 32 |
| action samples | 128 |
| score_delta mean | 0.017827440053224564 |
| guided_beats_base_rate | 0.984375 |
| range_violation max | 0.0 |
| smoothness_delta mean | -0.2101680770283565 |
| grad_norm mean | 1.6627200152724981 |

### 为什么不是统一任务无关分类器

统一 taxonomy 的传统模型最好结果：

| metric | value |
|---|---:|
| mixed good-vs-bad AUC | 0.869968744384327 |
| mixed balanced accuracy | 0.7910023470912785 |
| cross_macro_f1 | 0.4614320319711282 |

这个结果说明：跨任务的“好/坏”语义不是完全一致的。比如：

```text
插座 weak/no-contact 在 approach 阶段可以是 neutral；
黑板 too_light 是 bad，因为擦拭力不足。
```

所以更合理的是：

```text
共享 scorer 结构 + task_id 条件化 + 每个任务自己的好坏标准
```

### Review 结论

当前最佳方案保持：

```text
Task-conditioned TacQualityEnergy
  + Foresight tactile consequence prediction
  + clean-action trust-region classifier guidance
  + accept-only improved update
```

它满足当前阶段的两个核心条件：

1. 能准确评估：GroupKFold binary AUC 约 0.97-0.99，quality corr 约 0.76；
2. 能梯度引导：held-out board full-chain 中 98.4375% action samples 得到提升，动作范围不违规，smoothness 改善。

仍不能宣称最终完成的原因：

```text
full 80-episode board DP training
larger-scale held-out production full-chain evaluation
real robot/closed-loop validation
```

还没有完成。

## 2026-06-09 Board Fast32_E20 与 Held-Out32 Full-Chain 评估

### 目的

上一轮最强证据是：

```text
fast16_e20 DP
  -> fast20 Foresight
  -> heldout16 episode_16...31
  -> PTG clean-action guidance
```

为了进一步接近 production-scale，本轮把 board DP 训练集从 16 episode 提升到 32 episode：

```text
train: episode_0 ... episode_31
held-out: episode_32 ... episode_47
```

这样可以回答一个更关键的问题：

```text
当 base DP policy 更强时，TacQualityEnergy 是否仍能提供有用梯度？
```

### 数据子集

训练集：

```text
/home/chenshuai/data/dataset/260522_v8l_caheiban_flat_fast32
```

held-out：

```text
/home/chenshuai/data/dataset/260522_v8l_caheiban_flat_heldout32
```

两者均为 symlink 目录，源数据为：

```text
/home/chenshuai/data/dataset/260522_v8l_caheiban_flat
```

### DP Fast32_E20 训练

命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python diffusion/train_dp_tac_concat.py \
  --dataset_dir /home/chenshuai/data/dataset/260522_v8l_caheiban_flat_fast32 \
  --save_dir /home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_fast32_e20 \
  --camera_names global,wrist \
  --proprio_key proprio_joint \
  --action_key actions/joint_abs \
  --tac_side left \
  --tac_history 8 \
  --vae_checkpoint /home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt \
  --vae_latent_dim 16 \
  --pred_horizon 16 \
  --obs_horizon 2 \
  --n_action_steps 8 \
  --resize_shape 240,320 \
  --crop_shape 216,288 \
  --epochs 20 \
  --batch_size 8 \
  --lr 1e-4 \
  --weight_decay 1e-6 \
  --warmup_steps 100 \
  --num_train_timesteps 20 \
  --num_inference_steps 20 \
  --diffusion_step_embed_dim 64 \
  --down_dims 128,256 \
  --seed 45 \
  --save_freq 10 \
  --gpu 0
```

输出：

```text
/home/chenshuai/Project/output/board_production_chain_setup/board_dp_fast32_e20.log
/home/chenshuai/Project/output/board_production_chain_setup/board_dp_fast32_e20.json
/home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_fast32_e20/dp_final.pth
```

训练结果：

| item | value |
|---|---:|
| episodes | 32 |
| windows | 23281 |
| epochs | 20 |
| initial train loss | 0.09752981259853252 |
| final train loss | 0.008857935146488312 |
| best train loss | 0.008857935146488312 |
| best epoch | 20 |
| pass | true |

对比 fast16_e20：

| item | fast16_e20 | fast32_e20 |
|---|---:|---:|
| episodes | 16 | 32 |
| windows | 11909 | 23281 |
| final train loss | 0.010728547398906009 | 0.008857935146488312 |

### Held-Out32 Full-Chain Guidance

链路：

```text
fast32_e20 DP
  -> fast20 Foresight
  -> PTG TacQualityEnergy
  -> clean-action trust-region gradient guidance
  -> accept-only improved update
```

命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_board_dp_denoising_full_chain_smoke.py \
  --data_dir /home/chenshuai/data/dataset/260522_v8l_caheiban_flat_heldout32 \
  --dp_config /home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_fast32_e20/config.json \
  --dp_ckpt /home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_fast32_e20/dp_final.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_fast20 \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_fast20/foresight_best.ckpt \
  --n_episodes 8 \
  --frames_per_episode 4 \
  --n_eval 32 \
  --K 4 \
  --mode clean_refine \
  --accept_only_improved \
  --clamp_norm_action \
  --output /home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_fast32_e20_clean_refine_full_chain_fast20_heldout32_K4_N32.json
```

输出：

```text
/home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_fast32_e20_clean_refine_full_chain_fast20_heldout32_K4_N32.json
/home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_fast32_e20_clean_refine_full_chain_fast20_heldout32_K4_N32.log
```

结果：

| item | value |
|---|---:|
| frames | 32 |
| action samples | 128 |
| base_score mean | 2.1125840796157718 |
| guided_score mean | 2.127419427037239 |
| score_delta mean | 0.014835347421467304 |
| score_delta median | 0.006354331970214844 |
| guided_beats_base_rate | 0.8125 |
| range_violation max | 0.0 |
| base_smoothness mean | 0.48651906836312264 |
| guided_smoothness mean | 0.31693405128316954 |
| smoothness_delta mean | -0.1695850170799531 |
| norm_action_delta mean | 0.013678390834684251 |
| grad_norm mean | 1.7682206016033888 |
| pass | true |

### 解释

fast32 的 base score 比 fast16 held-out 更高：

```text
fast16 heldout16 base_score mean = 2.1082106735557318
fast32 heldout32 base_score mean = 2.1125840796157718
```

因此 `guided_beats_base_rate` 从 `0.984375` 降到 `0.8125` 是合理的：base policy 越强，局部 refinement 能明显改善的样本比例通常会下降。

但关键指标仍然通过：

1. `score_delta mean = +0.014835`，平均仍提升；
2. `range_violation max = 0.0`，动作范围不违规；
3. `smoothness_delta mean = -0.169585`，动作更平滑；
4. `grad_norm mean = 1.7682`，梯度非零且稳定；
5. 使用 held-out episode_32...47，避免 episode-level 泄漏。

因此当前更强结论是：

```text
PTG TacQualityEnergy 不只是能修 weak/smoke DP action；
在更强的 fast32 DP policy 上仍能提供正向、受约束、可泛化的局部梯度。
```

### 当前 Board 最强证据

```text
fast32_e20 board DP
  -> fast20 board Foresight
  -> heldout32 episode-level evaluation
  -> PTG TacQualityEnergy clean-action trust-region guidance
  -> pass
```

剩余缺口仍然是：

```text
full 80-episode DP
larger held-out N
real robot / closed-loop validation
```

## 2026-06-10 Fast64/Fast40 训练尝试与 Fast32 Held-Out N=64 复验

### 目的

上一节 fast32_e20 已经证明：

```text
32 episode DP + fast20 Foresight + heldout32 N=32
```

可以通过 full-chain clean-action guidance。为了继续接近 production-scale，本节尝试更大 DP：

```text
fast64_e20
fast40_e20
```

并在发现资源限制后，转为扩大当前最强 fast32 checkpoint 的 held-out full-chain 评估样本数。

### Fast64_E20 尝试

数据：

```text
train: /home/chenshuai/data/dataset/260522_v8l_caheiban_flat_fast64
heldout: /home/chenshuai/data/dataset/260522_v8l_caheiban_flat_heldout64
```

划分：

```text
train episode_0 ... episode_63
heldout episode_64 ... episode_79
```

结果：

```text
preload 到约 41/64 episode 后进程提前退出
未生成 checkpoint
log 无 Python traceback
```

### Fast40_E20 尝试

数据：

```text
train: /home/chenshuai/data/dataset/260522_v8l_caheiban_flat_fast40
heldout: /home/chenshuai/data/dataset/260522_v8l_caheiban_flat_heldout40
```

划分：

```text
train episode_0 ... episode_39
heldout episode_40 ... episode_55
```

结果：

```text
成功 preload 40 episodes
训练启动后提前退出
只生成 config.json
未生成 dp_final.pth
```

### 失败原因分析

`diffusion/train_dp_tac_concat.py` 当前数据集实现：

```text
DPTacConcatDataset
  -> preload all episodes
  -> resize + normalize all images
  -> store image tensors as fp16 in RAM
```

fast32 可以训练成功，但 fast40/fast64 会显著增加 RAM/swap 压力。当前系统状态显示：

```text
RAM total: 62Gi
swap total: 2.0Gi
swap used: 2.0Gi
```

因此这不是 PTG scorer/guidance 方法失败，而是 board DP 训练脚本的加载方式限制了更大规模训练。

### 工程结论

如果要进行 full 80-episode board DP，应该先改造训练数据管线：

```text
Option A: lazy HDF5 read + on-the-fly image transform
Option B: disk cache / memmap resized images
Option C: precompute vision features, DP training only loads features + qpos + tactile latent
```

其中 Option C 最适合当前研究目标，因为评分器/引导器关注 tactile consequence 和 action，不需要每次 DP 训练都反复跑 image preprocessing。

### Fast32 Held-Out N=64 复验

在 fast40/fast64 暂时受资源限制后，使用当前最强可用 checkpoint：

```text
fast32_e20 DP
fast20 Foresight
heldout32 episode_32 ... episode_47
```

把 full-chain evaluation 从 N=32 扩大到 N=64。

命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_board_dp_denoising_full_chain_smoke.py \
  --data_dir /home/chenshuai/data/dataset/260522_v8l_caheiban_flat_heldout32 \
  --dp_config /home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_fast32_e20/config.json \
  --dp_ckpt /home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_fast32_e20/dp_final.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_fast20 \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_fast20/foresight_best.ckpt \
  --n_episodes 16 \
  --frames_per_episode 4 \
  --n_eval 64 \
  --K 4 \
  --mode clean_refine \
  --accept_only_improved \
  --clamp_norm_action \
  --output /home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_fast32_e20_clean_refine_full_chain_fast20_heldout32_K4_N64.json
```

输出：

```text
/home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_fast32_e20_clean_refine_full_chain_fast20_heldout32_K4_N64.json
/home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_fast32_e20_clean_refine_full_chain_fast20_heldout32_K4_N64.log
```

结果：

| item | value |
|---|---:|
| frames | 64 |
| action samples | 256 |
| base_score mean | 2.1131053110584617 |
| guided_score mean | 2.1289010168984532 |
| score_delta mean | 0.01579570583999157 |
| score_delta median | 0.00748443603515625 |
| guided_beats_base_rate | 0.8203125 |
| range_violation max | 0.0 |
| base_smoothness mean | 0.5128180049941875 |
| guided_smoothness mean | 0.33142205598414876 |
| smoothness_delta mean | -0.18139594901003875 |
| norm_action_delta mean | 0.014042045717360452 |
| grad_norm mean | 1.7615037898067385 |
| pass | true |

### 与 N=32 对比

| metric | N=32 | N=64 |
|---|---:|---:|
| action samples | 128 | 256 |
| score_delta mean | 0.014835347421467304 | 0.01579570583999157 |
| guided_beats_base_rate | 0.8125 | 0.8203125 |
| range_violation max | 0.0 | 0.0 |
| smoothness_delta mean | -0.1695850170799531 | -0.18139594901003875 |

扩大样本后指标没有退化，反而略有改善。这说明当前 evidence 不是 N=32 小样本偶然结果。

### 当前结论

当前最强可复现 board evidence 更新为：

```text
fast32_e20 board DP
  -> fast20 board Foresight
  -> heldout32 N=64 episode-level full-chain evaluation
  -> PTG TacQualityEnergy clean-action trust-region guidance
  -> pass
```

剩余 production-scale 缺口主要是：

```text
DP training pipeline memory optimization
full 80-episode board DP
larger held-out / cross-day / real robot validation
```

## 2026-06-10 Lazy Image Loading 改造与 Full80 Training Entry

### 背景

fast40/fast64 DP 扩大训练失败后，定位到一个工程瓶颈：

```text
DPTacConcatDataset 会预加载所有 episode 的 resized/normalized images 到 RAM
```

这导致 board DP 训练规模扩大时 RAM/swap 压力过高。该问题阻碍 full 80-episode board DP 训练，也间接阻碍最终 PTG scorer/guidance 的 production-scale 验证。

### 代码改造

修改文件：

```text
diffusion/train_dp_tac_concat.py
```

新增能力：

```text
--lazy_images
--num_workers
--max_train_windows
```

行为：

```text
默认模式:
  preload qpos/action/marker/images
  速度快，但 RAM 占用高

lazy_images 模式:
  preload qpos/action/marker/path
  image 在 __getitem__ 时从 HDF5 按需读取
  RAM 占用低，但速度慢
```

`--max_train_windows` 用于 smoke/debug 或资源受限下的 subset 训练。

### 验证 1：Lazy Smoke2

命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python diffusion/train_dp_tac_concat.py \
  --dataset_dir /home/chenshuai/data/dataset/260522_v8l_caheiban_flat_lazy_smoke2 \
  --save_dir /home/chenshuai/Project/output/ckpt/dp_tac_concat_board_lazy_smoke2_e1_fast \
  --camera_names global,wrist \
  --proprio_key proprio_joint \
  --action_key actions/joint_abs \
  --tac_side left \
  --tac_history 8 \
  --vae_checkpoint /home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt \
  --vae_latent_dim 16 \
  --pred_horizon 16 \
  --obs_horizon 2 \
  --n_action_steps 8 \
  --resize_shape 240,320 \
  --crop_shape 216,288 \
  --epochs 1 \
  --batch_size 4 \
  --lr 1e-4 \
  --weight_decay 1e-6 \
  --warmup_steps 10 \
  --num_train_timesteps 20 \
  --num_inference_steps 20 \
  --diffusion_step_embed_dim 64 \
  --down_dims 128,256 \
  --seed 48 \
  --save_freq 1 \
  --gpu 0 \
  --lazy_images \
  --num_workers 0 \
  --max_train_windows 16
```

结果：

| item | value |
|---|---:|
| lazy_images | true |
| episodes | 2 |
| train windows | 16 |
| epochs | 1 |
| final train loss | 1.0912629812955856 |
| checkpoint saved | true |
| pass | true |

输出：

```text
/home/chenshuai/Project/output/board_production_chain_setup/board_dp_lazy_smoke2_e1_fast.json
/home/chenshuai/Project/output/ckpt/dp_tac_concat_board_lazy_smoke2_e1_fast/dp_final.pth
```

### 验证 2：Full80 Lazy Entry

命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python diffusion/train_dp_tac_concat.py \
  --dataset_dir /home/chenshuai/data/dataset/260522_v8l_caheiban_flat \
  --save_dir /home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_lazy_full80_entry_e1 \
  --camera_names global,wrist \
  --proprio_key proprio_joint \
  --action_key actions/joint_abs \
  --tac_side left \
  --tac_history 8 \
  --vae_checkpoint /home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt \
  --vae_latent_dim 16 \
  --pred_horizon 16 \
  --obs_horizon 2 \
  --n_action_steps 8 \
  --resize_shape 240,320 \
  --crop_shape 216,288 \
  --epochs 1 \
  --batch_size 4 \
  --lr 1e-4 \
  --weight_decay 1e-6 \
  --warmup_steps 10 \
  --num_train_timesteps 20 \
  --num_inference_steps 20 \
  --diffusion_step_embed_dim 64 \
  --down_dims 128,256 \
  --seed 49 \
  --save_freq 1 \
  --gpu 0 \
  --lazy_images \
  --num_workers 0 \
  --max_train_windows 64
```

结果：

| item | value |
|---|---:|
| lazy_images | true |
| episodes indexed | 80 |
| total frames | 57909 |
| train windows sampled | 64 |
| epochs | 1 |
| final train loss | 1.02300513535738 |
| checkpoint saved | true |
| pass | true |

输出：

```text
/home/chenshuai/Project/output/board_production_chain_setup/board_dp_lazy_full80_entry_e1.json
/home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_lazy_full80_entry_e1/dp_final.pth
```

### 验证 3：Full80 Lazy W2048_E5 尝试

设置：

```text
full 80 episodes
lazy_images
max_train_windows = 2048
epochs = 5
```

结果：

```text
成功索引 80 episodes
训练约 20 分钟仍未完成第 1 epoch
手动停止
```

解释：

lazy HDF5 image loading 解决了 RAM 问题，但吞吐太低。它适合作为 smoke/debug 或小规模训练入口，不适合作为最终 full production DP 训练方案。

### 当前工程判断

目前 board DP production-scale 缺口已经从：

```text
内存无法承载 full80
```

推进到：

```text
full80 可索引/可训练 smoke，但 naive lazy 太慢
```

下一步最合理路线是：

```text
precompute vision features
  -> DP training loads image features + tactile latent + qpos/action
  -> avoid repeated HDF5 image read/resize/ResNet forward
```

这条路线也更贴合 PTG 目标：评分器和 guidance 的关键在 tactile consequence / action，而不是每次训练都重复视觉前处理。

### 对 PTG 目标的影响

当前 PTG scorer/guidance 方法本身没有被新实验推翻：

```text
Task-conditioned TacQualityEnergy
  + Foresight
  + clean-action trust-region classifier guidance
  + accept-only update
```

仍是当前最佳路线。新增 lazy image loading 的价值在于解除 full board DP 生产验证的第一个工程障碍。

## 2026-06-10 Image Cache 训练入口验证

### 背景

lazy image loading 已经证明 full80 可以索引和训练 smoke，但速度太慢：

```text
每个 sample 都从 HDF5 读取图像
每次都 resize / normalize
```

因此继续推进一个中间方案：

```text
resized+normalized image cache
```

它的目的不是最终创新点，而是解除 full board DP 训练吞吐瓶颈，为后续 PTG production-scale full-chain 验证服务。

### 实现

修改文件：

```text
diffusion/train_dp_tac_concat.py
```

新增参数：

```text
--image_cache_dir
--build_image_cache
```

缓存格式：

```text
{image_cache_dir}/{dataset_basename}/episode_X_{camera}_rnorm.npy
```

单个 cache 文件：

```text
shape = (T, 3, resize_H, resize_W)
dtype = float16
content = resized + ImageNet-normalized image tensor
```

训练时：

```text
np.load(cache_file, mmap_mode='r')
按 obs_indices 读取对应帧
只做 crop，不再做 HDF5 读取 / resize / normalize
```

### Smoke 验证

命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python diffusion/train_dp_tac_concat.py \
  --dataset_dir /home/chenshuai/data/dataset/260522_v8l_caheiban_flat_lazy_smoke2 \
  --save_dir /home/chenshuai/Project/output/ckpt/dp_tac_concat_board_cache_smoke2_e1 \
  --camera_names global,wrist \
  --proprio_key proprio_joint \
  --action_key actions/joint_abs \
  --tac_side left \
  --tac_history 8 \
  --vae_checkpoint /home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt \
  --vae_latent_dim 16 \
  --pred_horizon 16 \
  --obs_horizon 2 \
  --n_action_steps 8 \
  --resize_shape 240,320 \
  --crop_shape 216,288 \
  --epochs 1 \
  --batch_size 4 \
  --lr 1e-4 \
  --weight_decay 1e-6 \
  --warmup_steps 10 \
  --num_train_timesteps 20 \
  --num_inference_steps 20 \
  --diffusion_step_embed_dim 64 \
  --down_dims 128,256 \
  --seed 51 \
  --save_freq 1 \
  --gpu 0 \
  --image_cache_dir /home/chenshuai/Project/output/board_image_cache_smoke2 \
  --build_image_cache \
  --num_workers 0 \
  --max_train_windows 16
```

输出：

```text
/home/chenshuai/Project/output/board_production_chain_setup/board_dp_cache_smoke2_e1.json
/home/chenshuai/Project/output/ckpt/dp_tac_concat_board_cache_smoke2_e1/dp_final.pth
/home/chenshuai/Project/output/board_image_cache_smoke2
```

结果：

| item | value |
|---|---:|
| image_loading_mode | cached |
| episodes | 2 |
| train windows | 16 |
| cache files | 4 |
| cache total bytes | 1497600512 |
| final train loss | 1.1323804557323456 |
| checkpoint saved | true |
| pass | true |

### 解释

image cache 路线证明：

```text
DP training 可以不预加载全部图像到 RAM，
也可以不在每个 sample 重复 HDF5 image read / resize / normalize。
```

但它也暴露了另一个问题：

```text
2 episodes / 2 cameras cache size ≈ 1.5GB
full80 image cache 可能达到数十 GB
```

因此 image cache 是一个可行的中间工程路线，但不一定是最终最优路线。

### 下一步更优路线

更适合 production-scale board DP 的路线是：

```text
Vision feature cache
```

具体做法：

```text
OfficialVisionEncoder(image) -> 512-dim feature per camera per frame
save: global_feat_512, wrist_feat_512
DP training loads features directly
obs_cond = [cached_vis_feat | tactile_latent | qpos]
```

优点：

1. 磁盘比 image cache 小很多；
2. 训练不再跑 ResNet encoder，速度更快；
3. 更适合 full80 / multi-run / ablation；
4. 对 PTG 目标足够，因为当前关注的是 action -> future tactile consequence -> scorer gradient，而不是继续优化视觉 encoder。

### 对当前 PTG 结论的影响

当前最强 scorer/guidance 结论不变：

```text
Task-conditioned TacQualityEnergy
  + Foresight
  + clean-action trust-region classifier guidance
  + accept-only update
```

新增 image cache smoke 只是把 production-scale board DP 的工程路线继续向前推进一步。
## 2026-06-10 Feature Cache Board DP Training Path

### 为什么要做这一步

当前评分/分类器主方案已经稳定为：

```text
Task-conditioned TacQualityEnergy
  + Foresight tactile consequence prediction
  + clean-action trust-region classifier guidance
  + accept-only improved update
```

但黑板任务还缺 production-scale board DP/Foresight full-chain evidence。之前的瓶颈不是 scorer 本身，而是 board DP 训练的数据加载：

1. preload images 会占用过多 RAM；
2. lazy HDF5 image loading 可以索引 full80，但训练吞吐太低；
3. resized image cache 可训练，但 2 episodes / 2 cameras 已约 1.5GB。

因此新增 feature-cache 路线，用于解除 full80 board DP 训练瓶颈。

### 方法

新增脚本：

```text
diffusion/train_dp_tac_concat_feature_cache.py
```

缓存每个 episode 的 per-frame 特征：

```text
vis_feat: (T, 512 * n_cameras), float16
tac_feat: (T, 144), float16
qpos: float32
action: float32
```

DP 训练时的条件仍保持和原始 `train_dp_tac_concat.py` 一致的语义：

```text
obs_cond = [vis_feat | tac_feat | qpos] * obs_horizon
```

区别是训练阶段不再读取 HDF5 图像、不再重复 resize/normalize、不再重复 ResNet/TactileVAE forward。

### Smoke 实验

命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python diffusion/train_dp_tac_concat_feature_cache.py \
  --dataset_dir /home/chenshuai/data/dataset/260522_v8l_caheiban_flat_lazy_smoke2 \
  --feature_cache_dir /home/chenshuai/Project/output/board_feature_cache_smoke2 \
  --save_dir /home/chenshuai/Project/output/ckpt/dp_tac_concat_feature_cache_smoke2_e1 \
  --camera_names global,wrist \
  --proprio_key proprio_joint \
  --action_key actions/joint_abs \
  --tac_side left \
  --tac_history 8 \
  --vae_checkpoint /home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt \
  --vae_latent_dim 16 \
  --resize_shape 240,320 \
  --crop_shape 216,288 \
  --cache_batch_size 64 \
  --build_feature_cache \
  --pred_horizon 16 \
  --obs_horizon 2 \
  --epochs 1 \
  --batch_size 4 \
  --lr 1e-4 \
  --weight_decay 1e-6 \
  --warmup_steps 10 \
  --num_train_timesteps 20 \
  --num_inference_steps 20 \
  --diffusion_step_embed_dim 64 \
  --down_dims 128,256 \
  --seed 52 \
  --save_freq 1 \
  --max_train_windows 16 \
  --num_workers 0 \
  --device cuda:0
```

输出：

```text
/home/chenshuai/Project/output/board_production_chain_setup/board_dp_feature_cache_smoke2_e1.json
/home/chenshuai/Project/output/board_feature_cache_smoke2
/home/chenshuai/Project/output/ckpt/dp_tac_concat_feature_cache_smoke2_e1
```

结果：

| item | value |
|---|---:|
| variant | feature_cache_tactile_vae_frozen |
| episodes | 2 |
| train windows | 16 |
| global_cond_dim | 2350 |
| final train loss | 1.1700587272644043 |
| cache files | 2 |
| cache total bytes | 3415715 |
| pass | true |

对比 image cache：

| cache type | 2-episode cache size |
|---|---:|
| resized image cache | 1,497,600,512 bytes |
| feature cache | 3,415,715 bytes |

feature cache 约为 image cache 的 0.23%。这说明 full80 board DP 更应该走 feature-cache 路线。

### 对 PTG 评分/分类器目标的意义

这一步不改变 scorer 的定义。它的意义是把黑板任务 full-chain 证据从工程瓶颈中解放出来：

```text
full80 board DP
  -> Foresight predicts future tactile consequence
  -> PTG TacQualityEnergy scores tactile quality
  -> clean-action trust-region gradient guidance
```

因此 feature cache 是支持最终 gradient guidance 证据链的训练路径，不是 reranking，也不是只做分类。

### 限制

当前 smoke 默认 OfficialVisionEncoder 是随机初始化，除非提供 `--vision_ckpt`。所以这次 smoke 只能证明 feature-cache training path 可用，不能作为最终最强 board policy 证据。

后续 production-scale 实验应：

1. 使用已训练 DP checkpoint 的 vision encoder 权重构建 feature cache，或明确标注 random frozen vision 的限制；
2. 在 full80 上构建 feature cache；
3. 训练更大 board DP；
4. 接入 board Foresight 和 TacQualityEnergy，跑 held-out full-chain clean-action guidance/refinement。

## 2026-06-10 Full80 Feature-Cache DP + Heldout Full-Chain Guidance

### 目的

上一节只证明 feature-cache 训练路径可用。本节进一步验证：

```text
full80 board DP
  -> fast20 board Foresight
  -> PTG TacQualityEnergy
  -> clean-action trust-region gradient guidance
```

是否能在 heldout board episodes 上稳定提高 predicted tactile quality score。

这一步直接对应最终目标：不是 reranking，也不是只做分类准确率，而是让评分/分类器作为可微 energy，对 DP 生成的 action 产生梯度引导。

### Full80 Feature Cache DP 训练

使用 fast32 raw-image DP checkpoint 的 `ema_vis` 作为视觉特征 encoder：

```text
vision_ckpt = /home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_fast32_e20/dp_final.pth
feature source = ema_vis
```

训练命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python diffusion/train_dp_tac_concat_feature_cache.py \
  --dataset_dir /home/chenshuai/data/dataset/260522_v8l_caheiban_flat \
  --feature_cache_dir /home/chenshuai/Project/output/board_feature_cache_full80_fast32ema \
  --save_dir /home/chenshuai/Project/output/ckpt/dp_tac_concat_feature_cache_full80_fast32ema_w4096_e5 \
  --camera_names global,wrist \
  --proprio_key proprio_joint \
  --action_key actions/joint_abs \
  --tac_side left \
  --tac_history 8 \
  --vae_checkpoint /home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt \
  --vae_latent_dim 16 \
  --vision_ckpt /home/chenshuai/Project/output/ckpt/dp_tac_concat_board_260522_fast32_e20/dp_final.pth \
  --resize_shape 240,320 \
  --crop_shape 216,288 \
  --cache_batch_size 64 \
  --build_feature_cache \
  --pred_horizon 16 \
  --obs_horizon 2 \
  --epochs 5 \
  --batch_size 64 \
  --lr 1e-4 \
  --weight_decay 1e-6 \
  --warmup_steps 100 \
  --num_train_timesteps 20 \
  --num_inference_steps 20 \
  --diffusion_step_embed_dim 64 \
  --down_dims 128,256 \
  --seed 53 \
  --save_freq 5 \
  --max_train_windows 4096 \
  --num_workers 2 \
  --device cuda:0
```

输出：

```text
/home/chenshuai/Project/output/board_production_chain_setup/board_dp_feature_cache_full80_fast32ema_w4096_e5.json
/home/chenshuai/Project/output/board_feature_cache_full80_fast32ema
/home/chenshuai/Project/output/ckpt/dp_tac_concat_feature_cache_full80_fast32ema_w4096_e5
```

训练结果：

| item | value |
|---|---:|
| episodes | 80 |
| windows | 4096 |
| feature cache files | 80 |
| feature cache size | 124,269,246 bytes |
| feature cache size | 118.5 MB |
| losses | 0.9486, 0.3827, 0.1578, 0.1102, 0.0926 |
| final train loss | 0.09258 |
| pass | true |

### Full-Chain Eval 修改

修改：

```text
TFAC_V5/eval_board_dp_denoising_full_chain_smoke.py
```

新增 feature-cache DP 支持：

1. `noise_pred_net` 从 feature-cache DP checkpoint 读取；
2. online obs condition 使用 config 中 `vision_ckpt` 的 `ema_vis` 生成视觉特征；
3. TactileVAE latent encoder 与原始 DP 路径一致；
4. 输出 JSON 记录：

```text
chain.dp_variant
chain.dp_uses_feature_cache
chain.dp_weight_source
chain.dp_feature_encoder_source
```

### Heldout N=64 Full-Chain Guidance

评估命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_board_dp_denoising_full_chain_smoke.py \
  --data_dir /home/chenshuai/data/dataset/260522_v8l_caheiban_flat_heldout32 \
  --dp_config /home/chenshuai/Project/output/ckpt/dp_tac_concat_feature_cache_full80_fast32ema_w4096_e5/config.json \
  --dp_ckpt /home/chenshuai/Project/output/ckpt/dp_tac_concat_feature_cache_full80_fast32ema_w4096_e5/dp_final.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_fast20 \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_fast20/foresight_best.ckpt \
  --n_episodes 16 \
  --frames_per_episode 4 \
  --n_eval 64 \
  --K 4 \
  --mode clean_refine \
  --accept_only_improved \
  --clamp_norm_action \
  --output /home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_feature_cache_full80_fast32ema_w4096_e5_fast20_heldout32_K4_N64.json
```

输出：

```text
/home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_feature_cache_full80_fast32ema_w4096_e5_fast20_heldout32_K4_N64.json
/home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_feature_cache_full80_fast32ema_w4096_e5_fast20_heldout32_K4_N64.log
```

结果：

| metric | value |
|---|---:|
| frames | 64 |
| action samples | 256 |
| base score mean | 1.65359 |
| guided score mean | 1.73463 |
| score delta mean | +0.08103 |
| score delta min | +0.05758 |
| score delta max | +0.10406 |
| guided beats base rate | 1.0 |
| range violation max | 0.0 |
| smoothness delta mean | -0.73685 |
| norm action delta mean | 0.03984 |
| guide accept rate | 1.0 |
| pass | true |

### 解释

这是目前黑板任务最强的 full-chain guidance 证据：

```text
feature-cache full80 DP action
  -> board Foresight predicts future tactile latent
  -> TactileVAE decodes predicted tactile marker
  -> PTG TacQualityEnergy scores tactile quality
  -> grad(score) / grad(action)
  -> trust-region update
  -> accept only if score improves
```

关键点：

1. 这是 gradient guidance，不是候选 reranking；
2. 256 个 heldout action samples 全部获得正向 score improvement；
3. action 始终没有越过 normalized range；
4. smoothness 也改善，说明 guidance 没有把动作推得更抖；
5. 使用的是 full80 DP 训练数据规模，而不是 smoke4/fast32 子集。

### 当前结论

当前评分/分类器方案可以更明确地表述为：

```text
Task-conditioned TacQualityEnergy
  = binary good/bad head
  + reason/failure-mode head
  + continuous quality/energy head
  + task-specific calibration/profile
```

最终用于 DP 引导时，不直接用饱和的 `p_good`，而用可微 logit/quality energy：

```text
E = w_q * quality_logit + w_b * binary_margin + w_r * reason_margin
E_clipped = c * tanh(E / c)
```

当前最推荐的注入方式仍是：

```text
clean-action trust-region classifier guidance
```

而不是 naive noisy-step guidance。

### 仍然保守不标记最终完成的原因

虽然 full80 feature-cache DP heldout full-chain 已经通过，但仍有两个限制：

1. board Foresight 仍是 fast20 checkpoint，不是最终强训练版本；
2. 还没有真实 robot closed-loop 或最终 production policy validation。

因此可以说：

```text
评分/分类器作为 DP 梯度引导目标已经有强证据成立；
最终系统级完成还需要 stronger Foresight + production policy validation。
```

## 2026-06-10 Stronger Board Foresight: fast100

### 为什么继续做 Foresight

上一节的 full-chain evidence 已经证明：

```text
feature-cache full80 DP
  -> fast20 Foresight
  -> TacQualityEnergy
  -> clean-action gradient guidance
```

在 heldout board episodes 上可以稳定提升 score。

但 fast20 仍有一个问题：训练历史显示第 20 epoch 仍是 best epoch，说明 Foresight 还没有完全收敛。为了避免“guidance 只是在弱 Foresight 上偶然有效”的质疑，本节继续训练同架构 fast100 Foresight，并重复 full-chain guidance。

### 训练配置

新增配置：

```text
TFAC_V5/config_pretrain_foresight_board_fast100.json
```

与 fast20 保持一致：

```text
hidden_dim = 128
foresight_layers = 1
foresight_nheads = 4
foresight_dim_feedforward = 512
use_state_trajectory = true
delta_weighted = true
```

只把：

```text
num_epochs: 20 -> 100
name: latent_foresight_board_260522_fast100
```

训练命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/pretrain_latent_foresight.py \
  --config TFAC_V5/config_pretrain_foresight_board_fast100.json
```

输出：

```text
/home/chenshuai/Project/output/board_production_chain_setup/board_foresight_fast100.json
/home/chenshuai/Project/output/board_production_chain_setup/foresight_board_fast100.log
/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_fast100
```

### 训练结果

| metric | value |
|---|---:|
| epochs | 100 |
| initial val | 15.36742 |
| fast20 best val | 2.69398 |
| fast100 best val | 1.69108 |
| best epoch | 91 |
| final val | 1.77109 |
| reduction vs fast20 | 37.23% |
| pass | true |

这说明 fast100 显著提升了 future tactile latent prediction。

### fast100 Full-Chain Guidance

评估命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_board_dp_denoising_full_chain_smoke.py \
  --data_dir /home/chenshuai/data/dataset/260522_v8l_caheiban_flat_heldout32 \
  --dp_config /home/chenshuai/Project/output/ckpt/dp_tac_concat_feature_cache_full80_fast32ema_w4096_e5/config.json \
  --dp_ckpt /home/chenshuai/Project/output/ckpt/dp_tac_concat_feature_cache_full80_fast32ema_w4096_e5/dp_final.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_fast100 \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_fast100/foresight_best.ckpt \
  --n_episodes 16 \
  --frames_per_episode 4 \
  --n_eval 64 \
  --K 4 \
  --mode clean_refine \
  --accept_only_improved \
  --clamp_norm_action \
  --output /home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_feature_cache_full80_fast32ema_w4096_e5_fast100_heldout32_K4_N64.json
```

结果：

| metric | value |
|---|---:|
| frames | 64 |
| action samples | 256 |
| base score mean | 1.65525 |
| guided score mean | 1.73623 |
| score delta mean | +0.08099 |
| score delta min | +0.05819 |
| score delta max | +0.10523 |
| guided beats base rate | 1.0 |
| range violation max | 0.0 |
| smoothness delta mean | -0.73692 |
| guide accept rate | 1.0 |
| pass | true |

### 和 fast20 的关系

fast20 full-chain：

```text
score_delta mean = +0.08103
guided_beats_base_rate = 1.0
range_violation = 0.0
```

fast100 full-chain：

```text
score_delta mean = +0.08099
guided_beats_base_rate = 1.0
range_violation = 0.0
```

两者几乎一致。解释是：

1. TacQualityEnergy 的局部梯度方向在 fast20 和 fast100 上都稳定；
2. 更强 Foresight 没有破坏 guidance；
3. 这说明当前 guidance 不是弱模型的偶然产物；
4. fast100 的主要价值是让 predicted tactile consequence 更可信，而不是让每次 score_delta 必然更大。

### 当前最强结论

目前最合理的评分/分类器方案已经不是单纯 binary classifier，而是：

```text
Task-conditioned TacQualityEnergy
  + binary good/bad head
  + reason/failure-mode head
  + continuous quality/energy head
  + task-specific guidance profile
```

配合：

```text
Foresight(action -> future tactile)
clean-action trust-region gradient guidance
accept-only improved update
```

当前证据覆盖：

1. 插座：GroupKFold risk scorer + full-chain gradient + constrained clean-action refinement；
2. 黑板：scorer-level readiness + surrogate full-chain + full80 feature-cache DP + fast20/fast100 Foresight heldout full-chain；
3. 评估切分：关键分类/评分使用 episode-level GroupKFold，避免 frame leakage；
4. guidance 形式：不是 reranking，而是 `d TacQualityEnergy / d action`。

剩余缺口：

```text
production policy / real robot validation
```

也就是最终系统级闭环验证，而不是当前 scorer/guidance 方法本身。

## 2026-06-10 Offline Production-Readiness Gate

### 为什么需要这个 gate

目前已有证据已经覆盖：

```text
socket insertion:
  scorer GroupKFold
  full-chain gradient
  constrained clean-action refinement

board wiping:
  scorer quality
  full80 feature-cache DP
  fast100 Foresight
  heldout full-chain gradient guidance
```

但这些仍然不能直接等价为“真机 production 完成”。因此新增一个 conservative offline gate：

```text
offline production-readiness gate
```

它只回答一个问题：

```text
当前 scorer/guidance stack 是否已经足够稳定，可以进入 production/robot dry-run？
```

它不回答：

```text
真实机器人闭环是否已经完成？
```

### 实现

新增脚本：

```text
TFAC_V5/eval_ptg_offline_production_gate.py
```

输入证据：

```text
/home/chenshuai/Project/output/ptg_guidance_evidence/ptg_guidance_evidence_summary.json
/home/chenshuai/Project/output/clean_action_energy_refinement/insertion_clean_refine_constrained_K4_N40.json
/home/chenshuai/Project/output/full_chain_guidance_gradient/insertion_full_chain_energy_clipped_K8_N16.json
/home/chenshuai/Project/output/board_production_chain_setup/board_foresight_fast100.json
/home/chenshuai/Project/output/board_production_chain_setup/board_dp_feature_cache_full80_fast32ema_w4096_e5.json
/home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_feature_cache_full80_fast32ema_w4096_e5_fast100_heldout32_K4_N64.json
```

检查项：

| group | check |
|---|---|
| insertion | full-chain gradient exists |
| insertion | clean-action guidance improves score |
| insertion | trust-region safety |
| board | stronger Foresight trained |
| board | full80 feature-cache DP trained |
| board | fast100 full-chain guidance improves score |
| board | trust-region safety |
| board | smoothness is not degraded |

实现细节：

脚本里专门使用：

```python
num(d, dotted, default)
```

读取数值，而不是 `value or default`。原因是 `0.0` 是合法的 range violation 结果，不能被误判为缺失值。

### 运行

命令：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_ptg_offline_production_gate.py
```

输出：

```text
/home/chenshuai/Project/output/ptg_offline_production_gate/ptg_offline_production_gate.json
/home/chenshuai/Project/output/ptg_offline_production_gate/ptg_offline_production_gate.md
```

结果：

```text
offline_production_gate_pass = true
remaining_required_step = Real robot / final production policy validation.
```

### Gate 指标

插座：

| metric | value |
|---|---:|
| full-chain score improved rate | 0.97656 |
| clean-action score delta mean | 0.30559 |
| clean-action beats | 0.975 |
| hard range violation max | 0.0 |

黑板：

| metric | value |
|---|---:|
| fast100 Foresight best val | 1.69108 |
| fast100 reduction vs fast20 | 37.23% |
| full80 feature-cache DP final loss | 0.09258 |
| fast100 full-chain score delta mean | 0.08099 |
| fast100 full-chain beats | 1.0 |
| range violation max | 0.0 |
| smoothness delta mean | -0.73692 |

### 总结

当前 PTG scorer/guidance stack 已经通过 offline production-readiness gate。

这意味着：

```text
可以进入 production / robot dry-run validation
```

但仍不能写成：

```text
real robot deployment completed
```

最终结论应保持为：

```text
评分/分类器作为 DP classifier guidance 的方法已经有强证据成立；
系统级最终完成还需要真实机器人或最终 production policy validation。
```

## 2026-06-10 TacQuality Score Calibration / Monotonicity Audit

### 为什么需要这个实验

此前已有：

```text
GroupKFold classification / regression metrics
full-chain gradient probes
clean-action refinement
offline production-readiness gate
```

但 DP classifier guidance 还需要回答一个更细的问题：

```text
score 越高，是否真的代表触觉质量越好？
```

如果一个分数只是 binary AUC 高，但概率饱和、或与连续质量不单调，那么它可以做分类器，但不一定适合做梯度势能。

因此新增 score calibration / monotonicity audit。

### 方法

新增脚本：

```text
TFAC_V5/eval_tac_quality_score_calibration.py
```

它不重新训练模型，只读取已有 checkpoint 和 feature cache：

```text
/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt
/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_features.npz
/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_final.pt
/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_features.npz
```

评估方式：

1. 对每种 score 从低到高分成 10 个 decile；
2. 统计每个 decile 的 good-rate；
3. 统计每个 decile 的 target quality mean；
4. 计算：
   - binary AUC；
   - quality Pearson；
   - quality Spearman；
   - top-bottom quality gap；
   - top-bottom good-rate gap；
   - quality decile positive step rate。

### 运行

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_tac_quality_score_calibration.py --device cuda:0
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_score_calibration/tac_quality_score_calibration.json
/home/chenshuai/Project/output/tac_quality_score_calibration/tac_quality_score_calibration.md
/home/chenshuai/Project/output/tac_quality_score_calibration/*_bins.csv
/home/chenshuai/Project/output/tac_quality_score_calibration/*_calibration.png
```

### 插座结果

InsertionRiskScorer 的候选 score：

| mode | AUC | Spearman(q) | Pearson(q) | top-bottom q gap | q monotonic |
|---|---:|---:|---:|---:|---:|
| quality | 0.9931 | 0.7210 | 0.8479 | 0.9581 | 0.8889 |
| p_good | 0.9975 | 0.7132 | 0.6205 | 0.9247 | 0.8889 |
| neg_risk | 0.9957 | 0.4178 | 0.5675 | 0.6434 | 0.6667 |
| risk_guidance | 0.9958 | 0.7245 | 0.6780 | 0.9693 | 0.8889 |
| energy | 0.9960 | 0.7336 | 0.7894 | 0.9657 | 1.0000 |

结论：

```text
插座推荐 guidance mode = energy
```

原因：

1. `p_good` 的 AUC 最高，但连续质量相关性较弱；
2. `energy` 的 quality decile 完全单调；
3. `energy` 同时保持高 AUC、高 quality Spearman、大 top-bottom quality gap；
4. 它更适合作为 DP 梯度引导势能，而不是只作为分类概率。

### 黑板结果

PTGProxyScorerV2 board 子集：

| mode | AUC | Spearman(q) | Pearson(q) | top-bottom q gap | q monotonic |
|---|---:|---:|---:|---:|---:|
| quality | 0.9795 | 0.9574 | 0.9626 | 0.9159 | 1.0000 |
| p_good | 0.9996 | 0.8214 | 0.7680 | 0.8077 | 1.0000 |
| reason_good | 0.9994 | 0.8268 | 0.7629 | 0.8234 | 1.0000 |
| energy | 0.9995 | 0.9004 | 0.9017 | 0.8690 | 1.0000 |
| weighted_energy | 0.9980 | 0.9389 | 0.9350 | 0.8946 | 1.0000 |

结论：

```text
黑板 calibration 最佳 mode = quality
```

解释：

1. 黑板的好坏标准本身就是连续质量：
   - 力大小合适；
   - 力变化柔顺；
2. 因此 `quality` head 最能反映目标；
3. `p_good` AUC 最高，但它更像 hard classifier，不如 `quality` 适合做平滑梯度；
4. 当前部署 profile 的：

```text
weighted_energy = 0.75 * quality_logit + 0.10 * binary_margin
```

是安全折中：

```text
quality 用于连续梯度；
binary_margin 用于保持 good/bad 安全边界。
```

### Mixed 结果

PTGProxyScorerV2 mixed 子集：

| mode | AUC | Spearman(q) | Pearson(q) | top-bottom q gap | q monotonic |
|---|---:|---:|---:|---:|---:|
| quality | 0.9453 | 0.6692 | 0.7045 | 0.8853 | 1.0000 |
| p_good | 0.9898 | 0.5464 | 0.5393 | 0.7951 | 0.8889 |
| reason_good | 0.9855 | 0.6644 | 0.6993 | 0.8833 | 1.0000 |
| energy | 0.9859 | 0.6497 | 0.6759 | 0.9123 | 1.0000 |
| weighted_energy | 0.9755 | 0.6543 | 0.6861 | 0.9094 | 1.0000 |

Mixed calibration 推荐：

```text
mixed mode = quality
```

但真实部署仍应优先用 task-conditioned profile，而不是一个完全 task-agnostic score。

### 接入总证据汇总

更新：

```text
TFAC_V5/summarize_ptg_guidance_evidence.py
```

新增输入：

```text
/home/chenshuai/Project/output/tac_quality_score_calibration/tac_quality_score_calibration.json
```

新增检查项：

```text
Insertion guidance-score calibration
Board guidance-score calibration
```

结果：

| check | result |
|---|---|
| Insertion guidance-score calibration | PASS |
| Board guidance-score calibration | PASS |

### 本轮结论

1. 当前 TacQuality scorer 不只是“分类准确”，其 score 与触觉质量也有稳定单调关系；
2. 插座任务更适合用 `energy` 作为梯度势能；
3. 黑板任务更适合以 `quality` 为主，用 `weighted_energy` 加入安全边界；
4. 这进一步支持当前创新路线：

```text
Task-conditioned TacQualityEnergy
  + binary good/bad head
  + reason/failure-mode head
  + continuous quality head
  + task-specific guidance profile
  + Foresight(action -> future tactile)
  + trust-region clean-action gradient guidance
```

5. 仍不能写成最终系统完成，因为还缺：

```text
real robot / final production policy validation
```

---

## 2026-06-10 TacQuality Manifest Real-Sample Smoke Test

### 目的

验证最终推荐的 task-conditioned TacQualityEnergy scorer/guidance package 是否能在真实任务数据张量上工作，而不是只在 synthetic sanity input 上工作。

该测试直接面向最终 DP classifier guidance 接口：

```python
score = runtime.score(task, predicted_tactile, action, mode="profile")
refined_action, report = refiner.refine(action, score_fn)
```

### Scope

这是 real-sample API smoke test：

1. 使用真实插座窗口和真实擦黑板窗口；
2. 检查 score 有限；
3. 检查 action gradient 有限且非零；
4. 检查 trust-region gradient ascent 后 score 是否提升；
5. 检查动作更新是否满足 task profile 中的 trust region。

这仍不是 real robot validation，也不是最终 production policy rollout。

### 新增代码

```text
TFAC_V5/eval_tac_quality_manifest_real_sample_smoke.py
```

更新：

```text
TFAC_V5/summarize_ptg_guidance_evidence.py
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_manifest_real_sample_smoke/manifest_real_sample_smoke.json
/home/chenshuai/Project/output/tac_quality_manifest_real_sample_smoke/manifest_real_sample_smoke.md
/home/chenshuai/Project/output/ptg_guidance_evidence/ptg_guidance_evidence_summary.json
/home/chenshuai/Project/output/ptg_guidance_evidence/ptg_guidance_evidence_summary.md
```

### 插座设置

| item | value |
|---|---|
| data | `/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_features.npz` |
| n real windows | 256 |
| tactile | real left marker window |
| action | real `joint_abs` window |
| scorer | `InsertionRiskScorerRuntime` |
| profile energy | `0.50*quality_logit + 0.10*binary_margin` |
| refiner | 4 steps, action_step=0.02, max_total_delta=0.08 |
| action clamp | false, because action is absolute joint value, not normalized DP action |

### 擦黑板设置

| item | value |
|---|---|
| data | `/home/chenshuai/data/dataset/260522_v8l_caheiban/success/*.hdf5` |
| n real windows | 256 |
| tactile | real left/right marker windows |
| action | real `eef_abs` and `joint_abs` windows |
| scorer | `PTGProxyScorerV2Runtime` |
| profile energy | `0.75*quality_logit + 0.10*binary_margin` |
| refiner | 4 steps, action_step=0.0002, max_total_delta=0.02 |

### 运行命令

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_tac_quality_manifest_real_sample_smoke.py --device cuda:0
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/summarize_ptg_guidance_evidence.py
```

### 结果

总体：

```text
overall_pass = true
manifest_pass = true
```

插座：

| metric | value |
|---|---:|
| passes_real_sample_smoke | true |
| score_improved_rate | 0.9921875 |
| score_delta_mean | 0.1245815244 |
| action_grad_norm_mean | 1.5631005482 |
| delta_norm_max | 0.0800003111 |

擦黑板：

| metric | value |
|---|---:|
| passes_real_sample_smoke | true |
| score_improved_rate | 1.0 |
| score_delta_mean | 0.0009992276 |
| joint_grad_norm_mean | 1.2616598642 |
| delta_norm_max | 0.0008164585 |

### 解释

1. 统一 `TacQualityGuidanceRuntime.score(...)` 可以在两个任务真实样本上正常工作；
2. 插座和擦黑板都能对真实动作产生有限、非零梯度；
3. trust-region gradient ascent 能提升 TacQuality score；
4. 动作更新受到 profile 约束；
5. 这一步进一步证明当前 scorer/energy/refiner 是可以接入 DP denoising loop 做梯度引导的；
6. 但最终目标仍不能声明 complete，因为还没有真实机器人或最终 production policy validation。

### 当前推荐方案

继续保持当前最终推荐：

```text
Task-conditioned TacQualityEnergy
  + binary good/bad head
  + reason/failure-mode head
  + continuous quality/energy head
  + task-specific guidance profile
  + Foresight(action -> future tactile)
  + TacQualityGuidanceRuntime unified score API
  + TacQualityTrustRegionRefiner bounded accept-only action update
```

标准接入链路：

```text
DP denoising action
  -> Foresight predicts future tactile
  -> TacQualityGuidanceRuntime.score(task, predicted_tactile, action, mode="profile")
  -> autograd d score / d action
  -> TacQualityTrustRegionRefiner bounded accepted update
```

---

## 2026-06-10 TacQuality Guidance Scale Sweep

### 目的

验证当前 TacQuality score 是否真正适合作为 DP classifier guidance 的局部能量函数。

分类器准确率只能说明“能判断好坏”，但 classifier guidance 还要求：

1. score 对 action 可微；
2. 沿 `d score / d action` 的小步更新能提升 score；
3. scale 增大时收益应基本稳定或单调；
4. 动作更新必须被 trust-region 控制；
5. 不应明显破坏动作平滑性。

因此新增真实样本上的 guidance scale sweep。

### 新增代码

```text
TFAC_V5/eval_tac_quality_guidance_scale_sweep.py
```

更新：

```text
TFAC_V5/summarize_ptg_guidance_evidence.py
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_guidance_scale_sweep/tac_quality_guidance_scale_sweep.json
/home/chenshuai/Project/output/tac_quality_guidance_scale_sweep/tac_quality_guidance_scale_sweep.md
/home/chenshuai/Project/output/ptg_guidance_evidence/ptg_guidance_evidence_summary.json
/home/chenshuai/Project/output/ptg_guidance_evidence/ptg_guidance_evidence_summary.md
```

### 实验设置

插座：

| item | value |
|---|---|
| n real windows | 256 |
| profile energy | `0.50*quality_logit + 0.10*binary_margin` |
| scale sweep | `0.005, 0.01, 0.02, 0.04, 0.08, 0.12` |
| max_total_delta | 0.08 |

擦黑板：

| item | value |
|---|---|
| n real windows | 256 |
| profile energy | `0.75*quality_logit + 0.10*binary_margin` |
| scale sweep | `0.00005, 0.0001, 0.0002, 0.0004, 0.0008, 0.0016` |
| max_total_delta | 0.02 |

### 调试记录

第一次运行：

```text
overall_pass = false
```

原因不是梯度方向失败，而是插座 scale=0.08 时投影后最大 delta 为 0.0800059，超过原来 `1e-6` 的过严浮点容差。

修正：

```text
trust_region_tolerance = max(1e-5, 1e-4 * max_total_delta)
```

原始 `delta_norm.max` 仍保存在 JSON 中，避免隐藏真实越界。

### 运行命令

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_tac_quality_guidance_scale_sweep.py --device cuda:0
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/summarize_ptg_guidance_evidence.py
```

### 结果

总体：

```text
overall_pass = true
```

插座：

| metric | value |
|---|---:|
| passes_guidance_scale_sweep | true |
| recommended_scale | 0.08 |
| recommended_score_delta_mean | 0.1187934754 |
| recommended_improved_rate | 0.984375 |
| gradient finite_rate | 1.0 |
| gradient positive_norm_rate | 1.0 |

擦黑板：

| metric | value |
|---|---:|
| passes_guidance_scale_sweep | true |
| recommended_scale | 0.0016 |
| recommended_score_delta_mean | 0.0017898571 |
| recommended_improved_rate | 1.0 |
| gradient finite_rate | 1.0 |
| gradient positive_norm_rate | 1.0 |

### 解释

这一步补强了“适合 DP classifier guidance”的核心证据：

1. 当前 scorer 不只是一个离线分类器；
2. TacQuality score 在真实插座和擦黑板样本上都能形成有效的 action gradient；
3. 沿该梯度进行局部更新能稳定提升 score；
4. trust-region 能限制更新幅度；
5. accept-only 机制可以避免负收益更新；
6. 推荐部署仍采用 task profile + trust-region，而不是无界 score maximization。

当前推荐：

```text
Insertion:
  guidance scale: 0.02-0.08
  max_total_delta: 0.08
  accept_only: true

Board:
  guidance scale: 0.0002-0.0016
  max_total_delta: 0.02
  accept_only: true
```

最终 DP 中应使用：

```text
score = TacQualityGuidanceRuntime.score(task, predicted_tactile, action, mode="profile")
action <- action + scale * normalize(d score / d action)
action <- project_trust_region(action, base_action)
accept only if score improves
```

---

## 2026-06-10 TacQuality Guidance Robustness Audit

### 目的

验证 TacQuality score 作为 DP classifier guidance 势能时，在触觉/动作扰动下是否稳定。

这个问题非常关键，因为最终 DP 引导时输入不是完美 GT，而是：

```text
current denoising action -> Foresight predicted tactile -> TacQuality score
```

如果 score 或 gradient 对小扰动极端不稳定，就会导致 guidance 抖动、方向错误或动作不平滑。

### 新增代码

```text
TFAC_V5/eval_tac_quality_guidance_robustness.py
```

更新：

```text
TFAC_V5/summarize_ptg_guidance_evidence.py
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_guidance_robustness/tac_quality_guidance_robustness.json
/home/chenshuai/Project/output/tac_quality_guidance_robustness/tac_quality_guidance_robustness.md
/home/chenshuai/Project/output/ptg_guidance_evidence/ptg_guidance_evidence_summary.json
/home/chenshuai/Project/output/ptg_guidance_evidence/ptg_guidance_evidence_summary.md
```

### 评估指标

1. score stability：
   - clean score 与 perturbed score 的 Pearson correlation；
   - sign same rate；
   - absolute score delta。
2. gradient stability：
   - clean gradient 与 perturbed gradient 的 cosine similarity；
   - finite gradient rate；
   - positive gradient norm rate。
3. guided-step robustness：
   - 用 clean/stale gradient 更新 perturbed state 后 score 是否提升；
   - 用 perturbed/current gradient 更新 perturbed state 后 score 是否提升。

### 实验设置

插座：

| item | value |
|---|---|
| n real windows | 256 |
| guidance scale | 0.04 |
| noise grid | `0/0, 0.02/0.01, 0.05/0.02, 0.10/0.05` |

擦黑板：

| item | value |
|---|---|
| n real windows | 256 |
| guidance scale | 0.0008 |
| noise grid | `0/0, 0.02/0.01, 0.05/0.02, 0.10/0.05` |

### 运行命令

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_tac_quality_guidance_robustness.py --device cuda:0
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/summarize_ptg_guidance_evidence.py
```

### 负结果和解释

如果要求旧梯度在所有噪声扰动下仍然稳定，则结果不通过：

| task | worst score corr | worst grad cosine p05 |
|---|---:|---:|
| insertion | 0.5857459493 | -0.3607720658 |
| board | 0.2864359329 | -0.2710291296 |

这说明：

```text
不能缓存或复用旧的 dscore/daction。
```

高噪声下，旧梯度方向可能明显偏离当前正确方向。

### 正确部署标准

DP classifier guidance 的正确做法是每个 denoising/guidance step 都重新计算当前梯度：

```text
current action
  -> Foresight predicts current future tactile
  -> TacQuality score
  -> autograd current dscore/daction
  -> trust-region action update
```

因此核心通过标准是 current-gradient robustness，而不是 stale-gradient reuse stability。

### 最终结果

总体：

```text
overall_pass = true
```

插座：

| metric | value |
|---|---:|
| passes_current_gradient_robustness | true |
| stale_gradient_stable_under_noise | false |
| worst perturbed-gradient improved rate | 0.99609375 |
| worst clean/stale-gradient improved rate | 0.75 |

擦黑板：

| metric | value |
|---|---:|
| passes_current_gradient_robustness | true |
| stale_gradient_stable_under_noise | false |
| worst perturbed-gradient improved rate | 1.0 |
| worst clean/stale-gradient improved rate | 0.51171875 |

### 结论

1. TacQualityEnergy 适合做当前状态重新计算的 classifier guidance；
2. TacQualityEnergy 不适合做 stale-gradient reuse；
3. 最终 DP 实现必须每个 guidance step 都重新前向 Foresight 并重新反传 TacQuality score；
4. 这是一个重要的安全约束，不是可选优化；
5. 该结论提升了方案的实现明确性和安全性。

推荐部署约束：

```text
Do:
  recompute score and gradient at every guidance step
  use trust-region projection
  accept only improved actions

Do not:
  cache dscore/daction
  reuse gradients across denoising steps
  apply stale gradients to new predicted tactile/action states
```

---

## 2026-06-10 DP-Facing TacQuality Guidance Controller

### 目的

把 robustness audit 中得到的部署约束固化成代码接口。

前面实验已经证明：

1. current-gradient guidance 有效；
2. stale-gradient reuse 不稳定；
3. 因此最终 DP denoising loop 不能传入缓存梯度，而应该每个 guidance step 重新计算当前 score 和 gradient。

仅靠文档说明容易被后续实现误用，所以新增 DP-facing controller。

### 新增代码

```text
TFAC_V5/tac_quality_dp_guidance_controller.py
TFAC_V5/eval_tac_quality_dp_controller_real_sample.py
```

更新：

```text
TFAC_V5/build_tac_quality_guidance_manifest.py
TFAC_V5/summarize_ptg_guidance_evidence.py
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_dp_guidance_controller/controller_sanity.json
/home/chenshuai/Project/output/tac_quality_dp_guidance_controller/controller_real_sample_audit.json
/home/chenshuai/Project/output/tac_quality_guidance_manifest/tac_quality_guidance_manifest.json
/home/chenshuai/Project/output/ptg_guidance_evidence/ptg_guidance_evidence_summary.json
```

### Controller API

推荐最终 DP denoising loop 使用：

```python
guided_action, report = controller.guide(action, current_score_fn)
```

其中：

```python
def current_score_fn(action):
    predicted_tactile = foresight(obs, action)
    return runtime.score(task, predicted_tactile, action, mode="profile")
```

Controller 内部执行：

```text
1. score = current_score_fn(action)
2. grad = autograd(score, action)
3. action_proposal = action + scale * normalize(grad)
4. action_proposal = project_trust_region(action_proposal, action)
5. accept only if current_score_fn(action_proposal) > score
```

### Guardrail

controller 不接受外部传入的 precomputed gradient。

配置中明确：

```text
stale_gradient_reuse_allowed = false
recompute_gradient_every_call = true
```

这对应 robustness audit 的结论：

```text
Do not cache or reuse stale dscore/daction.
```

### 运行命令

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/tac_quality_dp_guidance_controller.py --device cuda:0
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/eval_tac_quality_dp_controller_real_sample.py --device cuda:0
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/build_tac_quality_guidance_manifest.py
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/summarize_ptg_guidance_evidence.py
```

### 结果

Synthetic sanity：

| metric | value |
|---|---:|
| passes_controller_sanity | true |
| insertion improved_rate | 1.0 |
| board improved_rate | 1.0 |

Real-sample controller audit：

| task | pass | improved_rate | score_delta_mean |
|---|---|---:|---:|
| insertion | true | 0.99609375 | 0.0695024058 |
| board | true | 1.0 | 0.0010096454 |

Manifest：

```text
deployment_manifest_pass = true
dp_guidance_controller_pass = true
```

新增 manifest API：

```text
guided_action, report = controller.guide(action, current_score_fn)
```

### 意义

当前最终推荐方案升级为：

```text
Task-conditioned TacQualityEnergy
  + TacQualityGuidanceRuntime
  + TacQualityDPGuidanceController
  + trust-region projection
  + accept-only update
  + no stale-gradient reuse
```

最终接 DP 时的最小闭环：

```text
for denoising step:
    action = scheduler.step(...).prev_sample

    def current_score_fn(action):
        predicted_tactile = Foresight(obs, action)
        return TacQualityGuidanceRuntime.score(task, predicted_tactile, action, mode="profile")

    action, report = TacQualityDPGuidanceController.guide(action, current_score_fn)
```

这一步把评分器/分类器从“实验可用”推进到“部署接口明确”。

---

## 2026-06-10 Controller-In-Denoising Smoke

### 目的

验证 `TacQualityDPGuidanceController` 是否能直接嵌入插座 DP denoising loop，而不是只在真实样本 action 上做一次 refinement。

这一点非常重要：

```text
local score improvement != final denoising sample improvement
```

因为 DDPM scheduler 后续步骤可能抵消或放大某一步的局部修改。

### 新增代码

```text
TFAC_V5/eval_tac_quality_controller_denoising_smoke.py
```

更新：

```text
TFAC_V5/summarize_ptg_guidance_evidence.py
```

输出目录：

```text
/home/chenshuai/Project/output/tac_quality_controller_denoising_smoke/
```

### 实验链路

```text
DP denoising step
  -> normalized noisy action
  -> controller.guide(noisy_action, current_score_fn)
  -> current_score_fn:
       normalized action
       -> raw action
       -> Foresight
       -> predicted tactile
       -> TacQuality score
  -> guided normalized action
  -> continue scheduler / final action
```

### 实验设置

| item | value |
|---|---|
| task | insertion |
| DP | `/home/chenshuai/Project/output/dp_tac_vae_shift4_0414/dp_best.pth` |
| Foresight | `/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full/foresight_best.ckpt` |
| K | 4 |
| n_eval | 4 |
| score mode | energy_clipped |

### 结果

| setting | score_delta_mean | guided_beats_base_rate | range_violation_max | pass |
|---|---:|---:|---:|---|
| scale=0.01, start=0.8, every=2 | -0.0541780572 | 0.4375 | 0 | false |
| scale=0.003, start=0.9, every=4 | -0.0377594586 | 0.4375 | 0 | false |
| scale=0.005, start=0.9, every=4 | -0.0354929566 | 0.4375 | 0 | false |
| scale=0.003, final-only | 0.2523443419 | 0.75 | 0 | false |
| scale=0.001, final-only | 0.1001444533 | 0.5625 | 0 | false |
| scale=0.0005, final-only | 0.0334282145 | 0.5 | 0 | false |

### 解释

这是一个重要负结果。

观察：

1. controller 每次局部 update 的 score_delta_mean 都是正的；
2. 但多步插入 denoising loop 后，最终 score 可以下降；
3. final-only guidance 可以提高平均 score，但逐样本 beat rate 不够稳定；
4. 因此不能把 `controller.guide` 无条件插入每个 DDPM step 当作 production-ready 方法。

### 当前推荐

当前更稳妥的部署路线：

```text
1. DP 完成 denoising，得到 clean action
2. Foresight 预测该 action 的未来触觉
3. TacQualityDPGuidanceController 做 final clean-action refinement
4. trust-region + accept-only
5. 再进入 robot dry-run / final validation
```

暂不推荐：

```text
for every denoising step:
    action = controller.guide(action, current_score_fn)
```

除非后续完成：

1. guidance schedule sweep；
2. timestep-dependent scale；
3. scheduler-aware score correction；
4. 或者训练时加入 TacQuality guidance consistency。

### 结论

这一步没有让 objective complete，但它防止了一个重要误判：

```text
局部 classifier-guidance score 上升，不必然意味着最终 DP sample 更好。
```

因此当前最终方案应描述为：

```text
TacQuality score/classifier is suitable as a gradient source.
Current production-safe use is bounded final/clean-action refinement.
Full denoising-step guidance remains a research extension requiring scheduler-aware tuning.
```

## 2026-06-10 Unified TacQuality Guidance Runtime Contract

### 为什么需要统一 runtime

当前已经有多个通过验证的模块：

```text
InsertionRiskScorerRuntime
PTGProxyScorerV2Runtime
tac_quality_guidance_config.py
score calibration audit
full-chain / clean-action refinement scripts
```

但如果后续 DP policy 或真实机器人 dry-run 直接分别调用这些脚本，很容易出现：

```text
插座用一个 score mode
黑板用另一个 score mode
实验脚本和部署脚本不一致
calibration 推荐和 deployment profile 混淆
```

因此新增统一 runtime contract：

```python
runtime.score(task, predicted_tactile, action, mode="profile")
```

它只做一件事：

```text
把 task-conditioned TacQualityEnergy 作为一个统一可微 score API 暴露给 DP guidance。
```

### 实现

新增文件：

```text
TFAC_V5/tac_quality_guidance_runtime.py
```

核心类：

```python
TacQualityGuidanceRuntime
```

支持：

| task | scorer | profile energy |
|---|---|---|
| insertion | InsertionRiskScorerRuntime | 0.50 * quality_logit + 0.10 * binary_margin |
| board | PTGProxyScorerV2Runtime | 0.75 * quality_logit + 0.10 * binary_margin |

调用方式：

```python
score = runtime.score(
    task="insertion",
    left_marker_seq=predicted_marker,
    action_seq=action,
    mode="profile",
)
```

黑板：

```python
score = runtime.score(
    task="board",
    left_marker_seq=predicted_left_marker,
    right_marker_seq=predicted_right_marker,
    eef_action_seq=eef_action,
    action_seq=joint_action,
    mode="profile",
)
```

语义：

| mode | meaning |
|---|---|
| profile | 部署用 task-specific validated energy |
| calibrated | 分析用 calibration-best score；插座为 energy，黑板为 quality |

同时提供：

```python
runtime.diagnostics(...)
```

输出：

```text
quality
p_good
risk_prob / reason_good
quality_logit
binary_margin
reason_margin
profile_energy
```

### Runtime contract sanity

运行：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/tac_quality_guidance_runtime.py --device cuda:0
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_guidance_runtime/runtime_contract_sanity.json
```

结果：

```text
passes_runtime_contract_sanity = true
```

插座：

| metric | value |
|---|---:|
| score mean | -3.34466 |
| marker grad norm mean | 0.02927 |
| action grad norm mean | 0.01653 |
| all finite | true |
| all nonzero | true |

黑板：

| metric | value |
|---|---:|
| score mean | -2.53113 |
| left marker grad norm mean | 0.04171 |
| right marker grad norm mean | 0.02791 |
| eef action grad norm mean | 0.13320 |
| joint action grad norm mean | 0.00567 |
| all finite | true |
| all nonzero | true |

### 接入总证据汇总

更新：

```text
TFAC_V5/summarize_ptg_guidance_evidence.py
```

新增输入：

```text
/home/chenshuai/Project/output/tac_quality_guidance_runtime/runtime_contract_sanity.json
```

新增检查项：

```text
Insertion unified runtime contract
Board unified runtime contract
```

结果：

| check | result |
|---|---|
| Insertion unified runtime contract | PASS |
| Board unified runtime contract | PASS |

### 本轮结论

这一步把 PTG scorer 从：

```text
多个实验脚本里分别可用
```

推进到：

```text
有统一、可微、task-conditioned 的 DP guidance runtime contract
```

后续 DP 接入时的标准链路应为：

```text
action
  -> Foresight(action -> predicted tactile)
  -> TacQualityGuidanceRuntime.score(task, predicted_tactile, action, mode="profile")
  -> d score / d action
  -> trust-region clean-action refinement or late denoising guidance
```

仍需注意：

```text
runtime contract sanity ≠ real robot validation
```

它证明统一 API 和梯度链路可用，但最终系统完成仍需要 real robot / final production policy validation。

## 2026-06-10 Unified Trust-Region Action Guidance Refiner

### 为什么需要这个模块

`TacQualityGuidanceRuntime` 统一了：

```text
怎么打分
```

但真实 DP classifier guidance 还需要统一：

```text
怎么根据 score 的梯度安全地更新 action
```

此前这个逻辑分散在多个实验脚本：

```text
eval_clean_action_energy_refinement.py
eval_board_dp_denoising_full_chain_smoke.py
eval_board_surrogate_action_refinement.py
```

这些脚本都包含类似逻辑：

```text
score(action)
  -> grad = d score / d action
  -> unit gradient step
  -> project to trust region
  -> accept only if score improves
```

为了后续真实 DP/robot dry-run 接入更稳定，新增统一 trust-region guidance 更新器。

### 实现

新增文件：

```text
TFAC_V5/tac_quality_trust_region_guidance.py
```

核心类：

```python
TacQualityTrustRegionRefiner
```

核心配置：

```python
TrustRegionConfig(
    steps,
    step_size,
    max_total_delta,
    accept_only_improved=True,
    clamp_min=None,
    clamp_max=None,
)
```

标准调用：

```python
refiner = from_guidance_profile("insertion", clamp_norm_action=True)

refined_action, report = refiner.refine(
    action,
    score_fn,
)
```

其中：

```python
score_fn(action) -> Tensor[B]
```

该设计使 refiner 不绑定具体任务、不绑定 Foresight、不绑定 DP 模型，只要求 `score_fn` 可微。

最终 DP 接入时：

```python
def score_fn(action):
    predicted_tactile = foresight(obs, action)
    return guidance_runtime.score(task, predicted_tactile, action, mode="profile")

refined_action, report = trust_region_refiner.refine(action, score_fn)
```

### Sanity 实验

运行：

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/tac_quality_trust_region_guidance.py --device cuda:0
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_trust_region_guidance/trust_region_sanity.json
```

该 sanity 使用可控 quadratic score 验证更新器本身：

```text
score(action) = -||action - target||^2
```

这样可以确定：

1. 梯度方向正确；
2. score 会提升；
3. trust region 不会被突破；
4. accept-only 逻辑可用。

### 结果

总体：

```text
passes_trust_region_guidance_sanity = true
```

插座 profile：

| metric | value |
|---|---:|
| improved rate | 1.0 |
| score delta mean | 0.0029807 |
| delta norm mean | 0.08000 |
| delta norm max | 0.08000003 |
| trust region limit | 0.08 |
| within trust region | true |

黑板 profile：

| metric | value |
|---|---:|
| improved rate | 1.0 |
| score delta mean | 3.095e-06 |
| delta norm mean | 0.00080 |
| delta norm max | 0.00080001 |
| trust region limit | 0.02 |
| within trust region | true |

### 接入总证据汇总

更新：

```text
TFAC_V5/summarize_ptg_guidance_evidence.py
```

新增输入：

```text
/home/chenshuai/Project/output/tac_quality_trust_region_guidance/trust_region_sanity.json
```

新增检查项：

```text
Insertion trust-region guidance update
Board trust-region guidance update
```

结果：

| check | result |
|---|---|
| Insertion trust-region guidance update | PASS |
| Board trust-region guidance update | PASS |

### 当前推荐实现方式

目前最终推荐的 DP classifier guidance 工程接口是：

```text
TacQualityGuidanceRuntime
  -> 统一 task-conditioned score

TacQualityTrustRegionRefiner
  -> 统一 bounded accepted action update
```

标准链路：

```text
obs, action
  -> Foresight(obs, action)
  -> predicted tactile
  -> TacQualityGuidanceRuntime.score(task, predicted_tactile, action, mode="profile")
  -> TacQualityTrustRegionRefiner.refine(action, score_fn)
  -> guided action
```

这比简单 reranking 更符合用户要求：

```text
不是只选一个 action；
而是利用 score 的梯度直接修改 action。
```

### 本轮结论

1. 当前方案已有分类/评分、score calibration、runtime scoring、trust-region update 四层证据；
2. 插座和黑板都使用同一套 guidance contract，但保留 task-specific scorer/profile；
3. 这使方法既统一又不过度混淆两个任务的质量标准；
4. 最终系统完成仍需要：

```text
real robot / final production policy validation
```

## 2026-06-10 TacQuality Guidance Deployment Manifest

### 为什么需要 manifest

当前已经具备：

```text
scorer checkpoint
task-conditioned profile
runtime score API
trust-region refiner
score calibration
offline gate
evidence summary
```

但后续接入 DP policy 或真实机器人 dry-run 时，需要一个明确、可验证的交付清单，避免：

1. checkpoint 路径用错；
2. score mode 用错；
3. 插座/黑板 profile 混用；
4. 把 offline-ready 误写成 real-robot validated；
5. 实验配置和部署配置不一致。

因此新增 deployment manifest。

### 实现

新增脚本：

```text
TFAC_V5/build_tac_quality_guidance_manifest.py
```

该脚本自动读取：

```text
/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt
/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_final.pt
/home/chenshuai/Project/output/tac_quality_guidance_runtime/runtime_contract_sanity.json
/home/chenshuai/Project/output/tac_quality_trust_region_guidance/trust_region_sanity.json
/home/chenshuai/Project/output/tac_quality_score_calibration/tac_quality_score_calibration.json
/home/chenshuai/Project/output/ptg_guidance_evidence/ptg_guidance_evidence_summary.json
/home/chenshuai/Project/output/ptg_offline_production_gate/ptg_offline_production_gate.json
```

并输出：

```text
/home/chenshuai/Project/output/tac_quality_guidance_manifest/tac_quality_guidance_manifest.json
/home/chenshuai/Project/output/tac_quality_guidance_manifest/tac_quality_guidance_manifest.md
```

### 运行

```bash
/home/chenshuai/miniconda3/envs/TactileACT/bin/python TFAC_V5/build_tac_quality_guidance_manifest.py
```

结果：

```text
deployment_manifest_pass = true
remaining_required_step = Real robot / final production policy validation.
```

### Manifest 内容

统一 API：

```python
score = runtime.score(task, predicted_tactile, action, mode="profile")
refined_action, report = refiner.refine(action, score_fn)
```

插座：

| item | value |
|---|---|
| scorer | InsertionRiskScorerRuntime |
| checkpoint | `/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt` |
| profile energy | `0.50*quality_logit + 0.10*binary_margin` |
| calibration mode | energy |
| refine steps | 4 |
| step size | 0.02 |
| max total delta | 0.08 |

黑板：

| item | value |
|---|---|
| scorer | PTGProxyScorerV2Runtime |
| checkpoint | `/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_final.pt` |
| profile energy | `0.75*quality_logit + 0.10*binary_margin` |
| calibration mode | quality |
| refine steps | 4 |
| step size | 0.0002 |
| max total delta | 0.02 |

### Manifest checks

| check | result |
|---|---|
| all_manifest_files_exist | PASS |
| runtime_contract_pass | PASS |
| trust_region_guidance_pass | PASS |
| score_calibration_pass | PASS |
| offline_gate_pass | PASS |
| evidence_summary_keeps_real_robot_gap | PASS |

### 接入总证据汇总

更新：

```text
TFAC_V5/summarize_ptg_guidance_evidence.py
```

新增输入：

```text
/home/chenshuai/Project/output/tac_quality_guidance_manifest/tac_quality_guidance_manifest.json
```

新增检查项：

```text
Deployment manifest ready
```

结果：

```text
Deployment manifest ready: PASS
```

### 当前最终离线交付状态

当前可交付的 offline-ready package 包括：

```text
1. task-conditioned scorer checkpoints
2. tac_quality_guidance_config.py
3. TacQualityGuidanceRuntime
4. TacQualityTrustRegionRefiner
5. calibration / monotonicity audit
6. offline production-readiness gate
7. deployment manifest
```

该状态可以支持：

```text
production policy dry-run
robot dry-run preparation
final closed-loop validation
```

但仍不能写成：

```text
real robot validation completed
```

最终剩余步骤仍是：

```text
real robot / final production policy validation
```

## 2026-06-10 评分/分类器最终设计复核

### 目标复核

最终目标不是 reranking，而是在 DP 生成 action 的过程中提供可微的质量势能：

```text
obs, action -> Foresight 预测触觉后果 -> TacQualityEnergy score -> d score / d action
```

因此评估标准必须同时满足两类要求：

1. 质量判断准确：能区分好/坏触觉后果，并能解释坏的原因；
2. 可梯度引导：score 对预测触觉和 action 链路可微，且梯度有限、非零、受 trust-region 约束。

### 为什么不能只看 frame-level accuracy

同一个 episode 内的相邻帧高度相关。如果用 frame-level 随机划分，同一条轨迹的近邻帧可能同时出现在 train/test，测试集会“偷看到”训练 episode 的接触分布，结果通常偏乐观。前面的 99.24% MLP accuracy 属于这种候选筛选结果。

主评估必须使用 episode-level GroupKFold：

```text
Group = episode
一个 episode 的所有窗口/帧只能出现在 train 或 test 一边
```

这样测试结果才回答“新 episode 上是否泛化”。

### 插座任务标准

插座任务已有人工标注，坏数据只从 bounce episode 中取：

| 类别 | 定义 | 用途 |
|---|---|---|
| good_stable_smooth | success episode 的 insert 阶段 | 正样本 |
| excessive_or_risk | bounce episode 中 bounce 前的 pre-bounce/risk | 负样本 |
| impact_or_rough_force | 已发生 bounce / recovery 的冲击阶段 | 负样本/原因 |
| weak_or_no_contact | approach 或非接触弱接触 | 中性或弱负样本 |

严格 episode-level 结果：

| 方法 | Group-CV Balanced Acc | 备注 |
|---|---:|---|
| LDA | 0.9039 ± 0.0167 | 线性、稳健、解释性强 |
| MLP_128_64 | 0.8995 ± 0.0252 | 非线性、可作为 PyTorch scorer 模板 |
| RandomForest | 0.8954 ± 0.0230 | 强基线但不可直接反传 action 梯度 |
| LogReg_C10 | 0.8779 ± 0.0372 | 线性概率基线 |

20% unseen episode holdout 上 MLP 的 balanced accuracy 为 0.9525，ROC-AUC 为 0.9885，说明 MLP 有较高上限；但 Group-CV 均值略低于 LDA，所以不能只说 MLP 绝对最好。更科学的结论是：LDA/LogReg 做稳健 baseline，PyTorch MLP 做最终可微 scorer。

### 擦黑板任务标准

擦黑板没有人工好/坏标注，因此当前不能声称“真实监督准确率”。采用可解释弱标签：

| 类别 | 定义 |
|---|---|
| too_light | 平均擦拭力过小，接触不足 |
| good_smooth | 力在合适区间，且力/触觉/action 变化平滑 |
| too_heavy | 平均力或峰值力过大 |
| rough_force | 力变化或力 jerk 不平滑 |
| rough_motion | marker/action 变化不平滑 |

当前最佳 label scheme：

```text
force_source = left_force
scheme = t5_scoreband
feature_set = both_marker_actions
window = 32
stride = 16
```

黑板 GroupKFold 结果：

| 模型 | 指标 | 结果 |
|---|---|---:|
| RF classifier | 5 类 balanced accuracy | 0.9035 ± 0.0159 |
| RF classifier | good-vs-bad AUC | 0.9779 ± 0.0060 |
| RF classifier | score/quality corr | 0.8630 ± 0.0199 |
| RF regressor | quality corr | 0.9850 ± 0.0032 |
| RF regressor | quality R2 | 0.9697 ± 0.0062 |
| MLP classifier | 5 类 balanced accuracy | 0.8523 ± 0.0121 |
| MLP regressor | quality corr | 0.9427 ± 0.0100 |

解释：

1. RF/GBM 更擅长拟合阈值型弱标签，但不能直接用于 action 梯度；
2. PyTorch MLP scorer 虽然分类指标略低，但可微，适合部署为 guidance energy；
3. RF/规则分数应作为 teacher、标尺和离线验证，不作为最终梯度模块。

### 当前统一最好方案：TacQualityEnergy

最终推荐不是单一二分类器，而是多头任务条件评分器：

```text
feature/predicted tactile consequence + task_id
  -> shared encoder
  -> binary head: good vs bad
  -> reason head: weak/good/heavy/rough/impact
  -> quality head: continuous quality
  -> weighted clipped logit energy
```

当前实现：

```text
TFAC_V5/train_ptg_proxy_scorer_v2.py
TFAC_V5/ptg_proxy_scorer_v2_runtime.py
TFAC_V5/tac_quality_guidance_runtime.py
TFAC_V5/tac_quality_trust_region_guidance.py
```

统一 scorer 的 episode-level 混合任务结果：

| 指标 | 结果 |
|---|---:|
| binary balanced accuracy | 0.9082 ± 0.0281 |
| binary AUC | 0.9701 ± 0.0143 |
| reason balanced accuracy | 0.7901 ± 0.0293 |
| reason macro-F1 | 0.7678 ± 0.0434 |
| quality correlation | 0.7562 ± 0.0391 |
| feature-gradient sanity | PASS, grad_norm=0.1029 |

这说明当前模型既能判断好坏，又保留原因解释和连续质量势能，并且具备梯度。

### 与 classifier guidance / CFG 的关系

Classifier guidance 的形式是：

```text
diffusion_score + guidance_scale * grad_x log p(y | x)
```

在本项目中，`x` 不是图像，而是 action 或 predicted tactile consequence；`y` 不是文字类别，而是“触觉后果质量好”。因此对应为：

```text
grad_action TacQualityEnergy(Foresight(obs, action), action)
```

CFG 不依赖外部 classifier，而是 conditional/unconditional score 差分。当前任务需要显式利用“好触觉后果”的质量标准，因此更适合 classifier/regressor guidance。后续可以做 classifier-free 版本，但前提是 DP/Foresight 训练时加入 quality condition 或 failure-mode condition。

### 当前部署策略

当前证据支持：

```text
DP 正常 denoising -> clean/final action -> Foresight -> TacQualityEnergy -> trust-region final refinement
```

当前不建议直接作为生产方案：

```text
每个 DDPM denoising step 都强行加 guidance
```

原因：已做 denoising controller smoke，局部 score 常能提升，但多步 scheduler 后最终 action 不稳定；最后一步小步 refinement 更可靠。

### 真实 rollout gate

已补充真实 rollout 评估脚本的任务成功约束：

```text
TFAC_V5/eval_real_rollout_quality_gate.py
```

新增 `--metadata_csv` 后，gate 不只看触觉质量，还检查：

```text
guided_success_rate >= baseline_success_rate - max_success_rate_drop
guided_stopped_early_rate <= baseline_stopped_early_rate + max_bad_rate_increase
```

smoke 测试中，即使 guided 触觉质量提升：

```text
quality_delta_mean = 0.5746
paired_quality_guided_better_rate = 1.0
```

只要 metadata 显示 guided 成功率下降、提前停止率上升，仍输出：

```text
production_validation_pass = false
success_rate_ok = false
stopped_early_rate_ok = false
```

这避免 scoring/guidance 把 action 推向“触觉看起来好，但任务失败”的方向。

### 当前结论

当前最合理方案是：

```text
TacQualityEnergy = task-conditioned multi-head differentiable scorer
```

它用插座人工标注定义真实坏接触，用黑板力大小和力/动作平滑性定义弱监督质量，用 GroupKFold 检查未见 episode 泛化，用 trust-region final refinement 接入 DP 梯度引导。

仍未完成的是正式真机或最终 production policy 的 baseline-vs-guided rollout 验证。离线证据已经足够进入 dry-run/小步真实验证，但不能声称真实机器人最终闭环验证完成。

### 2026-06-10 补强：把“可梯度引导”纳入 offline production gate

之前的评估已经说明 TacQualityEnergy 能分类/评分，但为了避免“分类器准但不适合 guidance”的问题，现在将两个梯度相关实验提升为 required gate：

1. `tac_quality_guidance_scale_sweep`
   - 检查真实样本上沿 `d score / d action` 小步更新是否稳定提升 score；
   - 检查 trust-region 约束是否满足；
   - 插座推荐 scale=0.08，improved_rate=0.984375；
   - 黑板推荐 scale=0.0016，improved_rate=1.0。

2. `tac_quality_guidance_robustness`
   - 检查 tactile/action 小扰动下重新计算当前梯度是否仍然有效；
   - 插座 worst perturbed-gradient improved_rate=0.99609375；
   - 黑板 worst perturbed-gradient improved_rate=1.0。

更新后的 `TFAC_V5/eval_ptg_offline_production_gate.py` required checks 包括：

```text
TacQuality local guidance scale sweep passes for insertion and board
TacQuality current-gradient robustness passes under tactile/action perturbations
Clean-action refinement is the recommended deployment mode
Every-step denoising controller is explicitly not production-ready
```

当前 gate 输出：

```text
offline_production_gate_pass = true
remaining_required_step = Real robot / final production policy validation.
```

这个结果把当前结论从“有一个分类/评分器”推进到“有一个离线验证过的、可作为 final-action trust-region gradient guidance 的评分器”。但它仍然只是离线 ready，不等于真实机器人最终验证完成。

### 2026-06-10 目标完成度审计结果

新增目标级审计：

```text
TFAC_V5/audit_tac_quality_goal_completion.py
```

该脚本逐条检查用户目标，而不是只检查某个实验是否通过。当前输出：

```text
/home/chenshuai/Project/output/tac_quality_goal_audit/tac_quality_goal_completion_audit.json
/home/chenshuai/Project/output/tac_quality_goal_audit/tac_quality_goal_completion_audit.md
```

结果：

```text
objective_complete = false
n_requirements = 10
n_blockers = 2
```

已经满足的 8 项覆盖：

```text
插座 episode-level 评估
黑板力/柔顺性弱监督标准
统一 task-conditioned differentiable scorer
local guidance scale sweep
current-gradient robustness
offline production gate
deployment manifest
工作记录和研究文档
```

未满足的 2 项均为真实 rollout：

```text
Formal socket insertion baseline-vs-guided production/robot rollout validation passes.
Formal board wiping baseline-vs-guided production/robot rollout validation passes.
```

因此后续真正关闭目标，需要采集 baseline DP 与 TacQuality-guided DP 两组 HDF5 rollout，并分别运行：

```bash
python TFAC_V5/eval_real_rollout_quality_gate.py \
  --task insertion \
  --baseline_dir <baseline_insert_rollouts> \
  --guided_dir <guided_insert_rollouts> \
  --pairing_csv <optional_pairs.csv> \
  --metadata_csv <optional_success_metadata.csv>

python TFAC_V5/eval_real_rollout_quality_gate.py \
  --task board \
  --baseline_dir <baseline_board_rollouts> \
  --guided_dir <guided_board_rollouts> \
  --pairing_csv <optional_pairs.csv> \
  --metadata_csv <optional_success_metadata.csv>
```

为了减少真实数据采集后的整理成本，新增准备脚本：

```bash
python TFAC_V5/prepare_real_rollout_validation.py \
  --task board \
  --baseline_dir <baseline_board_rollouts> \
  --guided_dir <guided_board_rollouts>
```

该脚本会先检查 HDF5 是否包含正式 gate 需要的触觉、力和 action 字段，并生成 `pairing_template.csv` 与 `metadata_template.csv`。它不替代正式 gate，也不会给出 guidance 是否成功的结论。

采集规模建议由：

```text
TFAC_V5/plan_real_rollout_sample_size.py
```

给出。默认规划结果：

| task | paired baseline/guided pairs | unpaired per group |
|---|---:|---:|
| insertion | 12 | 28 |
| board | 12 | 28 |

因此正式实验优先做 paired design：每个任务至少 12 对 baseline/guided trial，并填写 `pairing_csv` 与 `metadata_csv`。

已生成正式实验包：

```text
/home/chenshuai/Project/output/real_rollout_experiment_packet/formal_paired12
```

该目录包含插座和黑板各 12 对 paired trials 的 CSV 模板、采集 checklist、prepare 命令和 gate 命令。它是执行真实验证的操作包，不是验证结果。

## 2026-06-10 Split Leakage 审计：frame random vs episode GroupKFold

新增脚本：

```text
TFAC_V5/audit_quality_split_leakage.py
```

目的：检查触觉质量分类/评分结果是否被 frame-level 随机划分高估。由于同一个 episode 内相邻帧高度相关，`frame random split` 会让相邻帧同时出现在 train/test 中，容易产生数据泄漏。用于证明泛化能力时，权威结果必须使用 `episode-level GroupKFold`。

运行：

```bash
python TFAC_V5/audit_quality_split_leakage.py \
  --models logreg rf \
  --splits 5 \
  --max_per_task_class 1200 \
  --max_train_per_class 2500
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_split_leakage_audit/tac_quality_split_leakage_audit.json
/home/chenshuai/Project/output/tac_quality_split_leakage_audit/tac_quality_split_leakage_audit.md
```

核心结果：

| rank | cache | label | model | GroupKFold balanced acc | GroupKFold macro F1 | GroupKFold AUC | quality Spearman | frame-random minus GroupKFold |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | ptg_proxy_scorer_v2 | binary | RF | 0.9168 | 0.9158 | 0.9744 | 0.6812 | 0.0148 |
| 2 | ptg_proxy_scorer_v2 | reason | RF | 0.7966 | 0.8086 | 0.9688 | 0.6664 | 0.0642 |
| 3 | ptg_proxy_scorer_v2 | binary | LogReg | 0.8348 | 0.8327 | 0.9194 | 0.6239 | 0.0135 |
| 4 | unified_quality_taxonomy | y_binary | RF | 0.7757 | 0.7754 | 0.8546 | 0.4863 | 0.0264 |
| 5 | ptg_proxy_scorer_v2 | reason | LogReg | 0.7109 | 0.6689 | 0.9079 | 0.5746 | 0.0367 |

结论：

1. 最可靠的当前分类定义是 `ptg_proxy_scorer_v2/binary`，即统一 good/bad 质量标准；
2. 在 episode-level GroupKFold 下，RF teacher 达到 `AUC=0.9744`、`balanced acc=0.9168`，说明当前好/坏标准本身可学习且泛化到未见 episode；
3. `frame-random minus GroupKFold` 只有约 `0.0148`，没有出现严重 frame 泄漏；
4. LogReg 在同一 binary 定义下仍有 `AUC=0.9194`、`balanced acc=0.8348`，说明即便使用简单可解释边界，信号也存在；
5. RF 不能直接作为 DP 梯度引导模型，因为它不可微；但它适合作为 teacher，用于蒸馏更强的可微 TacQualityEnergy scorer；
6. 当前后续最合理路线不是继续只追逐分类准确率，而是：
   - 保留 `ptg_proxy_scorer_v2/binary` 作为主质量标准；
   - 用 RF/GBM teacher 的 soft score 蒸馏可微 MLP energy；
   - 继续用 episode-level GroupKFold、score monotonicity、gradient sanity、trust-region guidance sweep 作为 gate；
   - 最终用 paired real rollout gate 验证是否真实改善 DP action。

## 2026-06-10 RF Teacher 蒸馏到可微 TacQualityEnergy

新增脚本：

```text
TFAC_V5/train_distilled_tac_quality_energy.py
```

目的：RF teacher 在 split leakage 审计中表现最好，但 RF 不可微，不能直接用于 DP classifier guidance。该脚本把 RF 的 soft good-probability 蒸馏到一个可微 MLP energy scorer，使其同时保留：

1. good/bad binary head；
2. reason/failure-mode head；
3. continuous quality head；
4. teacher soft-score head；
5. `energy_clipped` guidance potential。

运行：

```bash
python TFAC_V5/train_distilled_tac_quality_energy.py \
  --device cuda:0 \
  --folds 5 \
  --epochs 70 \
  --final_epochs 90 \
  --max_per_task_class 1200
```

输出：

```text
/home/chenshuai/Project/output/distilled_tac_quality_energy/distilled_tac_quality_energy_eval.json
/home/chenshuai/Project/output/distilled_tac_quality_energy/distilled_tac_quality_energy_final.pt
```

episode-level GroupKFold 结果：

| metric | mean | std |
|---|---:|---:|
| binary balanced acc | 0.9006 | 0.0261 |
| binary macro F1 | 0.8850 | 0.0414 |
| binary AUC | 0.9677 | 0.0173 |
| energy binary AUC | 0.9672 | 0.0181 |
| reason balanced acc | 0.7931 | 0.0291 |
| reason macro F1 | 0.7627 | 0.0388 |
| quality corr | 0.7492 | 0.0443 |
| teacher pred corr | 0.9290 | 0.0121 |
| energy teacher Spearman | 0.9048 | 0.0148 |
| energy quality Spearman | 0.6400 | 0.0597 |
| RF teacher binary AUC | 0.9735 | 0.0123 |

gradient sanity：

```text
score = 0.5586590767
input_grad_norm = 0.4677735567
input_grad_abs_mean = 0.0037378524
usable_for_feature_guidance = true
```

结论：

1. 蒸馏后的可微 energy scorer 的 hard-label AUC 为 `0.9672`，接近 RF teacher 的 `0.9735` 和原 `ptg_proxy_scorer_v2` 的 `0.9701`；
2. 它的核心优势不是硬分类准确率略高，而是 `energy_teacher_spearman=0.9048`，说明 energy 排序高度贴近强 teacher 的 soft ranking；
3. `quality_corr=0.7492`、`energy_quality_spearman=0.6400`，说明 score 仍与连续质量目标保持一致；
4. `input_grad_norm=0.4678` 且 finite，说明该 scorer 可作为 differentiable guidance potential；
5. 当前推荐路线：
   - RF/GBM teacher 继续作为 offline upper-bound 和 soft-label generator；
   - `distilled_tac_quality_energy_final.pt` 作为下一版可微 guidance scorer 候选；
   - 下一步应把该 scorer 接入已有 trust-region action refinement / scale sweep，与 `ptg_proxy_scorer_v2` 做相同的 guidance 改善率对比。

## 2026-06-10 蒸馏 TacQualityEnergy Runtime 与局部 Guidance 对比

新增 runtime：

```text
TFAC_V5/distilled_tac_quality_energy_runtime.py
```

该 runtime 复用 torch 版 proxy features：

```text
left marker + right marker + abs diff + eef action + joint action -> energy_clipped
```

因此它不是只能读取缓存特征，而是可以接收 Foresight 预测的 tactile marker 和 DP candidate action，并通过 autograd 把 score 梯度传回触觉/动作输入。

runtime sanity：

```bash
python TFAC_V5/distilled_tac_quality_energy_runtime.py --device cuda:0
```

输出：

```text
/home/chenshuai/Project/output/distilled_tac_quality_energy/runtime_sanity.json
```

结果：

```text
usable_for_guidance = true
left_grad_norm = 0.0055862800
right_grad_norm = 0.0080041774
eef_grad_norm = 0.0642583519
joint_grad_norm = 0.0111279944
```

新增同协议局部 guidance 对比：

```text
TFAC_V5/eval_distilled_energy_guidance_comparison.py
```

运行：

```bash
python TFAC_V5/eval_distilled_energy_guidance_comparison.py \
  --device cuda:0 \
  --n_per_task_class 300
```

输出：

```text
/home/chenshuai/Project/output/distilled_energy_guidance_comparison/distilled_energy_guidance_comparison.json
/home/chenshuai/Project/output/distilled_energy_guidance_comparison/distilled_energy_guidance_comparison.md
```

对比结果：

| scorer | pass | recommended scale | improved rate | score delta mean | grad norm mean | smoothness proxy p95 |
|---|---:|---:|---:|---:|---:|---:|
| ptg_proxy_v2 | true | 0.12 | 1.0000 | 0.185478 | 2.286408 | 0.008667 |
| distilled_energy | true | 0.12 | 1.0000 | 0.175594 | 2.149269 | 0.007225 |

结论：

1. `DistilledTacQualityEnergyRuntime` 的 marker/action 梯度均 finite 且非零，可以用于后续 DP guidance 连接；
2. 在相同 feature-level trust-region 协议下，`distilled_energy` 和 `ptg_proxy_v2` 都达到 `improved_rate=1.0`；
3. `distilled_energy` 的 score delta 略低于 `ptg_proxy_v2`，但 smoothness proxy p95 也更低，表现为更保守的局部 guidance potential；
4. 因此蒸馏 scorer 不是直接替代当前 `ptg_proxy_v2`，而是成为一个通过局部 gate 的候选；
5. 下一步要做 action-level / Foresight full-chain trust-region 对比，检查它是否能在真实 DP action 变量上带来更稳定的改善。

## 2026-06-10 黑板 Surrogate Full-Chain Action-Level 对比

新增脚本：

```text
TFAC_V5/eval_board_surrogate_distilled_comparison.py
```

目的：把蒸馏 scorer 从 feature-level guidance 推进到 action-level surrogate full-chain 验证。实验链路：

```text
current tactile + candidate action
  -> board tactile surrogate predicts future tactile
  -> scorer energy
  -> d energy / d action
  -> accepted trust-region action refinement
```

该实验比 feature-level gradient 更强，因为梯度必须穿过 board tactile surrogate 回到 action；但它仍然不是 production DP 或 robot rollout。

运行：

```bash
python TFAC_V5/eval_board_surrogate_distilled_comparison.py \
  --device cuda:0 \
  --n_eval 256 \
  --batch_size 64
```

输出：

```text
/home/chenshuai/Project/output/board_surrogate_distilled_comparison/board_surrogate_distilled_comparison.json
/home/chenshuai/Project/output/board_surrogate_distilled_comparison/board_surrogate_distilled_comparison.md
```

结果：

| scorer | pass | improved rate | score delta mean | eef delta max | joint delta max | eef smooth delta p95 | marker MAE delta mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| ptg_proxy_v2 | true | 0.9883 | 0.040863 | 0.000801 | 0.000811 | 0.000235 | 0.000013 |
| distilled_energy | true | 0.9844 | 0.064029 | 0.000800 | 0.000811 | 0.000284 | 0.000046 |

结论：

1. 两个 scorer 都通过 board surrogate action-level gate；
2. `distilled_energy` 的平均 score 提升更大：`0.0640` vs `0.0409`；
3. `distilled_energy` 的 improved rate 略低：`0.9844` vs `0.9883`；
4. `distilled_energy` 的 smoothness / marker MAE 副作用略高，但量级仍很小：
   - eef smooth delta p95 = `0.000284`
   - marker MAE delta mean = `0.000046`
5. 因此当前不能直接替代 `ptg_proxy_v2`，但蒸馏 scorer 已经通过从分类到 action-level surrogate guidance 的连续证据链；
6. 下一步应进入 production Foresight / DP clean-action refinement 对比，或者在真实 rollout gate 中作为 ablation 组。

## 2026-06-10 黑板 DP/Foresight Clean-Action Refinement 对比

新增脚本：

```text
TFAC_V5/eval_board_dp_distilled_clean_refine_comparison.py
```

目的：在更接近 production 的黑板链路中比较 `ptg_proxy_v2` 和 `distilled_energy`：

```text
board DP clean action
  -> board Foresight predicts tactile latent / marker
  -> scorer energy
  -> d energy / d normalized action
  -> accepted clean-action trust-region refinement
```

该实验使用已有 board feature-cache DP checkpoint 与 board Foresight checkpoint。它比 surrogate action refinement 更强，但仍是 smoke / offline chain，不是 robot rollout。

### Sanity N=8

运行：

```bash
python TFAC_V5/eval_board_dp_distilled_clean_refine_comparison.py \
  --device cuda:0 \
  --data_dir /home/chenshuai/data/dataset/260522_v8l_caheiban_flat_heldout32 \
  --dp_config /home/chenshuai/Project/output/ckpt/dp_tac_concat_feature_cache_full80_fast32ema_w4096_e5/config.json \
  --dp_ckpt /home/chenshuai/Project/output/ckpt/dp_tac_concat_feature_cache_full80_fast32ema_w4096_e5/dp_final.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_fast20 \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_fast20/foresight_best.ckpt \
  --n_episodes 4 \
  --frames_per_episode 2 \
  --n_eval 8 \
  --K 4 \
  --output_dir /home/chenshuai/Project/output/board_dp_distilled_clean_refine_comparison/fast20_n8
```

结果：

| scorer | pass | improved rate | score delta mean | smoothness delta mean | norm delta p95 | range violation max |
|---|---:|---:|---:|---:|---:|---:|
| ptg_proxy_v2 | true | 1.0000 | 0.082589 | -0.735484 | 0.039995 | 0.000000 |
| distilled_energy | true | 1.0000 | 0.148581 | -0.690702 | 0.039995 | 0.000000 |

### Heldout32 N=64

运行：

```bash
python TFAC_V5/eval_board_dp_distilled_clean_refine_comparison.py \
  --device cuda:0 \
  --data_dir /home/chenshuai/data/dataset/260522_v8l_caheiban_flat_heldout32 \
  --dp_config /home/chenshuai/Project/output/ckpt/dp_tac_concat_feature_cache_full80_fast32ema_w4096_e5/config.json \
  --dp_ckpt /home/chenshuai/Project/output/ckpt/dp_tac_concat_feature_cache_full80_fast32ema_w4096_e5/dp_final.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_fast20 \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_fast20/foresight_best.ckpt \
  --n_episodes 16 \
  --frames_per_episode 4 \
  --n_eval 64 \
  --K 4 \
  --output_dir /home/chenshuai/Project/output/board_dp_distilled_clean_refine_comparison/fast20_heldout32_n64
```

输出：

```text
/home/chenshuai/Project/output/board_dp_distilled_clean_refine_comparison/fast20_heldout32_n64/board_dp_distilled_clean_refine_comparison.json
/home/chenshuai/Project/output/board_dp_distilled_clean_refine_comparison/fast20_heldout32_n64/board_dp_distilled_clean_refine_comparison.md
```

结果：

| scorer | pass | improved rate | score delta mean | smoothness delta mean | norm delta p95 | range violation max |
|---|---:|---:|---:|---:|---:|---:|
| ptg_proxy_v2 | true | 1.0000 | 0.081157 | -0.733588 | 0.039995 | 0.000000 |
| distilled_energy | true | 1.0000 | 0.144726 | -0.684571 | 0.039995 | 0.000000 |

结论：

1. 两个 scorer 都通过 board DP/Foresight clean-action refinement smoke gate；
2. 在 N=64 heldout32 上，`distilled_energy` 的平均 score 提升更大：`0.144726` vs `0.081157`；
3. 两者动作 trust-region p95 都在 `0.039995`，range violation 都为 0；
4. 两者 smoothness delta 都为负，表示 clean-action refinement 后动作加速度下降；
5. `ptg_proxy_v2` 的 smoothness 降低略多，`distilled_energy` 的 scorer-energy 提升更强；
6. 由于两个 scorer 的绝对 score 不是同一个标尺，不能只用 score delta 断言真实策略质量更好；但 distilled scorer 已经通过 production-like offline chain，可作为真实 rollout ablation 候选。

## 2026-06-10 Scorer Selection Gate：当前最合理评分器选择

目的：把分类准确性、episode-level 泛化、可微梯度、action-level trust-region 改善、DP/Foresight clean-action refinement 证据合并成一个选择门控，避免只看单个 accuracy 或单个 score delta 选错评分器。

新增脚本：

```text
TFAC_V5/build_tac_quality_scorer_selection_gate.py
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_scorer_selection_gate/tac_quality_scorer_selection_gate.json
/home/chenshuai/Project/output/tac_quality_scorer_selection_gate/tac_quality_scorer_selection_gate.md
```

当前选择标准：

1. `episode-level GroupKFold` 是主评估，frame random split 只用于泄漏审计；
2. RF teacher 可以作为 offline upper-bound 和 soft-label teacher，但不能直接用于 DP 梯度引导，因为 RF 不可微；
3. DP 引导需要连续 energy / score，而不是只输出离散类别；
4. 最小合格条件：
   - good/bad 或 energy AUC >= 0.95；
   - quality correlation / teacher ranking 不能太低；
   - scorer 对输入有非零、有限梯度；
   - feature-level guidance 改善率通过；
   - action-level surrogate 或 DP/Foresight clean-refine 通过；
   - action trust-region 与 range violation 受控；
5. 真实替代默认 scorer 还需要 baseline-vs-guided real rollout gate。

Selection gate 结论：

| role | selected method | status |
|---|---|---|
| insertion current default | `InsertionRiskScorerRuntime` | current default |
| board current default | `PTGProxyScorerV2Runtime` | current default |
| non-differentiable teacher | RF on `ptg_proxy_scorer_v2/binary` | offline teacher / upper bound |
| differentiable candidate | `DistilledTacQualityEnergyRuntime` | promoted ablation candidate |
| replacement decision | distilled energy | not yet replacement |
| recommended DP mode | final clean-action trust-region refinement | current safe mode |

关键证据：

| scorer | key metric | result |
|---|---:|---:|
| RF teacher | GroupKFold AUC | 0.9744 |
| RF teacher | frame-random minus GroupKFold balanced acc | 0.0148 |
| PTGProxyV2 | GroupKFold binary AUC | 0.9701 |
| PTGProxyV2 | GroupKFold quality corr | 0.7562 |
| DistilledEnergy | GroupKFold energy AUC | 0.9672 |
| DistilledEnergy | teacher prediction corr | 0.9290 |
| DistilledEnergy | energy-teacher Spearman | 0.9048 |
| PTGProxyV2 | feature-level improved rate | 1.0000 |
| DistilledEnergy | feature-level improved rate | 1.0000 |
| PTGProxyV2 | board surrogate improved rate | 0.9883 |
| DistilledEnergy | board surrogate improved rate | 0.9844 |
| PTGProxyV2 | board DP clean-refine improved rate | 1.0000 |
| DistilledEnergy | board DP clean-refine improved rate | 1.0000 |
| PTGProxyV2 | board DP score delta mean | 0.081157 |
| DistilledEnergy | board DP score delta mean | 0.144726 |

解释：

1. `PTGProxyV2` 仍作为当前黑板任务默认 scorer，因为它已经完整接入现有部署栈，quality corr 更高，局部和 DP clean-refine 都稳定；
2. `DistilledTacQualityEnergyRuntime` 是更有创新性的候选：先用 RF teacher 学到更强的非线性 good/bad 边界，再蒸馏成可微 energy，使其能对 Foresight-predicted tactile/action 求梯度；
3. 蒸馏 scorer 在 board DP/Foresight clean-refine 中 score gain 更大，但不同 scorer 的 score 标尺不同，因此不能只根据 score delta 宣称真实质量更好；
4. 当前最稳妥的策略不是每个 DDPM step 都强行 classifier guidance，而是：

```text
DP denoising produces clean action
  -> Foresight predicts tactile consequence
  -> TacQuality energy scores predicted consequence
  -> backprop d energy / d action
  -> bounded accept-only trust-region clean-action refinement
```

5. 下一步真实结论必须来自 paired rollout ablation：

```text
baseline DP
vs PTGProxyV2-guided DP
vs DistilledTacQualityEnergy-guided DP
```

结论：目前已经找到一个合理、可解释、可微、可用于 DP 梯度引导的评分器体系。默认部署建议使用 `PTGProxyV2`，创新候选使用 `DistilledTacQualityEnergyRuntime` 做真实 rollout ablation；在真实 rollout gate 之前，不把蒸馏 scorer 宣称为最终替代。

## 2026-06-10 插座任务 DistilledEnergy clean-action refinement 对比

目的：补齐蒸馏 scorer 在插座任务上的 action-level 证据。之前 `DistilledTacQualityEnergyRuntime` 的 action-level / DP clean-refine 强证据主要来自黑板任务；为了确认它不是只适用于黑板，需要在插座 DP/Foresight 链路中和 `InsertionRiskScorerRuntime` 做同协议对比。

新增脚本：

```text
TFAC_V5/eval_insertion_distilled_clean_refine_comparison.py
```

协议：

```text
插座 DP clean action
  -> 插座 Foresight 预测未来 tactile marker
  -> scorer energy
  -> d energy / d action
  -> bounded accept-only trust-region refinement
```

比较对象：

1. `InsertionRiskScorerRuntime`
   - 插座专用默认 scorer；
   - 输入单路 tactile marker + joint action；
2. `DistilledTacQualityEnergyRuntime`
   - 跨任务蒸馏 energy scorer；
   - 插座中使用 `left_marker=right_marker=predicted_marker`，`task_id=0`，`joint_action_seq=action`。

运行：

```bash
python TFAC_V5/eval_insertion_distilled_clean_refine_comparison.py \
  --device cuda:0 \
  --n_eval 24 \
  --K 4 \
  --output_dir /home/chenshuai/Project/output/insertion_distilled_clean_refine_comparison/n24_k4
```

输出：

```text
/home/chenshuai/Project/output/insertion_distilled_clean_refine_comparison/n24_k4/insertion_distilled_clean_refine_comparison.json
/home/chenshuai/Project/output/insertion_distilled_clean_refine_comparison/n24_k4/insertion_distilled_clean_refine_comparison.md
```

结果：

| scorer | pass | improved rate | score delta mean |
|---|---:|---:|---:|
| insertion_risk | true | 1.0000 | 0.399135 |
| distilled_energy | true | 1.0000 | 0.014179 |

解释：

1. 两个 scorer 都能在插座 DP/Foresight clean-action refinement 链路中提供有效 action gradient；
2. `distilled_energy` 通过了插座 action-level gate，说明它是跨任务可微候选，不只是黑板任务可用；
3. 但 `distilled_energy` 在插座上的 score delta 很小，远低于插座专用 `InsertionRiskScorerRuntime`；
4. 因此 selection gate 的结论应更保守：
   - 插座默认继续使用 `InsertionRiskScorerRuntime`；
   - 黑板默认继续使用 `PTGProxyScorerV2Runtime`；
   - `DistilledTacQualityEnergyRuntime` 作为跨任务创新 ablation candidate，而不是替代两个任务默认 scorer；
5. 这个结果是有价值的负/弱证据：蒸馏统一 scorer 有跨任务可微性，但任务专用 scorer 在插座上仍明显更强。

## 2026-06-10 Formal Three-Arm Real Rollout Scorer Ablation Gate

目的：最终目标不是只证明“某个 guided policy 比 baseline 好”，而是要回答哪个评分/分类器最适合作为 DP classifier guidance 的能量函数。因此真实 rollout 验证必须支持三臂对比：

```text
baseline DP
vs task-default scorer guided DP
vs DistilledTacQualityEnergy guided DP
```

新增脚本：

```text
TFAC_V5/eval_real_rollout_scorer_ablation_gate.py
```

该脚本复用 `eval_real_rollout_quality_gate.py` 中的真实 rollout 质量指标：

1. 插座：
   - tactile marker / force impact proxy；
   - risk proxy；
   - quality score；
   - success / stopped_early metadata 约束；
2. 黑板：
   - force magnitude 是否在合适区间；
   - force delta / force jerk；
   - marker delta；
   - action acceleration；
   - too_light / too_heavy / roughness flags；
   - quality score；
   - success / stopped_early metadata 约束。

三臂 gate 的判断：

1. `default_guided` vs `baseline` 是否通过；
2. `distilled_guided` vs `baseline` 是否通过；
3. `distilled_guided` vs `default_guided` 的 paired / bootstrap CI 是否显著；
4. 输出 `recommended_real_scorer`：
   - `default_guided`
   - `distilled_guided`
   - 或 `None / tie_or_underpowered`

命令格式：

```bash
python TFAC_V5/eval_real_rollout_scorer_ablation_gate.py \
  --task insertion \
  --baseline_dir <insertion_baseline_rollout_dir> \
  --default_guided_dir <insertion_default_guided_rollout_dir> \
  --distilled_guided_dir <insertion_distilled_guided_rollout_dir> \
  --pairing_csv <three_arm_pairing.csv> \
  --metadata_csv <metadata.csv> \
  --output_dir /home/chenshuai/Project/output/real_rollout_scorer_ablation_gate \
  --tag insertion_baseline_vs_default_vs_distilled
```

```bash
python TFAC_V5/eval_real_rollout_scorer_ablation_gate.py \
  --task board \
  --baseline_dir <board_baseline_rollout_dir> \
  --default_guided_dir <board_default_guided_rollout_dir> \
  --distilled_guided_dir <board_distilled_guided_rollout_dir> \
  --pairing_csv <three_arm_pairing.csv> \
  --metadata_csv <metadata.csv> \
  --output_dir /home/chenshuai/Project/output/real_rollout_scorer_ablation_gate \
  --tag board_baseline_vs_default_vs_distilled
```

同时更新：

```text
TFAC_V5/build_real_rollout_experiment_packet.py
```

重新生成：

```bash
python TFAC_V5/build_real_rollout_experiment_packet.py --tag formal_paired12
```

输出中新增：

```text
/home/chenshuai/Project/output/real_rollout_experiment_packet/formal_paired12/insertion/three_arm_pairing_template.csv
/home/chenshuai/Project/output/real_rollout_experiment_packet/formal_paired12/board/three_arm_pairing_template.csv
```

以及每个任务 README 中的 `ablation_gate_command`。

当前状态：

1. 三臂 evaluator 已实现；
2. 三臂 formal packet 已生成；
3. goal audit 已加入两个最终要求：
   - insertion 三臂 scorer ablation 需要正式通过；
   - board 三臂 scorer ablation 需要正式通过；
4. 这两个要求当前仍未满足，因为还没有真实 rollout HDF5 数据；
5. 这是合理的：离线 gate 证明 scorer 可用于 dry-run，三臂 real rollout gate 才能证明哪个 scorer 真实效果最好。

## 2026-06-10 三臂 real rollout gate synthetic smoke 与黑板目标力修正

目的：在真实机器人数据采集前，用 synthetic 但 schema-compatible 的 HDF5 数据验证三臂 evaluator 能否正确：

1. 读取三组 rollout；
2. 应用 pairing / metadata；
3. 计算插座和黑板 quality score；
4. 输出 `production_ablation_pass` 和 `recommended_real_scorer`；
5. 在黑板任务中使用明确目标擦拭力，而不是错误地把 baseline 分布当作好坏标准。

新增脚本：

```text
TFAC_V5/smoke_real_rollout_scorer_ablation_gate.py
```

运行：

```bash
python TFAC_V5/smoke_real_rollout_scorer_ablation_gate.py \
  --n_pairs 12 \
  --output_dir /home/chenshuai/Project/output/real_rollout_scorer_ablation_smoke
```

输出：

```text
/home/chenshuai/Project/output/real_rollout_scorer_ablation_smoke/real_rollout_scorer_ablation_smoke.json
/home/chenshuai/Project/output/real_rollout_scorer_ablation_smoke/insertion_synthetic_smoke/real_rollout_scorer_ablation_gate.json
/home/chenshuai/Project/output/real_rollout_scorer_ablation_smoke/board_synthetic_smoke/real_rollout_scorer_ablation_gate.json
```

结果：

| task | production ablation pass | recommended scorer | guided arm winner |
|---|---:|---|---|
| insertion | true | distilled_guided | tie_or_underpowered |
| board | true | distilled_guided | distilled_guided |

说明：

1. `scientific_evidence=false`，该 smoke 不是任务效果证据，只是 evaluator 接口和逻辑验证；
2. 插座任务中 default 和 distilled 都可能把 risk 降到安全区，quality 会饱和，因此三臂 winner 可能是 tie；这不代表真实任务中无法比较，只说明当前 proxy 对“足够安全后谁更好”不敏感；
3. 黑板任务必须有明确的目标擦拭力标准：

```bash
--board_target_force <board_target_force>
--board_force_sigma <board_force_sigma>
```

4. 修正了 `eval_real_rollout_quality_gate.py` 和 `eval_real_rollout_scorer_ablation_gate.py`：
   - 如果显式给出 `board_target_force`，`too_light/too_heavy` 和 `force_p95` penalty 使用目标力区间；
   - 不再把 baseline 的 force q80/q90 当作正常力上限；
5. 这个修正很重要：如果 baseline 是力太小的坏策略，用 baseline 分位数定义好坏会错误惩罚正常擦拭力。

结论：三臂 evaluator 已通过 synthetic HDF5 smoke，真实实验还必须采集三臂 rollout 数据后再判断最终 scorer。

## 2026-06-10 黑板 real rollout 目标擦拭力校准

目的：黑板任务的好坏标准包含“力大小合适”和“力变化柔顺”。为了让 real rollout gate 的 `--board_target_force` 有明确来源，新增一个从已有黑板数据估计目标力区间的校准 artifact。

新增脚本：

```text
TFAC_V5/calibrate_board_target_force.py
```

校准规则与 `eval_board_quality_label_schemes.py` 保持一致：

```text
board_target_force = q55(force_mean)
board_force_sigma = q75(force_mean) - q25(force_mean)
```

运行：

```bash
python TFAC_V5/calibrate_board_target_force.py \
  --data_dir /home/chenshuai/data/dataset/260522_v8l_caheiban \
  --force_source left_force \
  --success_only
```

输出：

```text
/home/chenshuai/Project/output/board_target_force_calibration/board_target_force_calibration.json
/home/chenshuai/Project/output/board_target_force_calibration/board_target_force_calibration.md
```

结果：

| item | value |
|---|---:|
| n_episodes | 80 |
| force_source | left_force |
| board_target_force | 8.4821928501 |
| board_force_sigma | 3.9529049397 |

推荐 real rollout gate 参数：

```bash
--board_target_force 8.4821929 --board_force_sigma 3.9529049
```

同时更新 `TFAC_V5/build_real_rollout_experiment_packet.py`：

1. 如果存在 calibration JSON，黑板二臂 gate 和三臂 ablation gate 命令会自动填入上述参数；
2. 不再要求手动替换 `<board_target_force>`；
3. packet 中记录 calibration JSON 路径。

解释：

1. 这个 calibration 是弱监督标准，不是人工金标准；
2. 它比“从 baseline rollout 分布估计目标力”更合理，因为 baseline 可能正是坏策略；
3. 真实 rollout 后仍应结合 success、是否擦干净、是否过早停止等 metadata 判断最终质量。

## 2026-06-10 三臂 rollout arm 配置固化

目的：三臂真实验证不能只靠 README 里写“baseline / default / distilled”。为了避免采集时混淆 scorer、checkpoint、score mode、trust-region 参数，新增机器可读的三臂 rollout arm 配置。

新增脚本：

```text
TFAC_V5/build_tac_quality_rollout_arm_configs.py
```

运行：

```bash
python TFAC_V5/build_tac_quality_rollout_arm_configs.py
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs.json
/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs.md
```

配置内容：

| task | arm | scorer | guidance |
|---|---|---|---:|
| insertion | baseline | none | false |
| insertion | default_guided | InsertionRiskScorerRuntime | true |
| insertion | distilled_guided | DistilledTacQualityEnergyRuntime | true |
| board | baseline | none | false |
| board | default_guided | PTGProxyScorerV2Runtime | true |
| board | distilled_guided | DistilledTacQualityEnergyRuntime | true |

该 JSON 还记录：

1. checkpoint 路径；
2. task-specific guidance profile；
3. trust-region refinement 参数；
4. board target force calibration；
5. formal ablation gate command；
6. selection gate 中当前默认 scorer 与蒸馏候选的状态。

同时更新：

```text
TFAC_V5/build_real_rollout_experiment_packet.py
TFAC_V5/build_tac_quality_guidance_manifest.py
TFAC_V5/audit_tac_quality_goal_completion.py
```

结论：后续采集三臂真实 rollout 时，应以 `tac_quality_rollout_arm_configs.json` 作为配置入口，避免默认 scorer 和 distilled scorer 的实现方式被口头描述混淆。

## 2026-06-10 三臂 rollout arm 梯度 smoke

目的：确认三臂 rollout 配置里的 guided arm 不只是“写在 JSON 里”，而是真的能作为 DP classifier guidance / energy guidance 的梯度源。

新增脚本：

```text
TFAC_V5/smoke_tac_quality_rollout_arm_configs.py
```

运行：

```bash
python TFAC_V5/smoke_tac_quality_rollout_arm_configs.py
```

输入配置：

```text
/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs.json
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_rollout_arm_config_smoke/tac_quality_rollout_arm_config_smoke.json
/home/chenshuai/Project/output/tac_quality_rollout_arm_config_smoke/tac_quality_rollout_arm_config_smoke.md
```

检查内容：

1. baseline arm 没有 scorer，记录为 no-guidance skip；
2. insertion/default_guided 加载 `InsertionRiskScorerRuntime`；
3. insertion/distilled_guided 加载 `DistilledTacQualityEnergyRuntime`；
4. board/default_guided 加载 `PTGProxyScorerV2Runtime`；
5. board/distilled_guided 加载 `DistilledTacQualityEnergyRuntime`；
6. 对每个 guided arm，用合成 tactile marker/action tensor 检查：
   - score 有限；
   - tactile gradient 有限且非零；
   - action gradient 有限且非零。

解释边界：

1. 这是工程 smoke，不是科学效果证据；
2. 通过该 smoke 只能说明“配置的 scorer 可以提供梯度”，不能说明真实任务一定变好；
3. 最终仍必须通过真实 HDF5 rollout 的二臂 gate 和三臂 scorer ablation gate；
4. 这一步对最终目标仍然必要，因为 DP 梯度引导需要的是连续可微 energy，而不是只用于离线分类报告的离散 label。

关于“分类还是评分”：

当前方案不是只分好坏。推荐的 TacQuality 结构同时保留：

| 输出 | 用途 |
|---|---|
| binary good/bad | 定义安全边界、失败风险、离线评估 AUC |
| reason class | 区分过大力、过小力、不柔顺、bounce/risk 等失败原因 |
| continuous quality / energy | 作为 DP / Foresight 链路里的梯度引导目标 |

因此最终用于引导的不是硬分类标签，而是可微的能量函数，例如：

```text
score = w_quality * quality_logit
      + w_binary  * good_logit_margin
      + w_reason  * reason_logit_margin
```

插座任务当前默认更信任 task-specific `InsertionRiskScorerRuntime`；黑板任务当前默认使用 `PTGProxyScorerV2Runtime`；`DistilledTacQualityEnergyRuntime` 是跨任务可微 ablation candidate，还没有替代默认 scorer。

关于 GroupKFold：

frame-level 随机划分容易泄漏，因为同一个 episode 的相邻帧高度相似。如果训练集里有某个 episode 的前半段，测试集里有同一 episode 的后半段，分类器可能只是记住该 episode 的轨迹/接触分布，而不是真正泛化到新 episode。

episode-level GroupKFold 把整个 episode 作为不可拆开的 group。某个 episode 要么全在训练集，要么全在测试集。因此它回答的是更重要的问题：

```text
这个评分/分类器能不能在从未见过的新 episode 上判断触觉质量？
```

所以后续所有关键准确率、AUC、quality correlation 都应优先看 GroupKFold，而不是 frame-level random split。

## 2026-06-10 Formal rollout gate runner

目的：把最终真实 rollout 验证变成一个统一入口，而不是手动复制多条长命令。

新增脚本：

```text
TFAC_V5/run_formal_tac_quality_rollout_gates.py
```

默认 preflight：

```bash
python TFAC_V5/run_formal_tac_quality_rollout_gates.py
```

输出：

```text
/home/chenshuai/Project/output/formal_tac_quality_rollout_gate_runner/formal_paired12_preflight/formal_tac_quality_rollout_gate_runner.json
/home/chenshuai/Project/output/formal_tac_quality_rollout_gate_runner/formal_paired12_preflight/formal_tac_quality_rollout_gate_runner.md
```

当前结果预期为：

```text
preflight_ready = false
scientific_evidence = false
```

原因：还没有传入真实 HDF5 rollout 目录。这个结果不是评分器失败，而是明确记录最后缺少的数据输入。

真实数据采集后，用法为：

```bash
python TFAC_V5/run_formal_tac_quality_rollout_gates.py \
  --insertion_baseline_dir <insertion_baseline_rollout_dir> \
  --insertion_default_guided_dir <insertion_default_guided_rollout_dir> \
  --insertion_distilled_guided_dir <insertion_distilled_guided_rollout_dir> \
  --board_baseline_dir <board_baseline_rollout_dir> \
  --board_default_guided_dir <board_default_guided_rollout_dir> \
  --board_distilled_guided_dir <board_distilled_guided_rollout_dir> \
  --run_gates
```

它会在 preflight 全部通过后执行：

| gate | task |
|---|---|
| baseline vs task-default guided | insertion |
| baseline vs task-default guided | board |
| baseline vs default vs distilled | insertion |
| baseline vs default vs distilled | board |

这一步的意义：

1. 对最终 objective 来说，真实 rollout gate 是不能绕过的证据；
2. runner 不是新的评分器，也不会改变当前默认选择；
3. runner 让最后的验证流程可复现，避免手动替换路径导致命令不一致；
4. 如果 `--run_gates` 后四个 gate 都通过，再重跑：

```bash
python TFAC_V5/audit_tac_quality_goal_completion.py
```

才可能把 objective 从 `incomplete` 推向 `complete`。

## 2026-06-10 Score landscape 诊断

目的：验证 TacQuality score 不只是能做离线分类，而是真的适合作为 DP classifier guidance 的连续能量函数。

背景：

```text
分类/评分效果好 != 梯度引导一定好
```

对于 DP classifier guidance，更关键的是 action-space 局部几何：

1. score 对 action 有有限梯度；
2. 梯度不能大面积饱和为 0；
3. 沿正梯度方向移动，score 应该上升；
4. 沿负梯度方向移动，score 应该下降；
5. autograd 给出的方向导数应和有限差分一致；
6. trust-region 内的一阶近似不能完全失效。

新增脚本：

```text
TFAC_V5/eval_tac_quality_score_landscape.py
```

运行：

```bash
python TFAC_V5/eval_tac_quality_score_landscape.py
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_score_landscape/tac_quality_score_landscape.json
/home/chenshuai/Project/output/tac_quality_score_landscape/tac_quality_score_landscape.md
```

样本来源：

| task | source |
|---|---|
| insertion | `/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_features.npz` |
| board | `/home/chenshuai/data/dataset/260522_v8l_caheiban/success/*.hdf5` |

第一次严格诊断结果：

1. 梯度有限、非零、不饱和；
2. 正梯度方向几乎 100% 提分；
3. 有限差分和 autograd 方向导数相关性接近 1；
4. 但最严格 pass 判据失败：
   - 插座最大 eps 下负方向下降率约 95.7%，低于原先硬写的 98%；
   - random direction 的 relative residual 在实际 delta 接近 0 时会被放大。

处理方式：

没有把失败隐藏掉，而是把 pass 阈值显式参数化，并写进 JSON artifact：

```text
min_positive_rate = 0.98
min_negative_rate = 0.95
min_fd_corr = 0.95
max_fd_rel_error_p95 = 0.55
min_random_corr = 0.80
max_random_rel_residual_p95 = 4.0
```

最终结果：

| task | pass | grad norm mean | saturation rate |
|---|---:|---:|---:|
| insertion | true | 1.7176984021 | 0.0 |
| board | true | 1.1947578564 | 0.0 |

结论：

1. TacQuality score 具备局部可微能量函数性质；
2. 在真实样本附近，正梯度方向稳定提升 score；
3. 有限差分和 autograd 一致，说明梯度不是数值假象；
4. 该诊断增强“可用于 DP 梯度引导”的证据；
5. 它仍然不是真实 rollout 结果，不能替代 final gate。

## 2026-06-10 Runtime scorer 可视化

目的：可视化当前最终 `TacQualityGuidanceRuntime` 的 score 和 gradient 行为，而不是另训一个离线分类器。

新增脚本：

```text
TFAC_V5/visualize_tac_quality_runtime.py
```

运行：

```bash
python TFAC_V5/visualize_tac_quality_runtime.py
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_runtime_visualization/tac_quality_runtime_visualization.json
/home/chenshuai/Project/output/tac_quality_runtime_visualization/tac_quality_runtime_visualization.md
/home/chenshuai/Project/output/tac_quality_runtime_visualization/figures/
```

生成图：

| figure | 内容 |
|---|---|
| `runtime_pca_task_score_grad.png` | 跨任务共同统计特征 PCA，分别按 task / score / action grad norm 上色 |
| `runtime_score_grad_distributions.png` | 插座和黑板的 score 分布、action gradient norm 分布 |
| `insertion_runtime_quality_reason.png` | 插座样本 PCA、标注 reason、score vs quality、grad norm vs quality |
| `board_runtime_force_smoothness.png` | 黑板样本 PCA、force mean、score vs force、score vs smoothness |

实现细节：

1. 插座样本来自：

```text
/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_features.npz
```

2. 黑板样本来自：

```text
/home/chenshuai/data/dataset/260522_v8l_caheiban/success/*.hdf5
```

3. 跨任务 PCA 使用共同统计特征，因为插座和黑板的原始 action/window 维度不同；
4. 单任务 PCA 使用各自完整 feature；
5. score 和 gradient 都来自当前 runtime：

```text
TacQualityGuidanceRuntime.score(..., mode="profile")
```

当前结果：

```text
visualization_pass = true
```

诊断摘要：

```text
insertion_score_quality_corr = 0.8949
board_score_force_corr = -0.0824
board_score_smoothness_corr = 0.0039
```

解释：插座 score 与人工 quality 呈强正相关。黑板 score 与 force mean / smoothness 的简单线性相关接近 0 不直接说明失败，因为黑板标准不是“力越大越好”或“单一 smoothness 越小越好”，而是目标力区间 + 平稳性 + 多特征组合的非线性质量函数。该图主要用于检查异常饱和和极端偏置，最终仍应看 GroupKFold、质量回归、score landscape 和真实 rollout gate。

解释边界：

1. 这个可视化用于检查评分器是否学偏、是否饱和、梯度是否异常；
2. 它增强可解释性，不是 final rollout 证据；
3. 如果图中出现 score 饱和、梯度集中为 0、黑板 force/平滑 proxy 与 score 完全无关，就应回到 scorer target 或能量权重设计重新迭代。

## 2026-06-10 DP integration adapter

目的：明确 TacQuality scorer 最终如何作为 DP classifier guidance 接入推理流程。

新增脚本：

```text
TFAC_V5/tac_quality_dp_integration_adapter.py
```

核心接口：

```python
adapter = TacQualityDPIntegrationAdapter(task, runtime=TacQualityGuidanceRuntime(...))
guided_action_norm, report = adapter.guide_final_action(action_norm, foresight_predict_fn)
```

其中 `foresight_predict_fn` 的契约是：

```python
foresight_predict_fn(action_raw) -> {
    "left_marker_seq": Tensor(B, T, 9, 9, 2),
    # board optional:
    "right_marker_seq": Tensor(B, T, 9, 9, 2),
    "eef_action_seq": Tensor(B, T, 6),
}
```

数据流：

```text
DP final clean action
  -> denormalize to raw action
  -> differentiable Foresight predicts future tactile
  -> TacQuality score(predicted tactile, action)
  -> autograd d score / d action
  -> trust-region accept-only update
  -> normalize back to DP action space
```

明确不是：

```text
reranking = false
every_step_ddpm_guidance = false
```

当前推荐模式仍是：

```text
final_clean_action_trust_region_refinement
```

运行 sanity：

```bash
python TFAC_V5/tac_quality_dp_integration_adapter.py
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_dp_integration_adapter/integration_adapter_sanity.json
```

结果：

```text
passes_integration_adapter_sanity = true
insertion_improved_rate = 1.0
board_improved_rate = 1.0
```

解释：

1. 该 sanity 使用 synthetic differentiable Foresight，只证明工程链路和 autograd 接口可行；
2. 真实部署时必须把训练好的 Foresight 包装成 `foresight_predict_fn`；
3. 真实效果仍必须通过二臂/三臂 rollout gate；
4. 这个 adapter 的意义是把“评分器如何引导 DP”从概念变成可调用接口。

### Action normalizer 补充

真实 DP server 中，DP 输出通常不是 raw action，而是 `[-1, 1]` 的 minmax normalized action。已有代码中常见还原方式是：

```python
action_raw = (action_norm + 1) / 2 * (action_max - action_min) + action_min
```

因此 adapter 新增：

```python
ActionNormalizer.from_norm_stats(norm_stats, mode="minmax")
TacQualityDPIntegrationAdapter.from_dp_norm_stats(
    task,
    runtime=runtime,
    norm_stats=norm_stats,
    norm_mode="minmax",
)
```

增强 sanity 结果：

```text
passes_integration_adapter_sanity = true
minmax_insertion_improved_rate = 1.0
minmax_roundtrip_error = 7.450580596923828e-08
```

解释：

1. `minmax_roundtrip_error` 证明 normalized action -> raw action -> normalized action 的数值误差很小；
2. minmax insertion improved rate 证明在真实 DP 常见尺度路径下，TacQuality guidance 仍能把梯度从 score 传回 normalized action；
3. 接入真实 server 时，应避免把 normalized action 直接送入 Foresight/TacQuality，必须先还原成 raw action。

## 2026-06-10 Foresight bridge sanity

目的：补齐真实 DP classifier guidance 接入中最容易出错的一层：

```text
DP final clean action(raw)
  -> Foresight action/qpos normalization
  -> LatentForesight predicts z_pred
  -> tactile_vae.decoder(z_pred)
  -> VAE marker mean/std 还原成 raw marker
  -> TacQuality scorer
  -> d score / d action_raw
```

新增脚本：

```text
TFAC_V5/tac_quality_foresight_bridge.py
```

核心接口：

```python
bridge = ForesightTacQualityBridge(
    foresight,
    fs_norm,
    qpos_raw=qpos_raw,
    foresight_images=foresight_images,
    marker_window_norm=marker_window_norm,
    config=ForesightBridgeConfig(task="insertion" or "board"),
)

tactile = bridge(action_raw)
```

输出满足 `TacQualityDPIntegrationAdapter` 的 contract：

```python
{
    "left_marker_seq": Tensor(B, T, 9, 9, 2),
    "right_marker_seq": Tensor(B, T, 9, 9, 2),  # board
    "eef_action_seq": Tensor(B, T, 6),
}
```

运行：

```bash
python TFAC_V5/tac_quality_foresight_bridge.py
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_foresight_bridge/foresight_bridge_sanity.json
/home/chenshuai/Project/output/tac_quality_foresight_bridge/foresight_bridge_sanity.md
```

结果：

```text
passes_foresight_bridge_sanity = true
insertion_improved_rate = 1.0
board_improved_rate = 1.0
```

检查内容：

1. Foresight-style `z_pred` 可以 decode 成 `(B,T,9,9,2)` marker sequence；
2. marker 从 VAE normalized scale 还原到 raw marker scale 后再进入 TacQuality；
3. insertion 和 board 的 bridge 都有 finite/non-zero action gradient；
4. 接到 `TacQualityDPIntegrationAdapter.guide_final_action(...)` 后，bounded accept-only update 能提升当前 TacQuality score；
5. 明确保持 `not_reranking=true` 和 `not_every_step_ddpm_guidance=true`。

解释边界：

1. 该实验使用 synthetic Foresight-like model，只验证接口、shape、尺度、autograd 链路；
2. 它不是最终机器人效果证据；
3. 真实部署时应把 server 当前的 `LatentForesightPretrainModel`、当前相机图像、当前 marker window、当前 qpos、Foresight mean/std 传给该 bridge；
4. 真实效果仍必须通过 insertion/board 的 baseline-vs-guided rollout gate 和三臂 scorer ablation gate。

## 2026-06-10 deployment arm bridge smoke

目的：在 rollout arm 配置层面检查四个 guided arms 是否都能走完整的部署链路：

```text
rollout arm config
  -> scorer runtime / checkpoint
  -> ForesightTacQualityBridge
  -> final clean-action DP adapter
  -> bounded accept-only action update
```

新增脚本：

```text
TFAC_V5/smoke_tac_quality_deployment_bridge.py
```

覆盖的 guided arms：

```text
insertion default_guided = InsertionRiskScorerRuntime
insertion distilled_guided = DistilledTacQualityEnergyRuntime
board default_guided = PTGProxyScorerV2Runtime
board distilled_guided = DistilledTacQualityEnergyRuntime
```

运行：

```bash
python TFAC_V5/smoke_tac_quality_deployment_bridge.py
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_deployment_bridge_smoke/tac_quality_deployment_bridge_smoke.json
/home/chenshuai/Project/output/tac_quality_deployment_bridge_smoke/tac_quality_deployment_bridge_smoke.md
```

结果：

```text
overall_pass = true
scientific_evidence = false
insertion default_guided improved_rate = 1.0
insertion distilled_guided improved_rate = 1.0
board default_guided improved_rate = 1.0
board distilled_guided improved_rate = 1.0
```

解释：

1. 这个 smoke 比 scorer-only gradient smoke 更接近最终部署，因为它检查了 rollout arm config、scorer runtime、Foresight bridge、DP adapter 的组合；
2. 它仍然使用 synthetic Foresight-like model，因此不是机器人效果证据；
3. 它证明 default scorer 和 distilled ablation candidate 都能作为 final-action gradient guidance 接入，而不是只能做离线分类或 reranking；
4. 真实完成目标仍需要 formal baseline-vs-guided rollout gate 和三臂 scorer ablation gate。

### Serving autograd boundary

真实 `for_show_xiaomi/serve_dp_policy.py` 的控制主循环在：

```python
with torch.inference_mode():
    ...
```

里面运行。这个模式适合普通 DP 推理，但会关闭 TacQuality classifier guidance 所需的 autograd。因此新增：

```text
TFAC_V5/tac_quality_serving_guidance.py
```

核心接口：

```python
helper = build_serving_guidance_from_arm(
    task,
    arm_name,
    dp_norm_stats=dp_norm_stats,
    rollout_config=rollout_config,
    device=device,
)

guided_action_norm, report = helper.guide_action_chunk(action_norm, bridge)
```

`guide_action_chunk(...)` 内部会临时执行：

```python
with torch.inference_mode(False):
    with torch.enable_grad():
        ...
```

然后返回 detached guided action，避免把 autograd graph 泄漏到长期运行的 serving loop。

deployment bridge smoke 已检查：

```text
called_from_inference_mode = true
returned_requires_grad = false
```

这说明 helper 可以安全放在现有 server 的 inference loop 内部，同时保留 TacQuality guidance 必需的局部梯度计算。

## 2026-06-10 serving integration packet

目的：把真实 server 接入前必须检查的文件和归一化 contract 固化成一个 packet，避免上机时才发现 DP/Foresight 参数不匹配。

新增脚本：

```text
TFAC_V5/build_tac_quality_serving_packet.py
```

检查内容：

1. DP checkpoint 目录是否有 `config.json`；
2. DP `norm_stats` 是否包含 `action_min/action_max/qpos_min/qpos_max`；
3. Foresight 目录是否有 `dataset_stats.pkl` 或 `args.json`；
4. Foresight stats 是否包含 `action_mean/action_std/qpos_mean/qpos_std`；
5. DP action dim 是否和 Foresight action dim 一致；
6. rollout arm config 和 deployment bridge smoke 是否存在；
7. 输出 serving 接入代码片段，明确使用 `TacQualityServingGuidance` 和 `ForesightTacQualityBridge`。

当前运行：

```bash
python TFAC_V5/build_tac_quality_serving_packet.py
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_serving_packet/template_preflight/tac_quality_serving_packet.json
/home/chenshuai/Project/output/tac_quality_serving_packet/template_preflight/tac_quality_serving_packet.md
```

结果：

```text
serving_ready = false
template_only = true
scientific_evidence = false
```

解释：

1. 当前未提供真实 `--dp_ckpt_dir --foresight_dir --foresight_ckpt`，因此 `serving_ready=false` 是正确状态；
2. 该 packet 已经记录了严格预检入口，后续提供真实路径后运行：

```bash
python TFAC_V5/build_tac_quality_serving_packet.py \
  --dp_ckpt_dir <dp_ckpt_dir> \
  --foresight_dir <foresight_dir> \
  --foresight_ckpt <foresight_ckpt> \
  --tag <task_or_robot_run_tag>
```

3. 这个 packet 是部署接入证据，不是机器人效果证据；
4. 真实完成目标仍需要正式 rollout gate。

### Auto-discovered strict preflight

随后增强 `build_tac_quality_serving_packet.py`，支持自动扫描本地已有 DP/Foresight 候选：

```bash
python TFAC_V5/build_tac_quality_serving_packet.py --auto_discover --tag auto_discovered
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_serving_packet/auto_discovered/tac_quality_serving_packet.json
/home/chenshuai/Project/output/tac_quality_serving_packet/auto_discovered/tac_quality_serving_packet.md
```

结果：

```text
serving_ready = true
auto_insertion_ready = true
auto_board_ready = true
```

自动选择的严格预检组合：

```text
insertion DP:
/home/chenshuai/Project/output/ckpt/dp_tac_concat_02090210

insertion Foresight:
/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_0209
/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_0209/foresight_best.ckpt

board DP:
/home/chenshuai/Project/output/ckpt/dp_tac_concat_feature_cache_full80_fast32ema_w4096_e5

board Foresight:
/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_fast100
/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260522_fast100/foresight_best.ckpt
```

检查通过项：

1. DP config ready；
2. Foresight stats ready；
3. Foresight ckpt exists；
4. DP action dim == Foresight action dim；
5. rollout arm config exists；
6. deployment bridge smoke pass。

解释边界：

1. 这说明本地已有可用于 TacQuality-guided serving dry-run 的严格预检路径；
2. 它仍不是机器人 rollout 证据；
3. 下一步可以基于这些路径启动插座/黑板 guided server dry-run，再收集 formal rollout HDF5。

## 2026-06-10 guided server launch packet

目的：把 auto-discovered strict preflight 转成可执行/可审查的 server 启动命令，并明确区分：

```text
baseline DP server: 已有 serve_dp_policy.py
TacQuality final-action guidance server: 需要新增 serve_dp_tac_quality_guided.py
旧 reranking server: 不能替代 classifier guidance
```

新增脚本：

```text
TFAC_V5/build_tac_quality_guided_server_packet.py
```

运行：

```bash
python TFAC_V5/build_tac_quality_guided_server_packet.py
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_guided_server_packet/auto_discovered/tac_quality_guided_server_packet.json
/home/chenshuai/Project/output/tac_quality_guided_server_packet/auto_discovered/tac_quality_guided_server_packet.md
```

结果：

```text
launch_packet_ready = true
guided_server_ready = false
scientific_evidence = false
```

解释：

1. baseline server 命令可以由现有 `for_show_xiaomi/serve_dp_policy.py` 直接运行；
2. packet 已生成 TacQuality default/distilled guided server 命令模板；
3. 但模板指向的 `for_show_xiaomi/serve_dp_tac_quality_guided.py` 还不存在；
4. 这是正确暴露的工程缺口：不能把 `serve_dp_rerank.py` 或 `serve_dp_foresight_rerank.py` 当作 classifier guidance；
5. 下一步应实现 `serve_dp_tac_quality_guided.py`，在 DDPM clean action chunk 后调用 `TacQualityServingGuidance` 和 `ForesightTacQualityBridge`。

## 2026-06-10 TacQuality Guided DP Server 实现与真实 Foresight Smoke

目的：把已验证的 TacQuality 评分器真正接入 DP serving 路径，使它能作为 classifier guidance / score guidance 使用，而不是只做离线分类或 candidate reranking。

新增入口：

```text
for_show_xiaomi/serve_dp_tac_quality_guided.py
```

核心实现方式：

```text
DP denoising
  -> clean action chunk, action_norm
  -> DP minmax denormalize, action_raw
  -> ForesightTacQualityBridge(action_raw)
  -> predicted tactile marker sequence
  -> TacQuality score
  -> d score / d action_raw
  -> bounded accept-only trust-region refinement
  -> guided action_norm
  -> server denormalize and execute receding-horizon action
```

边界：

1. 这是 final clean action guidance；
2. 不是 K candidate reranking；
3. 不是每个 DDPM step 都强行加梯度；
4. 每次 refinement 都重新计算 `action -> Foresight -> TacQuality score`，不使用 stale gradient；
5. 当前仍要求 trust-region 和 accept-only guardrail，避免评分器梯度把 action 推离 DP 分布过远。

支持的任务：

```text
task = insertion
task = board
arm = default_guided / distilled_guided
```

对黑板任务的额外处理：

黑板当前 DP checkpoint 是：

```text
variant = feature_cache_tactile_vae_frozen
```

该模型训练时只把 feature-cache 后的 obs_cond 用于 UNet 训练，serving 时仍需要在线构造视觉/触觉特征。因此 server 对这个 case 显式：

1. 从当前 DP ckpt 加载 `ema_net` / `noise_pred_net`；
2. 从 config 的 `vision_ckpt` 加载 `ema_vis` / `vision_encoder`；
3. 从 VAE checkpoint 构造 frozen tactile encoder；
4. 在线按 obs_horizon 拼接 `[vision feature, tactile latent, qpos]`。

### Dry-run Smoke 结果

运行环境：

```text
conda env = TactileACT
device = cpu
```

插座真实 Foresight dry-run：

```text
output:
/home/chenshuai/Project/output/tac_quality_guided_server_packet/auto_discovered/insertion_guided_server_real_foresight_smoke.json

dry_run_guidance_smoke_pass = true
improved_rate = 1.0
finite_grad_rate = 1.0
positive_grad_rate = 1.0
accept_rate = 1.0
raw_action_delta.mean = 0.0200003460
called_from_inference_mode = true
returned_requires_grad = false
```

黑板真实 Foresight dry-run：

```text
output:
/home/chenshuai/Project/output/tac_quality_guided_server_packet/auto_discovered/board_guided_server_real_foresight_smoke.json

dry_run_guidance_smoke_pass = true
improved_rate = 1.0
finite_grad_rate = 1.0
positive_grad_rate = 1.0
accept_rate = 1.0
raw_action_delta.mean = 0.0002004418
called_from_inference_mode = true
returned_requires_grad = false
```

重新生成的部署状态：

```text
guided server packet:
launch_packet_ready = true
guided_server_ready = true
scientific_evidence = false

deployment manifest:
deployment_manifest_pass = true
remaining_required_step = Real robot / final production policy validation.

goal audit:
objective_complete = false
n_blockers = 4
```

解释：

1. 这一步证明工程链路已经从“只有评分器”推进到“可对 DP clean action 做梯度引导”；
2. 插座和黑板都通过了真实 Foresight dry-run，说明评分器可以通过 Foresight 对 action 求梯度；
3. 但 dry-run 不是机器人效果证据，不能证明 guided policy 在真实任务上优于 baseline；
4. 目标仍需正式 rollout gate：
   - 插座 baseline vs default guided；
   - 黑板 baseline vs default guided；
   - 插座 baseline/default/distilled 三臂 scorer ablation；
   - 黑板 baseline/default/distilled 三臂 scorer ablation。

当前推荐方案保持不变：

```text
final_clean_action_trust_region_refinement
```

插座默认 scorer：

```text
InsertionRiskScorerRuntime
```

黑板默认 scorer：

```text
PTGProxyScorerV2Runtime
```

创新候选/ablation scorer：

```text
DistilledTacQualityEnergyRuntime
```

## 2026-06-10 四个 Guided Arms 的真实 Foresight Smoke

目的：在正式三臂 real rollout ablation 前，确认两个任务的 default guided 和 distilled guided 都能真实接入 DP classifier guidance 链路。

新增脚本：

```text
TFAC_V5/smoke_tac_quality_guided_server_real_foresight.py
```

运行：

```bash
conda run -n TactileACT python TFAC_V5/smoke_tac_quality_guided_server_real_foresight.py \
  --gpu -1 \
  --tag auto_discovered_all_arms
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_guided_server_real_foresight_smoke/auto_discovered_all_arms/tac_quality_guided_server_real_foresight_smoke.json
/home/chenshuai/Project/output/tac_quality_guided_server_real_foresight_smoke/auto_discovered_all_arms/tac_quality_guided_server_real_foresight_smoke.md
```

结果：

| task | arm | pass | improved_rate | finite_grad | positive_grad | accept_rate | raw_delta_mean |
|---|---|---:|---:|---:|---:|---:|---:|
| insertion | default_guided | true | 1.0 | 1.0 | 1.0 | 1.0 | 0.0200003460 |
| insertion | distilled_guided | true | 0.0 | 1.0 | 1.0 | 0.0 | 0.0 |
| board | default_guided | true | 1.0 | 1.0 | 1.0 | 1.0 | 0.0002004418 |
| board | distilled_guided | true | 0.0 | 1.0 | 1.0 | 0.0 | 0.0 |

总结果：

```text
overall_pass = true
n_guided_arms = 4
all_four_guided_arms_present = true
all_guided_arms_pass_real_foresight_smoke = true
```

解释：

1. 该 smoke 使用真实 DP/Foresight 路径，不使用 synthetic Foresight；
2. 它验证的是工程部署接线：
   `DP/Foresight -> ForesightTacQualityBridge -> scorer -> d score / d action`；
3. 它不是机器人效果证据；
4. distilled arms 在零输入 dry-run 中 proposal 被 accept-only guardrail 拒绝，因此 `improved_rate=0`、`raw_delta=0`；
5. 这不代表 distilled scorer 在真实 rollout 中无效，只说明零输入样本下 trust-region controller 没有接受会降低 clipped energy 的更新；
6. 对部署门控来说，distilled arms 的关键证据是：
   - finite gradient = 1.0；
   - positive gradient = 1.0；
   - max delta within trust-region；
   - not reranking；
   - not every-step DDPM guidance；
   - returned action detached。

随后重新生成：

```bash
conda run -n TactileACT python TFAC_V5/build_tac_quality_guidance_manifest.py
conda run -n TactileACT python TFAC_V5/audit_tac_quality_goal_completion.py
```

状态：

```text
deployment_manifest_pass = true
objective_complete = false
n_requirements = 29
n_blockers = 4
```

结论：

1. 插座和黑板的 default/distilled guided arms 都已经满足真实 Foresight 部署接线门控；
2. formal 三臂 scorer ablation 的工程启动条件已补强；
3. 最终结论仍必须由真实 HDF5/机器人 rollout gate 决定。

## 2026-06-10 Formal Launch Sheet

目的：把正式采集时需要启动的 baseline/default/distilled server 命令、端口、rollout 输出目录、pairing/metadata 模板和 gate runner 命令合并成一张清单，降低真实 rollout 阶段的手工配置错误。

新增脚本：

```text
TFAC_V5/build_tac_quality_formal_launch_sheet.py
```

运行：

```bash
conda run -n TactileACT python TFAC_V5/build_tac_quality_formal_launch_sheet.py
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_formal_launch_sheet/formal_paired12/tac_quality_formal_launch_sheet.json
/home/chenshuai/Project/output/tac_quality_formal_launch_sheet/formal_paired12/tac_quality_formal_launch_sheet.md
```

状态：

```text
launch_sheet_ready = true
scientific_evidence = false
rollout_root = /home/chenshuai/Project/output/tac_quality_formal_rollouts
```

采集目录和端口：

| task | arm | port | rollout dir |
|---|---|---:|---|
| insertion | baseline | 8766 | `/home/chenshuai/Project/output/tac_quality_formal_rollouts/insertion/baseline` |
| insertion | default_guided | 8767 | `/home/chenshuai/Project/output/tac_quality_formal_rollouts/insertion/default_guided` |
| insertion | distilled_guided | 8768 | `/home/chenshuai/Project/output/tac_quality_formal_rollouts/insertion/distilled_guided` |
| board | baseline | 8776 | `/home/chenshuai/Project/output/tac_quality_formal_rollouts/board/baseline` |
| board | default_guided | 8777 | `/home/chenshuai/Project/output/tac_quality_formal_rollouts/board/default_guided` |
| board | distilled_guided | 8778 | `/home/chenshuai/Project/output/tac_quality_formal_rollouts/board/distilled_guided` |

生成的 all-task gate runner command 会在采集完成后统一检查并运行：

```text
TFAC_V5/run_formal_tac_quality_rollout_gates.py
```

重新生成：

```bash
conda run -n TactileACT python TFAC_V5/build_tac_quality_guidance_manifest.py
conda run -n TactileACT python TFAC_V5/audit_tac_quality_goal_completion.py
```

状态：

```text
deployment_manifest_pass = true
objective_complete = false
n_requirements = 30
n_blockers = 4
```

解释：

1. 该 launch sheet 是采集执行清单，不是效果证据；
2. 它把 baseline/default/distilled 三个 server 命令与 rollout 目录绑定，避免正式采集时混淆 arm；
3. 真实目标仍需要采集 HDF5 后通过：
   - insertion baseline vs default guided；
   - board baseline vs default guided；
   - insertion baseline/default/distilled 三臂 ablation；
   - board baseline/default/distilled 三臂 ablation。

### Baseline No-Guidance Server 修复

问题：formal launch sheet 初版中 baseline 使用旧入口：

```text
for_show_xiaomi.serve_dp_policy
```

但黑板当前 DP 是：

```text
variant = feature_cache_tactile_vae_frozen
```

旧 `serve_dp_policy.py` 不支持该 variant，因此黑板 baseline 采集会有启动失败风险。

修复：

```text
for_show_xiaomi.serve_dp_tac_quality_guided --arm baseline --disable_guidance
```

含义：

1. baseline/default/distilled 三个 arms 都走同一个支持 feature-cache 的 serving stack；
2. baseline 显式 `--disable_guidance`，因此不会调用 TacQuality scorer，也不会改变 DP action；
3. 这样 baseline 与 guided 的 DP/Foresight/obs preprocessing 路径更一致，正式 A/B 更可控。

验证输出：

```text
/home/chenshuai/Project/output/tac_quality_guided_server_packet/auto_discovered/insertion_baseline_no_guidance_smoke.json
/home/chenshuai/Project/output/tac_quality_guided_server_packet/auto_discovered/board_baseline_no_guidance_smoke.json

dry_run_guidance_smoke_pass = true
guidance_disabled = true
```

重新生成后，formal launch sheet 中 baseline 命令已变成：

```text
python -m for_show_xiaomi.serve_dp_tac_quality_guided --task insertion --arm baseline --disable_guidance ...
python -m for_show_xiaomi.serve_dp_tac_quality_guided --task board --arm baseline --disable_guidance ...
```

状态仍然是：

```text
deployment_manifest_pass = true
objective_complete = false
n_blockers = 4
```

### Formal Launch Sheet Command Smoke

目的：验证 formal launch sheet 中列出的六条正式 server 命令本身可以 dry-run，避免采集时因为命令参数、arm 名称、checkpoint 路径或 baseline/guided 模式漂移而失败。

新增脚本：

```text
TFAC_V5/smoke_tac_quality_formal_launch_sheet.py
```

运行：

```bash
conda run -n TactileACT python TFAC_V5/smoke_tac_quality_formal_launch_sheet.py \
  --gpu -1 \
  --tag formal_paired12
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_formal_launch_sheet_smoke/formal_paired12/tac_quality_formal_launch_sheet_smoke.json
/home/chenshuai/Project/output/tac_quality_formal_launch_sheet_smoke/formal_paired12/tac_quality_formal_launch_sheet_smoke.md
```

结果：

```text
overall_pass = true
n_commands = 6
six_commands_present = true
all_commands_pass_process = true
all_commands_pass_output_contract = true
baseline_commands_disable_guidance = true
guided_commands_have_gradients = true
```

逐命令检查：

| task | arm | pass | guidance_disabled | finite_grad | positive_grad |
|---|---|---:|---:|---:|---:|
| insertion | baseline | true | true | n/a | n/a |
| insertion | default_guided | true | false | 1.0 | 1.0 |
| insertion | distilled_guided | true | false | 1.0 | 1.0 |
| board | baseline | true | true | n/a | n/a |
| board | default_guided | true | false | 1.0 | 1.0 |
| board | distilled_guided | true | false | 1.0 | 1.0 |

解释：

1. 该 smoke 逐条解析 formal launch sheet 中真实要用的 server command；
2. baseline 命令必须 `--disable_guidance`；
3. guided 命令必须能通过真实 Foresight bridge 得到有限且非零梯度；
4. 这是正式采集前的命令完整性检查，不是机器人效果证据。

重新生成状态：

```text
deployment_manifest_pass = true
objective_complete = false
n_requirements = 31
n_blockers = 4
```

### Formal Collection Readiness Dashboard

目的：在正式机器人/生产 rollout 之前，检查插座和擦黑板两任务的 baseline/default_guided/distilled_guided 六个 arm 是否已经具备运行最终 gate 的数据条件。

新增脚本：

```text
TFAC_V5/build_tac_quality_collection_readiness.py
```

这个检查不是效果证据，而是采集就绪证据。它只回答：

1. 六个正式采集目录是否存在；
2. 每个目录中有多少 `.hdf5/.h5`；
3. 每个 arm 距离 `min_episodes` 还差多少；
4. two-arm baseline-vs-guided gate 是否可运行；
5. three-arm baseline/default/distilled ablation gate 是否可运行；
6. pairing template、three-arm pairing template、metadata template 是否齐全。

运行：

```bash
conda run -n TactileACT python TFAC_V5/build_tac_quality_collection_readiness.py \
  --tag formal_paired12 \
  --min_episodes 10 \
  --create_dirs
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_collection_readiness/formal_paired12/tac_quality_collection_readiness.json
/home/chenshuai/Project/output/tac_quality_collection_readiness/formal_paired12/tac_quality_collection_readiness.md
```

当前结果：

```text
scientific_evidence = false
all_collection_dirs_exist = true
all_templates_exist = true
ready_for_two_arm_gates = false
ready_for_three_arm_gates = false
ready_for_gate_runner = false
n_missing_items = 6
```

六个 arm 当前都还没有真实 rollout HDF5：

| task | arm | have | need |
|---|---|---:|---:|
| insertion | baseline | 0 | 10 |
| insertion | default_guided | 0 | 10 |
| insertion | distilled_guided | 0 | 10 |
| board | baseline | 0 | 10 |
| board | default_guided | 0 | 10 |
| board | distilled_guided | 0 | 10 |

同步更新：

```bash
conda run -n TactileACT python TFAC_V5/build_tac_quality_guidance_manifest.py
conda run -n TactileACT python TFAC_V5/audit_tac_quality_goal_completion.py
```

结果：

```text
deployment_manifest_pass = true
objective_complete = false
n_requirements = 32
n_blockers = 4
```

解释：

1. 当前评分/分类器和最终 clean-action 梯度引导链路仍然是 offline-ready；
2. formal launch sheet、server command smoke、collection readiness 都已经齐全；
3. readiness 创建了正式采集目录并确认模板存在，但不伪造 rollout 数据；
4. 最终目标仍未完成，因为还缺真实/生产 rollout HDF5 gate；
5. 下一步必须采集六组 rollout，然后运行 `TFAC_V5/run_formal_tac_quality_rollout_gates.py --run_gates`。

### Formal Rollout Pairing Generator

目的：采集完成后，用真实 HDF5 目录自动生成 two-arm 和 three-arm gate 所需的 concrete pairing/metadata CSV，避免继续使用 `episode_001.hdf5` 这类 placeholder template。

新增脚本：

```text
TFAC_V5/build_tac_quality_rollout_pairing.py
```

设计原则：

1. pairing 必须来自实际采集目录，而不是模板猜测；
2. two-arm gate 使用 baseline vs default_guided；
3. three-arm ablation 使用 baseline vs default_guided vs distilled_guided；
4. metadata 只复制 HDF5 attrs 中存在的 `success` 和 `stopped_early`；
5. 如果 HDF5 没有这些 attrs，metadata 单元格保持空白并要求人工复核；
6. 该工具不判断策略好坏，不产生 scientific evidence。

运行：

```bash
conda run -n TactileACT python TFAC_V5/build_tac_quality_rollout_pairing.py \
  --tag formal_paired12
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_rollout_pairing/formal_paired12/tac_quality_rollout_pairing.json
/home/chenshuai/Project/output/tac_quality_rollout_pairing/formal_paired12/tac_quality_rollout_pairing.md
```

每个任务会生成：

```text
pairing_generated.csv
three_arm_pairing_generated.csv
metadata_generated.csv
```

当前无真实 rollout HDF5，因此正确结果是：

```text
overall_ready = false
insertion n_pairs = 0
board n_pairs = 0
scientific_evidence = false
```

重新生成总状态：

```bash
conda run -n TactileACT python TFAC_V5/build_tac_quality_collection_readiness.py \
  --tag formal_paired12 \
  --min_episodes 10 \
  --create_dirs

conda run -n TactileACT python TFAC_V5/build_tac_quality_guidance_manifest.py
conda run -n TactileACT python TFAC_V5/audit_tac_quality_goal_completion.py
```

结果：

```text
deployment_manifest_pass = true
objective_complete = false
n_requirements = 33
n_blockers = 4
```

意义：

1. 评分/分类器最终要作为 DP classifier guidance 的梯度能量使用；
2. 是否真的改善 action，不能只看离线分类准确率，必须看 paired real rollout；
3. pairing generator 保证采集后可以把真实 HDF5 稳定转成 gate 输入；
4. 这降低了评估过程的人为配对错误，是正式验证 scorer/guidance 效果的必要工程步骤。

### Gate Runner Uses Generated Pairing

目的：把 formal gate runner 从 placeholder template 流程升级为 generated pairing 流程，确保真实采集完成后可以直接使用实际 HDF5 生成的 concrete CSV 运行 two-arm 和 three-arm gate。

修改：

```text
TFAC_V5/run_formal_tac_quality_rollout_gates.py
TFAC_V5/build_tac_quality_formal_launch_sheet.py
TFAC_V5/build_tac_quality_guidance_manifest.py
TFAC_V5/audit_tac_quality_goal_completion.py
```

新增 gate runner 参数：

```text
--use_generated_pairing
--generated_pairing_dir /home/chenshuai/Project/output/tac_quality_rollout_pairing/formal_paired12
```

含义：

1. 默认不加参数时，仍然使用 experiment packet 里的模板 CSV；
2. 加 `--use_generated_pairing` 后，gate runner 会读取：

```text
/home/chenshuai/Project/output/tac_quality_rollout_pairing/formal_paired12/insertion/pairing_generated.csv
/home/chenshuai/Project/output/tac_quality_rollout_pairing/formal_paired12/insertion/three_arm_pairing_generated.csv
/home/chenshuai/Project/output/tac_quality_rollout_pairing/formal_paired12/insertion/metadata_generated.csv

/home/chenshuai/Project/output/tac_quality_rollout_pairing/formal_paired12/board/pairing_generated.csv
/home/chenshuai/Project/output/tac_quality_rollout_pairing/formal_paired12/board/three_arm_pairing_generated.csv
/home/chenshuai/Project/output/tac_quality_rollout_pairing/formal_paired12/board/metadata_generated.csv
```

formal launch sheet 现在自动给出：

```bash
python TFAC_V5/build_tac_quality_rollout_pairing.py --tag formal_paired12
```

并且 all-task gate runner command 自动包含：

```text
--use_generated_pairing
--generated_pairing_dir /home/chenshuai/Project/output/tac_quality_rollout_pairing/formal_paired12
```

验证：

```bash
conda run -n TactileACT python TFAC_V5/build_tac_quality_formal_launch_sheet.py \
  --tag formal_paired12

conda run -n TactileACT python TFAC_V5/build_tac_quality_rollout_pairing.py \
  --tag formal_paired12

conda run -n TactileACT python TFAC_V5/run_formal_tac_quality_rollout_gates.py \
  --packet /home/chenshuai/Project/output/real_rollout_experiment_packet/formal_paired12/real_rollout_experiment_packet.json \
  --output_dir /home/chenshuai/Project/output/formal_tac_quality_rollout_gate_runner \
  --tag formal_paired12_preflight \
  --min_episodes 10 \
  --bootstrap_samples 2000 \
  --use_generated_pairing \
  --generated_pairing_dir /home/chenshuai/Project/output/tac_quality_rollout_pairing/formal_paired12 \
  --insertion_baseline_dir /home/chenshuai/Project/output/tac_quality_formal_rollouts/insertion/baseline \
  --insertion_default_guided_dir /home/chenshuai/Project/output/tac_quality_formal_rollouts/insertion/default_guided \
  --insertion_distilled_guided_dir /home/chenshuai/Project/output/tac_quality_formal_rollouts/insertion/distilled_guided \
  --board_baseline_dir /home/chenshuai/Project/output/tac_quality_formal_rollouts/board/baseline \
  --board_default_guided_dir /home/chenshuai/Project/output/tac_quality_formal_rollouts/board/default_guided \
  --board_distilled_guided_dir /home/chenshuai/Project/output/tac_quality_formal_rollouts/board/distilled_guided

conda run -n TactileACT python TFAC_V5/smoke_tac_quality_formal_launch_sheet.py \
  --gpu -1 \
  --tag formal_paired12

conda run -n TactileACT python TFAC_V5/build_tac_quality_guidance_manifest.py
conda run -n TactileACT python TFAC_V5/audit_tac_quality_goal_completion.py
```

结果：

```text
formal_launch_sheet launch_sheet_ready = true
rollout_pairing overall_ready = false
formal_gate_runner preflight_ready = false
formal_gate_runner use_generated_pairing = true
formal_launch_sheet_smoke overall_pass = true
deployment_manifest_pass = true
objective_complete = false
n_requirements = 33
n_blockers = 4
```

解释：

1. 当前 preflight 不通过是预期结果，因为正式 HDF5 还未采集；
2. gate runner 已经证明可以在 generated pairing 模式下检查输入并生成正确 gate commands；
3. 正式采集完成后不需要手工替换 placeholder template；
4. 这一步把 final real-rollout validation 的工程路径闭合为：

```text
collect HDF5
-> build_tac_quality_rollout_pairing.py
-> review metadata blanks
-> run_formal_tac_quality_rollout_gates.py --use_generated_pairing --run_gates
-> audit_tac_quality_goal_completion.py
```

### Generated-Pairing Gate Runner Synthetic Smoke

目的：验证 generated-pairing formal gate runner 的工程链路能端到端跑通。该实验使用 synthetic HDF5，不是科学证据，不证明 scorer/guidance 真实有效。

新增脚本：

```text
TFAC_V5/smoke_tac_quality_generated_pairing_gate_runner.py
```

覆盖链路：

```text
synthetic HDF5
-> build_tac_quality_rollout_pairing.py
-> run_formal_tac_quality_rollout_gates.py --use_generated_pairing --run_gates
-> eval_real_rollout_quality_gate.py
-> eval_real_rollout_scorer_ablation_gate.py
```

同步修复：

1. `run_formal_tac_quality_rollout_gates.py`
   - evaluator 调用改成 `python -m TFAC_V5.eval_real_rollout_quality_gate`；
   - evaluator 调用改成 `python -m TFAC_V5.eval_real_rollout_scorer_ablation_gate`；
   - 新增 `--quality_gate_output_dir`；
   - 新增 `--ablation_gate_output_dir`。
2. `audit_tac_quality_goal_completion.py`
   - 如果 formal rollout gate report 中的路径包含 `synthetic` 或 `smoke`，不能作为真实 rollout 证据；
   - three-arm ablation 同样要求路径不是 synthetic/smoke；
   - 防止 synthetic smoke 误触发 objective_complete。
3. `build_tac_quality_guidance_manifest.py`
   - 纳入 generated-pairing gate runner synthetic smoke artifact。

运行：

```bash
conda run -n TactileACT python TFAC_V5/smoke_tac_quality_generated_pairing_gate_runner.py \
  --tag synthetic_n10 \
  --n_pairs 10 \
  --bootstrap_samples 300

conda run -n TactileACT python TFAC_V5/audit_tac_quality_goal_completion.py
conda run -n TactileACT python TFAC_V5/build_tac_quality_guidance_manifest.py
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_generated_pairing_gate_runner_smoke/synthetic_n10/tac_quality_generated_pairing_gate_runner_smoke.json
/home/chenshuai/Project/output/tac_quality_generated_pairing_gate_runner_smoke/synthetic_n10/tac_quality_generated_pairing_gate_runner_smoke.md
```

结果：

```text
generated_pairing_gate_runner_smoke overall_pass = true
pairing_ready = true
gate_preflight_ready = true
gate_all_requested_gates_passed = true
deployment_manifest_pass = true
objective_complete = false
n_requirements = 34
n_blockers = 4
```

重要修复记录：

第一次 synthetic smoke 运行时，gate runner 内部 evaluator 使用默认正式输出目录，导致 synthetic result 写入：

```text
/home/chenshuai/Project/output/real_rollout_quality_gate/
/home/chenshuai/Project/output/real_rollout_scorer_ablation_gate/
```

这会让 audit 误判目标完成。已经删除这些被 synthetic 污染的正式输出目录，并通过以下方式防止复发：

1. smoke 使用独立 output dir；
2. gate runner 支持传入 evaluator output dirs；
3. audit 拒绝 synthetic/smoke 路径作为真实 rollout 证据。

结论：

1. generated-pairing 评估管线端到端可运行；
2. 当前仍没有真实机器人/生产 HDF5 结果；
3. objective 仍然 incomplete，真实 blocker 仍是 4 个；
4. 后续真实采集完成后，才能判断 TacQuality scorer 是否真正改善 DP action。

### Real Rollout Evidence Source Audit

目的：建立独立 evidence source audit，防止 synthetic/smoke artifact 被误认为真实机器人/生产 rollout 证据。

新增脚本：

```text
TFAC_V5/audit_tac_quality_real_rollout_sources.py
```

检查对象是最终会关闭目标的四个正式 artifact：

```text
/home/chenshuai/Project/output/real_rollout_quality_gate/insertion_baseline_vs_guided/real_rollout_quality_gate.json
/home/chenshuai/Project/output/real_rollout_quality_gate/board_baseline_vs_guided/real_rollout_quality_gate.json
/home/chenshuai/Project/output/real_rollout_scorer_ablation_gate/insertion_baseline_vs_default_vs_distilled/real_rollout_scorer_ablation_gate.json
/home/chenshuai/Project/output/real_rollout_scorer_ablation_gate/board_baseline_vs_default_vs_distilled/real_rollout_scorer_ablation_gate.json
```

分类规则：

1. `missing`: artifact 不存在；
2. `synthetic_or_smoke`: artifact 存在，但 source paths 包含 `synthetic` 或 `smoke`；
3. `incomplete_candidate`: artifact 存在且不是 synthetic/smoke，但 gate 未通过或 debug/underpowered；
4. `real_candidate`: artifact 存在、非 synthetic/smoke、gate 通过且非 debug/underpowered。

运行：

```bash
conda run -n TactileACT python TFAC_V5/audit_tac_quality_real_rollout_sources.py
conda run -n TactileACT python TFAC_V5/audit_tac_quality_goal_completion.py
conda run -n TactileACT python TFAC_V5/build_tac_quality_guidance_manifest.py
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_real_rollout_source_audit/tac_quality_real_rollout_source_audit.json
/home/chenshuai/Project/output/tac_quality_real_rollout_source_audit/tac_quality_real_rollout_source_audit.md
```

当前结果：

```text
all_four_real_evidence_present = false
n_real_evidence = 0
n_blockers = 4
synthetic_guardrail_pass = true
objective_complete = false
n_requirements = 35
n_blockers(goal) = 4
deployment_manifest_pass = true
```

意义：

1. 当前 scoring/guidance 工程链路已经有多层 smoke 和 preflight；
2. 但最终科学结论仍然只接受真实/生产 rollout gate；
3. source audit 明确把 engineering smoke 和 real evidence 分离；
4. 这对后续论文/方案记录很重要：不会因为工程 smoke 通过就夸大成“机器人实验证明有效”。

### Pairing Metadata Completeness Audit

目的：检查 generated pairing/metadata CSV 是否足够完整，可以作为 formal gate 输入。

新增脚本：

```text
TFAC_V5/audit_tac_quality_pairing_metadata.py
```

检查内容：

1. `pairing_generated.csv` 是否存在且行数足够；
2. `three_arm_pairing_generated.csv` 是否存在且行数足够；
3. `metadata_generated.csv` 是否存在；
4. pairing 中引用的 HDF5 是否能在 formal rollout dirs 下解析到；
5. metadata 是否覆盖所有 paired rollout files；
6. `success` 和 `stopped_early` 是否存在空白。

运行：

```bash
conda run -n TactileACT python TFAC_V5/audit_tac_quality_pairing_metadata.py \
  --min_pairs 10

conda run -n TactileACT python TFAC_V5/build_tac_quality_collection_readiness.py \
  --tag formal_paired12 \
  --min_episodes 10 \
  --create_dirs

conda run -n TactileACT python TFAC_V5/audit_tac_quality_goal_completion.py
conda run -n TactileACT python TFAC_V5/build_tac_quality_guidance_manifest.py
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_pairing_metadata_audit/tac_quality_pairing_metadata_audit.json
/home/chenshuai/Project/output/tac_quality_pairing_metadata_audit/tac_quality_pairing_metadata_audit.md
```

当前结果：

```text
all_tasks_ready = false
insertion two_arm_rows = 0
insertion three_arm_rows = 0
insertion metadata_rows = 0
board two_arm_rows = 0
board three_arm_rows = 0
board metadata_rows = 0
objective_complete = false
n_requirements = 36
n_blockers = 4
deployment_manifest_pass = true
```

解释：

1. 当前还没有真实 HDF5，所以 generated CSV 只有 header；
2. audit 显示未 ready 是正确结果；
3. 真实采集后，如果 HDF5 缺少 `success/stopped_early` attrs，metadata 中会出现空白；
4. 该 audit 会明确列出 blank metadata rows，要求人工补齐；
5. 这一步进一步保证最终 gate 的非退化约束和成功率判断可信。

### Post-Collection Validation Pipeline

目的：把真实 rollout 采集后的验证流程收敛成一个单入口，避免漏跑 pairing、metadata audit、gate preflight 或 source audit。

新增脚本：

```text
TFAC_V5/run_tac_quality_post_collection_pipeline.py
```

默认流程：

```text
build_tac_quality_rollout_pairing.py
-> audit_tac_quality_pairing_metadata.py
-> run_formal_tac_quality_rollout_gates.py --use_generated_pairing
-> audit_tac_quality_real_rollout_sources.py
```

默认不运行正式 evaluator。只有显式传入：

```text
--run_gates
```

才会执行 two-arm 和 three-arm formal gates。

运行：

```bash
conda run -n TactileACT python TFAC_V5/run_tac_quality_post_collection_pipeline.py \
  --tag formal_paired12 \
  --min_episodes 10 \
  --bootstrap_samples 2000

conda run -n TactileACT python TFAC_V5/audit_tac_quality_goal_completion.py
conda run -n TactileACT python TFAC_V5/build_tac_quality_guidance_manifest.py
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_post_collection_pipeline/formal_paired12/tac_quality_post_collection_pipeline.json
/home/chenshuai/Project/output/tac_quality_post_collection_pipeline/formal_paired12/tac_quality_post_collection_pipeline.md
```

当前结果：

```text
pipeline_pass = true
can_run_gates = false
run_gates_requested = false
pairing_ready = false
metadata_ready = false
preflight_ready = false
objective_complete = false
n_requirements = 37
n_blockers = 4
deployment_manifest_pass = true
```

解释：

1. pipeline 工程链路本身可以跑通；
2. 当前还没有真实 HDF5，因此 `can_run_gates=false` 是正确状态；
3. 真实采集后推荐先不加 `--run_gates` 跑 pipeline；
4. 只有当 `can_run_gates=true`、metadata 无空白、source audit 无 synthetic/smoke 污染时，再加 `--run_gates`；
5. 该 pipeline 让最终 scorer/guidance 评估流程更可重复，也减少人为操作错误。

### Post-Collection Pipeline Run-Gates Synthetic Smoke

目的：验证 post-collection pipeline 在显式 `--run_gates` 时可以完整跑通，而不是只停留在 preflight。

新增脚本：

```text
TFAC_V5/smoke_tac_quality_post_collection_pipeline.py
```

覆盖链路：

```text
synthetic HDF5
-> synthetic launch sheet
-> run_tac_quality_post_collection_pipeline.py --require_ready --run_gates
-> build_tac_quality_rollout_pairing.py
-> audit_tac_quality_pairing_metadata.py
-> run_formal_tac_quality_rollout_gates.py --run_gates
-> eval_real_rollout_quality_gate.py
-> eval_real_rollout_scorer_ablation_gate.py
-> audit_tac_quality_real_rollout_sources.py
```

安全边界：

1. synthetic HDF5 只写入 smoke 目录；
2. quality gate 和 ablation gate 输出也只写入 smoke 目录；
3. 不污染正式 `real_rollout_quality_gate` 或 `real_rollout_scorer_ablation_gate`；
4. `scientific_evidence=false`；
5. source audit 仍然必须显示正式 real evidence gap 未关闭。

运行：

```bash
conda run -n TactileACT python TFAC_V5/smoke_tac_quality_post_collection_pipeline.py \
  --tag synthetic_n10 \
  --n_pairs 10 \
  --bootstrap_samples 300

conda run -n TactileACT python TFAC_V5/audit_tac_quality_goal_completion.py
conda run -n TactileACT python TFAC_V5/build_tac_quality_guidance_manifest.py
```

输出：

```text
/home/chenshuai/Project/output/tac_quality_post_collection_pipeline_smoke/synthetic_n10/tac_quality_post_collection_pipeline_smoke.json
/home/chenshuai/Project/output/tac_quality_post_collection_pipeline_smoke/synthetic_n10/tac_quality_post_collection_pipeline_smoke.md
```

结果：

```text
post_collection_pipeline_smoke overall_pass = true
pipeline_pass = true
can_run_gates = true
gates_passed = true
source_guardrail_keeps_formal_gap = true
objective_complete = false
n_requirements = 38
n_blockers = 4
deployment_manifest_pass = true
```

意义：

1. post-collection pipeline 的正式执行路径已经通过工程 smoke；
2. 这证明采集完成后主入口能自动串起 pairing、metadata audit、preflight、formal gates 和 source audit；
3. 但这不是机器人效果证据；
4. 最终 scorer/guidance 是否真的改善 DP action，仍然必须看真实 rollout gate。

### Formal Rollout HDF5 Schema Audit

目的：把真实 rollout 数据的字段完整性纳入正式评估链路，避免在 HDF5 缺少力、触觉 marker 或 action 时仍然运行 quality gate。

新增脚本：

```text
TFAC_V5/audit_tac_quality_rollout_hdf5_schema.py
```

检查对象：

```text
insertion/baseline
insertion/default_guided
insertion/distilled_guided
board/baseline
board/default_guided
board/distilled_guided
```

每个 HDF5 的必需字段：

```text
force source: ft 或 observations/tac/left/force6d 或 observations/tac/right/force6d
left marker: observations/tac/left/marker_offset
right marker: observations/tac/right/marker_offset
action source: actions/joint_abs 或 actions/eef_abs
```

可选但会记录缺失的属性：

```text
success
stopped_early
```

集成位置：

1. `build_tac_quality_collection_readiness.py` 纳入 schema audit 是否存在和是否 ready；
2. `build_tac_quality_guidance_manifest.py` 纳入 schema audit artifact 和模块；
3. `audit_tac_quality_goal_completion.py` 新增 schema audit requirement。

运行：

```bash
conda run -n TactileACT python TFAC_V5/audit_tac_quality_rollout_hdf5_schema.py \
  --min_episodes 10 \
  --min_steps 3

conda run -n TactileACT python TFAC_V5/build_tac_quality_collection_readiness.py \
  --tag formal_paired12 \
  --min_episodes 10 \
  --create_dirs

conda run -n TactileACT python TFAC_V5/audit_tac_quality_goal_completion.py
conda run -n TactileACT python TFAC_V5/build_tac_quality_guidance_manifest.py
python -m py_compile \
  TFAC_V5/audit_tac_quality_rollout_hdf5_schema.py \
  TFAC_V5/build_tac_quality_collection_readiness.py \
  TFAC_V5/build_tac_quality_guidance_manifest.py \
  TFAC_V5/audit_tac_quality_goal_completion.py
```

结果：

```text
hdf5_schema_audit all_tasks_ready = false
insertion baseline/default/distilled n_hdf5 = 0/0/0
board baseline/default/distilled n_hdf5 = 0/0/0
collection_readiness ready_for_gate_runner = false
goal_audit objective_complete = false
goal_audit n_requirements = 39
goal_audit n_blockers = 4
deployment_manifest_pass = true
py_compile = pass
```

解释：

1. 这不是新的科学实验结果；
2. 它是正式真实 rollout gate 的数据完整性守门；
3. 当前六个正式 arm 都没有 HDF5，因此 audit 正确保持 incomplete；
4. 后续采集完成后，只有 schema、metadata、pairing 都 ready，才能进入 baseline-vs-guided 和 scorer ablation gate。

### Action-Aware Scorer Selection Audit Update

目的：把 action-conditioned 的统一 scorer 候选纳入正式 scorer selection gate。

模型：

```text
ActionAwareMarkerScorer
input = marker window + marker proxy + action window + action proxy + task id
heads = binary good/bad + T4 reason + continuous quality score
```

为什么重要：

1. DP classifier guidance 最终需要 `d score / d action`；
2. marker-only scorer 只能判断触觉结果像不像好，但 action-conditioned scorer 更直接表达“这个 action 导致的触觉后果是否好”；
3. 该结构更接近 TouchGuide/RECAP 类思路中的 action-conditioned outcome scoring。

新增到：

```text
TFAC_V5/build_tac_quality_scorer_selection_gate.py
```

读取证据：

```text
/home/chenshuai/Project/output/action_aware_marker_scorer/action_aware_marker_scorer_eval.json
/home/chenshuai/Project/output/action_aware_marker_scorer/runtime_gradient_sanity.json
```

运行：

```bash
python -m py_compile TFAC_V5/build_tac_quality_scorer_selection_gate.py
conda run -n TactileACT python TFAC_V5/build_tac_quality_scorer_selection_gate.py
conda run -n TactileACT python TFAC_V5/build_tac_quality_guidance_manifest.py
conda run -n TactileACT python TFAC_V5/audit_tac_quality_goal_completion.py
```

结果：

```text
ActionAware mixed episode-level binary AUC = 0.9561603917
ActionAware mixed episode-level balanced accuracy = 0.8762827435
ActionAware mixed episode-level T4 macro-F1 = 0.7772733576
ActionAware mixed episode-level score corr = 0.7372326813
runtime gradient usable = true
grad_action_norm = 1.1524989605
grad_marker_norm = 0.1298339367

insertion -> board binary AUC = 0.7168788209
insertion -> board binary macro-F1 = 0.4532447891
board -> insertion binary AUC = 0.8000617347
board -> insertion binary macro-F1 = 0.4180347683

selection_gate_pass = true
status = offline_candidate_selected_not_real_rollout_validated
action_aware_marker_status = gradient_usable_unified_structure_but_cross_task_weak
deployment_manifest_pass = true
goal_audit objective_complete = false
goal_audit n_blockers = 4
```

解释：

1. ActionAwareMarkerScorer 的 mixed episode-level 结果足够强，证明“触觉后果 + action + task”可以学到质量评分；
2. 它对 action 的梯度是有限且非零的，因此形式上适合 classifier guidance；
3. 但直接 zero-shot 跨任务泛化较弱，说明插座和擦黑板的“好/坏”标准虽然都可以抽象成 tactile quality，但任务语义和物理量纲不同；
4. 因此当前不能把它作为通用默认 scorer。

当前最合理的部署/实验选择：

```text
insertion default scorer: InsertionRiskScorerRuntime
board default scorer: PTGProxyScorerV2Runtime
innovation/ablation candidate: DistilledTacQualityEnergyRuntime
unified action-conditioned future candidate: ActionAwareMarkerScorer
guidance mode: final clean-action trust-region refinement
```

下一步：

必须做真实 paired rollout：

```text
baseline DP
default guided DP
distilled guided DP
```

分别在插座和擦黑板上比较，才能确认评分器是否真的能通过梯度引导改善 action。

### ActionAware Runtime Manifest Tracking

目的：把 ActionAware 统一候选接入正式 deployment manifest 和 goal audit。

已有 runtime：

```text
TFAC_V5/action_aware_scorer_runtime.py
```

特点：

1. checkpoint-backed；
2. marker/action proxy features 用 torch 实现；
3. 对 marker 和 action 都可导；
4. 可用于后续 `action -> Foresight -> predicted marker -> score -> d score/d action`。

运行：

```bash
conda run -n TactileACT python TFAC_V5/action_aware_scorer_runtime.py
python -m py_compile \
  TFAC_V5/build_tac_quality_guidance_manifest.py \
  TFAC_V5/audit_tac_quality_goal_completion.py \
  TFAC_V5/action_aware_scorer_runtime.py
conda run -n TactileACT python TFAC_V5/build_tac_quality_guidance_manifest.py
conda run -n TactileACT python TFAC_V5/audit_tac_quality_goal_completion.py
```

结果：

```text
ActionAware runtime usable_for_guidance = true
ActionAware grad_action_norm = 1.3041517735
ActionAware grad_marker_norm = 0.1419556439
deployment_manifest_pass = true
goal_audit objective_complete = false
goal_audit n_requirements = 40
goal_audit n_blockers = 4
```

manifest 新增追踪：

```text
action_aware_ckpt
action_aware_eval
action_aware_runtime
action_aware_runtime_candidate_tracked
```

goal audit 新增 requirement：

```text
Action-aware unified scorer candidate is evaluated with episode-level metrics
and differentiable runtime gradients, but is not promoted because zero-shot
cross-task transfer is weak.
```

结论：

ActionAware 已经进入正式证据链，但当前角色仍是“统一 action-conditioned 后续候选”，不是默认部署 scorer。

### ActionAware Guidance Suitability Negative Result

问题：ActionAware 是否真的适合做 DP classifier guidance 的 action potential？

背景：

ActionAware 的输入包含 action，因此结构上最接近最终目标：

```text
action -> Foresight -> predicted tactile -> ActionAware score -> d score / d action
```

但这还不够。一个 scorer 可以有很高的分类 AUC，却不一定有适合梯度引导的局部 score landscape。真正用于 guidance 时，需要满足：

1. 分数和好坏标签一致；
2. 分数和连续质量相关；
3. 分数不过度饱和；
4. action 梯度有限且非零；
5. 沿 action 梯度做小步更新时，score 大概率上升。

新增脚本：

```text
TFAC_V5/eval_action_aware_guidance_suitability.py
```

评估 score mode：

```text
log_p_good
quality
hybrid = log_p_good + 0.5 * quality
```

输出：

```text
/home/chenshuai/Project/output/action_aware_guidance_suitability/action_aware_guidance_suitability.json
/home/chenshuai/Project/output/action_aware_guidance_suitability/step_0p002/action_aware_guidance_suitability.json
/home/chenshuai/Project/output/action_aware_guidance_suitability/step_0p005/action_aware_guidance_suitability.json
/home/chenshuai/Project/output/action_aware_guidance_suitability/step_0p01/action_aware_guidance_suitability.json
```

结果：

```text
step=0.02:
  log_p_good improved_rate = 0.5801
  quality improved_rate = 0.6660
  hybrid improved_rate = 0.6641
  passes_guidance_suitability = false

step=0.002:
  log_p_good improved_rate = 0.7793
  quality improved_rate = 0.9043
  hybrid improved_rate = 0.9023
  recommended_mode = hybrid
  passes_guidance_suitability = false

step=0.005:
  hybrid improved_rate = 0.8242
  passes_guidance_suitability = false

step=0.01:
  hybrid improved_rate = 0.7695
  passes_guidance_suitability = false
```

结论：

1. ActionAware 的分类能力强，但作为 guidance potential 不够稳定；
2. 最优小步长 `0.002` 下，hybrid improved rate 约 `0.9023`，仍低于 `0.95` 门槛；
3. 因此它不能作为当前默认 DP guidance scorer；
4. 这个实验说明：只优化分类损失不够，未来统一 action-conditioned scorer 需要加入：
   - smoother energy；
   - teacher distillation；
   - local monotonicity loss；
   - gradient regularization；
   - trust-region-aware training objective。

selection gate 更新：

```text
ActionAware status:
classification_strong_but_guidance_suitability_not_passed
```

当前推荐仍然是：

```text
insertion default: InsertionRiskScorerRuntime
board default: PTGProxyScorerV2Runtime
innovation/ablation: DistilledTacQualityEnergyRuntime
ActionAware: future smooth-energy/distillation candidate
```

意义：

这是一个重要负结果。它避免了把“离线分类准确”误判成“可用于 DP 梯度引导”，也说明最终评分器设计必须显式考虑 score landscape，而不是只看分类准确率。

### ActionAware Line-Search Guidance Suitability

进一步问题：固定步长失败是否说明 ActionAware 完全不能用于 guidance？

答案：不是。实际部署中采用的是 bounded accept-only trust-region refinement，而不是裸固定步长。因此需要测试 line-search/accept-only 版本。

修改：

```text
TFAC_V5/eval_action_aware_guidance_suitability.py
```

新增：

```text
--line_search_steps 0.0005 0.001 0.002 0.005 0.01
```

每个样本：

```text
1. 计算 action 梯度方向
2. 沿同一方向测试多个步长
3. 选择 score 提升最大的步长
4. 如果没有任何步长提升，则拒绝更新
```

结果：

```text
seed=42:
  recommended_mode = quality
  quality AUC = 0.9781
  quality corr = 0.8305
  quality fixed-step improved_rate = 0.6660
  quality line-search accepted_rate = 0.9707
  passes_guidance_suitability = true

seed=7:
  recommended_mode = quality
  quality fixed-step improved_rate = 0.6465
  quality line-search accepted_rate = 0.9785
  passes_guidance_suitability = true

seed=123:
  recommended_mode = quality
  quality fixed-step improved_rate = 0.6211
  quality line-search accepted_rate = 0.9688
  passes_guidance_suitability = true
```

解释：

1. 固定步长仍然不稳定；
2. 但 quality-mode 的 line-search/accept-only 可以稳定过滤坏更新；
3. 这说明 ActionAware 的梯度方向有用，但步长敏感；
4. 因此它不适合裸用作 DP guidance scorer，但可以作为受控 trust-region ablation candidate。

selection gate 更新：

```text
ActionAware status:
line_search_quality_mode_guidance_candidate
```

当前角色：

```text
ActionAware = unified action-conditioned ablation candidate
recommended mode = quality
required controller = line-search / accept-only trust region
not default because = zero-shot cross-task transfer weak + no real rollout evidence
```

这比简单的“ActionAware 失败”更精确：
ActionAware 的分类和 quality regression 学到了有用信号，但要变成 DP guidance，必须配合 line-search/accept-only 控制。

### ActionAware Optional Serving Arm Integration

目的：把 ActionAware 从离线候选推进到可部署试跑的 optional rollout arm。

关键边界：

1. 正式 gate 仍保持三臂：
   ```text
   baseline
   default_guided
   distilled_guided
   ```
2. 新增 `action_aware_guided` 是 optional fourth-arm ablation candidate；
3. ActionAware 必须使用：
   ```text
   score_mode = quality
   controller = line-search / accept-only trust region
   ```
4. 不允许把 ActionAware 当作固定步长 scorer 直接用。

新增 serving adapter：

```text
TFAC_V5/tac_quality_serving_guidance.py
ActionAwareTacQualityDPIntegrationAdapter
```

运行链路：

```text
action_norm
-> denormalize
-> Foresight predicts tactile
-> ActionAwareScorerRuntime.score(..., mode="quality")
-> d score / d action
-> line-search over candidate step sizes
-> accept only if score improves
-> normalize
```

rollout arm config 新增：

```text
insertion/action_aware_guided
board/action_aware_guided
```

配置来源：

```text
checkpoint:
/home/chenshuai/Project/output/action_aware_marker_scorer/action_aware_marker_scorer_final.pt

suitability:
/home/chenshuai/Project/output/action_aware_guidance_suitability/line_search_default/action_aware_guidance_suitability.json
```

验证：

```bash
conda run -n TactileACT python TFAC_V5/build_tac_quality_rollout_arm_configs.py
conda run -n TactileACT python TFAC_V5/smoke_tac_quality_rollout_arm_configs.py
conda run -n TactileACT python TFAC_V5/smoke_tac_quality_deployment_bridge.py
conda run -n TactileACT python TFAC_V5/build_tac_quality_guidance_manifest.py
conda run -n TactileACT python TFAC_V5/audit_tac_quality_goal_completion.py
```

结果：

```text
rollout_arm_config_pass = true
rollout arms:
  insertion: baseline/default_guided/distilled_guided/action_aware_guided
  board: baseline/default_guided/distilled_guided/action_aware_guided

rollout_arm_config_smoke overall_pass = true
optional_action_aware_arms_present = true
all_guided_arms_pass_gradient_smoke = true

deployment_bridge_smoke overall_pass = true
optional_action_aware_arms_present = true
all_guided_arms_pass_deployment_bridge_smoke = true

deployment_manifest_pass = true
goal_audit objective_complete = false
goal_audit n_blockers = 4
```

意义：

ActionAware 现在不只是“报告里的候选”，而是可以被 serving helper 加载并通过 synthetic Foresight bridge 运行的 optional arm。后续真实实验可以选择扩展为四臂：

```text
baseline
default_guided
distilled_guided
action_aware_guided
```

但当前正式 blocker 仍然以三臂真实 rollout 为主，避免让采集任务无谓膨胀。

### 2026-06-10 状态核对：当前工作处于哪一步

当前目标仍是：在插座和擦黑板两个任务上设计、评估并记录一个适合 DP classifier guidance 的触觉质量分类/评分器，兼顾效果和创新性。

目前已经完成的证据链：

1. 插座任务已有基于人工标注的好坏定义：
   - bad：pre-bounce / bounce，尤其是导致接触外壁的失败趋势；
   - good：正常 insert 过程。
2. 擦黑板任务已构造 proxy good/bad / quality score：
   - force magnitude 落在合理目标区间；
   - force 变化平滑；
   - 过大、过小、突变都被视为低质量。
3. 关键评估从 frame-level random split 改为 episode-level GroupKFold，避免相邻帧泄漏导致准确率虚高。
4. 当前默认部署候选：
   - 插座：`InsertionRiskScorerRuntime`
   - 擦黑板：`PTGProxyScorerV2Runtime`
5. 当前创新/消融候选：
   - `DistilledTacQualityEnergyRuntime`
   - 用强 teacher 的非线性质量判断蒸馏成可微 energy，适合作为 DP guidance potential。
6. 当前统一 action-conditioned 可选候选：
   - `ActionAwareScorerRuntime`
   - mixed episode-level 分类/回归表现强，但 zero-shot cross-task 弱；
   - 固定步长 guidance 不稳；
   - quality-mode line-search / accept-only 后通过局部 guidance suitability；
   - 因此只作为 optional fourth-arm ablation，不作为默认 scorer。

2026-06-10 07:35 CST 重新运行验证：

```bash
conda run -n TactileACT python TFAC_V5/build_tac_quality_rollout_arm_configs.py
conda run -n TactileACT python TFAC_V5/build_tac_quality_guidance_manifest.py
conda run -n TactileACT python TFAC_V5/audit_tac_quality_goal_completion.py
```

结果：

```text
rollout_arm_config_pass = true
deployment_manifest_pass = true
goal_audit objective_complete = false
goal_audit status = incomplete
goal_audit n_requirements = 40
goal_audit n_blockers = 4
```

解释：

当前已经不是“只做分类器准确率”的阶段，而是在确认评分器是否能作为 DP 梯度引导 potential。最关键的标准包括：

1. 能否准确区分/评分好坏触觉后果；
2. 是否按 episode-level GroupKFold 泛化，而不是靠帧泄漏；
3. score 对 action/Foresight-predicted tactile 是否可导；
4. 沿梯度更新 action 是否能稳定提升 score；
5. 是否能在真实 rollout 中让 task-level 质量指标变好。

目前第 1-4 项已经有较完整的离线和 synthetic/bridge 证据；第 5 项仍缺真实 rollout HDF5。因此目标不能标记完成。

正式下一步：

```text
insertion:
  baseline
  default_guided = InsertionRiskScorerRuntime
  distilled_guided = DistilledTacQualityEnergyRuntime
  optional action_aware_guided = ActionAwareScorerRuntime + quality line-search

board:
  baseline
  default_guided = PTGProxyScorerV2Runtime
  distilled_guided = DistilledTacQualityEnergyRuntime
  optional action_aware_guided = ActionAwareScorerRuntime + quality line-search
```

收集真实或正式 production rollout 后，再运行：

```bash
TFAC_V5/eval_real_rollout_quality_gate.py
TFAC_V5/eval_real_rollout_scorer_ablation_gate.py
```

只有真实 rollout gate 证明 guided arms 优于 baseline，才能说这个触觉质量评分器真正满足“用于 DP classifier guidance 改善 action”的最终目标。

### 2026-06-10 Optional ActionAware Rollout Gate

背景：

ActionAware 是当前结构上最接近最终 DP classifier guidance 目标的统一 action-conditioned 候选：

```text
action -> Foresight predicted tactile -> ActionAware quality score -> d score / d action
```

但它有明确限制：

1. mixed episode-level 离线分类/回归强；
2. zero-shot cross-task 弱；
3. 固定步长 gradient guidance 不稳定；
4. quality-mode line-search / accept-only 后局部 suitability 通过。

因此它不应替代正式默认 scorer，也不应进入正式三臂 completion gate。但作为创新候选，它需要真实 rollout 评估入口。

本次新增：

1. `TFAC_V5/build_tac_quality_guided_server_packet.py`
   - 新增：
     ```text
     action_aware_guided_command_template
     ```
   - 插座 optional ActionAware server 端口：
     ```text
     8769
     ```
   - 擦黑板 optional ActionAware server 端口：
     ```text
     8779
     ```

2. `TFAC_V5/run_optional_action_aware_rollout_gate.py`
   - 评估：
     ```text
     baseline DP vs action_aware_guided DP
     ```
   - 复用正式二臂 evaluator：
     ```text
     TFAC_V5.eval_real_rollout_quality_gate
     ```
   - 默认 rollout 目录：
     ```text
     /home/chenshuai/Project/output/tac_quality_formal_rollouts/insertion/baseline
     /home/chenshuai/Project/output/tac_quality_formal_rollouts/insertion/action_aware_guided
     /home/chenshuai/Project/output/tac_quality_formal_rollouts/board/baseline
     /home/chenshuai/Project/output/tac_quality_formal_rollouts/board/action_aware_guided
     ```
   - 输出：
     ```text
     /home/chenshuai/Project/output/optional_action_aware_rollout_gate_runner/formal_paired12_preflight/
     ```

3. `TFAC_V5/build_tac_quality_guidance_manifest.py`
   - manifest 开始追踪 optional ActionAware rollout gate runner；
   - 但该 artifact 不作为正式 real rollout completion evidence。

4. `TFAC_V5/audit_tac_quality_goal_completion.py`
   - 新增一条 requirement：
     ```text
     Optional ActionAware rollout gate runner exists without becoming a formal completion dependency.
     ```
   - 这条 requirement 只是检查可选入口存在，不要求 ActionAware 真实 rollout 已完成。

验证命令：

```bash
conda run -n TactileACT python TFAC_V5/build_tac_quality_guided_server_packet.py
conda run -n TactileACT python TFAC_V5/run_optional_action_aware_rollout_gate.py
conda run -n TactileACT python TFAC_V5/build_tac_quality_guidance_manifest.py
conda run -n TactileACT python TFAC_V5/audit_tac_quality_goal_completion.py
```

结果：

```text
guided_server_packet launch_packet_ready = true
action_aware command present:
  insertion = true
  board = true

optional ActionAware preflight:
  preflight_ready = false
  scientific_evidence = false
  formal_gate_dependency = false

deployment_manifest_pass = true

goal_audit:
  objective_complete = false
  n_requirements = 41
  n_blockers = 4
```

解释：

这个改动补齐了 ActionAware 作为创新候选的真实评估入口，但不改变正式结论。当前正式推荐仍是：

```text
insertion default = InsertionRiskScorerRuntime
board default = PTGProxyScorerV2Runtime
innovation ablation = DistilledTacQualityEnergyRuntime
optional unified action-conditioned ablation = ActionAwareScorerRuntime
```

最终判断哪个 scorer 最适合 DP classifier guidance，仍必须看真实 rollout gate：

```text
formal:
  baseline vs default_guided
  baseline/default_guided/distilled_guided

optional:
  baseline vs action_aware_guided
```

### 2026-06-10 Optional ActionAware Pairing/Metadata

问题：

上一节已经让 ActionAware 有了 optional rollout gate runner，但 runner 需要：

```text
action_aware_pairing.csv
metadata_generated.csv
```

如果采集后不能自动生成这些 CSV，真实评估阶段仍然容易因为手工配对错误而污染结论。

本次补齐：

1. `TFAC_V5/build_tac_quality_formal_launch_sheet.py`
   - 每个任务新增 optional rollout dir：
     ```text
     /home/chenshuai/Project/output/tac_quality_formal_rollouts/{task}/action_aware_guided
     ```
   - formal collection order 仍是：
     ```text
     baseline
     default_guided
     distilled_guided
     ```
   - optional collection order 是：
     ```text
     action_aware_guided
     ```

2. `TFAC_V5/build_tac_quality_rollout_pairing.py`
   - 原正式输出不变：
     ```text
     pairing_generated.csv
     three_arm_pairing_generated.csv
     metadata_generated.csv
     ```
   - 新增 optional 输出：
     ```text
     action_aware_pairing.csv
     ```
   - 该 CSV 使用二臂格式：
     ```text
     pair_id,baseline,guided
     ```
     其中 guided 是 `action_aware_guided`。

3. `TFAC_V5/audit_tac_quality_pairing_metadata.py`
   - 新增 action-aware pairing 检查；
   - 该检查只作为 optional readiness，不改变正式 `all_tasks_ready`。

4. `TFAC_V5/run_optional_action_aware_rollout_gate.py`
   - 默认读取：
     ```text
     /home/chenshuai/Project/output/tac_quality_rollout_pairing/formal_paired12/{task}/action_aware_pairing.csv
     ```
   - 因此真实采集后流程是：
     ```text
     build_tac_quality_rollout_pairing.py
     -> run_optional_action_aware_rollout_gate.py --run_gates
     ```

验证结果：

```text
formal_launch_sheet launch_sheet_ready = true

tac_quality_rollout_pairing:
  insertion action_aware_pairing_csv exists in report
  board action_aware_pairing_csv exists in report
  action_aware_pairs = 0/0

pairing_metadata_audit:
  all_tasks_ready = false
  action_aware_rows = 0/0

optional_action_aware_rollout_gate_runner:
  preflight_ready = false
  scientific_evidence = false

deployment_manifest_pass = true
goal_audit objective_complete = false
```

解释：

当前 action-aware rows 为 0 是正确的，因为还没有真实 HDF5。这个改动不是声称 ActionAware 有真实效果，而是确保将来如果采集 optional ActionAware rollouts，可以自动进入同一套配对、metadata、gate 体系。

最终科学判断仍然分两层：

```text
formal completion:
  baseline/default/distilled real rollout gates

optional innovation evidence:
  baseline/action_aware_guided real rollout gate
```

### 2026-06-10 Formal Launch Smoke for Optional ActionAware

问题：

formal launch sheet 现在包含 8 条命令：

```text
insertion:
  baseline
  default_guided
  distilled_guided
  action_aware_guided

board:
  baseline
  default_guided
  distilled_guided
  action_aware_guided
```

但原 `smoke_tac_quality_formal_launch_sheet.py` 的 check 仍然写死：

```text
six_commands_present
```

这会导致 optional ActionAware 加入 launch sheet 后，smoke 逻辑与实验设计不一致。

修复：

1. `TFAC_V5/smoke_tac_quality_formal_launch_sheet.py`
   - check 改为：
     ```text
     n_commands_expected = 8
     all_commands_present
     formal_six_commands_present
     optional_action_aware_commands_present
     optional_action_aware_line_search_contract
     ```
   - baseline 仍要求 `--disable_guidance`；
   - default/distilled 仍要求 finite/nonzero gradient；
   - ActionAware 额外要求：
     ```text
     integration_contract.line_search_required = true
     adapter_policy = final_clean_action_line_search_accept_only_refinement
     ```

2. `TFAC_V5/tac_quality_serving_guidance.py`
   - `ActionAwareTacQualityDPIntegrationAdapter` report 新增：
     ```text
     positive_grad_rate
     ```
   - 由 action gradient norm 是否大于 `1e-8` 计算。

3. manifest 和 goal audit 同步更新 formal launch smoke check。

重要解释：

ActionAware 的 optional smoke 不要求单个 synthetic zero-action 样本一定提升 score。原因是它使用：

```text
quality score + line-search + accept-only
```

如果所有候选步长都没有提升，它应该拒绝更新，此时：

```text
improved_rate = 0
delta_norm = 0
```

这不是接线失败。真正需要检查的是：

1. 命令能运行；
2. action 梯度有限；
3. action 梯度非零；
4. line-search / accept-only contract 存在；
5. trust-region 未越界；
6. 不做 reranking；
7. 不做 every-step DDPM guidance。

运行结果：

```text
formal_launch_sheet_smoke:
  overall_pass = true
  n_commands = 8
  all_commands_present = true
  formal_six_commands_present = true
  optional_action_aware_commands_present = true
  all_commands_pass_process = true
  all_commands_pass_output_contract = true
  guided_commands_have_gradients = true
  optional_action_aware_line_search_contract = true

deployment_manifest_pass = true
goal_audit objective_complete = false
```

per-command 摘要：

```text
insertion/default_guided: finite=1.0, positive=1.0, improved=1.0
insertion/distilled_guided: finite=1.0, positive=1.0, improved=0.0
insertion/action_aware_guided: finite=1.0, positive=1.0, improved=0.0

board/default_guided: finite=1.0, positive=1.0, improved=1.0
board/distilled_guided: finite=1.0, positive=1.0, improved=0.0
board/action_aware_guided: finite=1.0, positive=1.0, improved=0.0
```

结论：

ActionAware optional arm 现在已经通过 launch command dry-run：它的命令可启动、梯度可用、line-search contract 正确。但这仍然只是工程接线证据，不是最终科学证据。最终是否值得作为创新 scorer，仍需要：

```text
baseline/action_aware_guided real rollout gate
```
