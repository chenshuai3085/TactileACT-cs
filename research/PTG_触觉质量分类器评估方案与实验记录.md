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
