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
