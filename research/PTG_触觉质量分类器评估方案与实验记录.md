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
