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
