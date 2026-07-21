# 2026-07-21 260617_v8l_caheiban Latent Energy Scorer 分数范围评估

## 目的

用户询问当前 `for_show_xiaomi/serve_board_dp_foresight_guided.py` 中使用的评分大概范围，以及分数是否属于 0~100。这里使用该 server 默认的 latent energy scorer 对 `/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban` 做离线评分统计。

## 评分模型

使用 checkpoint：

`/home/chenshuai/Project/output/board_latent_energy/ce_margin_e10/board_latent_energy_best.pt`

对应 server 中加载的模型：

`BoardLatentEnergyRuntime -> BoardLatentEnergyScorer`

当前 server 默认用于梯度引导的分数：

`expert_margin`

该分数不是 0~100，而是未归一化的 logit margin。代码中另有 `quality_0_100 = sigmoid(expert_margin) * 100`，这是 0~100 的派生展示分数，但默认不作为 guidance 主分数。

## 数据与设置

数据目录：

`/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban`

共发现 80 条 HDF5。`episode_1.hdf5` 缺少默认评分所需的 `observations/tac/left/marker_offset` 或 `actions/joint_abs` 字段，因此跳过；实际评估 79 条 episode。

评分方式：

- action 使用 HDF5 中的 `actions/joint_abs`。
- tactile latent 使用 HDF5 中的 left marker 通过 scorer checkpoint 对应的 TactileVAE 编码。
- chunk_len 使用 scorer checkpoint 中的默认长度。
- stride=4。
- 分别统计 contact-only 窗口和全窗口。

输出 JSON：

- `outputs/board_latent_energy_score_260617_v8l_caheiban/score_contact_only.json`
- `outputs/board_latent_energy_score_260617_v8l_caheiban/score_all_windows.json`

## Contact-only 窗口结果

有效窗口数：9568

预测类别计数：

| 类别 | 数量 |
|---|---:|
| expert | 3829 |
| pressure_too_small | 3262 |
| pressure_unstable | 1489 |
| pressure_too_large | 988 |

主要分数范围：

| 分数 | mean | min | p5 | p25 | p50 | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| expert_margin | -2.541 | -12.812 | -11.553 | -10.247 | -6.025 | 7.178 | 8.834 | 9.050 |
| quality_0_100 | 39.964 | 0.0003 | 0.001 | 0.0035 | 0.241 | 99.924 | 99.985 | 99.988 |
| score_good | 0.614 | -7.330 | -6.248 | -3.871 | -0.659 | 6.421 | 7.165 | 7.487 |

按预测类别看：

| 预测类别 | expert_margin mean | expert_margin p5/p50/p95 | quality_0_100 mean | quality_0_100 p5/p50/p95 |
|---|---:|---|---:|---|
| expert | 6.934 | 1.623 / 7.927 / 8.927 | 97.667 | 83.516 / 99.964 / 99.987 |
| pressure_too_small | -8.404 | -11.032 / -9.332 / -2.487 | 1.464 | 0.0016 / 0.0089 / 7.681 |
| pressure_too_large | -9.406 | -11.442 / -10.421 / -3.226 | 0.849 | 0.0011 / 0.0030 / 3.830 |
| pressure_unstable | -9.507 | -12.565 / -10.872 / -2.006 | 1.876 | 0.0003 / 0.0019 / 11.860 |

## 全窗口结果

有效窗口数：15864

预测类别计数：

| 类别 | 数量 |
|---|---:|
| expert | 3958 |
| pressure_too_small | 9147 |
| pressure_unstable | 1718 |
| pressure_too_large | 1041 |

主要分数范围：

| 分数 | mean | min | p5 | p25 | p50 | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| expert_margin | -5.332 | -12.812 | -11.240 | -10.488 | -9.502 | -0.142 | 8.741 | 9.050 |
| quality_0_100 | 24.932 | 0.0003 | 0.0013 | 0.0028 | 0.0075 | 46.453 | 99.984 | 99.988 |
| score_good | -0.771 | -7.372 | -6.010 | -3.474 | -2.629 | 2.741 | 7.112 | 7.487 |

## 结论

1. 当前 server 默认用于 guidance 的 `expert_margin` 不是 0~100 分数。它是未归一化的类别间隔分数，本次在 260617 数据上的总体范围约为 `-12.8 到 9.05`。
2. `quality_0_100` 才是 0~100 的派生分数，定义为 `sigmoid(expert_margin) * 100`，但它高度饱和：expert 类通常接近 98~100，bad 类通常接近 0。该分数更适合展示，不适合作为默认梯度引导目标。
3. contact-only 窗口里 expert 类的 `expert_margin` 大多为正，典型中位数约 `7.93`；非 expert 类大多为负，中位数约 `-9` 到 `-10.9`。
4. 全窗口中 `pressure_too_small` 占比显著上升，说明未接触/轻接触阶段会把整体均值拉低。若专利或图中想说明“当前实现的评分范围”，建议写“未归一化接触质量间隔分数，数值可正可负；当前数据中大致落在 -13 到 9，另可映射为 0~100 的展示分数”。

