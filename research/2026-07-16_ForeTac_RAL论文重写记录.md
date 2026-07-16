# 2026-07-16 ForeTac RA-L 论文重写记录

## 本次目标

根据项目主页 `tmp_vtm_ProjectPage/index.html` 和旧论文包
`paper/TacScore_RAL2026_source.zip`，将论文稿从旧的 TacScore/PTG 叙事重写为
当前网页一致的 ForeTac 叙事。

当前论文标题暂定为：

```text
ForeTac: Steering Robot Actions with Predicted Contact Consequences
```

## 论文主线

新的全文逻辑为：

```text
contact-rich manipulation 的核心难点
  -> 当前视觉/触觉策略多为 reactive，不能在执行前判断接触后果
  -> tactile representation、tactile/world-model prediction、inference-time guidance 三类工作之间存在缺口
  -> ForeTac 用 action-conditioned tactile foresight 预测 candidate action 的 future contact consequence
  -> contact-quality energy 将预测触觉后果转为可微 score
  -> trust-region guidance 在 DP late denoising 中做小幅局部动作修正
```

Introduction 已按“领域背景、现存问题、research gap、本文方法、贡献”重写，避免只做项目介绍。

## 证据边界

本次没有虚构在线真实机器人成功率。论文中明确区分：

- 已有证据：
  - TacVAE reconstruction 表；
  - board foresight horizon prediction 图和指标；
  - board guidance diagnostics 表和图；
- 待补证据：
  - 五任务 success/safe-success 主表；
  - no-guidance、reranking、horizon、score-mode、guidance-path 等 ablation。

因此当前稿件是“结构和方法完整、证据边界清楚”的投稿草稿，但不能作为最终投稿版本直接提交，除非补齐 controlled real-rollout 表。

## 文件改动

- `paper/main.tex`：完整重写为 ForeTac RA-L draft。
- `paper/README_RAL.md`：更新当前论文方向和 open items。
- `paper/figures/`：新增网页/诊断图转出的论文 figures。
- `paper/main.pdf`：由 `tectonic main.tex` 编译生成，6 页。
- `paper/ForeTac_RAL2026_source.zip`：新源码包。
- `paper/TacScore_RAL2026_source.zip`：覆盖为同版 ForeTac 源码包，避免旧路径打开旧稿。

## 编译结果

命令：

```bash
cd paper
tectonic main.tex
```

结果：

- 编译通过；
- 输出 `main.pdf`；
- PDF 页数为 6 页；
- 仅有 underfull hbox/vbox 排版警告，无缺图、缺引用或 LaTeX 错误。

## 后续优先补充

1. 完成五任务 paired online rollout：DP、DP+tactile、VLA baseline、reactive baseline、ForeAR、ForeTac。
2. 每个任务统计 success 与 safe success，额外保存 force/contact 失败原因。
3. 完成 ablation：no-guidance、reranking、single-step vs multi-step、score mode、guidance path。
4. 在线表补齐后，把 Discussion 中“pending controlled rollout study”的表述改为真实实验结论。
