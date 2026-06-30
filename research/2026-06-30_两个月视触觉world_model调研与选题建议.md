# 2026-06-30 两个月视触觉 World Model 调研与选题建议

## 一句话结论

两个月内最适合做的不是大规模视触觉视频生成 world model，而是一个小型、短 horizon、action-conditioned 的触觉/力觉 consequence model：

```text
current vision/proprio/tactile + candidate action chunk
  -> predict future tactile latent / marker / force-contact proxies
  -> TacQuality or force-quality score
  -> bounded DP guidance or candidate selection
  -> real rollout force/contact evidence
```

推荐主线：

> Force-aware Tactile Consequence World Model for Proactive Diffusion Policy Guidance

中文可以叫：

> 面向接触质量引导的轻量级视触觉后果世界模型

这条路线和现有仓库最匹配：已有 DP action prior、TactileVAE/latent foresight、TacQuality/energy scorer、board wiping 真机日志、force trace 分析和 guided serving 链路。两个月内可以形成完整闭环和可写的实验故事。

## 近期研究脉络

### 1. 触觉表征从任务专用走向 foundation / reusable encoder

- Sparsh 提出通用视觉式触觉自监督表征，在 46 万+ tactile images 上做 SSL，并在 TacBench 上显示自监督触觉预训练比任务专用端到端训练平均提升明显。对我们的启发是：触觉图像不一定要端到端从零学，应该尽量在 latent/proxy 空间建模，减少数据需求。
  - https://arxiv.org/abs/2410.24090
- AnySkin 关注传感器可替换性和跨实例泛化。对我们更偏工程启发：不要把贡献押在某个传感器的像素外观上，最好使用 marker/force/latent proxy。
  - https://arxiv.org/abs/2409.08276
- ManiFeel 是较系统的视触觉策略 benchmark，结论是触觉在视觉受限和接触密集任务中价值最大，但不同任务依赖的触觉形式不同。对我们启发：实验要选接触质量能被客观量化的任务，而不是泛泛地证明“加触觉更好”。
  - https://arxiv.org/abs/2505.18472

### 2. 2026 年触觉 world model / world action model 成为热点

- VT-WM 证明视触觉 world model 能改善接触物理想象，减少 vision-only rollout 中物体消失、瞬移、无接触移动等物理错误，并提升 zero-shot planning。
  - https://arxiv.org/abs/2602.06001
- OmniVTA 用大规模 OmniViTac 数据集、触觉 VAE、two-stream world model、contact-aware fusion policy 和 60Hz tactile reflex controller 形成完整系统。它很强，但数据和系统规模都太大，不适合两个月从零复刻。
  - https://arxiv.org/abs/2603.19201
- Dream-Tac 把 tactile world action model 用到 action generation，联合建模 action、future visual observation 和 tactile dynamics，并加入 contact-gated fusion / contact-aware attention。它说明“future tactile dynamics 引导动作生成”是当前热点。
  - https://arxiv.org/abs/2606.08737
- TacForeSight 是最贴近我们的小型方向：force-conditioned tactile world model 在 compact latent space 预测 short-horizon tactile latent dynamics，再把 predicted latents 作为 anticipatory contact priors 给 policy。它支持我们继续做短 horizon latent foresight，但也意味着“预测未来触觉 latent”本身已经不是足够独特的创新。
  - https://arxiv.org/abs/2606.11184
- FAWAM 强调 force 不应只是 observation，而要进入 prediction 和 execution-time correction。对擦黑板尤其关键，因为质量目标本质是力带和力平稳性。
  - https://arxiv.org/abs/2606.08555
- ContactWorld 的经验结论是：contact-rich world model 需要 spatially structured 且 temporally continuous 的表征；触觉是否有效取决于跨模态表征兼容性，而不只是把 modality 堆大。
  - https://arxiv.org/abs/2606.13877

### 3. Inference-time guidance 正好和现有 PTG/TacQuality 线契合

- TouchGuide 用 task-specific Contact Physical Model 在 diffusion/flow policy 推理时做 tactile guidance。它不是大规模 world model，而是用 limited expert demos 训练 feasibility score，引导采样动作满足接触约束。对我们很重要：证明“推理时触觉评分器改动作”是合理范式。
  - https://arxiv.org/abs/2601.20239
- ViTaL 把 vision/touch inference-time steering 组织成高低层：视觉做长 horizon mode selection，触觉做短 horizon diffusion editing，并使用 visuo-tactile latent world model 和 tactile verifier 评分 predicted tactile futures。它和我们最像，但我们的切入可以更具体：force/contact-quality energy + bounded DP guidance。
  - https://arxiv.org/abs/2606.14981

## 为什么两个月内不建议做“大型视触觉 world model”

大型路线通常包含：

```text
multi-camera video tokenizer
+ tactile image/force tokenizer
+ action-conditioned transformer/diffusion rollout
+ policy/planning loop
+ large multi-task dataset
+ real robot closed-loop validation
```

风险：

1. 数据量不够：OmniVTA 是 21k+ 轨迹、86 tasks；VT-WM/ContactWorld 也不是单任务几十条数据能支撑的叙事。
2. 原始 RGB future prediction 成本高，且和当前项目主要痛点不完全一致。
3. 两个月内最难的是“真实任务收益证据”，不是把模型结构写出来。
4. 我们已有的优势是 force trace、触觉 marker、DP serving、TacQuality guidance，不应浪费在复刻大模型框架上。

所以推荐压缩问题定义：

```text
不预测完整未来世界；
只预测和接触质量相关的 future tactile/force consequence。
```

## 推荐主线 A：小型 Force-aware Tactile Consequence Model

### 目标

给定当前观测和候选 action chunk，预测短 horizon 内的接触后果：

```text
input:
  current image feature
  current proprio / qpos
  current tactile marker latent or history
  candidate action chunk, e.g. 16 steps

output:
  future tactile latent / marker proxy
  future force proxy: left Fz, |F|, dF, jerk proxy
  contact logits
  force-band logits: too_small / good / too_large / oscillate
```

部署时：

```text
DP proposes action chunk
  -> consequence model predicts future contact
  -> quality score evaluates force/contact consequence
  -> trust-region gradient update or candidate selection
  -> execute
```

### 关键创新点

不要把创新写成“我们也预测未来触觉”。更强的说法是：

```text
Existing tactile world models:
  predicted tactile latent -> policy feature

Our small model:
  predicted tactile/force consequence -> differentiable contact-quality energy
  -> bounded correction of diffusion action chunks
```

这能避开 TacForeSight/Dream-Tac 的正面竞争，并利用我们已经做的 TacQuality/force-aware guidance。

### 最适合的任务

首选：擦黑板 board wiping。

原因：

1. 接触质量定义清楚：力在合理带内、力变化平滑、marker 稳定、动作不抖。
2. 现有数据有 positive / too_small / too_large / oscillate 类别。
3. 已有 force trace 分析工具和 real rollout logging。
4. 论文故事直观：视觉看不出擦拭压力，触觉/力觉可以。

备选：插孔 peg-in-hole。

优点是任务重要、接触强；风险是 outcome 比较离散，需要 success/bounce/retry 标注，短期更依赖真机成对实验。

### 两个月计划

第 1 周：数据和任务边界

- 固定主任务：推荐 board wiping。
- 整理 train/val/test episode split，避免同 episode 泄漏。
- 定义 contact phase、force-band、smoothness、success/failure 指标。
- 复查 left/right force 可用性；目前建议以 left force/Fz 为主。

第 2 周：最小 predictor baseline

- 输入 action chunk + qpos + current tactile latent。
- 输出 future marker latent + contact / force-band logits。
- 不做 RGB future generation。
- 指标：latent MSE/cosine、contact acc、band balanced acc。

第 3 周：加入 force-aware heads

- 增加 future force proxy regression：Fz、|F|、dF、短窗残差。
- 增加 multi-horizon score：不仅看最后一步，也看整个 chunk。
- 加权关注 contact frames，避免静止/未接触帧主导 loss。

第 4 周：quality energy / scorer 连接

- 训练或复用 TacQuality/force-band scorer。
- 跑 action-gradient audit：
  - finite grad rate
  - score delta
  - action delta norm
  - trust-region pass rate
  - predicted force-band improvement

第 5 周：接入 DP inference

- 两条实现路线二选一：
  - conservative：N 个 DP candidates reranking / small refinement；
  - stronger：denoising 或 clean action chunk 的 bounded gradient guidance。
- 先离线 replay，用真实窗口和 DP 生成动作比较 predicted quality。

第 6 周：真机 A/B

- board baseline vs guided，至少 5 对，最好 10 对。
- 每条记录 force_trace、score_delta、contact_gate、action_delta、episode metadata。
- 指标：in-band ratio、force abs error、force smoothness、jerk、任务完成度。

第 7 周：消融

- no future tactile，只用 current tactile。
- no force head，只预测 marker latent。
- no contact gate。
- reranking vs gradient guidance。
- different horizon/stride。

第 8 周：写作和图表

- 画 architecture：DP prior -> consequence WM -> quality energy -> bounded guidance。
- 图表包括 force curve、predicted vs real force、ablation table、real rollout pair table。
- 结论必须保守：预测/评分/引导链路和真实力曲线改善，不夸大泛化。

## 备选方向 B：Vision-conditioned Tactile Foresight

目标：

```text
current RGB + qpos + action chunk -> future tactile latent / contact probability
```

价值：在没有当前触觉或接触前阶段，也能预测即将发生的接触。

优点：

- 名字更像“小型视触觉 world model”。
- 可以作为 board/insertion 的预接触 warning。

风险：

- 视觉到未来触觉的映射更不适定，数据需求更大。
- 如果没有明确 force/contact reward，论文贡献容易变成“预测误差低”，不够机器人结果导向。

两个月可做版本：

- 不预测原始触觉图像。
- 只预测 tactile latent/contact/force-band。
- 用 RGB DINO/ResNet feature + qpos/action 作为条件。
- 和 full model 对比：vision-only foresight vs vision+tactile foresight。

适合作为主线 A 的 ablation 或第二阶段，不建议单独作为最终主线。

## 备选方向 C：Prediction-error-aware Guidance

目标：

```text
如果最近真实 tactile/force 和 world model 预测差距很大，
自动降低 guidance scale 或切回 baseline。
```

优点：

- 两个月内非常可做。
- 很像 Feedback World Model / safe guidance 的方向。
- 能解决真机部署中最实际的问题：预测错时不要乱引导。

形式：

```text
uncertainty/confidence =
  recent force prediction residual
  + recent tactile latent residual
  + contact gate mismatch

effective_guidance_scale =
  base_scale * sigmoid(-uncertainty)
```

缺点：单独做可能显得工程；最好和主线 A 绑定，作为 reliability module。

## 备选方向 D：State-gated Vision-Tactile Fusion Policy

参考 ReTac-ACT 的思路，用 proprio/contact state 动态调节视觉和触觉权重。

优点：

- 插孔任务直观。
- 实现成本低于 world model。

缺点：

- 容易变成又一个 fusion policy，创新弱于 consequence-guided DP。
- 对现有 TacQuality/foresight 资产利用少。

建议只作为 baseline：

```text
DP tactile concat
vs state-gated tactile concat
vs future-consequence-guided DP
```

## 我建议的最终题目形态

### 题目 1：最稳

> Proactive Tactile Quality Guidance for Diffusion Policies via Lightweight Force-aware Consequence Modeling

中文：

> 基于轻量级力触觉后果模型的扩散策略前瞻接触质量引导

主张：

- 用小型 world model 预测短 horizon 触觉/力觉后果；
- 用可微质量能量评价 predicted consequence；
- 在 DP action chunk 上做 bounded guidance；
- 在擦黑板真实 force trace 上证明力带和稳定性改善。

### 题目 2：更 world-model 命名

> A Lightweight Visuo-Tactile Consequence World Model for Contact-Rich Manipulation

主张：

- 不是完整视频 world model，而是 task-relevant consequence world model；
- 比大型 VT-WM 更小，面向真实部署；
- 支持 tactile/force prediction、quality scoring 和 policy correction。

风险：如果 real policy improvement 不强，题目会显得太大。

### 题目 3：更像系统

> Predict-Score-Guide: Tactile Consequence Modeling for Contact-Aware Diffusion Control

这个名字最清楚：

```text
Predict: 预测 future tactile/force consequence
Score: 质量能量评分
Guide: 引导 diffusion action
```

## 成功判据

最低可发表/可展示闭环：

1. consequence model 预测指标过关：
   - contact balanced acc > 0.85
   - force-band balanced acc > 0.85
   - force proxy MAE 明显好于 constant/current baseline
2. guidance audit 过关：
   - finite grad rate = 1.0
   - trust-region pass rate > 0.95
   - predicted quality improvement rate > 0.8
3. real board A/B 至少有一个强指标改善：
   - in-band force ratio 提升；
   - force abs error 下降；
   - contact-phase force delta/jerk 不变坏或更好；
   - task completion 不下降。

更强目标：

- 5-10 对 paired baseline/guided；
- guided 在 force band 和稳定性均提升；
- no-future / no-force / no-contact-gate 消融明显变差。

## 需要你拍板的关键问题

1. 主任务是否先定为擦黑板？
   - 我的建议：是。它最适合两个月做出真实 force/contact evidence。
2. world model 输出是否限定为 latent/proxy，不做原始 RGB/触觉图像生成？
   - 我的建议：限定为 tactile latent + marker/force/contact proxies。
3. 真机实验目标是 5 对还是 10 对 baseline/guided？
   - 我的建议：先 5 对跑通，若趋势明显再扩到 10 对。
4. 是否允许把插孔作为第二任务展示迁移？
   - 我的建议：如果第 5 周前 board 已经闭环，再加 insertion；否则不要分散。

## 当前最合理的下一步

先做一个一周内能验伪的 P0：

```text
board dataset
  -> action chunk + qpos + current tactile latent
  -> future force-band/contact prediction
  -> offline gradient/refinement audit
```

如果 P0 的 force-band/contact prediction 和 gradient audit 不过关，就不要进入真机；如果过关，再推进 DP guided serving 和 paired rollout。

