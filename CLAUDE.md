规则：
1、我叫chenshuai,你是我的科研助理，请你每次回答我的问题前，都要在前面都要加上“Hi,chenshuai。”
2、实验的idea请认真想和调研，关键的设计问题请问我。先说方案经过我的同意和讨论之后再动手，除非我要求你全权决定。
3、模型的修改以及架构修改请进行记录，方便我和你进行查看、每次修改代码的一个功能/实验跑通/改动较大时候就自行commit和git一下（不需要问我）
）
4、每次发现的重要实验结论都自行整理成文档的形式存储，另外修改了模型的话要记录在工作记录中（你单独创建一个文件叫做:工作记录.txt）

  git的指令：git push origin tacfore && git push upstream tacfore：https://github.com/chenshuai3085/TactileACT-cs.git
# TactileACT-cs

TFAC 

## 项目概述
触觉引导的机器人操作策略学习 (Think → Dream → Act)。目标论文: CoRL 2026。
核心思路: ACT + Foresight Transformer 预测未来触觉(动作条件化)+ 对比学习，用预测触觉通过 GatedFusion或者其他fusion方法引导第二个 decoder 输出更好的 action。

## 环境
- conda 环境: `TactileACT`
- Python 3.8.20, PyTorch 2.4.1+cu121
- 环境搭建: `bash setup_env.sh`
- detr 是本地包，需 `pip install -e detr/`

## 数据
  HDF5 文件格式                                                                                                                                                                                          
                                                                                                                                                                                                      
  每个 episode 一个文件 episode_X.hdf5：                                                                                                                                                                 
  /actions/joint_abs          (300, 7)     — 7维关节绝对角度
  /observations/proprio_joint (300, 7)     — 7维关节状态                                                                                                                                                 
  /observations/images/global (300, 200, 266, 3) — 全局相机                                                                                                                                              
  /observations/images/wrist  (300, 200, 266, 3) — 腕部相机
  /observations/tac/left/img  (300, H, W, 3)     — GelSight 触觉图像                                                                                                                                     
  /observations/tac/left/marker_offset (300, 9, 9, 2) — 标志点位移  

  视觉图像：                                                                                                                                                                                             
  - ImageNet 归一化（already_normalized=True，数据集中已做）
  - CLIP backbone 编码 → 512 维 tokens                                                                                                                                                                   
                                      
  触觉（两种模式）：                                                                                                                                                                                     
  - image 模式：GelSight 图像 → ImageNet 归一化 → Backbone → 512 维 embedding                                                                                                                            
  - marker 模式：marker_offset (9×9×2) → 逐通道归一化 (val - mean) / std → PointNet → 512 维                                                                                                             
    - 归一化统计量：mean=[0.572, -1.786], std=[1.596, 3.845]                                                                                                                                             
                                                                                                                                                                                                         
  qpos / action：                                                                                                                                                                                        
  - 均值方差归一化：(val - mean) / std                                                                                                                                                                   
  - 统计量从全部 episode 计算，存在 dataset_stats.pkl                                                                                                                                                    
                                                     
  训练/验证划分：                                                                                                                                                                                        
  - 80/20 随机划分，seed=1                                                                                                                                                                               
  - 训练 269 个，验证 68 个                                                                                                                                                                              
                                                                                                                                                                                                         
  Foresight 数据：                                                                                                                                                                                       
  - 每个样本同时加载当前帧 t 和未来帧 t+h的图像/触觉                                                                                                                                             
  - 作为 foresight GT target                                                                                                                                                                             
                                                                                                                                                                                                         
  Action Chunking：                                                                                                                                                                                      
  - chunk_size=20，每次预测未来 20 步动作                                                                                                                                                                               

## 代码结构 (主要)

```
TFAC/                        # V1: 基础 TFAC 模型
  tfac_model.py              # 核心: 共享encoder + 双decoder(A1/A2) + GatedFusion
  tfac_policy.py             # Policy wrapper
  foresight_transformer.py   # ForesightTransformer: SelfAttn([V;T]) + CrossAttn(A1) + FFN
  marker_encoder.py          # MarkerEncoder: Conv2D/PointNet, marker_offset(9×9×2) → embedding
  dataset.py                 # ForesightEpisodicDataset
  train.py                   # 训练脚本 (课程学习: 前75% GT, 后25% 预测)
  eval_foresight.py          # Foresight 评估
  eval_contrastive.py        # 对比学习评估

TFAC_V2/                     # V2: 时序版本 (输入历史 k 帧)
  foresight_transformer.py   # Factorized attention: Spatial → Temporal → Cross(A1) → FFN
  (其余文件结构同 V1)

TFAC_V3/                     # V3: 最完整版 (多帧预测 + sampling loss + 预训练)
  pretrain_clip.py           # Stage 0: Vision-Tactile CLIP 预训练 (ResNet18+PointNet, InfoNCE)
  pretrain_foresight.py      # Stage 1: Foresight 预训练 (冻结CLIP backbone, GT action条件)
  foresight_transformer.py   # 同 V2 factorized attention
  (其余文件结构同 V2)

tactile_foresight/           # 早期框架 (TouchGuide-inspired, DINOv2 空间), 非主要使用
diffusion/                   # 独立 Diffusion Policy 实现
```

## TFAC 架构 (Think → Dream → Act)
- **共享 backbone**: ResNet18 (CLIP 预训练) + input_proj
- **触觉编码**: MarkerEncoder (Conv2D/PointNet), marker_offset (9×9×2) → hidden_dim
- **CVAE encoder** + **Transformer encoder** (共享)
- **Decoder₁ (Draft, A1)**: 标准 ACT decoder, 输出 draft action
- **ForesightTransformer**: 以 A1.detach() 为条件, 从当前 V/T 预测未来触觉
  - V1: SelfAttn + CrossAttn(A1)
  - V2/V3: Spatial → Temporal → Cross(A1) (factorized attention, 支持历史帧)
- **ForesightContrastive**: V̂_future 和 T̂_future 投影做 InfoNCE 对比学习
- **GatedFusion**: memory + A1_feat + T̂_future 三路 softmax 门控融合
- **Decoder₂ (Final, A2)**: 以 enriched memory 为输入, 输出最终 action
- **训练**: 课程学习 (前75% epochs 用 GT future tactile, 后25% 用预测的)

## V1 → V2 → V3 差异
- **V2**: ForesightTransformer 加入 factorized temporal attention, 支持历史 k 帧
- **V3**: 增加 CLIP 预训练 + Foresight 预训练两个 stage, 多帧预测, sampling loss

## 常用命令

```bash
# 训练 TFAC V1
python TFAC/train.py --config TFAC/config_xiaomi.json

# 训练 TFAC V3 (完整流程)
bash clip_pretrain_xiaomi.sh       # Stage 0: CLIP 预训练
bash tactal_pretrain_xiaomi.sh     # Stage 1: Foresight 预训练
python TFAC_V3/train.py --config TFAC_V3/config_xiaomi.json  # Stage 2: 联合训练
```

## Git 分支
- `main`: 主分支
- `tacfore`: 触觉预测 (Tactile Foresight) 开发分支 (当前)
- `TactileACT_xiaomi`: 小米数据适配
- `tfdocs`: 文档分支

## 远程仓库
- `origin`: GitHub (chenshuai3085/TactileACT-cs)
- `xiaomi`: 小米内部 GitLab (chenshuai18/vtm-cs)
- `upstream`: 小米内部 GitLab (chenzhiyuan3/vtm) 

## 注意事项
- DINOv2 ViT-B/14 在 Python 3.8 下需要 `_patch_dinov2_for_py38()` 修补 PEP 604 语法
- 训练输出目录: `/home/chenshuai/Project/output/`
- tactile_foresight/ 是早期 TouchGuide-inspired 框架，现在主要使用 TFAC 系列




## TFAC演进过程

 TFAC 完整版本演进

  TFAC V1（TFAC/ 基础版）

  核心架构：Think → Dream → Act

  输入: 当前帧视觉 + 触觉 + qpos
           ↓
    CLIP Backbone + input_proj → vision tokens (N_v, B, 512)
    触觉: image mode → Backbone → tokens / marker mode → PointNet → 1 token
           ↓
    [latent, proprio, V_tokens, T_token] → TransformerEncoder (4层) → memory
           ↓
    Decoder Draft (3层) → A1 (草稿动作)
           ↓
    ForesightTransformer:
      SelfAttn([V;T]) → CrossAttn(Q=[V;T], K/V=A1) → FFN
      → 预测 t+h 的触觉/视觉
           ↓
    融合 (Gate/LTD/Token) → enriched memory
           ↓
    Decoder Final (7层) → A2 (最终动作)

  ForesightTransformer：
  - 2 层 ForesightLayer
  - 每层：SelfAttn → CrossAttn(A1) → FFN
  - 只看当前帧，无历史帧
  - 预测单帧未来触觉（t+h）

  融合方式（V4 扩展了三种）：
  - gate：三路（memory, A1, future_tac）逐维度 softmax 加权求和
  - ltd：LTD encoder 提取触觉变化 → FiLM 调制 memory
  - token（新加）：foresight 作为额外 token 追加到 memory

  触觉模式：
  - image：GelSight 图像 → Backbone → 512 维 embedding
  - marker：marker_offset (9×9×2) → PointNet → 512 维

  触觉预测头：
  - linear：Linear(512, 162) 直接映射
  - spatial：ConvTranspose2d 从 3×3 上采样到 9×9

  其他特性：
  - CVAE (训练时从 GT action 推断 z，推理时 z=0)
  - InfoNCE 对比学习对齐视觉-触觉
  - 课程学习 (curriculum_ratio 控制 GT/预测切换)
  - marker_offset 归一化
  - a2_init："zero" 或 "a1_refine"（A1 输出作为 A2 起点）

  ---
  TFAC V2（TFAC_V2/ 时序版）

  相对 V1 的核心改动：加入历史帧输入 + Factorized Attention

  输入: 历史 k 帧 + 当前帧的视觉/触觉
           ↓
    每帧独立过 Backbone → (k, N_total, B, D)
           ↓
    ForesightTransformer V2 (Factorized Attention):
      每层: Spatial SelfAttn → Temporal SelfAttn → CrossAttn(A1) → FFN

  ForesightTransformer V2：
  - 每层 ForesightLayer 有 3 种 attention：
    a. Spatial SelfAttn：同一时刻内 V/T tokens 交互（"这帧里视觉和触觉有什么关系"）
    b. Temporal SelfAttn：同一空间位置跨时间步交互（"这个位置过去 k 帧怎么变化的"）
    c. Cross-Attn(A1)：所有 tokens 与 draft action 交互
  - k=1 时退化为 V1 的简单 SelfAttn（自动兼容）

  历史编码：
  - _encode_history：历史 k 帧分别过 Backbone 得到 tokens
  - 最后一帧替换为当前帧的实际编码
  - 时序 position embedding 区分不同帧

  其他：
  - 同样支持 gate/ltd/token 融合、a2_init
  - 加了 spatial_tac_dec_layers 参数控制 SpatialTactileDecoder 层数
  - max_history=8 最多支持 8 帧历史

  ---CrossAttn │ Spatial + Temporal + CrossAttn │ 同 V2                   │
  ├──────────────────────┼──────────────────────┼────────────────────────────────┼─────────────────────────┤
  │ 历史帧输入           │ 无                   │ 支持 k 帧                      │ 支持 k 帧               │
  ├──────────────────────┼──────────────────────┼────────────────────────────────┼─────────────────────────┤
  │ 预测帧数             │ 单帧 (t+h)           │ 单帧                           │ 多帧 (t+1,...,t+H)      │
  ├──────────────────────┼──────────────────────┼────────────────────────────────┼─────────────────────────┤
  │ 预训练               │ 无                   │ 无                             │ CLIP + Foresight 两阶段 │
  ├──────────────────────┼──────────────────────┼────────────────────────────────┼─────────────────────────┤
  │ Sampling Loss        │ 无                   │ 无                             │ 有（自回归展开）        │
  ├──────────────────────┼──────────────────────┼────────────────────────────────┼─────────────────────────┤
  │ 触觉 loss            │ MSE                  │ MSE                            │ Smooth L1               │
  ├──────────────────────┼──────────────────────┼────────────────────────────────┼─────────────────────────┤
  │ 融合方式             │ gate/ltd/token       │ gate/ltd/token                 │ gate/ltd/token          │
  ├──────────────────────┼──────────────────────┼────────────────────────────────┼─────────────────────────┤
  │ 训练阶段             │ 1 阶段               │ 1 阶段                         │ 3 阶段                  │
  TFAC V3（TFAC_V3/ 预训练 + 多帧预测 + Sampling Loss）

  相对 V2 的核心改动：3 阶段训练 + 多帧预测 + Sampling Loss

  三阶段训练流程：

  Stage 0: Vision-Tactile CLIP 预训练 (pretrain_clip.py)
    ├── ResNet18 + PointNet 分别编码视觉/触觉
    ├── 投影到共享空间
    ├── InfoNCE loss 对齐同一时刻的视觉-触觉对
    └── 输出: 预训练好的 backbone 权重

  Stage 1: Foresight 预训练 (pretrain_foresight.py)
    ├── 冻结 Stage 0 的 backbone
    ├── 只训练 ForesightTransformer
    ├── 用 GT action 做条件（不需要 CVAE/Decoder）
    └── 输出: 预训练好的 foresight 权重

  Stage 2: 联合训练 (train.py)
    ├── 加载 Stage 0 + Stage 1 的权重
    ├── 完整 TFAC 架构端到端训练
    └── 所有模块一起更新

  多帧预测：
  - predict_horizon > 1 时，foresight 一次预测未来 H 帧触觉
  - 输出：t_hat (B, H, 9, 9, 2) 而不是 (B, 9, 9, 2)
  - loss：对每帧分别算 smooth_L1，再平均

  Sampling Loss（自回归展开）：
  步骤 0: 用当前触觉 → foresight → 预测 t+1 触觉 → 和 GT 算 loss
  步骤 1: 用预测的 t+1 触觉替换输入 → foresight → 预测 t+2 触觉 → 和 GT 算 loss
  步骤 2: 用预测的 t+2 触觉替换输入 → foresight → 预测 t+3 触觉 → 和 GT 算 loss
  ...
  最终 loss = 各步 loss 的平均
  - 模拟推理时 foresight 只能用自己预测结果的场景
  - 防止训练时依赖 GT 触觉、推理时 error 累积
  - sampling_steps=3，lambda_sampling=0.5

  Loss 组成（V3）：
  loss = l1_final + 0.2*l1_draft + 1.0*foresight_tac + 0.3*foresight_vis
       + 0.1*contrastive + 10*kl + 0.5*sampling

  ---
  版本对比总结

  ┌──────────────────────┬──────────────────────┬────────────────────────────────┬─────────────────────────┐
  │         特性         │          V1          │               V2               │           V3            │
  ├──────────────────────┼──────────────────────┼────────────────────────────────┼─────────────────────────┤
  │ ForesightTransformer │ SelfAttn + CrossAttn │ Spatial + Temporal + CrossAttn │ 同 V2                   │
  ├──────────────────────┼──────────────────────┼────────────────────────────────┼─────────────────────────┤
  │ 历史帧输入           │ 无                   │ 支持 k 帧                      │ 支持 k 帧               │
  ├──────────────────────┼──────────────────────┼────────────────────────────────┼─────────────────────────┤
  │ 预测帧数             │ 单帧 (t+h)           │ 单帧                           │ 多帧 (t+1,...,t+H)      │
  ├──────────────────────┼──────────────────────┼────────────────────────────────┼─────────────────────────┤
  │ 预训练               │ 无                   │ 无                             │ CLIP + Foresight 两阶段 │
  ├──────────────────────┼──────────────────────┼────────────────────────────────┼─────────────────────────┤
  │ Sampling Loss        │ 无                   │ 无                             │ 有（自回归展开）        │
  ├──────────────────────┼──────────────────────┼────────────────────────────────┼─────────────────────────┤
  │ 触觉 loss            │ MSE                  │ MSE                            │ Smooth L1               │
  ├──────────────────────┼──────────────────────┼────────────────────────────────┼─────────────────────────┤
  │ 融合方式             │ gate/ltd/token       │ gate/ltd/token                 │ gate/ltd/token          │
  ├──────────────────────┼──────────────────────┼────────────────────────────────┼─────────────────────────┤
  │ 训练阶段             │ 1 阶段               │ 1 阶段                         │ 3 阶段                  │
  └──────────────────────┴──────────────────────┴────────────────────────────────┴─────────────────────────┘

  当前主要在 V1 目录（TFAC/）上开发，V2/V3 的改进可以按需合并回来。



目前TFACv3 是当前我的方案 但是可能有些问题：
第一个：预测未来触觉总是不准确，显式预测mse监督也容易平均值，怎么改进/有没有很高的思路？
第二个：预测出来的结果如何来影响action？有没有有效和创新的方法和思路？
第三个: 目前的对比没有似乎没有起到特别好的效果，但是这个也想留着，怎么更好的监督，正负样本怎么选择？

目前选择的预测触觉的模态是标志点位移，这个标志点位移怎么才能预测的不错？？



或者你可以开放一下思维来思考一下。

提供给你一些论文的参考：[text](<../../Zotero/storage/DZ9VGC4S/Xue 等 - 2025 - Reactive Diffusion Policy Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation.pdf>)
[text](<../../Zotero/storage/63QP4GUK/Zheng 等 - 2026 - OmniVTA Visuo-Tactile World Modeling for Contact-Rich Robotic Manipulation.pdf>)
[text](<../../Zotero/storage/G4VHA9NB/Higuera 等 - 2026 - Visuo-Tactile World Models.pdf>)
[text](<../../Zotero/storage/C5L6QZIY/Heng 等 - 2025 - ViTacFormer Learning Cross-Modal Representation for Visuo-Tactile Dexterous Manipulation.pdf>)
[text](<../../Zotero/storage/PGELHA7K/Posadas-Nava 等 - ACTION CHUNKING WITH TRANSFORMERS FOR IMAGE-BASED SPACECRAFT GUIDANCE AND CONTROL.pdf>)
[text](<../../Zotero/storage/2E6NMX64/Du和Song - 2025 - DynaGuide Steering Diffusion Polices with Active Dynamic Guidance.pdf>)
[text](<../../Zotero/storage/467TIGGR/Ruan 等 - 2026 - ReTac-ACT A State-Gated Vision-Tactile Fusion Transformer for Precision Assembly.pdf>)
另外不能局限于这些论文 最新的有关触觉预测视觉预测的论文都可以看一看，深度的调研一下，找一些好一点的思路，其他相近的比如预测未来视觉等等也能参考。

最终就是两个目的，一个是怎么提升现在的触觉预测的设计，目前设计的缺点在哪？另一个就是任务成功率，这个触觉预测完之后怎么作用到ACTION上进行提升。
另外我的目的是发论文，创新点很重要，讲故事也很重要，别人没用过的东西用上也算是创新。
最后给出具体可行的完整方案。
