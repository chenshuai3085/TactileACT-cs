# TactileACT-cs

TFAC 

## 项目概述
触觉引导的机器人操作策略学习 (Think → Dream → Act)。目标论文: CoRL 2026。
核心思路: ACT + Foresight Transformer 预测未来触觉(动作条件化)+ 对比学习，用预测触觉通过 GatedFusion 引导第二个 decoder 输出更好的 action。

## 环境
- conda 环境: `TactileACT`
- Python 3.8.20, PyTorch 2.4.1+cu121
- 环境搭建: `bash setup_env.sh`
- detr 是本地包，需 `pip install -e detr/`

## 数据
- HDF5 数据集: `/home/chenshuai/data/dataset/260309_0310` (337 episodes x 300 timesteps)
- DINOv2 预提取特征: `/home/chenshuai/data/dataset/260309_0310_dino_features`
- 图像已做 ImageNet 归一化 (`already_normalized=True`)
- Action dim=7 (joint_abs), chunk_size=20

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
