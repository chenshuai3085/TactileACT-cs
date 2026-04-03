# TFAC 改进思路

## 0. VT-CLIP 预训练 (TFAC_V3 Stage 0)

**思路**: 用 InfoNCE 对比学习对齐视觉 (ResNet18) 和触觉 (PointNet/marker_offset)。

**与之前 CLIP 的区别**: 之前两边都是 ResNet18 (视觉图像 vs 触觉图像)，现在是 ResNet18 vs PointNet (marker_offset 9×9×2)。

**训练流程**:
```
Stage 0 (CLIP): ResNet18 + PointNet → 视觉-触觉对齐
Stage 1 (Foresight): 加载 Stage 0 权重 → 预测未来触觉
Stage 2 (主模型): 加载 Stage 1 权重 → 端到端训练 TFAC
```

**文件**: `TFAC_V3/pretrain_clip.py`, `TFAC_V3/config_clip.json`

---

## 1. Foresight 预训练 (TFAC_V3 Stage 1)

**现状**: Foresight 模块和主模型一起端到端训练，数据量受限于当前 337 条 episode。

**思路**: 单独预训练 foresight 模块（ForesightTransformer + SpatialTactileDecoder），再加载到主模型微调。

**数据**:
- 同任务数据 800-900 条（部分相机位置有偏移，但 GelSight 不受影响）
- 其他任务数据暂不混入，触觉动态差异大

**预训练方案**:
- 输入: 历史 k 帧 V/T tokens + GT action 序列
- 输出: 预测 t+H 的 marker_offset (9x9x2)
- Loss: smooth_L1
- 不需要 CVAE、Decoder1、Decoder2，训练更快

**主训练阶段**:
- 加载预训练的 foresight 权重
- foresight 用较小 lr（降 10 倍）微调
- 其余模块正常 lr 训练

---

## 2. Action 条件的课程学习

**现状**: Foresight 的 action 条件使用 A1（Decoder1 输出，detached），训练和推理一致。

**问题**: 训练早期 A1 质量差（接近随机），foresight 拿到的 action 条件噪声大，导致:
- 触觉预测困难，foresight_tac loss spike 多
- 早期学习效率低

**改进方案 — 课程策略**:
- 训练早期: 用 GT action 作为 foresight 条件（干净信号，易学习）
- 训练后期: 逐渐切换到 A1（适应推理时的分布）
- 切换比例可复用现有的 `curriculum_ratio` 参数逻辑

**备选方案**:
- 混合策略: 每个 batch 随机 50% GT / 50% A1
- 预训练全用 GT，主训练再切 A1

---

---

## 3. TFAC_V3 — Foresight 预训练 + 主模型微调（两阶段训练）

**核心思想**: 将 foresight 模块的训练从端到端中解耦，先单独预训练触觉预测能力，再整体微调。

### Stage 1: Foresight 预训练
- **模块**: ForesightTransformer + SpatialTactileDecoder（+ vision/tactile backbone）
- **数据**: 800-900 条同任务 episode（含相机偏移的旧数据也用上）
- **输入**: 历史 k 帧 vision/tactile tokens + **GT action** 序列
- **输出**: t+H 时刻的 marker_offset (9x9x2)
- **Loss**: smooth_L1（可选加 change_weight）
- **Vision backbone**: CLIP 预训练权重，冻结或极小 lr
- **优势**: 数据量大、无 CVAE/Decoder 干扰、GT action 信号干净

### Stage 2: 主模型训练
- **加载**: Stage 1 预训练的 foresight 权重
- **Foresight lr**: 主 lr 的 1/10（避免破坏预训练表征）
- **Action 条件**: 切换为 A1（Decoder1 输出），适应推理分布
- **其余模块**: 正常从头训练（CVAE、Decoder1、Decoder2、Gate Fusion）
- **数据**: 当前 337 条干净数据

### 与 V2 的区别
| | TFAC_V2 | TFAC_V3 |
|---|---|---|
| Foresight 训练 | 端到端，和主模型一起 | 两阶段，先预训练 |
| Action 条件 | A1（有噪声） | Stage1 用 GT，Stage2 用 A1 |
| 数据量 | 337 条 | Stage1: 800+条，Stage2: 337 条 |
| 时序输入 | 有（history_len=3） | 有（继承 V2） |

### 需要新增的代码
- `TFAC_V3/pretrain_foresight.py` — Stage 1 预训练脚本
- `TFAC_V3/train.py` — Stage 2 训练脚本（支持加载预训练权重）
- `TFAC_V3/config_pretrain.json` — Stage 1 配置
- `TFAC_V3/config_finetune.json` — Stage 2 配置

---

## 4. 其他待验证

- `foresight_change_weight`: 开启后 spike 增多（困难样本加权），但有助于接触时刻预测质量，暂时保留
- `foresight_horizon=10` vs `chunk_size=10`: 当前一致，后续可对比 horizon=8 的效果
- `foresight_nheads=8`: 已改为与主模型一致，待验证效果
- 多帧预测: 当前预测未来单帧，后续可考虑预测 H 帧（每帧都有监督）
