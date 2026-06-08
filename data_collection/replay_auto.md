# 自动轨迹采集系统 — 设计文档

## 1. 系统概述

本系统用于为 CQF（Contact Quality Filter）训练自动化地生成正负样本数据。核心思路：**程序化生成EEF轨迹 → 真机回放 → 同步录制多模态传感器数据**。

相比人工遥操作采集，本系统的优势：
- 正样本高度一致可控，消除人为不稳定因素
- 负样本失败模式精确定义，覆盖明确的异常类别
- 每条轨迹自动带微随机变化，既保证一致性又避免过拟合
- 批量化生产，效率远高于人工

### 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        generate_trajectories.py                       │
│                                                                       │
│  参数配置 (base ± jitter)                                             │
│       ↓                                                               │
│  Bezier曲线 + 柔顺波动 + 多频叠加 → EEF轨迹 (T, 6) .npy              │
│       ↓                                                               │
│  正样本 / 负样本(z_oscillate, z_too_high, z_too_low)                  │
└─────────────────────────────────────────────────────────────────────┘
                              │ .npy 文件
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           auto_replay.py                              │
│                                                                       │
│  加载轨迹 → rm_movep_canfd @20Hz → 同时采集:                          │
│    · 视觉 (global + wrist 相机)                                       │
│    · 触觉 (left + right GelSlim: img, marker_offset, force6d)         │
│    · 力/力矩传感器                                                    │
│    · 关节状态                                                         │
│       ↓                                                               │
│  保存为 HDF5 (与人工采集格式完全一致)                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 拟人化轨迹设计 — 核心理念

人工操作不是"走直线到目标"，而是**流畅、带惯性、有微小波动的连续运动**。程序化轨迹如果太规则（直线、匀速、恒定Z），训练出的模型会对真实数据泛化不良。因此，本系统在每个运动阶段都引入了拟人化设计。

### 2.1 核心设计原则

| 原则 | 做法 | 为什么 |
|------|------|--------|
| 不走直线 | 所有大位移用三次Bezier曲线 | 人的手臂运动是弧线，不是线性插值 |
| 不完全匀速 | 速度参数本身带±10%随机 | 人不可能精确匀速 |
| 不完全恒定Z | 擦拭时Z有超低频柔顺波动 | 表面不完美+手腕不稳 |
| 每条都不一样 | 所有参数 base±jitter | 避免模型记忆固定轨迹 |
| 不规则但平滑 | 频率极低(0.05-0.25Hz)，smoothstep过渡 | 抖≠自然，平滑的变化才像人 |

### 2.2 参数随机化策略

每条轨迹生成时，所有关键参数从 `base ± jitter` 范围内独立随机采样：

```
contact_z:     125mm ± 1mm      (每条的接触高度微不同)
x_start:       270mm ± 5mm      (起擦位置不同)
x_end:         420mm ± 5mm      (停擦位置不同)
y_start:       -10mm ± 1mm      (纵向起点不同)
pass_gap:      8mm ± 1mm        (换行间距不同)
wipe_speed:    10mm/s ± 10%     (擦拭快慢不同)
approach_speed: 16mm/s ± 10%    (下降快慢不同)
```

这意味着150条正样本中，没有任何两条的参数组合相同，但整体统计分布与真实人工数据（260522_v8l_caheiban）一致。

---

## 3. 各运动阶段的拟人化实现

### 3.1 Phase 1: Approach (从Reset位下降到接触面)

**问题**：直线下降看起来像机器，人的手是一个自然的弧线运动。

**方案**：三次Bezier曲线，控制点随机偏移。

```
P0 = reset_pose (高处)
P3 = contact_start (接触面)

P1 = P0 + 0.3*(P3-P0) + random_offset
P2 = P0 + 0.7*(P3-P0) + random_offset

B(t) = (1-t)³·P0 + 3(1-t)²t·P1 + 3(1-t)t²·P2 + t³·P3
```

控制点的随机偏移范围：
- X: ±5~10mm (手臂自然弧度)
- Y: ±5mm
- Z: -10~5mm (可能先平移一段再下降，或直接斜着下来)

**效果**：每条轨迹的approach路径都是不同曲率的平滑弧线，有的先平后降，有的直接斜切，但都很流畅。

### 3.2 Phase 2: Wiping (擦拭主体段)

#### 3.2.1 X方向：匀速推进

擦拭时X方向基本匀速（10mm/s ± 10%），这是合理的——人在擦东西时X方向确实接近匀速。

#### 3.2.2 Y方向：超低频微摆

**问题**：完全走直线不像人（人的手臂有自然摆动）。

**方案**：固定频率的正弦微摆，幅度极小（0.2mm），频率极低。

```python
y_jitter_freq = random.uniform(0.1, 0.3)   # 每条轨迹一个固定频率
y += 0.2mm * sin(2π * y_jitter_freq * t + phase)
```

**关键**：频率是生成轨迹时确定的常数，不是每步随机。每步随机会变成白噪声（抖动），而固定频率的正弦是缓慢飘移（自然）。

#### 3.2.3 Z方向：双频柔顺波动

**问题**：实际擦拭时Z不可能恒定——表面微凹凸、手腕柔顺控制都导致Z有微小波动。但波动必须极其缓慢，否则像Z方向的"抖动"。

**方案**：双频正弦叠加，频率极低。

```python
z_freq1 = random.uniform(0.05, 0.10)  # 主频: 10~20秒一个周期
z_freq2 = random.uniform(0.15, 0.25)  # 副频: 4~7秒一个周期
z_phase1, z_phase2 = random phases

z_offset = 1.5mm * (0.7*sin(2π*freq1*t + phase1) + 0.3*sin(2π*freq2*t + phase2))
```

**设计依据**：分析真实数据（260522_v8l_caheiban），接触段Z的去趋势标准差仅0.27mm，峰峰值1.38mm。1.5mm的双频叠加正好匹配这个分布。

**为什么不用随机噪声**：随机噪声（即使滤波后）看起来像传感器抖动，不像物理世界的柔顺。真实的Z波动来自表面形状和手腕刚度，本质上是低频的。

### 3.3 Pass过渡 (换行)

**问题**：擦完一行换到下一行时，如果是直线连接看起来很机械。

**方案**：Bezier曲线过渡 + Z方向微抬。

```python
# 换行时Z微微抬起1~2mm（人在换行时会略微松手再压下去）
z_lift = random.uniform(0.001, 0.002)  # 1~2mm

ctrl1[2] += z_lift                      # 中间拱起
ctrl2[2] += z_lift * random(0.5, 1.0)  # 非对称

# X方向有1~3mm的惯性超调（来不及立刻停住）
x_overshoot = random.uniform(0.001, 0.003)
```

**效果**：换行时轨迹有个小弧形，Z先微抬再压下，X有点惯性超调后回来——像人手在换行时的自然动作。

### 3.4 Phase 3: Return (抬回Reset位)

与Approach对称，同样用三次Bezier曲线 + 随机控制点，产生自然的上抬弧线。控制点Z偏移偏正值（倾向于先快速抬离表面），符合人"完成后松手抬起"的习惯。

---

## 4. 负样本设计

负样本的核心要求：**形态自然（像真实发生的异常），区别明确（CQF能学到什么是"不好的"）**。

### 4.1 z_too_high (接触力过小/无接触)

**生成方式**：直接修改 `contact_z` 参数（+5~10mm），然后调用正样本生成器。

```python
modified_params["contact_z"] = 125mm + random(5, 10)mm
# 重新生成完整轨迹 → approach/pass/return全部自然Bezier
```

**为什么不是事后加偏移**：如果在正样本上硬加Z偏移，approach终点和wiping起点之间会有突变（不自然）。直接修改参数重新生成，所有过渡自动平滑。

**物理含义**：末端执行器没有充分压到表面，接触力很小甚至完全没接触。在真机上会表现为触觉传感器几乎无变形。

### 4.2 z_too_low (接触力过大)

**生成方式**：同上，`contact_z` 减少3~6mm。

```python
modified_params["contact_z"] = 125mm - random(3, 6)mm
```

**物理含义**：压得太深，力过大。真机上触觉传感器会过度变形，可能触发力保护。

**为什么偏移范围不对称（high 5-10mm vs low 3-6mm）**：力和位移是非线性的——往下多压几毫米力的增加远大于上抬几毫米力的减少。3-6mm下压已经会产生很大的接触力差异。

### 4.3 z_oscillate (接触力不稳定)

**生成方式**：在正样本基础上，对擦拭段Z加不规则扰动。

**为什么不是简单正弦**：纯正弦（单一频率+幅度）看起来太规则，像有意施加的周期扰动，不像真实的不稳定接触。

**实际做法**：多频叠加 + 低通滤波随机游走

```python
# 3~5个随机频率叠加 (模拟多种不稳定源)
n_components = random(3, 6)
freqs = random.uniform(0.3, 2.5, n_components)
amps = random.uniform(2, 6, n_components) mm
phases = random phases

# 低通滤波随机游走 (模拟持握力的漂移)
walk = cumsum(randn * 0.8mm) → moving_average → clip(±3mm)

# 最终扰动 = 多频叠加 + 随机游走
perturb[t] = Σ amp_i * sin(2π*freq_i*t + phase_i) + walk[t]

# 渐入渐出: 擦拭开头/结尾30步用smoothstep平滑过渡
perturb[:30] *= smoothstep(0→1)
perturb[-30:] *= smoothstep(1→0)
```

**物理含义**：操作者手不稳、持握力变化、表面不平等导致的Z方向不规则波动。接触力忽大忽小。

**渐入渐出的必要性**：approach终点和wiping起点处Z要连续，不能突然开始抖动。smoothstep保证了从稳定到不稳定的自然过渡。

---

## 5. 数据参数来源

所有默认参数均从真实人工采集数据中统计得出：

**数据源**：`/home/chenshuai/data/dataset/260522_v8l_caheiban/success/` (80 episodes)

| 统计项 | 真实数据值 | 系统参数 |
|--------|-----------|----------|
| 接触Z高度 | 125.6 ± 0.8 mm | contact_z = 125mm, jitter = ±1mm |
| 接触段Z波动(去趋势) | std = 0.27mm, p-p = 1.38mm | z_compliance = ±1.5mm |
| Y方向抖动(去趋势) | std = 0.96mm | Y微摆 ±0.2mm (偏保守) |
| X擦拭速度 | mean = 9.4mm/s | wipe_speed = 10mm/s ± 10% |
| 擦拭X范围 | [270, 420]mm | x_start=270, x_end=420, ±5mm |
| 擦拭方向 | 右→左起始 | going_left = (pass_idx%2==0) |

---

## 6. 回放系统 (auto_replay.py)

### 6.1 执行流程

```
每条轨迹:
  1. 用户按Enter确认
  2. rm_movel 复位到 reset_pose (阻塞等待到位)
  3. 逐步执行:
     for step in trajectory:
       ├── 读取力传感器 → 安全检查 (Fz>25N 或 Fxy>20N 则停)
       ├── rm_movep_canfd(target_quat) → 发送运动指令
       ├── 记录 action (target_euler = 我们发的指令)
       ├── sleep 控制频率 @20Hz
       └── get_obs() → 记录 state (实际位姿+图像+触觉)
  4. 抬起25mm
  5. 保存为 episode_X.hdf5
```

### 6.2 Action vs State

| 字段 | 含义 | 来源 |
|------|------|------|
| `actions/eef_abs` | EEF目标位姿 (我们发出的指令) | 轨迹文件中的目标 |
| `actions/joint_abs` | 执行时刻的实际关节角 | 实时状态回调 |
| `observations/proprio_eef` | 实际EEF位姿 | 实时状态回调 |
| `observations/proprio_joint` | 实际关节角 | 实时状态回调 |
| `observations/images/*` | 视觉图像 | RealSense相机 |
| `observations/tac/left/*` | 左手触觉 | GelSlim传感器 |
| `observations/tac/right/*` | 右手触觉 | GelSlim传感器 |

训练时：`observations` 作为输入，`actions/joint_abs` 作为预测目标。

### 6.3 传感器配置

- **视觉**：2个RealSense (global + wrist)，裁剪后 266×200
- **触觉**：2个GelSlim Mini (left + right)，240×240
  - 采集模态：img (RGB图像)、marker_offset (9×9×2 标志点位移)、force6d (6维力)
- **力/力矩**：6维力传感器（安全保护用）
- **关节**：7自由度关节角 + 电流

### 6.4 安全机制

| 机制 | 触发 | 动作 |
|------|------|------|
| Z力过大 | Fz > 25N | 立即停止，保存已录数据 |
| XY力过大 | Fxy > 20N | 立即停止 |
| 每条确认 | 两条之间 | 等待Enter |
| 干跑检查 | --dry_run | 检查范围/速度/安全限制 |
| 暂停/恢复 | 空格键 | 暂停当前回放 |

---

## 7. 生成的数据规格

### 7.1 文件结构

```
/home/chenshuai/data/trajectories/
├── wipe_pos/                  — 正样本 150条 (straight, 2pass, cz=125mm)
├── wipe_neg_z_oscillate/      — 负样本 60条 (Z不规则抖动)
├── wipe_neg_z_too_high/       — 负样本 60条 (Z偏高+5~10mm)
└── wipe_neg_z_too_low/        — 负样本 60条 (Z偏低-3~6mm)
```

### 7.2 轨迹文件格式

每个 `.npy` 文件是 `(T, 6)` 的 float32 数组：`[x, y, z, rx, ry, rz]`（米/弧度）。

典型时长：40~50秒（800~1000步 @20Hz）。

### 7.3 采集后的HDF5格式

```
episode_X.hdf5
├── actions/
│   ├── eef_abs                (T, 6)   float32
│   └── joint_abs              (T, 7)   float32
├── observations/
│   ├── proprio_eef            (T, 6)   float32
│   ├── proprio_joint          (T, 7)   float32
│   ├── images/
│   │   ├── global             (T, 200, 266, 3) uint8
│   │   └── wrist              (T, 200, 266, 3) uint8
│   └── tac/
│       ├── left/
│       │   ├── img            (T, 240, 240, 3) uint8
│       │   ├── marker_offset  (T, 9, 9, 2)     float32
│       │   └── force6d        (T, 6)            float32
│       └── right/
│           ├── img            (T, 240, 240, 3) uint8
│           ├── marker_offset  (T, 9, 9, 2)     float32
│           └── force6d        (T, 6)            float32
├── ft                         (T, 6)   float32
└── joint_current              (T, 7)   float32
```

---

## 8. 使用命令

```bash
# === Step 1: 生成轨迹 ===
# 正样本 150条
python data_collection/generate_trajectories.py \
    --task wipe --type positive --pattern straight \
    --n_passes 2 --batch 150 --randomize \
    --save_dir /home/chenshuai/data/trajectories/wipe_pos

# 负样本各60条
python data_collection/generate_trajectories.py \
    --task wipe --type negative --failure_mode z_oscillate \
    --batch 60 --randomize \
    --save_dir /home/chenshuai/data/trajectories/wipe_neg_z_oscillate

python data_collection/generate_trajectories.py \
    --task wipe --type negative --failure_mode z_too_high \
    --batch 60 --randomize \
    --save_dir /home/chenshuai/data/trajectories/wipe_neg_z_too_high

python data_collection/generate_trajectories.py \
    --task wipe --type negative --failure_mode z_too_low \
    --batch 60 --randomize \
    --save_dir /home/chenshuai/data/trajectories/wipe_neg_z_too_low

# === Step 2: 安全检查 (干跑) ===
python data_collection/auto_replay.py \
    --traj_dir /home/chenshuai/data/trajectories/wipe_pos \
    --save_dir /tmp/test --dry_run

# === Step 3: 真机采集 ===
python data_collection/auto_replay.py \
    --traj_dir /home/chenshuai/data/trajectories/wipe_pos \
    --save_dir /home/chenshuai/data/dataset/auto_wipe_pos \
    --n_repeats 1
```

---

## 9. 设计决策记录

| 决策 | 选项 | 选择 | 理由 |
|------|------|------|------|
| 轨迹曲线类型 | 线性/二次/三次Bezier/样条 | 三次Bezier | 4个控制点，足够表达自然弧线又不会过拟合 |
| Z柔顺频率 | 高频(1-5Hz) / 低频(0.05-0.25Hz) | 低频 | 真实数据分析显示Z变化极慢，高频像抖动不像柔顺 |
| 负样本z_too_high实现 | 事后加偏移 / 修改参数重新生成 | 修改参数重新生成 | 事后加偏移导致approach→wipe突变，修改参数则全程平滑 |
| 负样本z_oscillate波形 | 单正弦 / 多频叠加+随机游走 | 多频+游走 | 单正弦太规则不像真实不稳定，多频+游走更接近真实抖动 |
| Y方向微摆 | 每步随机 / 固定频率正弦 | 固定频率正弦 | 每步随机=白噪声=抖动，固定频率=缓慢飘移=自然 |
| Pass过渡 | 直线 / Bezier+微抬 | Bezier+微抬1-2mm | 人换行时会略微松手再压下去 |
| 参数随机化 | 全局固定 / 每条独立随机 | 每条独立随机(base±jitter) | 避免模型过拟合到固定轨迹，同时保证分布一致 |
