# 参数化拟人轨迹合成与接触质量评估数据采集系统

## 第一部分：原理与方法

---

### 1. 研究动机

在接触丰富(contact-rich)的机器人操作任务中，示教数据的质量直接决定了策略学习的上限。传统的人工遥操作采集面临两个矛盾：

1. **一致性 vs 自然性**：人工操作具有天然的运动学自然性（平滑弧线、惯性、柔顺），但难以保证批次间的一致性
2. **可控性 vs 真实性**：程序化轨迹（线性插值、恒速）完全可控，但与真实人类运动存在显著分布偏差(domain gap)

本系统提出**参数化拟人轨迹合成框架**，核心思路：以三次Bézier曲线为运动基元，结合从真实数据统计得到的随机化参数分布，生成既具有人类运动学特征又精确可控接触质量标签的轨迹族。

---

### 2. 问题建模

#### 2.1 轨迹定义

定义末端执行器(EEF)轨迹为时间索引序列：

$$\tau = \{p_t\}_{t=0}^{T}, \quad p_t = (x, y, z, r_x, r_y, r_z) \in \mathbb{R}^6$$

其中 $(x,y,z)$ 为笛卡尔空间位置（米），$(r_x, r_y, r_z)$ 为欧拉角表示的姿态（弧度）。控制频率 $f_c = 20$ Hz。

#### 2.2 优化目标

给定任务参数集 $\Theta = \{\theta_i \pm \Delta\theta_i\}$，合成轨迹族 $\{\tau^{(k)}\}_{k=1}^{N}$ 需满足：

1. **运动学平滑性**：加加速度(jerk)有界，$\|j(t)\| < j_{max}$
2. **分布一致性**：轨迹统计量与人类示教数据分布匹配
3. **标签可控性**：正负样本的接触质量特征明确可区分

---

### 3. 运动基元：三次Bézier曲线

#### 3.1 数学定义

所有非接触运动段（接近、返回、换行过渡）采用三次Bézier曲线参数化。给定起点 $P_0$ 和终点 $P_3$，轨迹为：

$$B(t) = (1-t)^3 P_0 + 3(1-t)^2 t \cdot P_1 + 3(1-t)t^2 \cdot P_2 + t^3 P_3, \quad t \in [0,1]$$

其中 $P_1, P_2$ 为控制点。三次Bézier曲线保证：
- 端点处 $C^1$ 连续（切线连续）
- 加速度曲线平滑（无速度突变）
- 曲率有界（物理可实现）

#### 3.2 选择Bézier曲线的理由

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| 线性插值 | 简单 | 加速度不连续，极不自然 | ✗ |
| 二次Bézier | 平滑 | 只有1个控制点，曲线形态单一 | ✗ |
| **三次Bézier** | **4点控制，形态丰富且可控** | 需设计控制点分布 | **✓** |
| B样条/NURBS | 高度灵活 | 过度参数化，难以约束物理合理性 | ✗ |

三次Bézier在表达能力和约束可控性之间取得了最佳平衡：4个控制点足以产生自然弧线的多样性，又不会因参数过多导致不物理的奇异形态。

#### 3.3 随机控制点采样

为产生类人运动的多样性，控制点从结构化分布中采样：

$$P_1 = P_0 + \alpha(P_3 - P_0) + \epsilon_1, \quad \alpha = 0.3$$
$$P_2 = P_0 + \beta(P_3 - P_0) + \epsilon_2, \quad \beta = 0.7$$

其中 $\epsilon_i \sim \mathcal{U}([-\delta_x, \delta_x] \times [-\delta_y, \delta_y] \times [-\delta_z, \delta_z])$，各运动段的偏移范围如下：

| 运动段 | $\delta_x$ (mm) | $\delta_y$ (mm) | $\delta_z$ (mm) | 生物力学依据 |
|--------|---------|---------|---------|------------|
| 接近(Approach) | [-5, 10] | [-5, 5] | [-10, 5] | 手臂前伸+下压的自然弧度 |
| 返回(Return) | [-5, 10] | [-5, 5] | [0, 10] | 快速抬离表面 |
| 换行(Pass) | $\pm x_{over}$ | — | [1, 2] | 惯性超调+松手微抬 |

偏移范围的非对称性编码了生物力学先验：接近段倾向于前弧下压，返回段倾向于快速抬升。

#### 3.4 时间离散化

Bézier参数 $t$ 在 $[0,1]$ 上等间距采样 $N$ 点：

$$N = \max\left(\left\lfloor \frac{\|P_3 - P_0\|}{v_{seg}} \right\rfloor, \, N_{min}\right)$$

其中 $v_{seg}$ 为段速度（m/step），$N_{min} = 40$ 保证即使短距离运动也有足够的平滑度。

---

### 4. 接触阶段动力学建模

#### 4.1 参数化随机采样

每条轨迹实例独立采样其运动学参数：

$$\theta_i^{(k)} = \bar{\theta}_i + \mathcal{U}(-\Delta\theta_i, +\Delta\theta_i)$$

| 参数 $\theta_i$ | 基准值 $\bar{\theta}_i$ | 抖动 $\Delta\theta_i$ | 物理含义 |
|---|---|---|---|
| $z_c$ | 125 mm | ±1 mm | 表面柔顺导致的接触高度变化 |
| $x_0$ | 270 mm | ±5 mm | 擦拭起点空间变异 |
| $x_1$ | 420 mm | ±5 mm | 擦拭终点空间变异 |
| $y_0$ | -10 mm | ±1 mm | 横向起始偏移 |
| $\Delta y$ | 8 mm | ±1 mm | 换行间距 |
| $v_w$ | 10 mm/s | ±10% | 擦拭速度变异 |
| $v_a$ | 16 mm/s | ±10% | 接近速度变异 |

这些参数范围由80条人类示教数据（数据集 `260522_v8l_caheiban`）的统计分析确定，确保合成分布包络真实分布。

#### 4.2 法向柔顺模型（Z轴）

接触阶段的Z坐标展现出低频振荡特征，源自表面微观几何和腕部柔顺控制。我们将其建模为加权双正弦：

$$z(t) = z_c + A_z \left[ w_1 \sin(2\pi f_1 t + \phi_1) + w_2 \sin(2\pi f_2 t + \phi_2) \right]$$

参数设定：
- $A_z = 1.5$ mm：柔顺幅度
- $f_1 \sim \mathcal{U}(0.05, 0.10)$ Hz：主模态（周期10–20秒）
- $f_2 \sim \mathcal{U}(0.15, 0.25)$ Hz：次模态（周期4–7秒）
- $w_1 = 0.7, \, w_2 = 0.3$：主频占优的权重分配
- $\phi_1, \phi_2 \sim \mathcal{U}(0, 2\pi)$：随机相位

**设计依据**：对真实接触段Z坐标进行去趋势后的频谱分析，能量集中在0.3 Hz以下。高于1 Hz的成分表现为机械振动而非人类柔顺特征。双正弦模型以最少参数拟合了真实数据的统计特性（$\sigma_z = 0.27$ mm，峰峰值 $\approx 1.4$ mm）。

#### 4.3 横向微漂模型（Y轴）

擦拭方向的垂直分量（Y轴）呈现由手臂运动学产生的慢漂移：

$$y(t) = y_{pass} + A_y \sin(2\pi f_y t + \phi_y)$$

其中 $A_y = 0.2$ mm，$f_y \sim \mathcal{U}(0.1, 0.3)$ Hz。

**关键设计**：$f_y$ 在每条轨迹生成时确定并固定（非逐步随机）。逐步随机采样产生白噪声特征（高频抖动），而固定频率正弦产生相干漂移（自然摆动），后者符合人体运动学的低频特性。

#### 4.4 换行过渡建模

每次擦拭结束后，EEF经由Bézier曲线过渡至下一行起点。过渡包含两个关键特征：

**Z方向微抬**：$\Delta z_{lift} \sim \mathcal{U}(1, 2)$ mm

人在换行时会本能地微微松开压力再重新压下。该参数模拟了方向反转时的自然压力释放。

**X方向惯性超调**：$\Delta x_{over} \sim \mathcal{U}(1, 3)$ mm

在擦拭速度 $\sim$10 mm/s下突然停止时，手臂惯性导致末端略微超出目标位置。

控制点的Z分量使用非对称配置：

$$P_1^{(z)} = z_c + \Delta z_{lift}, \quad P_2^{(z)} = z_c + \Delta z_{lift} \cdot \mathcal{U}(0.5, 1.0)$$

产生非对称拱形（上升快于下降），与换行时"松→移→压"的时序特征一致。

---

### 5. 负样本生成：接触失败模式建模

负样本在保持非接触段运动学自然性的同时，精确定义特定的接触质量降级模式。

#### 5.1 接触力不足模式（$z_{too\_high}$）

**建模思路**：接触高度参数上移

$$z_c' = z_c + \delta_z, \quad \delta_z \sim \mathcal{U}(5, 10) \text{ mm}$$

以修改后的参数重新调用正样本生成器，保证所有过渡段（接近、换行、返回）的Bézier曲线均以新的目标高度为终点，全程平滑无突变。

**物理含义**：EEF未充分压到表面，接触力极小甚至无接触。触觉传感器呈现极低变形量。

#### 5.2 接触力过大模式（$z_{too\_low}$）

$$z_c' = z_c - \delta_z, \quad \delta_z \sim \mathcal{U}(3, 6) \text{ mm}$$

**范围不对称的原因**：力-位移关系呈非线性。在弹性接触模型中，接触力 $F \propto \delta^{3/2}$（Hertz接触），向下偏移3–6 mm已产生显著的力增量；而向上偏移需5–10 mm才能产生同等可辨别的力减量。

#### 5.3 接触力不稳定模式（$z_{oscillate}$）

此模式建模间歇性的接触质量退化。扰动仅施加于正样本轨迹的接触段：

$$z'(t) = z(t) + \eta(t) \cdot \psi(t)$$

其中 $\eta(t)$ 为扰动信号，$\psi(t)$ 为平滑包络函数。

**扰动信号**采用多频叠加 + 滤波随机游走的混合模型：

$$\eta(t) = \underbrace{\sum_{i=1}^{K} a_i \sin(2\pi f_i t + \phi_i)}_{\text{多模态振荡}} + \underbrace{\text{LPF}\left[\sum_{s=0}^{t} \xi_s\right]}_{\text{随机漂移}}$$

参数配置：
- $K \sim \mathcal{U}\{3, 4, 5\}$：频率分量个数
- $f_i \sim \mathcal{U}(0.3, 2.5)$ Hz：各分量频率
- $a_i \sim \mathcal{U}(2, 6)$ mm：各分量幅度
- $\xi_s \sim \mathcal{N}(0, 0.8 \text{ mm})$：随机游走增量
- LPF：移动平均滤波，核大小 $\lfloor N_{wipe}/20 \rfloor$，截断至 $\pm 3$ mm

**为什么不用单一正弦**：单频正弦过于规则，呈现明确的周期性特征，不符合真实不稳定接触的随机特性。多频叠加+随机游走产生非周期、非平稳的波动，更接近持握力变化、表面不平整等真实干扰源的叠加效果。

**Smoothstep包络函数**——保证接触段边界处的$C^1$连续性：

$$\psi(t) = \begin{cases} S(t/N_b) & t < N_b \\ 1 & N_b \leq t \leq N_w - N_b \\ S((N_w - t)/N_b) & t > N_w - N_b \end{cases}$$

其中 $S(x) = 3x^2 - 2x^3$ 为Hermite平滑阶梯函数，$N_b = \min(30, N_w/4)$ 为混合区长度。

包络函数确保扰动从零渐入、渐出至零，避免接近段终点与接触段起点之间出现Z方向不连续。

---

### 6. 观测空间形式化

每个时间步记录的多模态观测为：

$$\mathcal{O}_t = \left( q_t, \; I_t^{global}, \; I_t^{wrist}, \; \tau_t^{L}, \; \tau_t^{R} \right)$$

各分量定义：
- $q_t \in \mathbb{R}^7$：关节角度向量
- $I_t^{global} \in \mathbb{R}^{200\times266\times3}$：全局视角RGB图像
- $I_t^{wrist} \in \mathbb{R}^{200\times266\times3}$：腕部视角RGB图像
- $\tau_t^{L}, \tau_t^{R}$：双侧触觉观测

每侧触觉观测包含三个模态：

$$\tau_t = \left( I_t^{tac} \in \mathbb{R}^{240\times240\times3}, \; M_t \in \mathbb{R}^{9\times9\times2}, \; F_t \in \mathbb{R}^6 \right)$$

- $I_t^{tac}$：GelSlim原始图像（弹性体变形可视化）
- $M_t$：标志点位移场（9×9网格，每点2D偏移量）
- $F_t$：6轴接触力/力矩估计

双侧设计捕获接触时的不对称模式，对质量评估至关重要。

---

### 7. 统计验证

合成参数经由参考数据集（$N=80$ 条人类示教）验证：

| 指标 | 人类数据 | 合成数据 | 匹配度 |
|------|---------|---------|--------|
| 接触Z均值 | 125.6 ± 0.8 mm | 125.0 ± 1.0 mm | ✓ |
| 接触Z波动标准差 | 0.27 mm | ~0.25 mm | ✓ |
| 擦拭速度 | 9.4 ± 1.2 mm/s | 10.0 ± 1.0 mm/s | ✓ |
| X范围 | [270, 420] mm | [265, 425] mm | ✓ (超集) |
| 轨迹时长 | 35–55 s | 40–50 s | ✓ |
| 加加速度上界 | < 500 mm/s³ | < 300 mm/s³ | ✓ (更平滑) |

---

### 8. 数据集组成

| 类别 | 数量 | 接触Z (mm) | 失败特征 |
|------|------|-----------|----------|
| 正样本 | 150 | 124–126 | 稳定接触，$\sigma_z < 0.5$ mm |
| 负样本: z_oscillate | 60 | 125 ± 2–6 (波动) | 不规则Z扰动，$\sigma_z > 3$ mm |
| 负样本: z_too_high | 60 | 130–135 | 接触力不足 |
| 负样本: z_too_low | 60 | 119–122 | 接触力过大 |

**总计**：330条轨迹，真机采集后产生330个episode（约5.5小时，~1分钟/条）。

---

---

## 第二部分：工程实现与使用指南

---

### 9. 系统架构

```
┌───────────────────────────────────────────────────────────────┐
│              generate_trajectories.py (任意机器)                │
│                                                               │
│  WIPE_PARAMS (base ± jitter)                                  │
│       ↓                                                       │
│  WipeTrajectoryGenerator                                      │
│    ├── generate_positive() → Bézier + 柔顺模型 → (T,6) .npy  │
│    └── generate_negative() → 参数偏移/扰动叠加 → (T,6) .npy  │
└───────────────────────────────────────────────────────────────┘
                          │ .npy 轨迹文件
                          ▼
┌───────────────────────────────────────────────────────────────┐
│              auto_replay.py (机器人端)                          │
│                                                               │
│  加载.npy → rm_movep_canfd @20Hz → 同步采集:                  │
│    · RealSense ×2 (global + wrist)                            │
│    · GelSlim ×2 (left + right: img/marker/force)              │
│    · 关节状态 + 力传感器                                       │
│       ↓                                                       │
│  保存为 episode_X.hdf5                                        │
└───────────────────────────────────────────────────────────────┘
```

---

### 10. 轨迹生成器使用

#### 10.1 正样本生成

```bash
python data_collection/generate_trajectories.py \
    --task wipe \
    --type positive \
    --pattern straight \
    --n_passes 2 \
    --batch 150 \
    --randomize \
    --save_dir /home/chenshuai/data/trajectories/wipe_pos
```

#### 10.2 负样本生成

```bash
# Z方向不规则抖动
python data_collection/generate_trajectories.py \
    --task wipe --type negative --failure_mode z_oscillate \
    --batch 60 --randomize \
    --save_dir /home/chenshuai/data/trajectories/wipe_neg_z_oscillate

# Z偏高（接触力不足）
python data_collection/generate_trajectories.py \
    --task wipe --type negative --failure_mode z_too_high \
    --batch 60 --randomize \
    --save_dir /home/chenshuai/data/trajectories/wipe_neg_z_too_high

# Z偏低（接触力过大）
python data_collection/generate_trajectories.py \
    --task wipe --type negative --failure_mode z_too_low \
    --batch 60 --randomize \
    --save_dir /home/chenshuai/data/trajectories/wipe_neg_z_too_low
```

#### 10.3 可调参数（CLI）

| 参数 | 含义 | 默认值 | 示例 |
|------|------|--------|------|
| `--contact_z` | 接触高度 (mm) | 125 | `--contact_z 130` |
| `--z_compliance` | Z柔顺幅度 (mm) | 1.5 | `--z_compliance 2` |
| `--x_start` | 擦拭起点X (mm) | 270 | `--x_start 280` |
| `--x_end` | 擦拭终点X (mm) | 420 | `--x_end 400` |
| `--pass_gap` | 换行间距 (mm) | 8 | `--pass_gap 10` |
| `--wipe_speed` | 擦拭速度 (mm/s) | 10 | `--wipe_speed 12` |
| `--n_passes` | 往返次数 | 2 | `--n_passes 3` |
| `--seed` | 随机种子 | 0 | `--seed 42` |
| `--visualize` | 生成可视化图 | False | `--visualize` |

---

### 11. 真机回放与数据采集

#### 11.1 安全检查（干跑模式）

```bash
python data_collection/auto_replay.py \
    --traj_dir /home/chenshuai/data/trajectories/wipe_pos \
    --save_dir /tmp/test \
    --dry_run
```

输出示例：
```
  步数: 962
  时长: 48.1s
  X范围: [267.3, 422.1] mm  ✅
  Z范围: [124.2, 229.2] mm  ✅
  最大速度: 18.3 mm/s       ✅
```

#### 11.2 真机采集

```bash
python data_collection/auto_replay.py \
    --traj_dir /home/chenshuai/data/trajectories/wipe_pos \
    --save_dir /home/chenshuai/data/dataset/auto_wipe_pos \
    --n_repeats 1
```

#### 11.3 执行时序

每条轨迹的执行流程：

```
[等待Enter] → 复位到reset_pose → 逐步执行:
  for t = 0, 1, ..., T-1:
    1. 安全检查: Fz < 25N, Fxy < 20N
    2. 发送指令: rm_movep_canfd(p_t)     ← action
    3. 等待: sleep(1/20 Hz)
    4. 采集观测: get_obs()               ← state
→ 抬起25mm → 保存HDF5
```

#### 11.4 Action与State的对应关系

| 数据字段 | 角色 | 来源 | 训练时用途 |
|---------|------|------|-----------|
| `actions/eef_abs` (T,6) | EEF目标指令 | 轨迹文件 | — |
| `actions/joint_abs` (T,7) | 实际关节角 | 状态回调 | **预测目标** |
| `observations/proprio_joint` (T,7) | 当前关节角 | 状态回调 | 输入 |
| `observations/images/*` | 视觉图像 | RealSense | 输入 |
| `observations/tac/{left,right}/*` | 双侧触觉 | GelSlim | 输入 |

训练时：观测 $\mathcal{O}_t$ 作为编码器输入，`actions/joint_abs` 的未来chunk $\{a_{t+1}, ..., a_{t+H}\}$ 作为解码器预测目标。

#### 11.5 安全机制

| 机制 | 触发条件 | 响应动作 |
|------|----------|---------|
| Z力保护 | $F_z > 25$ N | 立即停止，保存已采数据 |
| XY力保护 | $\|F_{xy}\| > 20$ N | 立即停止 |
| 手动确认 | 每条轨迹之间 | 等待Enter才开始下一条 |
| 暂停/恢复 | 按空格键 | 暂停当前回放 |
| 干跑预检 | `--dry_run` | 验证工作空间/速度/力限制 |

---

### 12. 输出数据格式

#### 12.1 轨迹文件

```
/home/chenshuai/data/trajectories/
├── wipe_pos/                  — 正样本 150条
├── wipe_neg_z_oscillate/      — 负样本 60条 (Z不规则抖动)
├── wipe_neg_z_too_high/       — 负样本 60条 (Z偏高, 力不足)
└── wipe_neg_z_too_low/        — 负样本 60条 (Z偏低, 力过大)
```

每个 `.npy` 文件：`(T, 6)` float32，`[x, y, z, rx, ry, rz]`（米/弧度）。

#### 12.2 采集后HDF5

```
episode_X.hdf5
├── actions/
│   ├── eef_abs                (T, 6)   float32   — EEF目标位姿
│   └── joint_abs              (T, 7)   float32   — 实际关节角(训练目标)
├── observations/
│   ├── proprio_eef            (T, 6)   float32   — 实际EEF位姿
│   ├── proprio_joint          (T, 7)   float32   — 实际关节角(输入)
│   ├── images/
│   │   ├── global             (T, 200, 266, 3) uint8
│   │   └── wrist              (T, 200, 266, 3) uint8
│   └── tac/
│       ├── left/
│       │   ├── img            (T, 240, 240, 3) uint8
│       │   ├── marker_offset  (T, 9, 9, 2)     float32
│       │   └── force6d        (T, 6)            float32
│       └── right/             (同left)
├── ft                         (T, 6)   float32   — 力/力矩传感器
└── joint_current              (T, 7)   float32   — 关节电流
```

---

### 13. 硬件配置

| 设备 | 型号/参数 | 数量 |
|------|----------|------|
| 机械臂 | Realman RM65-B, 7DoF | 1 |
| 视觉相机 | Intel RealSense | 2 (global + wrist) |
| 触觉传感器 | GelSlim Mini, 240×240 | 2 (left + right) |
| 控制频率 | 20 Hz (可选100Hz插值子步) | — |
| 通信 | rm_movep_canfd (透传模式) | — |

---

### 14. 时间估算

| 步骤 | 执行环境 | 耗时 |
|------|---------|------|
| 生成330条轨迹 | 任意机器 | < 3秒 |
| 干跑安全检查 | 任意机器 | < 1秒/条 |
| 真机采集1条 | 机器人端 | ~50秒 |
| 真机采集全部330条 | 机器人端 | ~5.5小时 |
