# 自动轨迹采集系统

## 目的

为CQF（Contact Quality Filter）训练构造正负样本数据。通过程序化生成EEF轨迹，真机回放并录制传感器数据，获得比人工采集更稳定（正样本）或可控失败（负样本）的数据。

## 系统架构

```
generate_trajectories.py          auto_replay.py
┌─────────────────────┐          ┌──────────────────────────┐
│  轨迹生成器          │          │  真机回放 + 录制          │
│                     │  .npy    │                          │
│  参数 → EEF轨迹序列  │ ──────→ │  加载轨迹 → movep执行     │
│  (T, 6) xyz+euler   │          │  同时录制obs → 保存HDF5   │
└─────────────────────┘          └──────────────────────────┘
     任何机器                          机器人机器
```

## 支持的任务

| 任务 | 正样本 | 负样本 |
|------|--------|--------|
| 擦黑板 (wipe) | straight / zigzag / sine | z_oscillate / z_too_high / z_too_low |
| 插插座 (insert) | 基于success模板+微扰 | misalign / too_fast / wrong_angle |

---

## 完整使用流程（以擦黑板为例）

### Step 1: 生成轨迹

```bash
# 生成10条不同的正样本轨迹 (直线来回2次, 每条有随机扰动)
python data_collection/generate_trajectories.py \
    --task wipe \
    --type positive \
    --pattern straight \
    --n_passes 2 \
    --batch 10 \
    --randomize \
    --save_dir /home/chenshuai/data/trajectories/wipe_pos

# 生成5条负样本轨迹
python data_collection/generate_trajectories.py \
    --task wipe \
    --type negative \
    --batch 5 \
    --randomize \
    --save_dir /home/chenshuai/data/trajectories/wipe_neg
```

#### 生成参数说明

| 参数 | 含义 | 可选值 |
|------|------|--------|
| `--task` | 任务类型 | `wipe` / `insert` |
| `--type` | 样本类型 | `positive` / `negative` / `both` |
| `--pattern` | 擦拭路径模式 | `straight` / `zigzag` / `sine` |
| `--n_passes` | 擦拭往返次数 | 1=单程, 2=来回, 3=来回来, ... |
| `--pass_gap` | 多pass间Y间距(mm) | 默认8mm |
| `--batch` | 生成轨迹条数 | 配合--randomize使用 |
| `--randomize` | 每条加随机扰动 | 使每条轨迹不同 |
| `--failure_mode` | 负样本失败类型 | 见下方负样本说明 |
| `--seed` | 随机种子 | 可复现 |
| `--visualize` | 生成可视化图 | 最多可视化前3条 |

#### 正样本轨迹结构（时间顺序）

```
Phase 1: Approach
  ├── 从RESET(Z=229mm) 水平移动到擦拭起点上方
  └── 垂直下降到接触高度(Z≈135mm)

Phase 2: Wiping
  ├── 第1 pass: X方向匀速擦过去, Z恒定
  ├── 过渡到第2 pass起点(Y偏移pass_gap)
  └── 第2 pass: X方向擦回来, Z恒定

Phase 3: Return
  ├── 垂直抬起到RESET高度(Z=229mm)
  └── 水平回到RESET位置
```

#### 随机扰动范围

| 维度 | 扰动范围 | 说明 |
|------|----------|------|
| 接触点 X | ±4mm | 开始擦的位置不同 |
| 接触点 Y | ±4mm | 上下偏移不同 |
| 接触高度 Z | ±2mm | 压入深度微调 |
| 擦拭速度 | ±10% | 快慢不同 |
| 擦拭路径 Y | ±0.5mm正弦抖动 | 不走完美直线 |

#### 负样本失败模式

| failure_mode | 描述 | 效果 |
|-------------|------|------|
| `z_oscillate` | Z方向大幅正弦波动(±5-10mm, 0.5-2Hz) | 接触力忽大忽小，不稳定 |
| `z_too_high` | Z整体偏高(+5-10mm) | 压不到/太轻，接触不良 |
| `z_too_low` | Z整体偏低(-3-5mm) | 压得太死，力过大 |

---

### Step 2: 检查安全性（推荐）

```bash
python data_collection/auto_replay.py \
    --traj_dir /home/chenshuai/data/trajectories/wipe_pos \
    --save_dir /tmp/test \
    --dry_run
```

输出示例：
```
==================================================
  DRY RUN — 轨迹检查
==================================================
  步数: 1098
  时长: 54.9s
  起点: xyz=[0.3897, -0.0053, 0.2292]
  终点: xyz=[0.3903, -0.0053, 0.2292]
  X范围: [0.2700, 0.4200]
  Z范围: [0.1350, 0.2292]
  速度: mean=13.9 mm/s, max=23.2 mm/s

  起点距RESET: 0.7 mm ✅
  最低Z: 135.0 mm ✅
  最大速度: 23.2 mm/s ✅
==================================================
```

确认所有轨迹都显示 ✅ 后再上真机。

---

### Step 3: 真机回放 + 录制

```bash
python data_collection/auto_replay.py \
    --traj_dir /home/chenshuai/data/trajectories/wipe_pos \
    --filter "wipe_pos" \
    --save_dir /home/chenshuai/data/dataset/auto_wipe_pos \
    --n_repeats 1
```

#### 回放参数说明

| 参数 | 含义 |
|------|------|
| `--traj` | 单条轨迹文件 (.npy) |
| `--traj_dir` | 轨迹目录 (回放所有.npy) |
| `--filter` | 文件名过滤 (如 "wipe_pos") |
| `--save_dir` | HDF5保存目录 |
| `--n_repeats` | 每条轨迹重复次数 (已randomize则设1) |
| `--dry_run` | 干跑模式 (不连机器人) |
| `--no_tactile` | 不采集触觉 (调试用) |
| `--max_force_z` | Z力上限(N), 默认25 |
| `--interpolation` | 开启子步插值 (100Hz) |

#### 执行交互流程

```
==================================================
  待采集: episode_0 (轨迹: wipe_pos_straight_p2_000, repeat 1/1)
  按 Enter 开始 (先复位→再回放), Ctrl+C 退出
==================================================
                        ← 按Enter
🔄 复位到初始位置...        ← rm_movel回RESET, 阻塞等待到位
✅ 复位完成
▶️  开始回放...             ← 开始按轨迹执行+录制
  step 50/1098 (2.5s), pos=[0.315,-0.010,0.135]
  step 100/1098 (5.0s), ...
  ...
  step 1098/1098 (54.9s), pos=[0.390,-0.005,0.229]
  ↑ 抬起 25mm 完成          ← 结束动作
✅ 已保存 1098 步 → .../episode_0.hdf5

==================================================
  待采集: episode_1 (轨迹: wipe_pos_straight_p2_001, repeat 1/1)
  按 Enter 开始 (先复位→再回放), Ctrl+C 退出
==================================================
                        ← 按Enter继续 / Ctrl+C退出
```

#### 安全机制

| 机制 | 触发条件 | 动作 |
|------|----------|------|
| 力过大停止 | Fz > 25N 或 Fxy > 20N | 立即停止,保存已录数据 |
| 手动停止 | Ctrl+C | 退出程序 |
| 每条确认 | 每条轨迹之间 | 等待Enter才继续 |
| dry_run预检 | 执行前 | 检查范围/速度/安全 |

---

### Step 4: 确认输出

```bash
# 查看采集了多少条
ls /home/chenshuai/data/dataset/auto_wipe_pos/ | wc -l

# 检查单条数据格式
python -c "
import h5py
f = h5py.File('/home/chenshuai/data/dataset/auto_wipe_pos/episode_0.hdf5', 'r')
def show(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f'{name}: {obj.shape} {obj.dtype}')
f.visititems(show)
f.close()
"
```

输出的HDF5格式与人工采集完全一致：
```
actions/eef_abs                      (T, 6)    float32
actions/joint_abs                    (T, 7)    float32
ft                                   (T, 6)    float32
joint_current                        (T, 7)    float32
observations/images/global           (T, 200, 266, 3) uint8
observations/images/wrist            (T, 200, 266, 3) uint8
observations/proprio_eef             (T, 6)    float32
observations/proprio_joint           (T, 7)    float32
observations/tac/left/img            (T, 240, 240, 3) uint8
observations/tac/left/marker_offset  (T, 9, 9, 2) float32
observations/tac/left/force6d        (T, 6)    float32
observations/tac/right/img           (T, 240, 240, 3) uint8
observations/tac/right/marker_offset (T, 9, 9, 2) float32
observations/tac/right/force6d       (T, 6)    float32
```

---

## 时间估算

| 步骤 | 在哪跑 | 耗时 |
|------|--------|------|
| 生成10条轨迹 | 任何机器 | < 1秒 |
| dry_run检查 | 任何机器 | < 1秒 |
| 真机采集1条 (2pass) | 机器人机器 | ~55秒 |
| 真机采集10条 | 机器人机器 | ~10分钟 |
| 真机采集50条 | 机器人机器 | ~50分钟 |

---

## 快速上手命令

```bash
# === 擦黑板正样本 (推荐配置) ===
# 生成
python data_collection/generate_trajectories.py \
    --task wipe --type positive --pattern straight \
    --n_passes 2 --batch 20 --randomize \
    --save_dir /home/chenshuai/data/trajectories/wipe_pos_straight

# 检查
python data_collection/auto_replay.py \
    --traj_dir /home/chenshuai/data/trajectories/wipe_pos_straight \
    --save_dir /tmp/test --dry_run

# 采集
python data_collection/auto_replay.py \
    --traj_dir /home/chenshuai/data/trajectories/wipe_pos_straight \
    --save_dir /home/chenshuai/data/dataset/auto_wipe_pos \
    --n_repeats 1


# === 擦黑板负样本 (3种Z扰动) ===
python data_collection/generate_trajectories.py \
    --task wipe --type negative --batch 10 --randomize \
    --save_dir /home/chenshuai/data/trajectories/wipe_neg

# 指定单种负样本模式
python data_collection/generate_trajectories.py \
    --task wipe --type negative --failure_mode z_oscillate \
    --batch 10 --randomize \
    --save_dir /home/chenshuai/data/trajectories/wipe_neg

python data_collection/auto_replay.py \
    --traj_dir /home/chenshuai/data/trajectories/wipe_neg \
    --save_dir /home/chenshuai/data/dataset/auto_wipe_neg \
    --n_repeats 1


# === 插插座正样本 ===
python data_collection/generate_trajectories.py \
    --task insert --type positive --batch 10 --randomize \
    --save_dir /home/chenshuai/data/trajectories/insert_pos

python data_collection/auto_replay.py \
    --traj_dir /home/chenshuai/data/trajectories/insert_pos \
    --save_dir /home/chenshuai/data/dataset/auto_insert_pos \
    --n_repeats 1 --max_force_z 20
```

---

## 轨迹参数来源

所有默认参数从已有人工采集数据中统计得出：

**擦黑板** (来源: `/home/chenshuai/data/dataset/260519_v8l/peg_in_hole_0519/`, 80 episodes)
- 起始位姿: [0.3903, -0.0053, 0.2292, 3.14, -0.007, -2.838]
- 接触高度: ~135mm (数据均值159mm, 取保守值)
- X擦拭范围: [270mm, 420mm], 约150mm
- Y范围: [-10mm, 50mm], 约60mm
- 人工Fz: mean=7.5N, std=8.2N (程序化后应更稳定)

**插插座** (来源: `/home/chenshuai/data/dataset/0209-0210/`, 156 success + 164 bounce)
- 起始位姿: [0.3903, -0.0052, 0.2293, 3.141, -0.007, -2.838]
- 完全插入Z: 161mm
- Z下降: 68mm
- XY漂移: ~45mm (自然弧线)

---

## 文件结构

```
data_collection/
├── README.md                    ← 本文档
├── generate_trajectories.py     ← 轨迹生成器
├── auto_replay.py               ← 真机回放+录制
└── copy_rename_hdf5_files.py    ← 数据整理工具
```
