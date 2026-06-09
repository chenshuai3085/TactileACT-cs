#!/usr/bin/env python3
"""
自动轨迹生成器 — 为CQF正负样本训练构造EEF轨迹

生成的轨迹格式: (T, 6) [x, y, z, rx, ry, rz] (欧拉角, 与数据集格式一致)
控制频率: 20Hz

用法:
    python data_collection/generate_trajectories.py --task wipe --visualize
    python data_collection/generate_trajectories.py --task insert --visualize
"""
import numpy as np
import argparse
import json
from pathlib import Path


# ============ 擦黑板任务参数 (从数据中提取) ============
# 所有带 _jitter 后缀的是随机浮动范围(±), randomize=True时生效
WIPE_PARAMS = {
    "reset_pose": [0.3903, -0.0053, 0.2292, 3.14, -0.007, -2.838],
    "contact_z": 0.1245,           # base接触高度 (m), 正样本接触Z约束在0.124~0.125
    "contact_z_jitter": 0.00025,   # 接触高度浮动 ±0.25mm
    "z_compliance": 0.00025,       # 擦拭中Z柔顺小波动 ±0.25mm, 随机化后不超过0.124~0.125
    "contact_z_range": [0.124, 0.125],
    "negative_z_ranges": {
        "z_oscillate": [0.1234, 0.1275],
        "z_too_high": [0.125, 0.1275],
        "z_too_low": [0.1234, 0.1244],
    },
    "orientation": [3.14, -0.006, -2.86],
    "x_start": 0.270,             # base擦拭起点X (m)
    "x_start_jitter": 0.005,      # 起点浮动 ±5mm → (265~275)
    "x_end": 0.420,               # base擦拭终点X (m)
    "x_end_jitter": 0.005,        # 终点浮动 ±5mm → (415~425)
    "y_start": -0.01,             # Y起始位置 (m)
    "y_start_jitter": 0.001,      # Y起始浮动 ±1mm
    "pass_gap": 0.008,            # base pass间Y间距 (m) 8mm
    "pass_gap_jitter": 0.001,     # pass间距浮动 ±1mm → (7~9mm)
    "wipe_speed": 0.0005,         # base擦拭速度 m/step (20Hz → 10mm/s, 匹配0522)
    "wipe_speed_jitter": 0.10,    # 速度浮动比例 ±10%
    "approach_speed": 0.0008,     # 下降速度 m/step (20Hz → 16mm/s)
    "approach_speed_jitter": 0.10,  # 下降速度浮动 ±10%
    "y_range": [-0.01, 0.05],     # Y方向范围 (zigzag/sine用)
    "control_hz": 20,
}

# ============ 插插座任务参数 (从数据中提取) ============
INSERT_PARAMS = {
    "reset_pose": [0.3903, -0.0052, 0.2293, 3.141, -0.007, -2.838],
    "target_xy": [0.357, 0.013],       # 插座中心XY (从success数据终点)
    "insert_z": 0.161,                 # 完全插入的Z高度
    "contact_z": 0.178,                # 大约开始接触的Z
    "orientation": [3.14, -0.006, -2.86],
    "approach_speed": 0.0005,          # 接近速度 m/step
    "insert_speed": 0.0002,            # 插入速度 m/step (慢, 精确)
    "control_hz": 20,
}


class WipeTrajectoryGenerator:
    """擦黑板轨迹生成器"""

    def __init__(self, params=None):
        self.p = params or WIPE_PARAMS

    def generate_positive(self, pattern="straight", n_passes=1,
                          randomize=False, seed=None):
        """
        生成正样本轨迹: approach + 稳定擦拭 + 抬回初始位置

        所有参数从params的base值出发, randomize=True时每个参数独立加随机浮动,
        保证每条轨迹相似但完全不同。

        Args:
            pattern: "straight" 直线 | "zigzag" Z字形 | "sine" 正弦波
            n_passes: 往返次数 (1=单程, 2=一来一回, 3=来回来, ...)
            randomize: 加随机扰动(让每条轨迹不同)
            seed: 随机种子
        Returns:
            trajectory: (T, 6) EEF轨迹 (含结束抬回)
            metadata: dict 轨迹描述信息
        """
        if seed is not None:
            np.random.seed(seed)

        p = self.p
        hz = p["control_hz"]
        rx, ry, rz = p["orientation"]

        # === 从base值 ± jitter随机化所有参数 ===
        if randomize:
            cz = p["contact_z"] + np.random.uniform(-p["contact_z_jitter"], p["contact_z_jitter"])
            x_lo = p["x_start"] + np.random.uniform(-p["x_start_jitter"], p["x_start_jitter"])
            x_hi = p["x_end"] + np.random.uniform(-p["x_end_jitter"], p["x_end_jitter"])
            y0 = p["y_start"] + np.random.uniform(-p["y_start_jitter"], p["y_start_jitter"])
            pass_gap = p["pass_gap"] + np.random.uniform(-p["pass_gap_jitter"], p["pass_gap_jitter"])
            pass_gap = max(pass_gap, 0.003)  # 最少3mm
            wipe_speed = p["wipe_speed"] * (1.0 + np.random.uniform(-p["wipe_speed_jitter"], p["wipe_speed_jitter"]))
            approach_speed = p["approach_speed"] * (1.0 + np.random.uniform(-p["approach_speed_jitter"], p["approach_speed_jitter"]))
        else:
            cz = p["contact_z"]
            x_lo = p["x_start"]
            x_hi = p["x_end"]
            y0 = p["y_start"]
            pass_gap = p["pass_gap"]
            wipe_speed = p["wipe_speed"]
            approach_speed = p["approach_speed"]

        trajectory = []

        # === Phase 1: Approach (贝塞尔曲线下降, 模拟人手自然随意移动) ===
        start = np.array(p["reset_pose"][:3])
        # 擦拭从x_hi(右侧)开始
        contact_start = np.array([x_hi, y0, cz])
        approach_dist = np.linalg.norm(contact_start - start)
        n_approach = max(int(approach_dist / approach_speed), 40)

        # 三次贝塞尔: P0=start, P3=contact_start, P1/P2为随机控制点
        # 产生平滑但不规则的自然弧线
        mid = (start + contact_start) / 2
        if randomize:
            # 控制点在中间附近随机偏移, 产生不同的弧线形态
            ctrl1 = start + 0.3 * (contact_start - start)
            ctrl1[0] += np.random.uniform(-0.005, 0.010)  # X略偏
            ctrl1[1] += np.random.uniform(-0.005, 0.005)  # Y略偏
            ctrl1[2] += np.random.uniform(-0.010, 0.005)  # Z可能先平再下

            ctrl2 = start + 0.7 * (contact_start - start)
            ctrl2[0] += np.random.uniform(-0.005, 0.010)
            ctrl2[1] += np.random.uniform(-0.005, 0.005)
            ctrl2[2] += np.random.uniform(-0.010, 0.000)
        else:
            ctrl1 = start + 0.33 * (contact_start - start)
            ctrl2 = start + 0.67 * (contact_start - start)

        for i in range(n_approach):
            t = (i + 1) / n_approach
            # 三次贝塞尔: B(t) = (1-t)^3*P0 + 3*(1-t)^2*t*P1 + 3*(1-t)*t^2*P2 + t^3*P3
            pos = ((1-t)**3 * start +
                   3 * (1-t)**2 * t * ctrl1 +
                   3 * (1-t) * t**2 * ctrl2 +
                   t**3 * contact_start)
            trajectory.append([pos[0], pos[1], pos[2], rx, ry, rz])

        # === Phase 2: Wiping (第一pass从右→左) ===
        wipe_length = x_hi - x_lo
        n_wipe_single = int(wipe_length / wipe_speed)

        # Z柔顺波动: 超低频正弦, 非常平滑缓慢 (匹配0522真实数据)
        z_compliance = p.get("z_compliance", 0.0015)
        z_freq1 = np.random.uniform(0.05, 0.10)  # 主频: 10~20秒一个完整周期
        z_freq2 = np.random.uniform(0.15, 0.25)  # 副频: 4~7秒一个周期
        z_phase1 = np.random.uniform(0, 2 * np.pi)
        z_phase2 = np.random.uniform(0, 2 * np.pi)
        # Y抖动频率 (每条轨迹固定一个, 不是每步随机)
        y_jitter_freq = np.random.uniform(0.1, 0.3)  # 3~10秒一个周期
        y_jitter_phase = np.random.uniform(0, 2 * np.pi)
        rough_cfg = p.get("contact_z_roughness")
        rough_step = 0
        rough_knots = rough_values = rough_noise = None
        if rough_cfg:
            rough_interval = max(2, int(rough_cfg.get("interval_steps", 10)))
            total_wipe_est = max(n_passes * n_wipe_single + 1, 2)
            rough_knots = np.arange(0, total_wipe_est + rough_interval, rough_interval)
            rough_amp = float(rough_cfg.get("amp", 0.0003))
            rough_noise_std = float(rough_cfg.get("noise_std", 0.00005))
            rough_values = np.random.uniform(-rough_amp, rough_amp, size=len(rough_knots))
            rough_noise = np.random.randn(total_wipe_est) * rough_noise_std

        for pass_idx in range(n_passes):
            # 第一pass从右→左(X减小), 第二pass左→右, 交替
            going_left = (pass_idx % 2 == 0)

            # 每个pass的Y位置
            if pass_idx == 0:
                y_pass = y0
            else:
                this_gap = pass_gap
                if randomize:
                    this_gap += np.random.uniform(-0.001, 0.001)
                y_pass = y0 + pass_idx * this_gap

            # pass间过渡 (贝塞尔弧线, 自然小弧度转弯)
            if pass_idx > 0 and len(trajectory) > 0:
                prev_pos = np.array(trajectory[-1][:3])
                next_x = x_hi if going_left else x_lo
                target_pos = np.array([next_x, y_pass, prev_pos[2]])
                trans_dist = np.linalg.norm(target_pos - prev_pos)
                n_trans = max(int(trans_dist / wipe_speed), 10)

                if randomize:
                    # pass换行: X方向微超调1~3mm, Z方向微抬1~2mm模拟换行时的自然弧线
                    x_overshoot = np.random.uniform(0.001, 0.003)
                    z_lift = np.random.uniform(0.001, 0.002)
                    ctrl1_t = prev_pos.copy()
                    if going_left:
                        ctrl1_t[0] = prev_pos[0] - x_overshoot
                    else:
                        ctrl1_t[0] = prev_pos[0] + x_overshoot
                    ctrl1_t[1] = prev_pos[1] + 0.4 * (y_pass - prev_pos[1])
                    ctrl1_t[2] += z_lift

                    ctrl2_t = target_pos.copy()
                    ctrl2_t[1] = prev_pos[1] + 0.6 * (y_pass - prev_pos[1])
                    ctrl2_t[2] += z_lift * np.random.uniform(0.5, 1.0)
                else:
                    ctrl1_t = prev_pos + 0.33 * (target_pos - prev_pos)
                    ctrl2_t = prev_pos + 0.67 * (target_pos - prev_pos)

                for i in range(n_trans):
                    t_blend = (i + 1) / n_trans
                    pos = ((1-t_blend)**3 * prev_pos +
                           3 * (1-t_blend)**2 * t_blend * ctrl1_t +
                           3 * (1-t_blend) * t_blend**2 * ctrl2_t +
                           t_blend**3 * target_pos)
                    if p.get("clip_transition_z") and "contact_z_range" in p:
                        pos[2] = np.clip(pos[2], p["contact_z_range"][0], p["contact_z_range"][1])
                    trajectory.append([pos[0], pos[1], pos[2], rx, ry, rz])

            for i in range(n_wipe_single):
                t = (i + 1) / n_wipe_single
                if going_left:
                    x = x_hi - t * wipe_length  # 右→左
                else:
                    x = x_lo + t * wipe_length  # 左→右

                if pattern == "straight":
                    y = y_pass
                    if randomize:
                        t_sec = (len(trajectory) - n_approach) / hz
                        y += 0.0002 * np.sin(2 * np.pi * y_jitter_freq * t_sec + y_jitter_phase)

                elif pattern == "zigzag":
                    y_lo_val = p["y_range"][0]
                    y_hi_val = p["y_range"][1]
                    n_sweeps = 4 if not randomize else np.random.choice([3, 4, 5])
                    phase = (t * n_sweeps) % 1.0
                    if phase < 0.5:
                        y = y_lo_val + (y_hi_val - y_lo_val) * (phase * 2)
                    else:
                        y = y_hi_val - (y_hi_val - y_lo_val) * ((phase - 0.5) * 2)

                elif pattern == "sine":
                    y_center = (p["y_range"][0] + p["y_range"][1]) / 2
                    y_amp = (p["y_range"][1] - p["y_range"][0]) / 3
                    if randomize:
                        y_amp *= np.random.uniform(0.7, 1.3)
                    freq = 3.0 if not randomize else np.random.uniform(2.0, 4.0)
                    y = y_center + y_amp * np.sin(2 * np.pi * freq * t)

                else:
                    y = y_pass

                # Z柔顺波动
                t_sec = (len(trajectory) - n_approach) / hz
                z_offset = z_compliance * (
                    0.7 * np.sin(2 * np.pi * z_freq1 * t_sec + z_phase1) +
                    0.3 * np.sin(2 * np.pi * z_freq2 * t_sec + z_phase2)
                )
                z = cz + z_offset
                if rough_cfg and rough_knots is not None:
                    rough_idx = min(rough_step, len(rough_noise) - 1)
                    z += np.interp(rough_step, rough_knots, rough_values) + rough_noise[rough_idx]
                    rough_step += 1
                if "contact_z_range" in p:
                    z = np.clip(z, p["contact_z_range"][0], p["contact_z_range"][1])
                trajectory.append([x, y, z, rx, ry, rz])

        # === Phase 3: 贝塞尔曲线抬回 (同样自然弧线) ===
        last_pos = np.array(trajectory[-1][:3])
        reset_pos = np.array(p["reset_pose"][:3])
        return_dist = np.linalg.norm(reset_pos - last_pos)
        n_return = max(int(return_dist / approach_speed), 40)

        if randomize:
            ctrl1_r = last_pos + 0.3 * (reset_pos - last_pos)
            ctrl1_r[0] += np.random.uniform(-0.005, 0.010)
            ctrl1_r[1] += np.random.uniform(-0.005, 0.005)
            ctrl1_r[2] += np.random.uniform(0.000, 0.010)

            ctrl2_r = last_pos + 0.7 * (reset_pos - last_pos)
            ctrl2_r[0] += np.random.uniform(-0.005, 0.010)
            ctrl2_r[1] += np.random.uniform(-0.005, 0.005)
            ctrl2_r[2] += np.random.uniform(-0.005, 0.010)
        else:
            ctrl1_r = last_pos + 0.33 * (reset_pos - last_pos)
            ctrl2_r = last_pos + 0.67 * (reset_pos - last_pos)

        for i in range(n_return):
            t = (i + 1) / n_return
            pos = ((1-t)**3 * last_pos +
                   3 * (1-t)**2 * t * ctrl1_r +
                   3 * (1-t) * t**2 * ctrl2_r +
                   t**3 * reset_pos)
            trajectory.append([pos[0], pos[1], pos[2], rx, ry, rz])

        trajectory = np.array(trajectory, dtype=np.float32)
        metadata = {
            "task": "wipe",
            "type": "positive",
            "pattern": pattern,
            "n_passes": n_passes,
            "contact_z_mm": cz * 1000,
            "x_range_mm": [x_lo * 1000, x_hi * 1000],
            "y_start_mm": y0 * 1000,
            "pass_gap_mm": pass_gap * 1000,
            "wipe_speed_mm_s": wipe_speed * hz * 1000,
            "randomize": randomize,
            "seed": seed,
            "duration_steps": len(trajectory),
            "duration_sec": len(trajectory) / hz,
        }
        return trajectory, metadata

    def generate_negative(self, failure_mode="z_oscillate", n_passes=2, seed=None):
        """
        生成负样本轨迹: 在正样本基础上添加Z方向扰动

        Args:
            failure_mode:
                "z_oscillate" — Z在base上下不规则波动(接触力不稳)
                "z_too_high"  — contact_z偏高(压不到/太轻)
                "z_too_low"   — contact_z偏低(压得太死)
            n_passes: 传递给正样本的pass数
            seed: 随机种子
        """
        if seed is not None:
            np.random.seed(seed)

        p = self.p
        cz = p["contact_z"]
        hz = p["control_hz"]
        neg_ranges = p.get("negative_z_ranges", {})

        # z_too_high / z_too_low: 直接修改contact_z重新生成
        # 这样approach/pass过渡/return全部都是自然的Bezier曲线
        if failure_mode == "z_too_high":
            z_lo, z_hi = neg_ranges.get("z_too_high", [0.125, 0.1275])
            center = (z_lo + z_hi) / 2 + np.random.uniform(-0.0003, 0.0003)
            modified_params = dict(self.p)
            modified_params["contact_z"] = center
            modified_params["contact_z_jitter"] = 0.00020
            modified_params["z_compliance"] = 0.00025
            modified_params["contact_z_range"] = [z_lo, z_hi]
            modified_params["contact_z_roughness"] = {
                "amp": 0.00045,
                "noise_std": 0.00008,
                "interval_steps": 10,
            }
            modified_params["clip_transition_z"] = True
            temp_gen = WipeTrajectoryGenerator(params=modified_params)
            base_traj, base_meta = temp_gen.generate_positive(
                pattern="straight", n_passes=n_passes, randomize=True, seed=seed)

            metadata = {
                "task": "wipe",
                "type": "negative",
                "failure_mode": failure_mode,
                "perturbation_desc": f"range=[{z_lo*1000:.1f},{z_hi*1000:.1f}]mm",
                "n_passes": n_passes,
                "contact_z_mm": modified_params["contact_z"] * 1000,
                "contact_z_range_mm": [z_lo * 1000, z_hi * 1000],
                "duration_steps": len(base_traj),
                "duration_sec": len(base_traj) / hz,
            }
            return base_traj.astype(np.float32), metadata

        elif failure_mode == "z_too_low":
            z_lo, z_hi = neg_ranges.get("z_too_low", [0.1234, 0.1244])
            center = (z_lo + z_hi) / 2 + np.random.uniform(-0.00012, 0.00012)
            modified_params = dict(self.p)
            modified_params["contact_z"] = center
            modified_params["contact_z_jitter"] = 0.00010
            modified_params["z_compliance"] = 0.00015
            modified_params["contact_z_range"] = [z_lo, z_hi]
            modified_params["contact_z_roughness"] = {
                "amp": 0.00018,
                "noise_std": 0.00004,
                "interval_steps": 10,
            }
            modified_params["clip_transition_z"] = True
            temp_gen = WipeTrajectoryGenerator(params=modified_params)
            base_traj, base_meta = temp_gen.generate_positive(
                pattern="straight", n_passes=n_passes, randomize=True, seed=seed)

            metadata = {
                "task": "wipe",
                "type": "negative",
                "failure_mode": failure_mode,
                "perturbation_desc": f"range=[{z_lo*1000:.1f},{z_hi*1000:.1f}]mm",
                "n_passes": n_passes,
                "contact_z_mm": modified_params["contact_z"] * 1000,
                "contact_z_range_mm": [z_lo * 1000, z_hi * 1000],
                "duration_steps": len(base_traj),
                "duration_sec": len(base_traj) / hz,
            }
            return base_traj.astype(np.float32), metadata

        # z_oscillate: 在正样本上加不规则扰动
        base_traj, base_meta = self.generate_positive(
            pattern="straight", n_passes=n_passes, randomize=True, seed=seed)

        # 找擦拭阶段: Z低于contact_z + 10mm的区间
        z_thresh = cz + 0.010
        wipe_mask = base_traj[:, 2] < z_thresh
        wipe_indices = np.where(wipe_mask)[0]
        if len(wipe_indices) == 0:
            wipe_start, wipe_end = 0, len(base_traj)
        else:
            wipe_start, wipe_end = wipe_indices[0], wipe_indices[-1]

        traj = base_traj.copy()
        n_wipe = wipe_end - wipe_start

        # 平滑过渡: approach末端和return开头也要渐变，不能突变
        # 过渡区长度: approach末尾30步渐入, wiping结束后30步渐出
        n_blend = min(30, n_wipe // 4)

        def _smooth_blend(n):
            """smoothstep 0→1"""
            t = np.linspace(0, 1, n)
            return t * t * (3 - 2 * t)

        # z_oscillate: 多频叠加 + 随机游走, 最终限制在安全Z范围内
        z_lo, z_hi = neg_ranges.get("z_oscillate", [0.1234, 0.1275])
        n_components = np.random.randint(3, 6)
        freqs = np.random.uniform(0.25, 1.6, n_components)
        amps = np.random.uniform(0.00035, 0.00095, n_components)
        phases = np.random.uniform(0, 2 * np.pi, n_components)

        # 随机游走 (低通滤波的噪声)
        walk_steps = np.random.randn(n_wipe) * 0.00012
        walk = np.cumsum(walk_steps)
        kernel_size = max(5, n_wipe // 20)
        kernel = np.ones(kernel_size) / kernel_size
        walk_smooth = np.convolve(walk, kernel, mode='same')
        walk_smooth = np.clip(walk_smooth, -0.0009, 0.0009)

        rough_interval = 8
        rough_knots = np.arange(0, n_wipe + rough_interval, rough_interval)
        rough_values = np.random.uniform(-0.00075, 0.00075, size=len(rough_knots))
        rough_noise = np.random.randn(n_wipe) * 0.00008

        # 计算全段扰动
        perturb = np.zeros(n_wipe)
        for j in range(n_wipe):
            t_sec = j / hz
            perturb[j] = sum(a * np.sin(2 * np.pi * f * t_sec + p)
                             for a, f, p in zip(amps, freqs, phases))
            perturb[j] += walk_smooth[j]
            perturb[j] += np.interp(j, rough_knots, rough_values) + rough_noise[j]

        # 渐入渐出
        blend_in = _smooth_blend(n_blend)
        blend_out = _smooth_blend(n_blend)[::-1]
        perturb[:n_blend] *= blend_in
        perturb[-n_blend:] *= blend_out

        traj[wipe_start:wipe_end, 2] += perturb
        traj[wipe_start:wipe_end, 2] = np.clip(traj[wipe_start:wipe_end, 2], z_lo, z_hi)
        total_amp = np.max(amps) * 1000
        desc = f"multi-freq({n_components}), peak~{total_amp:.1f}mm+walk, range=[{z_lo*1000:.1f},{z_hi*1000:.1f}]mm"

        metadata = {
            "task": "wipe",
            "type": "negative",
            "failure_mode": failure_mode,
            "perturbation_desc": desc,
            "n_passes": n_passes,
            "contact_z_mm": cz * 1000,
            "contact_z_range_mm": [z_lo * 1000, z_hi * 1000],
            "duration_steps": len(traj),
            "duration_sec": len(traj) / hz,
        }
        return traj.astype(np.float32), metadata


class InsertTrajectoryGenerator:
    """插插座轨迹生成器"""

    def __init__(self, params=None, data_dir=None):
        self.p = params or INSERT_PARAMS
        self.data_dir = data_dir
        self._template_traj = None

    def _load_template(self):
        """从success数据中提取平均轨迹作为模板"""
        if self._template_traj is not None:
            return self._template_traj

        import h5py
        import pickle
        import os

        data_dir = self.data_dir or '/home/chenshuai/data/dataset/0209-0210'
        ann_path = os.path.join(data_dir, 'annotations.pkl')

        with open(ann_path, 'rb') as f:
            ann = pickle.load(f)

        # 收集所有success的EEF轨迹
        success_trajs = []
        for k, v in ann.items():
            if k == '_meta':
                continue
            if v.get('type') != 'success':
                continue
            ep_file = os.path.join(data_dir, f'{k}.hdf5')
            if not os.path.exists(ep_file):
                continue
            with h5py.File(ep_file, 'r') as f:
                eef = f['observations/proprio_eef'][:]  # (300, 6)
            success_trajs.append(eef)

        # 所有success都是300帧, 直接取平均
        self._template_traj = np.mean(success_trajs, axis=0).astype(np.float32)
        print(f"[InsertTrajGen] Loaded template from {len(success_trajs)} success episodes")
        return self._template_traj

    def generate_positive(self, xy_noise_mm=1.0, seed=None):
        """
        生成正样本: 基于success模板 + 微小扰动

        Args:
            xy_noise_mm: XY方向加的高斯噪声 (mm)
            seed: 随机种子
        Returns:
            trajectory: (300, 6)
            metadata: dict
        """
        if seed is not None:
            np.random.seed(seed)

        template = self._load_template()
        traj = template.copy()

        # 加微小XY扰动 (模拟不同插入位置)
        if xy_noise_mm > 0:
            noise_xy = np.random.randn(2) * (xy_noise_mm / 1000.0)
            traj[:, 0] += noise_xy[0]
            traj[:, 1] += noise_xy[1]

        metadata = {
            "task": "insert",
            "type": "positive",
            "xy_noise_mm": xy_noise_mm,
            "duration_steps": len(traj),
        }
        return traj, metadata

    def generate_negative(self, failure_mode="misalign", seed=None):
        """
        生成负样本

        Args:
            failure_mode:
                "misalign"    — XY偏移大(对不准孔, 会碰壁)
                "too_fast"    — 下降速度过快
                "wrong_angle" — 姿态偏转(歪着插)
        """
        if seed is not None:
            np.random.seed(seed)

        template = self._load_template()
        traj = template.copy()
        p = self.p

        if failure_mode == "misalign":
            # XY大偏移: 3~6mm (足以碰到孔壁)
            offset = (np.random.randn(2) * 0.003) + np.sign(np.random.randn(2)) * 0.003
            # 从approach中段开始加偏移
            apply_from = 50 + np.random.randint(0, 30)
            for i in range(apply_from, len(traj)):
                blend = min(1.0, (i - apply_from) / 20.0)
                traj[i, 0] += offset[0] * blend
                traj[i, 1] += offset[1] * blend

        elif failure_mode == "too_fast":
            # 压缩时间轴: 300帧的内容压到200帧内执行完
            # 后100帧保持在底部不动
            fast_len = 200
            indices = np.linspace(0, 299, fast_len).astype(int)
            fast_part = template[indices]
            hold_part = np.tile(template[-1:], (100, 1))
            traj = np.vstack([fast_part, hold_part])

        elif failure_mode == "wrong_angle":
            # 姿态偏转: rx/ry加偏移
            angle_offset = np.random.uniform(0.02, 0.05) * np.sign(np.random.randn())
            apply_from = 60
            for i in range(apply_from, len(traj)):
                blend = min(1.0, (i - apply_from) / 30.0)
                traj[i, 4] += angle_offset * blend  # ry偏转

        metadata = {
            "task": "insert",
            "type": "negative",
            "failure_mode": failure_mode,
            "duration_steps": len(traj),
        }
        return traj.astype(np.float32), metadata


def visualize_trajectory(traj, metadata, save_path=None):
    """可视化轨迹的XYZ和姿态"""
    import matplotlib.pyplot as plt

    T = len(traj)
    t = np.arange(T) / 20.0  # 转成秒

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"{metadata['task']} | {metadata['type']} | {metadata.get('pattern', metadata.get('failure_mode', ''))}")

    # XY平面
    ax = axes[0, 0]
    ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=0.8)
    ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=8, label='start')
    ax.plot(traj[-1, 0], traj[-1, 1], 'r^', markersize=8, label='end')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('XY Plane')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Z高度随时间
    ax = axes[0, 1]
    ax.plot(t, traj[:, 2] * 1000, 'r-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Z (mm)')
    ax.set_title('Z Height vs Time')
    ax.grid(True, alpha=0.3)

    # XZ侧视图
    ax = axes[1, 0]
    ax.plot(traj[:, 0], traj[:, 2] * 1000, 'g-', linewidth=0.8)
    ax.plot(traj[0, 0], traj[0, 2] * 1000, 'go', markersize=8)
    ax.plot(traj[-1, 0], traj[-1, 2] * 1000, 'r^', markersize=8)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (mm)')
    ax.set_title('XZ Side View')
    ax.grid(True, alpha=0.3)

    # 速度
    ax = axes[1, 1]
    if T > 1:
        vel = np.diff(traj[:, :3], axis=0) * 20  # m/s
        speed = np.linalg.norm(vel, axis=1) * 1000  # mm/s
        ax.plot(t[1:], speed, 'purple', linewidth=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (mm/s)')
        ax.set_title(f'EEF Speed (mean={speed.mean():.1f} mm/s)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.savefig('/tmp/traj_preview.png', dpi=120, bbox_inches='tight')
        print(f"Saved: /tmp/traj_preview.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="生成CQF训练用轨迹")
    parser.add_argument("--task", choices=["wipe", "insert"], required=True)
    parser.add_argument("--type", choices=["positive", "negative", "both"], default="both")
    parser.add_argument("--pattern", type=str, default=None,
                        help="wipe: straight/zigzag/sine")
    parser.add_argument("--failure_mode", type=str, default=None,
                        help="wipe: z_oscillate/z_too_high/z_too_low; "
                             "insert: misalign/too_fast/wrong_angle")
    parser.add_argument("--n_passes", type=int, default=1,
                        help="擦黑板往返次数 (1=单程, 2=来回, 3=来回来, ...)")
    parser.add_argument("--contact_z", type=float, default=None,
                        help="base接触高度 (mm), 如125")
    parser.add_argument("--z_compliance", type=float, default=None,
                        help="擦拭中Z柔顺波动幅度 (mm), 如5表示±5mm")
    parser.add_argument("--x_start", type=float, default=None,
                        help="base擦拭起点X (mm), 如270")
    parser.add_argument("--x_end", type=float, default=None,
                        help="base擦拭终点X (mm), 如420")
    parser.add_argument("--pass_gap", type=float, default=None,
                        help="base pass间Y间距 (mm), 如8")
    parser.add_argument("--wipe_speed", type=float, default=None,
                        help="base擦拭速度 (mm/s), 如12")
    parser.add_argument("--batch", type=int, default=1,
                        help="批量生成条数 (配合--randomize每条不同)")
    parser.add_argument("--randomize", action="store_true",
                        help="每条轨迹加随机扰动(接触点/速度/路径微变)")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    save_dir = Path(args.save_dir) if args.save_dir else Path("/tmp/traj_gen")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "wipe":
        params = WIPE_PARAMS.copy()
        if args.contact_z is not None:
            params["contact_z"] = args.contact_z / 1000.0
        if args.z_compliance is not None:
            params["z_compliance"] = args.z_compliance / 1000.0
        if args.x_start is not None:
            params["x_start"] = args.x_start / 1000.0
        if args.x_end is not None:
            params["x_end"] = args.x_end / 1000.0
        if args.pass_gap is not None:
            params["pass_gap"] = args.pass_gap / 1000.0
        if args.wipe_speed is not None:
            params["wipe_speed"] = args.wipe_speed / 1000.0 / params["control_hz"]  # mm/s → m/step
        gen = WipeTrajectoryGenerator(params=params)

        if args.type in ("positive", "both"):
            patterns = [args.pattern] if args.pattern else ["straight", "zigzag", "sine"]
            for pat in patterns:
                for batch_idx in range(args.batch):
                    seed = args.seed + batch_idx if args.randomize else args.seed
                    traj, meta = gen.generate_positive(
                        pattern=pat, n_passes=args.n_passes,
                        randomize=args.randomize, seed=seed)

                    suffix = f"_p{args.n_passes}"
                    if args.batch > 1:
                        suffix += f"_{batch_idx:03d}"
                    fname = f"wipe_pos_{pat}{suffix}"

                    print(f"[+] {fname}: {len(traj)} steps, {meta['duration_sec']:.1f}s")
                    if args.visualize and batch_idx < 3:
                        visualize_trajectory(traj, meta,
                                             save_path=str(save_dir / f"{fname}.png"))
                    np.save(save_dir / f"{fname}.npy", traj)

        if args.type in ("negative", "both"):
            modes = [args.failure_mode] if args.failure_mode else ["z_oscillate", "z_too_high", "z_too_low"]
            for mode in modes:
                for batch_idx in range(args.batch):
                    seed = args.seed + batch_idx + 1000 if args.randomize else args.seed
                    traj, meta = gen.generate_negative(failure_mode=mode, seed=seed)

                    suffix = ""
                    if args.batch > 1:
                        suffix = f"_{batch_idx:03d}"
                    fname = f"wipe_neg_{mode}{suffix}"

                    print(f"[-] {fname}: {len(traj)} steps, {meta['duration_sec']:.1f}s, "
                          f"perturbation={meta.get('perturbation_desc', mode)}")
                    if args.visualize and batch_idx < 3:
                        visualize_trajectory(traj, meta,
                                             save_path=str(save_dir / f"{fname}.png"))
                    np.save(save_dir / f"{fname}.npy", traj)

    elif args.task == "insert":
        gen = InsertTrajectoryGenerator()

        if args.type in ("positive", "both"):
            traj, meta = gen.generate_positive(xy_noise_mm=1.0, seed=args.seed)
            print(f"[+] Insert positive: {len(traj)} steps")
            if args.visualize:
                visualize_trajectory(traj, meta,
                                     save_path=str(save_dir / "insert_pos.png"))
            np.save(save_dir / "insert_pos.npy", traj)

        if args.type in ("negative", "both"):
            modes = [args.failure_mode] if args.failure_mode else ["misalign", "too_fast", "wrong_angle"]
            for mode in modes:
                traj, meta = gen.generate_negative(failure_mode=mode, seed=args.seed)
                print(f"[-] Insert negative ({mode}): {len(traj)} steps")
                if args.visualize:
                    visualize_trajectory(traj, meta,
                                         save_path=str(save_dir / f"insert_neg_{mode}.png"))
                np.save(save_dir / f"insert_neg_{mode}.npy", traj)

    print(f"\nAll trajectories saved to: {save_dir}")


if __name__ == "__main__":
    main()
