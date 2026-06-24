#!/usr/bin/env python3
"""
自动轨迹回放 + 录制脚本 (高跟随版本, 完全独立, 无外部项目依赖)

依赖:
    - Robotic_Arm SDK (realman)
    - realman_env (相机、触觉)
    - numpy, h5py, scipy, cv2

用法:
    # 干跑模式 (不连机器人, 只检查轨迹)
    python auto_replay_high_follow.py --traj trajectories/wipe_pos_000.npy --save_dir ./data --dry_run

    # 真机回放单条
    python auto_replay_high_follow.py --traj trajectories/wipe_pos_000.npy --save_dir ./data

    # 批量回放
    python auto_replay_high_follow.py --traj_dir ./trajectories/ --filter "wipe_pos" --save_dir ./data

将此脚本和轨迹文件放在机器人端任意目录即可运行。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import threading
import re
import socket
from datetime import datetime
from pathlib import Path
from collections import deque
import multiprocessing as mp
import queue

import select

import numpy as np
from scipy.spatial.transform import Rotation as R


# ============ 配置 ============
class AutoReplayConfig:
    ROBOT_IP = "192.168.1.18"
    CONTROL_HZ = 20
    N_SUBSTEPS = 5
    CTRL_DT = 1.0 / (CONTROL_HZ * N_SUBSTEPS)
    USE_INTERPOLATION = False

    # 安全限制
    MAX_FORCE_Z = 25.0
    MAX_FORCE_XY = 20.0

    # 结束后抬起高度 (m)
    LIFT_AFTER_DONE = 0.025

    # 相机配置
    IMAGE_SIZE = (266, 200)   # (W, H)
    CAMERAS = {
        "global": {
            "serial": "130322273140",
            "dim": (640, 480),
            "fps": 30,
            "exposure": 15000,
            "crop": lambda x: x[90:290, 224:490],
        },
        "wrist": {
            "serial": "230322271557",
            "dim": (640, 480),
            "fps": 30,
            "exposure": 15000,
            "crop": lambda x: x[:, :],
        }
    }

    # 触觉传感器
    TACTILE_SENSORS = {
        "left": {"sn": "GF2250032BAE6", "dim": (240, 240)},
        "right": {"sn": "GF2250002C848", "dim": (240, 240)},
    }
    TACTILE_MODALITIES = ("img", "marker_offset", "force6d")

    # HDF5
    H5_COMPRESSION = "gzip"

    # 复位位姿
    RESET_POSE = [0.390345, -0.005217, 0.229337, 3.141, -0.007, -2.838]

    # 实时显示
    VISUALIZE_LIVE = False
    VISUALIZE_FZ_WINDOW = 300


# ============ 工具函数 ============
def poseuler_to_posquat(pose_euler, order="xyz"):
    """欧拉角(6D) → 四元数(7D) [x,y,z,qw,qx,qy,qz]"""
    pos = pose_euler[:3]
    euler = pose_euler[3:]
    quat = R.from_euler(order, euler).as_quat()  # [qx,qy,qz,qw]
    return np.concatenate([pos, [quat[3], quat[0], quat[1], quat[2]]])


def check_force_safety(ft, config):
    """检查力是否超限"""
    if ft is None:
        return True, ""
    fz = abs(ft[2])
    fxy = np.sqrt(ft[0]**2 + ft[1]**2)
    if fz > config.MAX_FORCE_Z:
        return False, f"Fz={fz:.1f}N > {config.MAX_FORCE_Z}N"
    if fxy > config.MAX_FORCE_XY:
        return False, f"Fxy={fxy:.1f}N > {config.MAX_FORCE_XY}N"
    return True, ""


def check_pause():
    """非阻塞检测空格键, 返回是否需要暂停"""
    if select.select([sys.stdin], [], [], 0)[0]:
        key = sys.stdin.read(1)
        if key == ' ':
            return True
    return False


def _live_visualizer_process(data_queue, stop_event, window, fps):
    import cv2

    history = deque(maxlen=max(int(window), 2))

    def to_bgr(img, size):
        if img is None:
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)
        out = img
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        elif out.shape[-1] == 3:
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return cv2.resize(out, size, interpolation=cv2.INTER_NEAREST)

    def put_label(img, text):
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(img, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)

    def draw_fz_graph(size):
        width, height = size
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)

        left, right = 70, width - 25
        top, bottom = 35, height - 45
        cv2.rectangle(canvas, (left, top), (right, bottom), (90, 90, 90), 1)
        cv2.putText(canvas, "Robot FT Z force over time", (left, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "time (s)", ((left + right) // 2 - 35, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Fz (N)", (8, (top + bottom) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        data = list(history)
        if not data:
            cv2.putText(canvas, "Waiting for force samples...", (left + 20, top + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
            return canvas

        xs = np.array([d[0] for d in data], dtype=np.float32) / fps
        ys = np.array([d[1] for d in data], dtype=np.float32)
        y_min = float(np.min(ys))
        y_max = float(np.max(ys))
        if abs(y_max - y_min) < 1e-3:
            y_min -= 1.0
            y_max += 1.0
        else:
            pad = max(0.5, 0.12 * (y_max - y_min))
            y_min -= pad
            y_max += pad

        x_min = float(xs[0])
        x_max = float(xs[-1]) if float(xs[-1]) > x_min else x_min + 1.0 / fps

        # 坐标刻度
        for frac in np.linspace(0, 1, 5):
            y = int(bottom - frac * (bottom - top))
            val = y_min + frac * (y_max - y_min)
            cv2.line(canvas, (left, y), (right, y), (55, 55, 55), 1)
            cv2.putText(canvas, f"{val:+.1f}", (8, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        for frac in np.linspace(0, 1, 5):
            x = int(left + frac * (right - left))
            val = x_min + frac * (x_max - x_min)
            cv2.line(canvas, (x, top), (x, bottom), (45, 45, 45), 1)
            cv2.putText(canvas, f"{val:.1f}", (x - 15, bottom + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        pts = []
        for x_val, y_val in zip(xs, ys):
            x = int(left + (float(x_val) - x_min) / (x_max - x_min) * (right - left))
            y = int(bottom - (float(y_val) - y_min) / (y_max - y_min) * (bottom - top))
            pts.append((x, y))
        if len(pts) >= 2:
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False,
                          (0, 220, 255), 2, cv2.LINE_AA)

        latest = ys[-1]
        latest_t = xs[-1]
        cv2.circle(canvas, pts[-1], 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, f"current: t={latest_t:.2f}s  Fz={latest:+.3f} N",
                    (left + 15, top + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 255), 2, cv2.LINE_AA)
        return canvas

    latest = None
    window_name = "AutoReplay Live: cameras + Fz"
    cam_size = (420, 316)
    graph_size = (840, 300)
    try:
        while not stop_event.is_set():
            try:
                while True:
                    msg = data_queue.get_nowait()
                    if msg is None:
                        stop_event.set()
                        break
                    if msg.get("reset"):
                        history.clear()
                        latest = None
                        continue
                    latest = msg
                    if "fz" in msg and msg["fz"] is not None:
                        history.append((int(msg["frame_idx"]), float(msg["fz"])))
            except queue.Empty:
                pass

            global_img = to_bgr(None if latest is None else latest.get("global"), cam_size)
            wrist_img = to_bgr(None if latest is None else latest.get("wrist"), cam_size)
            put_label(global_img, "global camera")
            put_label(wrist_img, "wrist camera")
            graph = draw_fz_graph(graph_size)
            canvas = np.vstack([np.hstack([global_img, wrist_img]), graph])
            cv2.imshow(window_name, canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break
            time.sleep(0.01)
    finally:
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass


class LiveReplayVisualizer:
    """独立进程显示两路相机和Z向力曲线；主采集进程不等待绘图。"""

    def __init__(self, window=300, fps=20, queue_size=2):
        ctx = mp.get_context("spawn")
        self.queue = ctx.Queue(maxsize=max(int(queue_size), 1))
        self.stop_event = ctx.Event()
        self.process = ctx.Process(
            target=_live_visualizer_process,
            args=(self.queue, self.stop_event, int(window), float(fps)),
            daemon=True,
        )
        self.frame_idx = 0
        self.enabled = False

    def start(self):
        if self.enabled:
            return
        self.process.start()
        self.enabled = True

    def _send_nonblocking(self, msg):
        if not self.enabled:
            return
        try:
            self.queue.put_nowait(msg)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(msg)
            except queue.Full:
                pass

    def reset(self):
        self.frame_idx = 0
        self._send_nonblocking({"reset": True})

    def update(self, obs):
        ft = obs.get("ft")
        fz = float(ft[2]) if ft is not None and len(ft) >= 3 else None
        self._send_nonblocking({
            "frame_idx": self.frame_idx,
            "fz": fz,
            "global": obs.get("global"),
            "wrist": obs.get("wrist"),
        })
        self.frame_idx += 1

    def close(self):
        if not self.enabled:
            return
        self.stop_event.set()
        try:
            self.queue.put_nowait(None)
        except Exception:
            pass
        self.process.join(timeout=2.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)
        self.enabled = False


# ============ 机器人环境 (自包含) ============
class ReplayEnv:
    """
    精简的机器人环境, 只保留回放需要的功能:
    - 机械臂连接 + 状态回调
    - 相机采集
    - 触觉采集
    - movep_canfd 执行
    """

    def __init__(self, config: AutoReplayConfig):
        import cv2
        self.cv2 = cv2
        self.cfg = config
        self._state_lock = threading.Lock()
        self._state_snapshot = None
        self.last_pose_quat = None

        # --- 1. 连接机械臂 ---
        from Robotic_Arm.rm_robot_interface import (
            RoboticArm, rm_thread_mode_e,
            rm_realtime_push_config_t, rm_realtime_arm_state_callback_ptr,
        )
        self._rm_callback_ptr_cls = rm_realtime_arm_state_callback_ptr

        print("📡 连接机械臂...")
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = self.arm.rm_create_robot_arm(config.ROBOT_IP, 8080)
        if handle.id == -1:
            raise RuntimeError(f"❌ 连接失败: {config.ROBOT_IP}")
        self.arm.rm_clear_system_err()
        print(f"✅ 机械臂已连接: {config.ROBOT_IP}")

        # 实时状态推送
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((config.ROBOT_IP, 8080))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            raise RuntimeError("❌ 获取本机IP失败")

        for port in [8098, 8099, 8097]:
            cfg_push = rm_realtime_push_config_t(1, True, port, 2, local_ip)
            ret = self.arm.rm_set_realtime_push(cfg_push)
            if ret == 0:
                print(f"✅ 实时推送 (端口: {port})")
                break

        self._arm_callback_enabled = True
        self._cb_ptr = self._rm_callback_ptr_cls(self._arm_state_callback)
        self.arm.rm_realtime_arm_state_call_back(self._cb_ptr)
        time.sleep(0.5)

        # --- 2. 相机 ---
        self.cameras = {}
        if config.CAMERAS:
            from realman_env.camera.rs_capture import RSCapture
            for name, cam_cfg in config.CAMERAS.items():
                cap = RSCapture(
                    name=name,
                    serial_number=cam_cfg["serial"],
                    dim=cam_cfg["dim"],
                    fps=cam_cfg["fps"],
                    depth=False,
                    exposure=cam_cfg["exposure"],
                )
                self.cameras[name] = cap
            print(f"✅ 相机: {list(self.cameras.keys())}")

        # --- 3. 触觉 ---
        self.tac_cap = None
        if config.TACTILE_SENSORS:
            from realman_env.tactile.tactile_capture import TactileCapture
            from realman_env.camera.video_capture import VideoCapture
            tac = TactileCapture(
                tactile_sensors=config.TACTILE_SENSORS,
                modalities=config.TACTILE_MODALITIES,
            )
            self.tac_cap = VideoCapture(tac)
            print(f"✅ 触觉: {list(config.TACTILE_SENSORS.keys())}")

        print("✅ 环境初始化完成\n")

    def _arm_state_callback(self, data):
        """机械臂状态回调"""
        if not self._arm_callback_enabled:
            return
        try:
            pose_euler = np.array([
                data.waypoint.position.x, data.waypoint.position.y, data.waypoint.position.z,
                data.waypoint.euler.rx, data.waypoint.euler.ry, data.waypoint.euler.rz,
            ], dtype=np.float32)
            pose_quat = np.array([
                data.waypoint.position.x, data.waypoint.position.y, data.waypoint.position.z,
                data.waypoint.quaternion.w, data.waypoint.quaternion.x,
                data.waypoint.quaternion.y, data.waypoint.quaternion.z,
            ], dtype=np.float32)
            joint = np.array(data.joint_status.joint_position, dtype=np.float32)

            force = np.array(data.force_sensor.zero_force[:3], dtype=np.float32)
            torque = np.array(data.force_sensor.zero_force[3:], dtype=np.float32)
            ft_raw = np.concatenate([force, torque])

            # 低通滤波
            with self._state_lock:
                prev = self._state_snapshot
            if prev is not None and prev.get("ft") is not None:
                dt = 5e-3
                cutoff = 50.0
                gain = dt / (dt + 1.0 / (2 * np.pi * cutoff))
                ft = (gain * ft_raw + (1 - gain) * prev["ft"]).astype(np.float32)
            else:
                ft = ft_raw.copy()

            with self._state_lock:
                self._state_snapshot = {
                    "pose": pose_euler,
                    "pos_quat": pose_quat,
                    "joint": joint,
                    "ft": ft,
                    "ts": time.perf_counter(),
                }
        except Exception:
            pass

    def get_state_snapshot(self):
        with self._state_lock:
            snap = self._state_snapshot
            if snap is None:
                return None
            return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in snap.items()}

    def get_obs(self):
        """采集一帧完整观测"""
        import cv2

        snap = self.get_state_snapshot()
        if snap is None:
            time.sleep(0.05)
            snap = self.get_state_snapshot()
            if snap is None:
                raise RuntimeError("未收到机械臂状态")

        obs = {}

        # 相机
        for name, camera in self.cameras.items():
            ret, img = camera.read()
            if ret and img is not None:
                if name in self.cfg.CAMERAS:
                    img = self.cfg.CAMERAS[name]["crop"](img)
                img = cv2.resize(img, self.cfg.IMAGE_SIZE)
                obs[name] = img[..., ::-1]  # BGR→RGB

        # 触觉
        if self.tac_cap is not None:
            ret, tac_frame = self.tac_cap.read()
            if ret and tac_frame is not None and isinstance(tac_frame, dict):
                for side, side_payload in tac_frame.items():
                    if side_payload is None or not isinstance(side_payload, dict):
                        continue
                    for modality, value in side_payload.items():
                        if value is not None:
                            obs[f"tac_{side}_{modality}"] = value

        # Proprio
        obs["proprio_eef"] = snap["pose"].copy()
        obs["proprio_joint"] = snap["joint"].copy()
        obs["ft"] = snap["ft"].copy()

        return obs

    def close(self):
        print("\n🧹 关闭环境...")
        self._arm_callback_enabled = False
        time.sleep(0.02)
        for camera in self.cameras.values():
            camera.close()
        if self.tac_cap is not None:
            self.tac_cap.close()
        self.arm.rm_delete_robot_arm()
        print("✅ 环境已关闭")


# ============ 回放逻辑 ============
def replay_one_episode(env, trajectory, config, verbose=True, visualizer=None):
    """回放一条轨迹, 返回(observations, action_buffers, stopped_early)"""
    T = len(trajectory)
    observations = []
    action_buffers = {"eef_abs": [], "joint_abs": []}
    stopped_early = False

    obs = env.get_obs()
    observations.append(obs)
    if visualizer is not None:
        visualizer.update(obs)

    for step in range(T):
        t0 = time.perf_counter()

        # 暂停检测 (空格键)
        if check_pause():
            print(f"\n⏸️  已暂停 (step {step}/{T}) — 按空格继续")
            # 等待再次按空格恢复
            while True:
                time.sleep(0.05)
                if check_pause():
                    break
            print("▶️  继续回放...")

        target_euler = trajectory[step]
        target_quat = poseuler_to_posquat(target_euler)

        # 安全检查
        snap = env.get_state_snapshot()
        if snap is not None and snap.get("ft") is not None:
            safe, reason = check_force_safety(snap["ft"], config)
            if not safe:
                print(f"\n⚠️  力过大停止! step={step}/{T}, {reason}")
                stopped_early = True
                break

        # 发送运动指令
        if config.USE_INTERPOLATION and env.last_pose_quat is not None:
            path = np.linspace(env.last_pose_quat, target_quat, config.N_SUBSTEPS + 1)[1:]
            t_base = time.perf_counter()
            for k, p in enumerate(path, start=1):
                env.arm.rm_movep_canfd(p.tolist(), True)
                t_target = t_base + k * config.CTRL_DT
                while time.perf_counter() < t_target:
                    time.sleep(0.001)
        else:
            env.arm.rm_movep_canfd(target_quat.tolist(), True)

        env.last_pose_quat = target_quat.copy()

        # 记录动作
        action_buffers["eef_abs"].append(target_euler.astype(np.float32))
        if snap is not None:
            action_buffers["joint_abs"].append(snap["joint"].astype(np.float32))
        else:
            action_buffers["joint_abs"].append(np.zeros(7, dtype=np.float32))

        # 控制频率
        elapsed = time.perf_counter() - t0
        time.sleep(max(0, (1.0 / config.CONTROL_HZ) - elapsed))

        # 采集obs
        obs = env.get_obs()
        observations.append(obs)
        if visualizer is not None:
            visualizer.update(obs)

        if verbose and (step + 1) % 50 == 0:
            print(f"  step {step+1}/{T} ({(step+1)/config.CONTROL_HZ:.1f}s), "
                  f"pos=[{target_euler[0]:.3f},{target_euler[1]:.3f},{target_euler[2]:.3f}]")

    # 对齐
    observations = observations[:len(action_buffers["eef_abs"])]

    # 结束动作: 抬起
    lift_height = config.LIFT_AFTER_DONE
    last_euler = trajectory[min(step, T-1)].copy()
    lift_target = last_euler.copy()
    lift_target[2] += lift_height
    lift_quat = poseuler_to_posquat(lift_target)

    start_quat = env.last_pose_quat if env.last_pose_quat is not None else poseuler_to_posquat(last_euler)
    n_lift = 10
    for i in range(n_lift):
        t_blend = (i + 1) / n_lift
        p = start_quat + t_blend * (lift_quat - start_quat)
        env.arm.rm_movep_canfd(p.tolist(), True)
        time.sleep(1.0 / config.CONTROL_HZ)
    env.last_pose_quat = lift_quat.copy()

    if verbose:
        print(f"  ↑ 抬起 {lift_height*1000:.0f}mm 完成")

    return observations, action_buffers, stopped_early


# ============ 保存 ============
def save_episode(observations, action_buffers, save_path, config):
    """保存为HDF5"""
    import h5py

    ep_len = len(observations)
    if ep_len < 5:
        print(f"⚠️  数据太短({ep_len}步), 跳过")
        return False

    tactile_keys = [k for k in observations[0].keys() if k.startswith("tac_")]
    data_dict = {}

    # 图像
    for cam in ["global", "wrist"]:
        if cam in observations[0]:
            data_dict[f"observations/images/{cam}"] = np.array(
                [obs[cam] for obs in observations], dtype=np.uint8)

    # Proprio
    if "proprio_eef" in observations[0]:
        data_dict["observations/proprio_eef"] = np.array(
            [obs["proprio_eef"] for obs in observations], dtype=np.float32)
    if "proprio_joint" in observations[0]:
        data_dict["observations/proprio_joint"] = np.array(
            [obs["proprio_joint"] for obs in observations], dtype=np.float32)

    # 力
    if "ft" in observations[0]:
        data_dict["ft"] = np.array(
            [obs["ft"] for obs in observations], dtype=np.float32)

    # 动作
    for name, buf in action_buffers.items():
        if buf:
            data_dict[f"actions/{name}"] = np.array(buf, dtype=np.float32)

    # 触觉
    for k in tactile_keys:
        suffix = k[len("tac_"):]
        if "_" not in suffix:
            side, modality = suffix, "img"
        else:
            side, modality = suffix.split("_", 1)
        data_dict[f"observations/tac/{side}/{modality}"] = np.array(
            [obs[k] for obs in observations])

    # 写HDF5
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    compression = config.H5_COMPRESSION
    with h5py.File(str(save_path), "w") as f:
        for key, value in data_dict.items():
            if compression and str(compression).lower() != "none":
                f.create_dataset(key, data=value, compression=str(compression))
            else:
                f.create_dataset(key, data=value)

    print(f"✅ 已保存 {ep_len} 步 → {save_path}")
    return True


# ============ Dry Run ============
def dry_run(trajectory, config):
    """干跑: 不连机器人, 检查轨迹"""
    T = len(trajectory)
    print(f"\n{'='*50}")
    print(f"  DRY RUN — 轨迹检查")
    print(f"{'='*50}")
    print(f"  步数: {T}")
    print(f"  时长: {T/config.CONTROL_HZ:.1f}s")
    print(f"  起点: xyz=[{trajectory[0,0]:.4f}, {trajectory[0,1]:.4f}, {trajectory[0,2]:.4f}]")
    print(f"  终点: xyz=[{trajectory[-1,0]:.4f}, {trajectory[-1,1]:.4f}, {trajectory[-1,2]:.4f}]")
    print(f"  X范围: [{trajectory[:,0].min():.4f}, {trajectory[:,0].max():.4f}]")
    print(f"  Y范围: [{trajectory[:,1].min():.4f}, {trajectory[:,1].max():.4f}]")
    print(f"  Z范围: [{trajectory[:,2].min():.4f}, {trajectory[:,2].max():.4f}]")

    vel = np.diff(trajectory[:, :3], axis=0) * config.CONTROL_HZ
    speed = np.linalg.norm(vel, axis=1) * 1000
    print(f"  速度: mean={speed.mean():.1f} mm/s, max={speed.max():.1f} mm/s")

    acc = np.diff(vel, axis=0) * config.CONTROL_HZ
    acc_norm = np.linalg.norm(acc, axis=1) * 1000
    print(f"  加速度: mean={acc_norm.mean():.0f} mm/s², max={acc_norm.max():.0f} mm/s²")

    reset = np.array(config.RESET_POSE[:3])
    start_dist = np.linalg.norm(trajectory[0, :3] - reset) * 1000
    print(f"\n  起点距RESET: {start_dist:.1f} mm", end="")
    print(" ✅" if start_dist < 50 else " ⚠️  较远!")

    z_min = trajectory[:, 2].min()
    print(f"  最低Z: {z_min*1000:.1f} mm", end="")
    print(" ✅" if z_min > 0.120 else " ⚠️  可能撞桌面!")

    max_speed = speed.max()
    print(f"  最大速度: {max_speed:.1f} mm/s", end="")
    print(" ✅" if max_speed < 50 else " ⚠️  较快!")

    print(f"\n{'='*50}\n")


# ============ 主函数 ============
def get_next_episode_index(save_dir):
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return 0
    existing = []
    pat = re.compile(r"^episode_(\d+)\.hdf5$")
    for f in save_dir.iterdir():
        m = pat.match(f.name)
        if m:
            existing.append(int(m.group(1)))
    return max(existing) + 1 if existing else 0


def load_completed_replays(log_path: Path):
    """Load successfully completed (trajectory, repeat) pairs from jsonl log."""
    completed = set()
    if not log_path.exists():
        return completed

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("status") != "success":
                continue
            if item.get("stopped_early"):
                continue
            traj_path = item.get("traj_path")
            repeat_idx = item.get("repeat_idx")
            if traj_path is None or repeat_idx is None:
                continue
            completed.add((str(Path(traj_path).resolve()), int(repeat_idx)))
    return completed


def append_replay_log(log_path: Path, record: dict):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _resolve_trajectory_path(raw_path: str, script_dir: Path) -> Path:
    """Resolve a trajectory path against common locations.

    Priority:
    1. As provided (absolute or relative to current working directory)
    2. Relative to the script directory
    3. Common subdirectories under both cwd and script directory
    """
    raw = Path(raw_path)
    candidates = []

    def _add(path: Path):
        if path not in candidates:
            candidates.append(path)

    _add(raw)
    if not raw.is_absolute():
        _add(script_dir / raw)

        common_roots = [
            Path.cwd(),
            script_dir,
            Path.cwd() / "wipe_pos",
            script_dir / "wipe_pos",
            Path.cwd() / "trajectories",
            script_dir / "trajectories",
        ]
        for root in common_roots:
            _add(root / raw.name)

    for path in candidates:
        if path.exists():
            return path.resolve()

    tried = "\n  - " + "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"未找到轨迹文件: {raw_path}\n尝试过的路径:{tried}")


def _resolve_trajectory_dir(raw_dir: str, script_dir: Path) -> Path:
    raw = Path(raw_dir)
    candidates = []

    def _add(path: Path):
        if path not in candidates:
            candidates.append(path)

    _add(raw)
    if not raw.is_absolute():
        _add(script_dir / raw)
        _add(Path.cwd() / raw)
        _add(script_dir / raw.name)
        _add(Path.cwd() / raw.name)

    for path in candidates:
        if path.exists():
            return path.resolve()

    tried = "\n  - " + "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"未找到轨迹目录: {raw_dir}\n尝试过的路径:{tried}")


def main():
    parser = argparse.ArgumentParser(description="自动轨迹回放+录制 (独立版)")
    parser.add_argument("--traj", type=str, default=None, help="单条轨迹 (.npy)")
    parser.add_argument("--traj_dir", type=str, default=None, help="轨迹目录")
    parser.add_argument("--filter", type=str, default=None, help="文件名过滤")
    parser.add_argument("--save_dir", type=str, required=True, help="HDF5保存目录")
    parser.add_argument("--n_repeats", type=int, default=1, help="每条重复次数")
    parser.add_argument("--dry_run", action="store_true", help="干跑模式")
    parser.add_argument("--no_tactile", action="store_true", help="不采集触觉")
    parser.add_argument("--no_camera", action="store_true", help="不采集相机")
    parser.add_argument("--max_force_z", type=float, default=30.0, help="Z力上限(N)")
    parser.add_argument("--interpolation", action="store_true", help="开启子步插值")
    parser.add_argument("--robot_ip", type=str, default="192.168.1.18", help="机械臂IP")
    parser.add_argument("--start_index", type=int, default=0,
                        help="手动从排序后的第几个轨迹文件开始(0-based), 例如38表示跳过前38条")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                        help="根据采集日志自动跳过已成功回放的轨迹(repeat级别), 默认开启")
    parser.add_argument("--no_resume", dest="resume", action="store_false",
                        help="关闭自动续采跳过逻辑")
    parser.add_argument("--resume_log", type=str, default=None,
                        help="采集日志jsonl路径, 默认保存到 save_dir/collection_log.jsonl")
    parser.add_argument("--visualize_live", action="store_true",
                        help="实时显示global/wrist相机和机器人FT Z向力曲线, 不启用报警")
    parser.add_argument("--visualize_fz_window", type=int, default=300,
                        help="实时Fz曲线显示最近多少个采样点")
    args = parser.parse_args()

    config = AutoReplayConfig()
    config.ROBOT_IP = args.robot_ip
    config.MAX_FORCE_Z = args.max_force_z
    config.USE_INTERPOLATION = args.interpolation
    config.VISUALIZE_LIVE = args.visualize_live
    config.VISUALIZE_FZ_WINDOW = args.visualize_fz_window
    if args.no_tactile:
        config.TACTILE_SENSORS = {}
    if args.no_camera:
        config.CAMERAS = {}

    script_dir = Path(__file__).resolve().parent

    # 收集轨迹文件
    traj_files = []
    if args.traj:
        traj_path = _resolve_trajectory_path(args.traj, script_dir)
        traj_files.append(traj_path)
    elif args.traj_dir:
        traj_dir = _resolve_trajectory_dir(args.traj_dir, script_dir)
        for f in sorted(traj_dir.glob("*.npy")):
            if args.filter and args.filter not in f.stem:
                continue
            traj_files.append(f)
    else:
        print("错误: 请指定 --traj 或 --traj_dir")
        sys.exit(1)

    if not traj_files:
        print("错误: 未找到轨迹文件")
        sys.exit(1)

    if args.start_index < 0:
        print("错误: --start_index 不能小于0")
        sys.exit(1)
    if args.start_index >= len(traj_files):
        print(f"错误: --start_index={args.start_index} 超出轨迹数量 {len(traj_files)}")
        sys.exit(1)
    if args.start_index > 0:
        print(f"↪️  手动从轨迹索引 {args.start_index} 开始, 跳过前 {args.start_index} 条")
        traj_files = traj_files[args.start_index:]

    print(f"📋 待检查 {len(traj_files)} 条轨迹, 每条重复 {args.n_repeats} 次")
    for f in traj_files:
        print(f"  - {f.name}")

    # Dry run
    if args.dry_run:
        for f in traj_files:
            traj = np.load(str(f))
            print(f"\n--- {f.name} ---")
            dry_run(traj, config)
        return

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ep_idx = get_next_episode_index(save_dir)
    resume_log = Path(args.resume_log) if args.resume_log else save_dir / "collection_log.jsonl"
    completed = load_completed_replays(resume_log) if args.resume else set()

    replay_items = []
    skipped_done = 0
    for traj_file in traj_files:
        traj_key = str(Path(traj_file).resolve())
        for rep in range(args.n_repeats):
            if args.resume and (traj_key, rep) in completed:
                skipped_done += 1
                continue
            replay_items.append((traj_file, rep))

    print(f"\n{'='*50}")
    print(f"  自动回放模式")
    print(f"  保存目录: {save_dir}")
    print(f"  起始索引: episode_{ep_idx}")
    print(f"  续采日志: {resume_log}")
    print(f"  已跳过成功项: {skipped_done}")
    print(f"  本次待回放项: {len(replay_items)}")
    print(f"{'='*50}")

    if not replay_items:
        print("✅ 没有需要回放的轨迹")
        return

    total_saved = 0
    env = None
    visualizer = None
    old_term_settings = None
    try:
        # 终端设为raw模式 (支持非阻塞按键检测: 空格暂停)
        import termios
        import tty
        old_term_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        # 真机模式
        env = ReplayEnv(config)
        if config.VISUALIZE_LIVE:
            visualizer = LiveReplayVisualizer(
                window=config.VISUALIZE_FZ_WINDOW,
                fps=config.CONTROL_HZ,
            )
            visualizer.start()

        last_traj_file = None
        traj = None
        for traj_file, rep in replay_items:
            if last_traj_file != traj_file:
                traj = np.load(str(traj_file))
                last_traj_file = traj_file
                print(f"\n📂 轨迹: {traj_file.name} ({len(traj)} steps, {len(traj)/config.CONTROL_HZ:.1f}s)")
            assert traj is not None
            if visualizer is not None:
                visualizer.reset()

            print(f"\n{'='*50}")
            print(f"  待采集: episode_{ep_idx} (轨迹: {traj_file.stem}, repeat {rep+1}/{args.n_repeats})")
            print(f"  按 Enter 开始 (先复位→再回放), Ctrl+C 退出")
            print(f"{'='*50}")

            # 等待Enter键 (raw模式下手动检测)
            while True:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key in ('\n', '\r'):
                        break
                time.sleep(0.05)

            # 复位
            print("🔄 复位到初始位置...")
            env.arm.rm_movel(config.RESET_POSE, v=15, r=0, connect=0, block=1)
            time.sleep(1.0)
            snap = env.get_state_snapshot()
            if snap is not None:
                env.last_pose_quat = snap["pos_quat"].copy()
            print("✅ 复位完成")
            time.sleep(0.5)

            # 回放
            print("▶️  开始回放...")
            obs_list, act_bufs, stopped = replay_one_episode(
                env, traj, config, verbose=True, visualizer=visualizer)

            if stopped:
                print("⚠️  提前停止, 仍保存已录数据")

            # 保存
            save_path = save_dir / f"episode_{ep_idx}.hdf5"
            success = save_episode(obs_list, act_bufs, save_path, config)
            if success:
                append_replay_log(resume_log, {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "status": "success",
                    "episode_idx": ep_idx,
                    "save_path": str(save_path.resolve()),
                    "traj_path": str(Path(traj_file).resolve()),
                    "traj_name": traj_file.name,
                    "repeat_idx": rep,
                    "repeat_total": args.n_repeats,
                    "num_steps_saved": len(obs_list),
                    "stopped_early": bool(stopped),
                })
                ep_idx += 1
                total_saved += 1

    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断")

    finally:
        if visualizer is not None:
            visualizer.close()
        if env is not None:
            env.close()
        # 恢复终端设置
        if old_term_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term_settings)
        print(f"\n✅ 完成! 共保存 {total_saved} 条episode → {save_dir}")


if __name__ == "__main__":
    main()
