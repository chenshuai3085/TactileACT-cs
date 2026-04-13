#!/usr/bin/env python3
import threading
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent))

import numpy as np
import time
import h5py
import cv2
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from Robotic_Arm.rm_robot_interface import *

# ========== 视觉采集模式配置 ==========
# True:  使用异步模式（read() < 1ms，推荐）
# False: 使用阻塞模式（read() ~65ms，原始模式）
USE_ASYNC_VISION = True  # 将在命令行参数解析后覆盖
# ====================================

# ========== 触觉采集模式配置 ==========
# True:  使用最新帧覆盖模式（延迟 0-33ms，推荐用于训练）
# False: 使用队列缓存模式（延迟 33-66ms，原始模式）
USE_LATEST_TAC_MODE = True  # 将在命令行参数解析后覆盖
# =====================================

# ========== 性能调试打印 ==========
# True:  打印详细性能指标（movep/obs/cam/tac 耗时）
# False: 不打印（正常采集时使用）
PRINT_PERFORMANCE = False  # 将在命令行参数解析后覆盖
# ===================================

# 延迟选择采集实现：在 main() 中根据命令行决定具体类
# 这里先占位，避免提前绑定导致命令行切换不生效
RSCapture = None
VideoCapture = None

USE_SIGMA7 = True
CONTROL_MODE = "abs"  # "inc" 增量模式，后面绝对控制时会用 "abs"
if USE_SIGMA7 and CONTROL_MODE == "inc":
    from realman_env.spacemouse.sigma7_expert_inc import Sigma7Expert as TeleopExpert
elif USE_SIGMA7 and CONTROL_MODE == "abs":
    from realman_env.spacemouse.sigma7_expert_abs import Sigma7Expert as TeleopExpert
elif USE_SIGMA7 and CONTROL_MODE == "van":
    from realman_env.spacemouse.sigma7_expert_vanilla import Sigma7Expert as TeleopExpert
else:
    from realman_env.spacemouse.spacemouse_expert import SpaceMouseExpert as TeleopExpert


# ⚠️ 直接导入原版导纳控制器（不重写！）
from realman_env.robot_servers.admittance_controller import AdmittanceController

from realman_env.tactile.tactile_capture import TactileCapture


# ============ 工具 / 任务坐标系配置 ============
# 夹爪相对末端：绕 Z 轴正转 15° 安装
TOOL_Z_ROT_DEG = 15.0
TOOL_Z_ROT_RAD = np.deg2rad(TOOL_Z_ROT_DEG)

# 轴尾端相对末端原点，沿末端 Z 轴的偏移（单位：米）
# ⚠️ 这里需要你填真实的 dz 值（正方向：沿末端 Z 指向夹爪/轴的方向）
TOOL_Z_OFFSET = 0.20  # TODO: 比如 0.08

R_ET = np.array([  # 把 {T} 中的向量表示成 {E} 的旋转矩阵，满足 R_ET @ v_T = v_E
    [np.cos(TOOL_Z_ROT_RAD), -np.sin(TOOL_Z_ROT_RAD), 0.0],
    [np.sin(TOOL_Z_ROT_RAD),  np.cos(TOOL_Z_ROT_RAD), 0.0],
    [0.0,                     0.0,                    1.0],
])
p_TE = np.array([0.0, 0.0, TOOL_Z_OFFSET])

# ============ 配置（与原版pih.py对齐）============
class Config:
    # 机械臂
    ROBOT_IP = "192.168.1.18"
    CONTROL_HZ = 20  # ⚠️ 主频率
    N_SUBSTEPS = 5   # ⚠️ 子步数（与原版RMController一致）
    CTRL_DT = 1.0 / (CONTROL_HZ * N_SUBSTEPS)  # 子步时间间隔
    # 是否使用插值
    USE_INTERPOLATION = False
    # 插值起点的选择
    # "measured"：用当前实际位姿（强烈推荐）
    # "commanded"：用上一条命令（仅用于调试/对比）
    INTERP_START_MODE = "measured"
    
    # 导纳控制参数（与原版admittance_controller.py一致）
    ADMITTANCE_ENABLED = False  # ⚠️ 启用导纳控制
    ADMITTANCE_DT = 0.01  # 子步时间间隔 (100Hz)
    # 注意：其他导纳参数在AdmittanceController类中已定义，直接使用原版
    # SpaceMouse缩放（与pih.py的ACTION_SCALE一致）
    # FT_SCALE = np.array([0.12, 0.2])  # 力反馈缩放  # 第一轮数据采集时：0.1,0.08 第4次0.15-0.12
    # FT_SCALE = np.array([0.12, 2])  # 260114
    # FT_SCALE = np.array([0.12, 1.5])  # 260114
    # FT_SCALE = np.array([0.12, 0.5])  # 260130
    # FT_SCALE = np.array([0.12, 0.7])  # 260202
    FT_SCALE = np.array([0.15, 0.7])  # 260209


    if USE_SIGMA7 and CONTROL_MODE == "inc":
        # 增量版：可以稍大一点，因为每步是小增量
        ACTION_SCALE = np.array([10, 1.5])
    elif USE_SIGMA7 and CONTROL_MODE == "abs":
        # 绝对版：偏移直接叠加到起点，建议先保守
        # ACTION_SCALE = np.array([1.5, 0.6])  # 第一轮数采集时：1.0,0.4
        ACTION_SCALE = np.array([1, 0.4])  # 第一轮数采集时：1.0,0.4
    else:
        ACTION_SCALE = np.array([0.002, 0.0125])

    # 数据采集
    MAX_EPISODE_LENGTH = 500
    DATA_DIR = Path("/home/czy/dataset/pih/260413/peg_in_hole_0413")

    # 当遥操作设备（如 Sigma7）从“启用->禁用”时，是否自动结束当前 episode
    # - True: 关掉 Sigma7 开关就立刻结束（动态步数）
    # - False: 仍然以 MAX_EPISODE_LENGTH 或按 q 结束
    END_ON_TELEOP_DISABLE = True
    # ===== 状态存储类型：可选 "eef" / "joint" =====
    # "eef"  -> 保存绝对末端位姿  (observations/proprio_eef)
    # "joint"-> 保存绝对关节角    (observations/proprio_joint)
    STATE_TYPES = ("eef", "joint")  # 想只存一种就改成 ("eef",) 或 ("joint",)
    ACTION_TYPES = ("eef_abs", "joint_abs")  # ("eef_abs", "eef_rel", "joint_abs")

    # 相机配置
    CAMERAS = {
        "global": {
            "serial": "130322273140",
            "dim": (640, 480),
            "fps": 30,
            "exposure": 15000,
            # 下面的 x 赋值，维度一是行y，维度二是列x ——— NumPy / OpenCV 风格： x[y1:y2, x1:x2] 
            # img[start_row:end_row, start_col:end_col] 
            "crop": lambda x: x[90:290, 224:490],  # 12/10数采 H200 W266 img = img.crop((224, 90, 490, 290)) x1 y1 x2 y2
        },
        "wrist": {
            "serial": "230322271557",
            "dim": (640, 480),
            "fps": 30,
            "exposure": 15000,
            "crop": lambda x: x[:, :],
        }
    }
    TACTILE_SENSORS = {
        "left": {
            "sn": "GF2250032BAE6",
            "dim": (240, 240),
        },
        "right": {
            "sn": "GF2250002C848",
            "dim": (240, 240),
        }
    }
    # 触觉要采集/保存的模态（由 tactile_capture.py 支持）
    # 可选：img/diff/depth/marker_img/marker_current/marker_offset/force6d
    TACTILE_MODALITIES = ("img",)

    # HDF5 压缩："gzip" 或 "none"
    H5_COMPRESSION = "gzip"
    
    # IMAGE_SIZE = (640, 480)
    IMAGE_SIZE = (266, 200)   # OpenCV 语义是 (W, H)
    # 是否启动可视化（在单独线程显示相机/触觉），默认关闭
    VISUALIZE_SENSORS = False
    # 可视化时是否显示裁剪后的图像（默认 False，命令行可用 --visualize-cropped）
    VISUALIZE_CROPPED = False

    # 是否实时可视化 6 维力传感器（obs['ft']），默认关闭
    VISUALIZE_FT = False
    # 力曲线刷新频率（Hz），过高可能占用 CPU
    VISUALIZE_FT_HZ = 20
    # 力曲线窗口（显示最近多少个 step 的数据）。若一条 episode 约 200 步，设为 >=200 可完整显示。
    VISUALIZE_FT_WINDOW = 300

    # 第五次数采不加noise
    # RESET_POSE = [0.344388, 0.015012, 0.23486, 3.137, 0.047, -2.774]
    RESET_POSE = [0.390345, -0.005217, 0.229337, 3.141, -0.007, -2.838]

# ============ 工具函数（复现原版transformations.py）============


def poseuler_to_posquat(pose_euler, order="xyz"):
    """欧拉角转四元数（与原版一致）"""
    pos = pose_euler[:3]
    euler = pose_euler[3:]
    quat = R.from_euler(order, euler).as_quat()  # [qx,qy,qz,qw]
    return np.concatenate([pos, [quat[3], quat[0], quat[1], quat[2]]])  # [x,y,z,qw,qx,qy,qz]


# ============ 简化版环境 ============
class SimplePIHEnv:
    """
    关键特性（与原版RMController一致）：
    - 绝对位姿控制
    - desired_tcp_pose作为积分器状态
    - 导纳控制器（力反馈补偿）
    - 子步插值（n_substeps=5, 100Hz）
    """
    
    def __init__(self, config: Config):
        self._state_lock = threading.Lock()
        self._state_snapshot = None  # dict: {"pose":..., "pos_quat":..., "joint":..., "ft":..., "ts":...}
        self.cfg = config
        self.step_count = 0
        
        # ✅ 提前初始化回调中会用到的属性（避免 AttributeError）
        self.ft_bias = np.zeros(6, dtype=float)
        self.ft_bias_ready = False
        self._arm_callback_enabled = True
        self.admittance = None  # 稍后根据配置初始化
        
        # 1. 连接机械臂
        print("📡 连接机械臂...")
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(config.ROBOT_IP, 8080)
        if self.handle.id == -1:
            raise RuntimeError(f"❌ 连接失败: {config.ROBOT_IP}")
        self.arm.rm_clear_system_err()
        print(f"✅ 机械臂已连接: {config.ROBOT_IP} (handle: {self.handle.id})")
        
        # 设置实时状态回调（带重试机制）
        print("⚙️  配置实时推送...")
        # 尝试获取本机IP（在192.168.1.x网段）
        import socket
        try:
            # 创建UDP socket连接到机械臂IP来获取本机IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((config.ROBOT_IP, 8080))
            local_ip = s.getsockname()[0]
            s.close()
            print(f"   检测到本机IP: {local_ip}")
        except:  # 报错并退出
            print("❌ 获取本机IP失败，无法配置实时推送")
            sys.exit(1)

        # 尝试不同的端口（如果8098被占用）
        for port in [8098, 8099, 8097]:
            cfg = rm_realtime_push_config_t(1, True, port, 2, local_ip)
            ret = self.arm.rm_set_realtime_push(cfg)
            if ret == 0:
                print(f"✅ 实时推送已配置 (端口: {port})")
                break
        else:
            print("⚠️  警告：实时推送配置失败，将使用轮询模式")

        self.arm_state_cb_ptr = rm_realtime_arm_state_callback_ptr(self._arm_state_callback)
        self.arm.rm_realtime_arm_state_call_back(self.arm_state_cb_ptr)
        # 控制回调是否生效（用于优雅关闭时避免回调继续访问已释放资源）
        self._arm_callback_enabled = True
        time.sleep(0.5)  # 等待回调建立
        
        # 2. 初始化相机
        self.cameras = {}
        
        for name, cam_cfg in config.CAMERAS.items():
            rs_cap = RSCapture(
                name=name,
                serial_number=cam_cfg["serial"],
                dim=cam_cfg["dim"],
                fps=cam_cfg["fps"],
                depth=False,
                exposure=cam_cfg["exposure"]
            )
            # 视觉采集与触觉独立：相机直接使用 RSCapture
            self.cameras[name] = rs_cap
        
        print(f"✅ 相机已初始化: {list(self.cameras.keys())}")
        
        # 3. 初始化触觉传感器（可选）
        self.tac_cap = None
        tactile_sensors = getattr(config, "TACTILE_SENSORS", None)

        if tactile_sensors:
            modalities = getattr(config, "TACTILE_MODALITIES", ("img",))
            tac = TactileCapture(tactile_sensors=tactile_sensors, modalities=modalities)
            # 触觉采集根据 USE_LATEST_TAC_MODE 选择 VideoCapture 实现
            self.tac_cap = VideoCapture(tac)
            print(f"✅ 触觉传感器已初始化: {list(tactile_sensors.keys())}")
        
        # 4. 初始化SpaceMouse
        self.spacemouse = TeleopExpert()
        print("✅ SpaceMouse已初始化")

        # --- Sigma7 绝对控制用到的起点状态 ---
        # 增量模式下这两个也存在，但不会用，不影响。
        self.sigma7_prev_enabled = False
        self.sigma7_robot_start_pose = None  # 机械臂起点位姿
        
        # ⚠️ 导纳控制器（直接使用原版！）
        if config.ADMITTANCE_ENABLED:
            self.admittance = AdmittanceController(dt=config.ADMITTANCE_DT)
            print("✅ 导纳控制器已初始化")
        # else: admittance 已在 __init__ 开头初始化为 None
        
        # ft_bias 和 ft_bias_ready 已在 __init__ 开头初始化

        # ⚠️ 关键：期望位姿状态（积分器）
        self.desired_tcp_pose = None  # [x,y,z, rx,ry,rz] 当前期望位姿
        self.last_pose_quat = None    # [x,y,z, qw,qx,qy,qz] 上一步指令位姿（用于子步插值）
        
        # --- (optional) caches for precise time alignment ---
        self._cmd_pose_t = None      # 本步动作（执行前就确定的目标位姿）

        print("✅ 环境初始化完成\n")

        # 最小化改动：按需启动外部可视化模块（在单独文件中实现）
        if getattr(config, 'VISUALIZE_SENSORS', False):
            try:
                # 直接导入 repo 内同级的 visualize_sensors 模块
                from visualize_sensors import SensorVisualizer
                use_cropped = getattr(config, 'VISUALIZE_CROPPED', False)
                self._sensor_visualizer = SensorVisualizer(self, use_cropped=use_cropped)
                self._sensor_visualizer.start()
                print("👁️  传感器可视化已启动")
            except Exception as e:
                print("⚠️ 无法启动传感器可视化:", e)

        # 可选：实时可视化 6 维力/矩（ft）
        self._ft_visualizer = None
        if getattr(config, 'VISUALIZE_FT', False):
            try:
                hz = getattr(config, 'VISUALIZE_FT_HZ', 20)
                window = getattr(config, 'VISUALIZE_FT_WINDOW', 300)
                from visualize_ft import ForceVisualizer
                self._ft_visualizer = ForceVisualizer(self, hz=hz, window=window)
                self._ft_visualizer.start()
                print(f"📈 六维力曲线可视化已启动 (hz={hz}, window={window})")
            except Exception as e:
                print("⚠️ 无法启动六维力曲线可视化:", e)

    
    def _arm_state_callback(self, data):
        """机械臂状态回调（读取位姿和力传感器）"""
        # 如果回调已被禁用（通常在 close() 期间），直接返回
        if not getattr(self, '_arm_callback_enabled', True):
            return

        try:
            pose_euler = np.array([
                data.waypoint.position.x,
                data.waypoint.position.y,
                data.waypoint.position.z,
                data.waypoint.euler.rx,
                data.waypoint.euler.ry,
                data.waypoint.euler.rz
            ]).astype(np.float32)
            pose_quat = np.array([
                data.waypoint.position.x,
                data.waypoint.position.y,
                data.waypoint.position.z,
                data.waypoint.quaternion.w,
                data.waypoint.quaternion.x,
                data.waypoint.quaternion.y,
                data.waypoint.quaternion.z
            ]).astype(np.float32)

            joint = np.array(
                data.joint_status.joint_position, dtype=np.float32
            )

            joint_current = np.array(
                data.joint_status.joint_current, dtype=np.float32
            )

            # 原始传感器数据（假设在 TCP 坐标系 {E} 下）
            force_tcp = np.array(data.force_sensor.zero_force[:3], dtype=np.float32)
            torque_tcp = np.array(data.force_sensor.zero_force[3:], dtype=np.float32)
            unfiltered_ft_tcp = np.concatenate([force_tcp, torque_tcp])

            # 低通滤波：用“上一次 snapshot 里的 ft”作为滤波状态（而不是 self.current_ft）
            with self._state_lock:
                prev = self._state_snapshot

            if prev is None or prev.get("ft") is None:  # 如果没有，直接初始化
                ft = unfiltered_ft_tcp.copy()

            else:
                dt = 5e-3  # UDP回调频率
                cutoff_freq = 50.0
                gain = dt / (dt + 1.0 / (2 * np.pi * cutoff_freq))
                ft = (gain * unfiltered_ft_tcp + (1 - gain) * prev["ft"]).astype(np.float32)

                
            # 上面，即：ft 就是 TCP 坐标系 {E} 下的力/矩，需要做“轴端映射”。
            # 新增：把力反馈给 sigma7 @ 2025-11-26  @ 2025-12-01 改为做轴端映射
            if USE_SIGMA7 and hasattr(self, "spacemouse"):
                    F_E = ft[:3]
                    tau_E = ft[3:]
                    # --- ① TCP {E} → 轴端任务坐标系 {T} ---
                    # 轴端相对末端的偏移：p_TE = [0, 0, dz]，在 {E} 下表达
                    # τ_T = R_ET^T (τ_E - p × F_E),  F_T = R_ET^T F_E
                    F_T = R_ET.T @ F_E
                    tau_T = R_ET.T @ (tau_E - np.cross(p_TE, F_E))

                    # --- ② 可以选择只用 任务相关分量，比如：
                    #    - F_T[2]: 插入方向阻力
                    #    - F_T[0:2]: 侧向碰撞
                    #    这里先把 3 维力都保留，扭矩先弱化处理
                    F_T_for_sigma = F_T.copy()
                    tau_T_for_sigma = 0.2 * tau_T  # 先给个小一点的转矩比例，避免“扭着手腕”

                    # --- ③ 再转回 TCP {E}，因为 sigma7_expert_abs 期望的是“realman 坐标系风格” ---
                    F_E_for_sigma = R_ET @ F_T_for_sigma
                    tau_E_for_sigma = R_ET @ tau_T_for_sigma

                    # --- ④ 在这个坐标系下做偏置估计 + 映射给 sigma7 ---
                    # 1）如果偏置还没估计好：在 reset 后的前若干个回调中做平均
                    if not self.ft_bias_ready:
                        acc = getattr(self, "_ft_bias_acc", None)
                        if acc is None:
                            # 第一帧
                            self._ft_bias_acc = np.concatenate([F_E_for_sigma, tau_E_for_sigma])
                            self._ft_bias_count = 1
                        else:
                            self._ft_bias_acc += np.concatenate([F_E_for_sigma, tau_E_for_sigma])
                            self._ft_bias_count += 1
                        # 比如累计 100 帧 ≈ 0.5s（5ms 回调频率）
                        if self._ft_bias_count >= 100:
                            self.ft_bias = self._ft_bias_acc / self._ft_bias_count
                            self.ft_bias_ready = True
                            del self._ft_bias_acc
                            del self._ft_bias_count
                            print("[V3] 力传感器偏置估计完成(给sigma7):", self.ft_bias)

                    # 2）给 sigma7 的力反馈：用去偏置后的值
                    if self.ft_bias_ready:
                            ft_for_sigma = np.concatenate([F_E_for_sigma, tau_E_for_sigma]) - self.ft_bias
                    else:
                        ft_for_sigma = np.concatenate([F_E_for_sigma, tau_E_for_sigma])
                        
                    f = ft_for_sigma[:3]# - np.array([-1.04241695e+01, -3.91363890e+01, -1.59821867e+01])
                    tau = ft_for_sigma[3:]# - np.array([1.10645416e+00, -3.77675319e-01, -2.64234962e-02])
                    k_f, k_tau = self.cfg.FT_SCALE
                    self.spacemouse.set_force(k_f * f, k_tau * tau)

            # 更新状态快照
            ts = time.perf_counter()
            with self._state_lock:
                self._state_snapshot = {
                    "pose": pose_euler.astype(np.float32, copy=False),
                    "pos_quat": pose_quat.astype(np.float32, copy=False),
                    "joint": joint.astype(np.float32, copy=False),
                    "joint_current": joint_current.astype(np.float32, copy=False),
                    "ft": ft.astype(np.float32, copy=False),
                    "ts": ts,
                }
                # 兼容旧字段（如果别处还在读）
                self.current_pose = self._state_snapshot["pose"]
                self.current_pos_quat = self._state_snapshot["pos_quat"]
                self.current_joint = self._state_snapshot["joint"]
                self.current_joint_current = self._state_snapshot["joint_current"]
                self.current_ft = self._state_snapshot["ft"]

        except Exception as e:
            # 打印完整 traceback 以便定位具体出错源（文件/设备/调用）
            import traceback
            print(f"⚠️  回调错误: {repr(e)}")
            traceback.print_exc()
    
    def reset(self):
        """重置环境"""
        print("\n🔄 重置环境...")
        self.step_count = 0
        
        # 清除错误
        self.arm.rm_clear_system_err()
        time.sleep(0.1)
        
        # 移动到复位位置
        print(f"🎯 移动到复位位置: {self.cfg.RESET_POSE[:3]}")
        ret = self.arm.rm_movel(self.cfg.RESET_POSE, v=15, r=0, connect=0, block=1)
        if ret != 0:
            print(f"⚠️  复位移动失败")
        time.sleep(1.0)
        
        # ⚠️ 等待回调更新状态（确保获取到最新位姿）
        # 等待一帧稳定的 snapshot
        t0 = time.perf_counter()
        snap = self.get_state_snapshot()
        while snap is None and (time.perf_counter() - t0) < 2.0:
            time.sleep(0.01)
            snap = self.get_state_snapshot()

        if snap is None:
            raise RuntimeError("reset() 超时：未收到机械臂状态回调")

        self.desired_tcp_pose = snap["pose"].copy()
        self.last_pose_quat = snap["pos_quat"].copy()
        print(f"✅ 期望位姿初始化: {self.desired_tcp_pose}")

        # ⚠️ 重置导纳控制器（在删除滤波器之前！）
        if self.admittance is not None:
            # 确保current_ft存在
            if hasattr(self, 'current_ft'):
                self.admittance.reset(self.current_ft.copy())
                print("✅ 导纳控制器已重置")
            else:
                print("⚠️  等待力传感器数据...")
                time.sleep(0.5)
                if hasattr(self, 'current_ft'):
                    self.admittance.reset(self.current_ft.copy())
                    print("✅ 导纳控制器已重置")
        
        # 重置力传感器滤波器（删除属性让回调重新初始化）
        if hasattr(self, 'current_ft'):
            delattr(self, 'current_ft')
        
        # ✅ 下一条 episode 重新估计力传感器偏置
        self.ft_bias_ready = False
        if hasattr(self, '_ft_bias_acc'):
            delattr(self, '_ft_bias_acc')
        if hasattr(self, '_ft_bias_count'):
            delattr(self, '_ft_bias_count')
            
        # 获取初始观测
        obs = self._get_obs()

        # 新一条 episode：清空 ft 曲线历史（只影响显示，不影响写盘）
        if getattr(self, '_ft_visualizer', None) is not None:
            try:
                self._ft_visualizer.reset()
            except Exception:
                pass
        
        print("✅ 重置完成，准备采集\n")
        return obs
    
    def step(self, spacemouse_action, enabled: bool=True):
        """
        执行一步（与原版RMController逻辑完全一致）
        
        原版流程 (realman_server.py + admittance_controller.py):
        1. SpaceMouse delta -> sm_to_pose() -> desired_pose (20Hz主频率)
        2. ROS订阅desired_pose
        3. 子步插值: path = linspace(last_pose, desired_pose, n_substeps=5)
        4. 每个子步 (100Hz):
           a. 导纳控制: delta = admittance.compute_delta(ft)
           b. 补偿位姿: p[:3] += delta[:3]
           c. 发送命令: rm_movep_canfd(p)
           d. 等待: sleep(dt - computation_time)
        
        简化版流程（无ROS，直接复现）:
        同上，只是不通过ROS通信，直接在代码中完成
        """
        start_time = time.perf_counter()
        
        snap = self.get_state_snapshot()
        ft_now = snap["ft"] if (snap is not None and snap.get("ft") is not None) else None

        # ⚠️ 关键1：SpaceMouse delta积分到期望位姿（20Hz主频率）
        pos_scale, rot_scale = self.cfg.ACTION_SCALE
        
        if USE_SIGMA7 and CONTROL_MODE == "abs":
            # ===== Sigma7 绝对控制 =====
            # 1) 读取 teleop 是否开启
            enabled = self.spacemouse.is_enabled() if hasattr(self.spacemouse, "is_enabled") else True
            # print(f"[V3] abs mode, enabled={enabled}, action_norm={np.linalg.norm(spacemouse_action):.4f}")

            # 2) 从关 -> 开：记录“机器人起点”
            if enabled and not self.sigma7_prev_enabled:
                # 以当前 desired_tcp_pose 作为 robot_start_pose
                self.sigma7_robot_start_pose = self.desired_tcp_pose.copy()
                print("[V3] robot_start_pose set to:", self.sigma7_robot_start_pose)
            self.sigma7_prev_enabled = enabled

            if enabled and self.sigma7_robot_start_pose is not None:
                # spacemouse_action: [Δx_dev, Δy_dev, Δz_dev, Δrx_dev, Δry_dev, Δrz_dev]
                device_offset = spacemouse_action.copy()

                # ---- ① 把 sigma7 返回的偏移看作“任务坐标系 {T} 下的偏移” ----
                dx_task = device_offset[:3] * pos_scale
                drot_task = device_offset[3:] * rot_scale

                # ② 位置：任务 → TCP
                dx_tcp = R_ET @ dx_task

                # ③ 姿态：任务 → TCP
                #    先在 {T} 里构造增量旋转矩阵，再用 R_ET 做基变换
                R_delta_task = R.from_rotvec(drot_task)             # R_Δ^T
                R_delta_task_mat = R_delta_task.as_matrix()         # 3x3

                # 同一物理旋转在 {E} 中的表示：R_Δ^E = R_ET * R_Δ^T * R_ET^T
                R_delta_tcp_mat = R_ET @ R_delta_task_mat @ R_ET.T  # 3x3
                R_delta_tcp = R.from_matrix(R_delta_tcp_mat)

                # ④ 作用到当前 TCP 姿态
                # 内旋xyz，即 R=Rx * Ry * Rz，等价于外旋zyx
                R_start = R.from_euler("xyz", self.sigma7_robot_start_pose[3:])
                R_target = R_delta_tcp * R_start

                target_pose = self.sigma7_robot_start_pose.copy()
                target_pose[:3] += dx_tcp
                target_pose[3:] = R_target.as_euler("xyz")
                self.desired_tcp_pose = target_pose
            else:
                # teleop 关闭时，不改 desired_tcp_pose
                pass

        # ✅ 缓存：本步命令（动作）
        self._cmd_pose_t = self.desired_tcp_pose.copy()

        # ⚠️ 关键2：子步插值（与原版RMController一致）
        # last_pose_quat: 上一步指令位姿（四元数）, current_pos_quat: 当前实际位姿（四元数）
        desired_pose_quat = poseuler_to_posquat(self.desired_tcp_pose, order="xyz")  # qw,qx,qy,qz格式

        # ===== decide interpolation start =====
        if self.cfg.INTERP_START_MODE == "measured":
            # ✅ 当前真实执行到的位姿（最稳）
            snap = self.get_state_snapshot()
            start_pose_quat = snap["pos_quat"].copy()
        elif self.cfg.INTERP_START_MODE == "commanded":
            # ⚠️ 上一条命令，仅用于对比
            start_pose_quat = self.last_pose_quat.copy()
        
        # ===== execute =====
        if self.cfg.USE_INTERPOLATION:
            # linspace生成n_substeps个子步
            path = np.linspace(start_pose_quat, desired_pose_quat, self.cfg.N_SUBSTEPS + 1)[1:]
            
            t_base = start_time  # 以 进入 step 开始时刻为基准，按绝对时间轴发送子步，避免 sleep漂移累积
            # ⚠️ 关键3：执行每个子步（100Hz子频率）
            movep_times = []  # 记录每个子步的 movep 耗时
            for k, p in enumerate(path, start=1):
                # 导纳控制补偿（如果启用且力传感器数据已就绪）
                if self.admittance is not None and ft_now is not None:
                    pose_delta = self.admittance.compute_delta(
                        ft_now.copy(),
                        control_axis=[0, 1, 2]  # 仅xyz位置控制
                    )
                    p[:3] += pose_delta[:3]  # 应用位置补偿
                
                # 发送命令（计时）
                t_movep_start = time.perf_counter()
                self.arm.rm_movep_canfd(p.tolist(), 0)  # qw,qx,qy,qz格式
                t_movep_end = time.perf_counter()
                movep_times.append((t_movep_end - t_movep_start) * 1000)
                
                # 控制子步频率 —— 目标发送时刻：t_base + k*DT（100Hz）
                t_target = t_base + k * self.cfg.CTRL_DT
                # 绝对时间 sleep（更稳）
                while True:
                    now = time.perf_counter()
                    dt_sleep = t_target - now
                    if dt_sleep <= 0:
                        break
                    time.sleep(min(dt_sleep, 0.001))
            
            # 获取观测
            obs = self._get_obs()
            
            # 性能打印（可选）
            if PRINT_PERFORMANCE:
                if movep_times:
                    avg_movep = np.mean(movep_times)
                    max_movep = np.max(movep_times)
                    print(f"[perf] movep: {len(movep_times)}子步 avg={avg_movep:.2f}ms max={max_movep:.2f}ms")
        else:
            # 不插值，直接发送目标位姿
            p = desired_pose_quat.copy()
            
            # 导纳控制补偿（如果启用且力传感器数据已就绪）
            if self.admittance is not None and ft_now is not None:
                pose_delta = self.admittance.compute_delta(
                    ft_now.copy(),
                    control_axis=[0, 1, 2]  # 仅xyz位置控制
                )
                p[:3] += pose_delta[:3]  # 应用位置补偿
            # 发送命令（计时）
            if PRINT_PERFORMANCE:
                t_movep_start = time.perf_counter()
            self.arm.rm_movep_canfd(p.tolist(), 0)  # qw,qx,qy,qz格式
            if PRINT_PERFORMANCE:
                t_movep_end = time.perf_counter()
                movep_time = (t_movep_end - t_movep_start) * 1000
            
            # 获取观测
            obs = self._get_obs()  # 发送后立即读取
            
            # 控制频率 + 性能打印（可选）
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0, (1.0 / self.cfg.CONTROL_HZ) - elapsed)
            
            if PRINT_PERFORMANCE:
                print(f"[perf] movep={movep_time:.2f}ms | 循环={elapsed*1000:.1f}ms + sleep={sleep_time*1000:.1f}ms = {(elapsed+sleep_time)*1000:.1f}ms")
            
            time.sleep(sleep_time)

        # 可选：每步 push 一条 ft 给后台可视化线程（严格按 step 计数）
        if getattr(self, '_ft_visualizer', None) is not None:
            try:
                self._ft_visualizer.push(obs.get('ft', None))
            except Exception:
                pass

        # ===== update last commanded pose =====
        self.last_pose_quat = desired_pose_quat.copy()
        
        if enabled:
            self.step_count += 1
        
        # 判断是否结束
        done = self.step_count >= self.cfg.MAX_EPISODE_LENGTH

        return obs, done
    

    def get_state_snapshot(self):
        """
        返回一次性一致状态（浅拷贝 dict；数组本身是 numpy 对象）
        """
        with self._state_lock:
            snap = self._state_snapshot
            if snap is None:
                return None
            # dict 拷贝一份，避免外部改引用
            return {
                "pose": snap["pose"].copy(),
                "pos_quat": snap["pos_quat"].copy(),
                "joint": snap["joint"].copy(),
                "joint_current": snap["joint_current"].copy(),
                "ft": snap["ft"].copy() if snap.get("ft") is not None else None,
                "ts": snap["ts"],
            }


    def _get_obs(self):
        """获取观测（与原版格式完全一致）"""
        t_start = time.perf_counter()
        
        snap = self.get_state_snapshot()
        if snap is None:
            # 回调还没来，稍等一下（或直接 raise）
            time.sleep(0.01)
            snap = self.get_state_snapshot()
            if snap is None:
                raise RuntimeError("未收到机械臂状态回调，无法生成观测")
        obs = {}
        
        # 1. 相机图像
        t_cam_start = time.perf_counter()
        for name, camera in self.cameras.items():
            ret, img = camera.read()
            if ret and img is not None:
                if name in self.cfg.CAMERAS:
                    img = self.cfg.CAMERAS[name]["crop"](img)
                img = cv2.resize(img, self.cfg.IMAGE_SIZE)
                obs[name] = img[..., ::-1]  # BGR->RGB
        t_cam_end = time.perf_counter()
        # 2. 触觉图像（可选）
        t_tac_start = time.perf_counter()
        if self.tac_cap is not None:
            ret, tac_frame = self.tac_cap.read()  # VideoCapture 现在返回 (ret, frame)
            if ret and tac_frame is not None and isinstance(tac_frame, dict):
                # 新版多传感器：dict[side] -> {modality: value} 或 None
                for side, side_payload in tac_frame.items():
                    if side_payload is None:
                        continue
                    if not isinstance(side_payload, dict):
                        continue
                    for modality, value in side_payload.items():
                        if value is None:
                            continue
                        obs[f"tac_{side}_{modality}"] = value
        t_tac_end = time.perf_counter()
        
        # 3. Proprio: 根据 Config.STATE_TYPES 选择要导出的状态
        state_types = getattr(self.cfg, "STATE_TYPES", ("eef",))

        if "eef" in state_types:
            # Debug: 验证欧拉角和四元数转换一致性
            # tmpEuler = self.current_pose[3:]
            # print("current euler[rx ry rz]:", tmpEuler)
            # tmpQuat = self.current_pos_quat[3:]  # qw,qx,qy,qz
            # print("current quat [qw,qx,qy,qz]:", tmpQuat)
            # # 将tmpEuler按照 ZYX内旋顺序【realman机械臂 格式】-大写内旋
            # eulerMat = R.from_euler("ZYX", tmpEuler[::-1]).as_matrix()
            # print("euler to rot mat:\n", eulerMat)
            # quatMat = R.from_quat([tmpQuat[1], tmpQuat[2], tmpQuat[3], tmpQuat[0]]).as_matrix()
            # print("quat to rot mat:\n", quatMat)
            # # 比较两种方式得到的旋转矩阵是否一致
            # print("rot mat difference:\n", eulerMat - quatMat)

            # ✅ measured eef pose at time t (来自回调 current_pose)
            pose = snap["pose"]
            pos = pose[:3]
            euler = pose[3:]
            obs["proprio_eef"] = np.concatenate([pos, euler], axis=0).astype(np.float32)

        if "joint" in state_types:
            # 绝对关节角
            obs["proprio_joint"] = snap["joint"].astype(np.float32)
        
        # 添加关节电流数据
        obs["joint_current"] = snap["joint_current"].astype(np.float32)
        
        # 4. 力传感器：6 维 F/T，直接记为 ft (shape: (6,))
        obs["ft"] = snap["ft"].astype(np.float32)

        # 性能统计（可选打印）
        if PRINT_PERFORMANCE:
            t_obs_end = time.perf_counter()
            obs_total_ms = (t_obs_end - t_start) * 1000
            cam_ms = (t_cam_end - t_cam_start) * 1000
            tac_us = (t_tac_end - t_tac_start) * 1e6
            tac_count = len([k for k in obs.keys() if k.startswith("tac_")])
            print(f"   [_get_obs] total={obs_total_ms:.2f}ms, cam={cam_ms:.2f}ms, tac={tac_us:.1f}μs×{tac_count}")

        return obs
    
    def close(self):
        """关闭环境"""
        print("\n🧹 关闭环境...")
        
        # 禁用实时回调，避免回调在资源被释放后继续访问导致错误
        try:
            self._arm_callback_enabled = False
        except Exception:
            pass

        # 给回调线程一点时间返回
        time.sleep(0.01)

        self.spacemouse.close()
        for camera in self.cameras.values():
            camera.close()
        if self.tac_cap is not None:
            self.tac_cap.close()
        # 停止可视化（如果已启动）
        if hasattr(self, '_sensor_visualizer'):
            self._sensor_visualizer.stop()
        if getattr(self, '_ft_visualizer', None) is not None:
            try:
                self._ft_visualizer.stop()
            except Exception:
                pass
        self.arm.rm_delete_robot_arm()
        print("✅ 环境已关闭")


# ============ 数据采集 ============
def collect_episode(env: SimplePIHEnv, episode_idx: int, save_path: Path):
    """采集一个episode"""
    
    print(f"\n{'='*60}")
    print(f"📹 Episode {episode_idx} 开始采集")
    print("  - 按 'q' 结束当前episode")
    print(f"{'='*60}\n")
    
    # 重置环境
    obs = env.reset()
    
    # 数据缓冲区
    observations = []
    # 按类型分别存动作，由 Config.ACTION_TYPES 控制
    action_types = getattr(env.cfg, "ACTION_TYPES", ("eef_abs",))
    action_buffers = {k: [] for k in action_types}
    
    # 键盘监听
    stop_flag = [False]
    def on_press(key):
        try:
            if key.char == 'q':
                stop_flag[0] = True
        except:
            pass
    
    from pynput import keyboard
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    # 控制循环
    pbar = tqdm(total=env.cfg.MAX_EPISODE_LENGTH, desc="Recording")
    done = False
    # 只在 teleop 启用时计数的步数（用于 warmup 丢弃前几步）
    warmup_steps = 3  # 设为0意味着不丢弃
    enabled_step_idx = 0
    had_enabled = False
    prev_enabled = None
    end_on_disable = bool(getattr(env.cfg, "END_ON_TELEOP_DISABLE", False))

    try:
        while not done and not stop_flag[0]:            
            # 1) 先读取当前观测（obs 已经是上一步 step 后的观测）,因为 env.step() 会更新 obs
            # 1) 先拷贝所有观测（遇到 ndarray 就 copy 一下）
            obs_t = {}
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    try:
                        obs_t[k] = v.copy()  # env.obs是不断更新的，需要copy
                    except Exception:
                        obs_t[k] = v
                else:
                    obs_t[k] = v
            
            # 2) 读取SpaceMouse
            spacemouse_action, buttons = env.spacemouse.get_action()
            # 判定当前是否启用遥操作：
            # - 对 SpaceMouse：没有 is_enabled()，默认始终为 True
            # - 对 Sigma7：使用按钮开关，由 is_enabled() 决定
            enabled = True
            if hasattr(env.spacemouse, "is_enabled"):
                try:
                    enabled = env.spacemouse.is_enabled()
                except Exception:
                    enabled = True

            # ✅ 动态 episode：如果 Sigma7 由“开 -> 关”，自动结束当前 episode
            # 这样不需要等 MAX_EPISODE_LENGTH 走完。
            if (
                end_on_disable
                and had_enabled
                and (prev_enabled is True)
                and (enabled is False)
            ):
                # 发送一次“保持当前位姿”的控制命令，让机械臂稳一下再退出循环
                try:
                    env.step(spacemouse_action, enabled=False)
                except Exception:
                    pass
                print("\n⏹️  检测到遥操作关闭，结束当前 episode")
                break
            prev_enabled = enabled

            # 3) 先step，在 step 内部 会更新 desired_tcp_pose等，
            #    并通过 子步插值 + 导纳控制 真正驱动机械臂
            #    enabled=False 时，不会累加 env.step_count（episode 不会被“吃掉”）
            obs_tp1, done = env.step(spacemouse_action, enabled=enabled)
            obs = obs_tp1  # obs 永远表示 “最新观测”
            if enabled:
                had_enabled = True

                # 4) 记录动作
                if enabled_step_idx >= warmup_steps:
                    # 过了 warmup 才真正写入数据
                    observations.append(obs_t)
                    
                    # ==== 4.1 计算所有可能的动作表示 ====
                    action_types = getattr(env.cfg, "ACTION_TYPES", ("eef_abs",))

                    cmd_pose_t = env._cmd_pose_t  # step 内部缓存的 t时刻 命令位姿
                    # (1) 绝对末端位姿
                    if "eef_abs" in action_types:
                        pos = cmd_pose_t[:3]
                        euler = cmd_pose_t[3:]
                        eef_abs = np.concatenate([pos, euler], axis=0).astype(np.float32)
                        action_buffers["eef_abs"].append(eef_abs)

                    # (2) 末端相对位姿增量：6D [dx,dy,dz, d_rx,d_ry,d_rz]
                    # ✅ reference: measured pose at time t (from obs_t)
                    if "eef_rel" in action_types:
                        pose_t = obs_t.get("proprio_eef", None)
                        if pose_t is None:
                            raise RuntimeError("ACTION_TYPES 包含 eef_rel，但 obs_t 缺少 proprio_eef；请确保 STATE_TYPES 包含 'eef'")

                        pos_t = pose_t[:3].astype(np.float32)
                        euler_t = pose_t[3:].astype(np.float32)

                        # ✅ command: desired pose at time t (after step updates desired_tcp_pose)
                        des_pos = cmd_pose_t[:3].astype(np.float32)
                        des_euler = cmd_pose_t[3:].astype(np.float32)

                        dpos = des_pos - pos_t
                        R_cur = R.from_euler("xyz", euler_t)
                        R_des = R.from_euler("xyz", des_euler)
                        R_rel = R_cur.inv() * R_des
                        drot = R_rel.as_rotvec().astype(np.float32)  # [d_rx, d_ry, d_rz]

                        a_eef_rel = np.concatenate([dpos, drot], axis=0).astype(np.float32)
                        action_buffers["eef_rel"].append(a_eef_rel)

                    # (3) 绝对关节角：joint_abs
                    if "joint_abs" in action_types:
                        arm_model = rm_robot_arm_model_e.RM_MODEL_RM_75_E
                        force_type = rm_force_type_e.RM_MODEL_RM_SF_E
                        # 初始化算法的机械臂及末端型号
                        algo_handle = Algo(arm_model, force_type)
                        # 逆运动学求解
                        joint_t = obs_t["proprio_joint"]
                        pose_tp1 = cmd_pose_t
                        params = rm_inverse_kinematics_params_t(joint_t.tolist(), pose_tp1.tolist(), 1)
                        ik_solution = algo_handle.rm_algo_inverse_kinematics(params)
                        action_buffers["joint_abs"].append(np.array(ik_solution[1], dtype=np.float32))
                        # print("IK 解:", ik_solution)  # (0, [-0.206063911318779, 33.7917594909668, 3.664539098739624, 77.0719985961914, -0.9778476357460022, 66.70684814453125, -17.95758819580078])
                        # print("直接读取的关节角:", env.current_joint)
                        # print("关节角差异:", ik_solution[1] - env.current_joint)


                    # ==== 4.2 更新进度条 ====
                    pbar.update(1)
                    pbar.set_description("Recording")
                else:
                    # 热身阶段：不写数据，只更新提示
                    pbar.set_description(
                        f"Recording [WARMUP {enabled_step_idx+1}/{warmup_steps}]"
                    )
                enabled_step_idx += 1
            else:
                # teleop 未启用：不写数据、不更新 pbar 进度，只显示 PAUSED
                pbar.set_description("Recording [PAUSED]")
    
    except KeyboardInterrupt:
        print("\n⏸️  用户中断")
    
    finally:
        listener.stop()
        pbar.close()

    # 👇 把“是否开启过遥操作”的判断放在 finally 外面
    if not had_enabled:
        print("⚠ 本条 episode 未曾开启遥操作（had_enabled=False），不记录此 episode。")
        return

    
    # 检查数据长度
    ep_len = len(observations)
    if ep_len < 10:
        print(f"⚠️  数据太短（{ep_len}步），跳过保存")
        return False
    
    print(f"\n💾 保存数据到 {save_path}...")
    
    # 转换为numpy数组并保存
    # detect tactile keys (new only): tac_<side>_<modality>
    tactile_keys = [k for k in observations[0].keys() if k.startswith("tac_")]  # 如果没有触觉传感器则为空列表
    
    data_dict = {
        "observations/images/global": np.array([obs["global"] for obs in observations], dtype=np.uint8),
        "observations/images/wrist": np.array([obs["wrist"] for obs in observations], dtype=np.uint8),
    }
    # ====== 保存所有 proprio 类型 ======
    # 例如：proprio, proprio_eef, proprio_joint ...
    proprio_keys = [k for k in observations[0].keys() if k.startswith("proprio")]
    for k in proprio_keys:
        h5key = f"observations/{k}"  # 比如 observations/proprio_joint
        data_dict[h5key] = np.array([obs[k] for obs in observations], dtype=np.float32)

    # ====== 保存关节电流数据 ======
    # 'joint_current' 键，则堆成 (T,7) 存到 "joint_current"
    data_dict["joint_current"] = np.array([obs["joint_current"] for obs in observations], dtype=np.float32)
    
    # ====== 保存 6 维力传感器数据 (ft) ======
    #  'ft' 键，则堆成 (T,6) 存到顶层数据集 "ft"
    data_dict["ft"] = np.array([obs["ft"] for obs in observations], dtype=np.float32)
    
    # ====== 保存所有动作类型 ======
    # action_buffers: dict[name -> list[array]]
    for name, buf in action_buffers.items():
        if not buf:
            continue
        arr = np.array(buf, dtype=np.float32)
        h5key = f"actions/{name}"   # 例如 actions/eef_abs, actions/eef_rel, actions/joint_abs
        data_dict[h5key] = arr

    # save tactile channels individually
    for k in tactile_keys:
        # tac_left_img -> observations/tac/left/img
        suffix = k[len("tac_"):]
        if "_" not in suffix:
            # 如果没有 modality，视为 img（但正常情况下不应出现，因为采集时总是指定了 modality，不会只有 tac_left 这种裸键。 如果不采集触觉呢？—— 那就根本不会有 tac_ 开头的键）
            side, modality = suffix, "img"
        else:
            side, modality = suffix.split("_", 1)  # 1表示只分割一次

        h5key = f"observations/tac/{side}/{modality}"
        values = [obs[k] for obs in observations]
        arr = np.array(values)
        # 常见：img(=warped_img)/diff/marker_img 为 uint8；depth/marker_current/marker_offset/force6d 为 float32
        data_dict[h5key] = arr
    
    # 保存初始位姿（9D格式：7D quaternion + 2个占位符，与原版一致）
    # init_pose_quat = poseuler_to_posquat(env.desired_tcp_pose)  # 7D
    # 原版格式：[x,y,z, qw,qx,qy,qz, 0, gripper]，V3无夹爪所以只保存7D
    # data_dict["misc/init_pose"] = init_pose_quat.astype(np.float32)
    
    # 保存为HDF5
    compression = getattr(env.cfg, "H5_COMPRESSION", "gzip")
    with h5py.File(save_path, "w") as f:
        for key, value in data_dict.items():
            if compression and str(compression).lower() != "none":
                f.create_dataset(key, data=value, compression=str(compression))
            else:
                f.create_dataset(key, data=value)
    
    print(f"✅ 已保存 {ep_len} 步数据")
    print(f"   - 观测: {list(observations[0].keys())}")  # 列出观测键
    print(f"   - 动作类型: {list(action_buffers.keys())}")
    print(f"   - 动作1的维度: {action_buffers[next(iter(action_buffers))][0].shape}")
    # next(iter(action_buffers)) → 取字典的第一个 key
    
    return True


def main():
    import argparse
    import copy
    import re
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50, help="采集episode数量")
    parser.add_argument(
        "--index-mode",
        choices=["append", "fill"],
        default="append",
        help=(
            "episode 索引策略：append=续上已有最大索引（默认）；"
            "fill=从0开始优先补漏（会跳过已存在文件）。"
        ),
    )
    parser.add_argument("--visualize", action="store_true", help="开启相机/触觉可视化（可选）")
    parser.add_argument("--visualize-cropped", action="store_true", help="可视化时显示裁剪后的图像")
    parser.add_argument("--visualize-ft", action="store_true", help="实时可视化六维力/矩 (ft)")
    parser.add_argument(
        "--async-vision",
        type=lambda x: x.lower() == "true",
        default=True,
        help="视觉异步采集（true/false，默认 true）",
    )
    parser.add_argument(
        "--latest-tac",
        type=lambda x: x.lower() == "true",
        default=True,
        help="触觉最新帧模式（true/false，默认 true）",
    )
    parser.add_argument(
        "--print-perf",
        type=lambda x: x.lower() == "true",
        default=False,
        help="打印性能指标（true/false，默认 false）",
    )
    parser.add_argument(
        "--end-on-disable",
        type=lambda x: x.lower() == "true",
        default=True,
        help="遥操作从开->关时自动结束episode（true/false，默认 true）",
    )
    parser.add_argument(
        "--tactile",
        choices=["none", "left", "right", "both"],
        default=None,
        help=(
            "触觉采集开关：none 不采；left/right 只采单侧；both 采双侧。"
            "默认不传则使用 Config.TACTILE_SENSORS"
        ),
    )
    parser.add_argument(
        "--tac-out",
        type=str,
        default=None,
        help=(
            "触觉输出模态，逗号分隔，例如 img,diff,depth,marker_img,marker_current,marker_offset,force6d。"
            "默认不传则使用 Config.TACTILE_MODALITIES"
        ),
    )
    parser.add_argument(
        "--h5-compress",
        choices=["gzip", "none"],
        default=None,
        help="HDF5 压缩：gzip(默认)/none(关闭压缩以加速写盘)；不传则用 Config.H5_COMPRESSION",
    )
    args = parser.parse_args()

    def _scan_existing_episode_indices(data_dir: Path):
        """扫描 data_dir 下 episode_*.hdf5，返回已存在的 index 列表（升序去重）"""
        if not data_dir.exists():
            return []
        pat = re.compile(r"^episode_(\d+)\.hdf5$")
        indices = []
        for p in data_dir.iterdir():
            if not p.is_file():
                continue
            m = pat.match(p.name)
            if m:
                try:
                    indices.append(int(m.group(1)))
                except Exception:
                    pass
        return sorted(set(indices))
    
    # 根据命令行参数选择具体采集实现（延迟绑定，确保切换生效）
    global RSCapture, VideoCapture, USE_ASYNC_VISION, USE_LATEST_TAC_MODE, PRINT_PERFORMANCE
    USE_ASYNC_VISION = args.async_vision
    USE_LATEST_TAC_MODE = args.latest_tac
    PRINT_PERFORMANCE = args.print_perf

    if USE_ASYNC_VISION:
        from realman_env.camera.rs_capture_async import RSCapture as RSCapture
    else:
        from realman_env.camera.rs_capture import RSCapture as RSCapture

    if USE_LATEST_TAC_MODE:
        from realman_env.camera.video_capture_latest import VideoCapture as VideoCapture
    else:
        from realman_env.camera.video_capture import VideoCapture as VideoCapture
    
    # 创建配置和环境
    config = Config()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    existing_indices = _scan_existing_episode_indices(config.DATA_DIR)
    existing_max = max(existing_indices) if existing_indices else -1

    # 起始索引仅由 index-mode 决定
    if args.index_mode == "append":
        start_idx = existing_max + 1
    else:  # fill
        start_idx = 0

    if args.tac_out:
        config.TACTILE_MODALITIES = tuple(
            m.strip().lower() for m in args.tac_out.split(",") if m.strip()
        )

    if args.h5_compress:
        config.H5_COMPRESSION = args.h5_compress

    # 动态 episode：允许命令行覆盖
    config.END_ON_TELEOP_DISABLE = args.end_on_disable

    # ===== 触觉：允许命令行覆盖（none/left/right/both） =====
    tactile_cfg = getattr(config, "TACTILE_SENSORS", None)
    if tactile_cfg:
        tactile_cfg = copy.deepcopy(tactile_cfg)

    if args.tactile == "none":
        tactile_cfg = None
    elif args.tactile == "left":
        tactile_cfg = {"left": tactile_cfg.get("left")} if tactile_cfg else None
    elif args.tactile == "right":
        tactile_cfg = {"right": tactile_cfg.get("right")} if tactile_cfg else None
    elif args.tactile == "both":
        tactile_cfg = tactile_cfg

    config.TACTILE_SENSORS = tactile_cfg
    
    print("\n" + "="*60)
    print("🚀 V3版PIH数据采集（带导纳控制）")
    print("="*60)
    print(f"📊 配置信息:")
    print(f"   - 控制频率: {config.CONTROL_HZ} Hz")
    print(f"   - 子步数: {config.N_SUBSTEPS}")
    print(f"   - 导纳控制: {'✅ 启用' if config.ADMITTANCE_ENABLED else '❌ 禁用'}")
    print(f"   - 位置缩放: {config.ACTION_SCALE[0]}")
    print(f"   - 旋转缩放: {config.ACTION_SCALE[1]}")
    print(f"   - 最大步数: {config.MAX_EPISODE_LENGTH}")
    print(f"   - 目标episodes: {args.episodes}")
    print(f"   - 索引策略: {args.index_mode} (start_idx={start_idx}, existing_max={existing_max})")
    print(f"   - 可视化: {'✅ 启用' if args.visualize else '❌ 禁用'}")
    print(f"   - 视觉异步: {'✅ 启用' if USE_ASYNC_VISION else '❌ 禁用'}")
    print(f"   - 触觉最新帧: {'✅ 启用' if USE_LATEST_TAC_MODE else '❌ 禁用'}")
    print("="*60 + "\n")
    
    # 根据命令行参数控制可视化（最小侵入：通过 config 传入）
    if args.visualize:
        config.VISUALIZE_SENSORS = True
    if args.visualize_cropped:
        config.VISUALIZE_CROPPED = True
    if args.visualize_ft:
        config.VISUALIZE_FT = True

    env = SimplePIHEnv(config)
    
    try:
        success_count = 0
        is_first_collection = True
        i = start_idx
        while success_count < args.episodes:
            save_path = config.DATA_DIR / f"episode_{i}.hdf5"

            if save_path.exists():
                if args.index_mode == "fill":
                    # fill 模式：跳过已有文件，继续向后找空位
                    print(f"⏭️  跳过已存在的 episode_{i}")
                    i += 1
                    continue
                else:
                    # append 模式：理论上不该存在；若存在则直接往后推一位
                    print(f"⚠️  episode_{i} 已存在（append 模式下异常），自动改用下一个索引")
                    i += 1
                    continue
            
            
            # 每次采集前都询问
            if is_first_collection:
                user_input = input(f"\n准备开始采集 Episode {i}，继续? (Enter继续 / q退出): ")
            else:
                user_input = input("继续下一个episode? (Enter继续 / q退出): ")
            
            if user_input.lower() == 'q':
                print("👋 用户选择退出")
                break
            
            # ✅ 每个 episode 开始前，都尝试让遥操作设备回到中心
            # 在 collect_episode 之前插入一个 recenter 调用（有就用，没有就跳过），
            # 保证不论 inc / abs / spacemouse，都尝试 recenter：
            if hasattr(env, "spacemouse") and hasattr(env.spacemouse, "recenter"):
                print("🔄 遥操作设备 recenter 中...")
                env.spacemouse.recenter(timeout=10.0)

            success = collect_episode(env, i, save_path)
            if success:
                success_count += 1
            is_first_collection = False
            i += 1
            
            print(f"\n📈 进度: {success_count}/{args.episodes} episodes 完成\n")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  采集被用户中断")
    
    finally:
        env.close()
        print(f"\n🎉 采集完成！共保存 {success_count} 个episodes")
        print(f"📁 数据目录: {config.DATA_DIR}")


if __name__ == "__main__":
    main()
