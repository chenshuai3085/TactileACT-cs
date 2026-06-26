"""
Robotiq Gripper Server - No ROS Version (Fixed)
修复后的版本: 寄存器格式与pyrobotiqgripper一致
"""
if __name__ == "__main__":
    import pathlib
    import sys
    sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import time
import serial
from realman_env.robot_servers.gripper_server import GripperServer


class RobotiqGripperServerNoROS(GripperServer):
    """Robotiq 2F夹爪控制器 - 无ROS版本"""

    def __init__(self, gripper_port="/dev/ttyUSB1"):
        super().__init__()

        try:
            self.serial_port = serial.Serial(
                port=gripper_port,
                baudrate=115200,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0
            )
            print(f"✅ Gripper connected on {gripper_port}")
        except Exception as e:
            print(f"❌ Failed to connect gripper: {e}")
            self.serial_port = None

        self.gripper_pos = 0.0
        self.OUTPUT_REGISTER = 1000
        self.INPUT_REGISTER = 2000
        time.sleep(0.5)

    def _calculate_crc(self, data):
        """计算Modbus CRC16校验码"""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc

    def _send_command(self, register_values):
        """发送Modbus RTU命令"""
        if self.serial_port is None:
            print("⚠️ Gripper not connected")
            return

        slave_id = 0x09
        function_code = 0x10
        start_address = self.OUTPUT_REGISTER
        num_registers = 3
        byte_count = 6

        command = bytearray([
            slave_id, function_code,
            (start_address >> 8) & 0xFF, start_address & 0xFF,
            0x00, 0x03,  # num_registers = 3
            byte_count
        ])

        for value in register_values[:3]:
            command.append((value >> 8) & 0xFF)
            command.append(value & 0xFF)

        crc = self._calculate_crc(command)
        command.append(crc & 0xFF)
        command.append((crc >> 8) & 0xFF)

        try:
            self.serial_port.write(command)
            time.sleep(0.05)
            self.serial_port.read(100)
        except Exception as e:
            print(f"❌ Command failed: {e}")

    def activate_gripper(self):
        """激活夹爪"""
        print("Activating gripper...")
        # Step 1: Reset (all zeros)
        self._send_command([0x0000, 0x0000, 0x0000])
        time.sleep(0.5)
        # Step 2: Activate (rACT=1)
        self._send_command([0x0100, 0x0000, 0x0000])
        time.sleep(2.0)
        print("✅ Activated")

    def open(self):
        """打开夹爪"""
        print("Opening...")
        self._send_command([0x0900, 0x0000, 0xFF00])

    def close(self):
        """关闭夹爪"""
        print("Closing...")
        self._send_command([0x0900, 0x00FF, 0xFF14])

    def move(self, position):
        """移动到指定位置 (0-255)"""
        position = int(max(0, min(255, position)))
        print(f"Moving to {position}...")
        self._send_command([0x0900, position, 0xFF14])

    def __del__(self):
        if hasattr(self, 'serial_port') and self.serial_port:
            self.serial_port.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python robotiq_gripper_server_no_ros_fixed.py /dev/ttyUSB0")
        sys.exit(1)

    gripper = RobotiqGripperServerNoROS(sys.argv[1])

    print("\n1. Activating...")
    gripper.activate_gripper()
    time.sleep(2)

    # print("\n2. Opening...")
    # gripper.open()
    # time.sleep(2)

    # print("\n3. Closing...")
    # gripper.close()
    # time.sleep(2)

    # print("\n4. Moving to 100...")
    # gripper.move(100)
    # time.sleep(2)

    # print("\n5. Moving to 145...")
    # gripper.move(145)
    # time.sleep(2)

    # print("\n6. Opening...")
    # gripper.open()
    # time.sleep(2)

    while True:
        ctr = input("Enter position (0-255): ")
        if not ctr.isdigit():
            print("Exiting...")
            break
        print("\n4. Moving...")
        gripper.move(float(ctr))
        time.sleep(2)

    print("\n✅ Test completed!")
