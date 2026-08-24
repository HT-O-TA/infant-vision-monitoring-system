#!/usr/bin/env python
# coding: utf-8

import cv2
import json
import threading
import time
import requests
from picamera2 import Picamera2, Preview
import libcamera
import smbus
import math

# ======================
# 电机控制类（完整实现版，根据官方代码）
# ======================
class YB_Pcb_Car(object):
    def get_i2c_device(self, address, i2c_bus):
        self._addr = address
        if i2c_bus is None:
            return smbus.SMBus(1)
        else:
            return smbus.SMBus(i2c_bus)

    def __init__(self):
        # 创建 I2C 设备
        try:
            self._device = self.get_i2c_device(0x16, 1)
            print("电机控制器初始化成功")
        except Exception as e:
            print(f"初始化失败: {str(e)}")
            self._device = None

    def write_u8(self, reg, data):
        try:
            self._device.write_byte_data(self._addr, reg, data)
        except Exception as e:
            print(f'write_u8 I2C error: {e}')

    def write_reg(self, reg):
        try:
            self._device.write_byte(self._addr, reg)
        except Exception as e:
            print(f'write_reg I2C error: {e}')

    def write_array(self, reg, data):
        try:
            self._device.write_i2c_block_data(self._addr, reg, data)
        except Exception as e:
            print(f'write_array I2C error: {e}')

    def Ctrl_Car(self, l_dir, l_speed, r_dir, r_speed):
        try:
            reg = 0x01
            data = [l_dir, l_speed, r_dir, r_speed]
            self.write_array(reg, data)
        except Exception as e:
            print(f'Ctrl_Car I2C error: {e}')
            
    def Control_Car(self, speed1, speed2):
        try:
            if speed1 < 0:
                dir1 = 0
            else:
                dir1 = 1
            if speed2 < 0:
                dir2 = 0
            else:
                dir2 = 1 
            self.Ctrl_Car(dir1, int(math.fabs(speed1)), dir2, int(math.fabs(speed2)))
        except Exception as e:
            print(f'Control_Car I2C error: {e}')

    def Car_Run(self, speed1, speed2):
        # 前进（正方向旋转）
        try:
            self.Ctrl_Car(1, speed1, 1, speed2)
        except Exception as e:
            print(f'Car_Run I2C error: {e}')

    def Car_Stop(self):
        # 急停，利用专用寄存器停止
        try:
            reg = 0x02
            self.write_u8(reg, 0x00)
        except Exception as e:
            print(f'Car_Stop I2C error: {e}')

    def Car_Back(self, speed1, speed2):
        # 后退（反方向旋转）
        try:
            self.Ctrl_Car(0, speed1, 0, speed2)
        except Exception as e:
            print(f'Car_Back I2C error: {e}')

    def Car_Left(self, speed1, speed2):
        # 左转：通常设置左电机反转，右电机正转
        try:
            self.Ctrl_Car(0, speed1, 1, speed2)
        except Exception as e:
            print(f'Car_Left I2C error: {e}')

    def Car_Right(self, speed1, speed2):
        # 右转：通常设置左电机正转，右电机反转
        try:
            self.Ctrl_Car(1, speed1, 0, speed2)
        except Exception as e:
            print(f'Car_Right I2C error: {e}')

    def Car_Spin_Left(self, speed1, speed2):
        # 原地左旋
        try:
            self.Ctrl_Car(0, speed1, 1, speed2)
        except Exception as e:
            print(f'Car_Spin_Left I2C error: {e}')

    def Car_Spin_Right(self, speed1, speed2):
        # 原地右旋
        try:
            self.Ctrl_Car(1, speed1, 0, speed2)
        except Exception as e:
            print(f'Car_Spin_Right I2C error: {e}')

    def Ctrl_Servo(self, servo_id, angle):
        # 控制舵机，确保角度在 0~180 范围内
        try:
            if angle < 0:
                angle = 0
            elif angle > 180:
                angle = 180
            reg = 0x03
            data = [servo_id, angle]
            self.write_array(reg, data)
        except Exception as e:
            print(f'Ctrl_Servo I2C error: {e}')


# ======================
# 机器人核心控制逻辑（客户端）
# ======================
class RobotController:
    def __init__(self):
        self.command = None   # 来自服务器的控制命令
        self.car = YB_Pcb_Car()   # 初始化官方完整版本的电机控制器
        self.running = True
        self.lock = threading.Lock()

    def image_worker(self):
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"format": 'RGB888', "size": (640, 480)}
        )
        # 这里根据需要配置镜像翻转（官方示例中使用 hflip 和 vflip）
        config["transform"] = libcamera.Transform(hflip=1, vflip=1)
        
        try:
            picam2.configure(config)
            with picam2:
                picam2.start()
                print("摄像头已启动")
                while self.running:
                    frame = picam2.capture_array()
                    self.image_display(frame)
                    self.send_frame(frame)
                    time.sleep(0.01)
        except Exception as e:
            print(f"摄像头错误: {str(e)}")
        finally:
            try:
                picam2.stop()
            except Exception:
                pass

    def send_frame(self, frame):
        # 将当前帧发送至服务器，并根据返回信息更新控制命令
        _, img_encoded = cv2.imencode('.jpg', frame)
        try:
            response = requests.post(
                'http://192.168.128.87:5000/api/process',
                files={'image': ('frame.jpg', img_encoded.tobytes())},
                timeout=2
            )
            if response.status_code == 200:
                try:
                    json_data = response.json()
                    new_cmd = json_data.get("control", "stop")
                    print(f"【服务器指令】{new_cmd}")
                    
                    with self.lock:
                        self.command = new_cmd
                        print(f"【更新指令】{self.command}")
                except json.JSONDecodeError:
                    print("无效的JSON响应")
        except Exception as e:
            print(f"传输错误: {str(e)}")

    def image_display(self, frame):
        # 这里简单输出图像尺寸与帧率
        h, w = frame.shape[:2]
        print(f"图像尺寸: {w}x{h}, 帧率: {1/0.01:.1f} FPS")

    def motor_worker(self):
        CONTROL_INTERVAL = 0.1
        while self.running:
            with self.lock:
                cmd = self.command
                self.command = None

            if not cmd:
                time.sleep(CONTROL_INTERVAL)
                continue
                    
            valid_cmds = ["forward", "backward", "left", "right", "stop"]
            if cmd not in valid_cmds:
                print(f"【无效指令】{cmd}")
                time.sleep(CONTROL_INTERVAL)
                continue
                    
            print(f"【执行指令】{cmd}")
            try:
                if cmd == "forward":
                    # 调用前进：根据实际需要设置速度参数
                    self.car.Car_Run(10, 10)
                elif cmd == "backward":
                    self.car.Car_Back(5, 5)
                elif cmd == "left":
                    self.car.Car_Left(40, 80)
                elif cmd == "right":
                    self.car.Car_Right(80, 40)
                elif cmd == "stop":
                    self.car.Car_Stop()
            except Exception as e:
                print(f"电机控制过程中出现异常: {e}")

            time.sleep(CONTROL_INTERVAL)

    def emergency_stop(self):
        self.running = False
        self.car.Car_Stop()


# ======================
# 主程序入口
# ======================
if __name__ == '__main__':
    controller = RobotController()

    # 启动摄像头和电机控制两个线程
    threads = [
        threading.Thread(target=controller.image_worker, daemon=True),
        threading.Thread(target=controller.motor_worker, daemon=True)
    ]
    for t in threads:
        t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("正在紧急停止...")
        controller.emergency_stop()
