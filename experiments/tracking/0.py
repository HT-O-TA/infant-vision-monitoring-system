from flask import Flask, request, Response, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial.distance import cosine
import serial
import io
import threading
import queue

# 初始化Flask应用
app = Flask(__name__)

# 全局配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SERIAL_PORT = '/dev/ttyUSB0'
BAUDRATE = 9600

# 初始化模型
yolo_model = YOLO("yolo11x-pose.pt").to(DEVICE)
resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).to(DEVICE)
resnet.eval()

# 初始化DeepSORT
tracker = DeepSort(max_age=20, n_init=4, nn_budget=150)

# 小车控制初始化
ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)

# 线程安全的队列处理
frame_queues = {}

# 目标特征初始化（示例）
TARGET_IMAGE = cv2.imread("target_1.png")
TARGET_FEATURES = resnet(torch.tensor(TARGET_IMAGE).permute(2,0,1).unsqueeze(0).float()/255).detach().cpu().numpy().flatten()

def process_stream(client_id):
    """持续处理视频流的线程函数"""
    while True:
        try:
            # 从队列获取帧数据
            frame = frame_queues[client_id].get(timeout=5)
            
            # YOLO姿态估计
            results = yolo_model.track(frame, persist=True, verbose=False)
            
            # DeepSORT跟踪
            tracks = []
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                # 提取关键点（YOLO-Pose）
                keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints else []
                
                # 生成检测对象
                detections = []
                for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                    if conf > 0.5 and cls == 0:  # 仅处理人类
                        x1, y1, x2, y2 = map(int, box)
                        cropped = frame[y1:y2, x1:x2]
                        # features = resnet(torch.tensor(cropped).detach().cpu().numpy().flatten()
                        features = resnet(torch.tensor(cropped, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)).detach().cpu().numpy().flatten()
                        detections.append(((x1, y1, x2, y2), conf, "person", features))
                        
            # 更新跟踪器
            tracks = tracker.update_tracks(detections, frame=frame)
            
            # 目标锁定逻辑
            target_id = None
            min_distance = float('inf')
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_features = track.features
                distance = cosine(track_features, TARGET_FEATURES)
                if distance < min_distance:
                    min_distance = distance
                    target_id = track.track_id
            
            # 小车控制
            if target_id is not None:
                for track in tracks:
                    if track.track_id == target_id:
                        x1, y1, x2, y2 = track.to_ltrb()
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        img_center = (frame.shape[1]//2, frame.shape[0]//2)
                        
                        # 生成控制指令
                        dx = cx - img_center[0]
                        if abs(dx) > 50:
                            cmd = b'left' if dx < 0 else b'right'
                        else:
                            cmd = b'forward'
                        ser.write(cmd)
            
            # 编码返回帧
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            
        except queue.Empty:
            del frame_queues[client_id]
            break

@app.route('/video_feed', methods=['POST'])
def video_feed():
    client_id = request.remote_addr  # 简单使用客户端IP作为唯一标识
    
    if client_id not in frame_queues:
        frame_queues[client_id] = queue.Queue(maxsize=10)
        threading.Thread(target=process_stream, args=(client_id,)).start()
    
    # 接收视频帧
    file = request.files['frame']
    frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    try:
        frame_queues[client_id].put_nowait(frame)
    except queue.Full:
        pass
    
    return Response(status=204)

@app.route('/control', methods=['POST'])
def manual_control():
    """手动控制接口"""
    command = request.json.get('cmd')
    if command in ['left', 'right', 'forward', 'stop']:
        ser.write(command.encode())
        return jsonify(status="success")
    return jsonify(status="invalid command")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)