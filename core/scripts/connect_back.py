#!/usr/bin/env python
# coding: utf-8

import copy
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b7
from torchvision import models
from ultralytics import YOLO
from scipy.spatial.distance import cosine
from src.blazeface import blaze_detect
from src.model_expression import CNN3
from src.utils import index2emotion, cv2_img_add_text
from src.body import Body
from src.hand import Hand
from src import util
from flask import Flask, request, jsonify
from torchvision.models import resnet50
import os
import time
import uuid
import requests

app = Flask(__name__)

# ---------------------------
# 全局变量配置
# ---------------------------
# 默认目标图像目录（用户 target 文件在该目录下命名为 target_{user_id}.jpg）
TARGET_DIR = "./target"
target_features = None
target_user_id = None  # 当前 target 对应的用户 id

# 新增：客户端地址（主动通知的目标地址）
CLIENT_URL = "http://192.168.128.100:5000"  # 修改为实际客户端地址和端口

# ---------------------------
# 初始化模型等全局资源
# ---------------------------
model = YOLO("yolo11x.pt")
resnet = efficientnet_b7(pretrained=False).to(device)
weights_path = r"model/efficientnet_b7.pth"
resnet.load_state_dict(torch.load(weights_path))
resnet.eval()

def load_model():
    model_ex = CNN3()
    model_ex.load_weights('./model/cnn3_best_weights.h5')
    return model_ex

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

# ---------------------------
# 图像处理相关函数
# ---------------------------

def process_frame(frame, body=True, hands=True, target_box=None, expand=30):
    canvas = copy.deepcopy(frame)
    if target_box:
        x1, y1, x2, y2 = target_box
        x1, y1, x2, y2 = x1 - expand, y1 - expand, x2 + expand, y2 + expand
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            print("Error: target_box is invalid after adjustment.")
            return frame
        cropped_frame = frame[y1:y2, x1:x2]
        if cropped_frame.size == 0:
            print("Error: cropped_frame is empty.")
            return frame
        if body:
            candidate, subset = body_estimation(cropped_frame)
            if candidate.shape[0] == 0:
                print("No body keypoints detected.")
            else:
                candidate[:, 0] += x1
                candidate[:, 1] += y1
                canvas = util.draw_bodypose(canvas, candidate, subset)
        if hands:
            if candidate.shape[0] > 0:
                hands_list = util.handDetect(candidate, subset, cropped_frame)
                all_hand_peaks = []
                for x, y, w, is_left in hands_list:
                    peaks = hand_estimation(frame[y:y + w, x:x + w, :])
                    peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
                    peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
                    all_hand_peaks.append(peaks)
                canvas = util.draw_handpose(canvas, all_hand_peaks)
    return canvas

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_t, y1_t, x2_t, y2_t = box2
    inter_x1 = max(x1, x1_t)
    inter_y1 = max(y1, y1_t)
    inter_x2 = min(x2, x2_t)
    inter_y2 = min(y2, y2_t)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_t - x1_t) * (y2_t - y1_t)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

def is_face_in_target(face, target_box, iou_threshold=0.0802):
    face_x, face_y, face_w, face_h = face
    target_x1, target_y1, target_x2, target_y2 = target_box
    face_box = (face_x, face_y, face_x + face_w, face_y + face_h)
    iou = compute_iou(face_box, target_box)
    return iou >= iou_threshold

def generate_faces(face_img, img_size=48):
    if face_img is None or face_img.size == 0:
        return []
    face_img = face_img / 255.
    face_img = cv2.resize(face_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    resized_images = [face_img,
                    face_img[2:45, :],
                    face_img[1:47, :],
                    cv2.flip(face_img, 1)]
    for i in range(len(resized_images)):
        resized_images[i] = cv2.resize(resized_images[i], (img_size, img_size))
        resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
    return np.array(resized_images)

def predict_expression(frame, faces, target_box, iou_threshold=0.0802):
    if faces is None or len(faces) == 0:
        return frame
    model_ex = load_model()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for (x, y, w, h) in faces:
        if not is_face_in_target((x, y, w, h), target_box, iou_threshold):
            continue
        face = frame_gray[y: y + h, x: x + w]
        if face is None or face.size == 0:
            continue
        face_resized = generate_faces(face)
        if len(face_resized) == 0:
            continue
        results = model_ex.predict(face_resized)
        result_sum = np.sum(results, axis=0).reshape(-1)
        label_index = np.argmax(result_sum, axis=0)
        emotion = index2emotion(label_index)
        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 0, 0), thickness=2)
        frame = cv2_img_add_text(frame, emotion, x + 30, y + 30, (255, 255, 255), 20)
    return frame

def extract_image_features(image, model):
    try:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(image)
        return features.cpu().numpy().flatten()
    except Exception as e:
        print(f"特征提取失败: {str(e)}")
        return None

# ---------------------------
# 新增：封装 CarController 类，用于生成控制指令
# ---------------------------
class CarController:
    def control_car(self, x1, y1, x2, y2, frame_shape):
        # 计算目标中心
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        img_width, img_height = frame_shape[1], frame_shape[0]
        img_center = (img_width // 2, img_height // 2)
        dx = cx - img_center[0]
        dy = cy - img_center[1]
        width = x2 - x1
        height = y2 - y1
        area = width * height

        small_area_threshold = int(img_width * img_height * 0.02)
        large_area_threshold = int(img_width * img_height * 0.15)
        if abs(dx) > int(img_width * 0.2):
            return 'spin_left' if dx < 0 else 'spin_right'
        elif abs(dx) > int(img_width * 0.05):
            return 'curve_left' if dx < 0 else 'curve_right'
        if area > large_area_threshold:
            return 'forward'
        elif area < small_area_threshold:
            return 'backward'
        return 'stop'

# ---------------------------
# 新增：用于设置 target 的接口，方案二实现核心——接收用户 id 并加载对应目标
# ---------------------------
@app.route('/set_target', methods=['POST'])
def set_target():
    global target_features, target_user_id
    try:
        user_id = request.form.get("user_id")
        if not user_id:
            return jsonify({"error": "No user_id provided"}), 400
        target_path = os.path.join(TARGET_DIR, f"target_{user_id}.jpg")
        print(f"正在加载目标图像: {target_path}")
        if not os.path.exists(target_path):
            print("错误：目标图像不存在!")
            return jsonify({"error": f"Target image for user {user_id} does not exist"}), 404
        target_image = cv2.imread(target_path)
        if target_image is None:
            print("错误：无法读取目标图像")
            return jsonify({"error": "Cannot read target image"}), 500
        target_features = extract_image_features(target_image, resnet)
        target_user_id = user_id
        print(f"目标特征加载成功，特征维度: {target_features.shape}")

        # 主动通知客户端启动采集
        notify_url = f"{CLIENT_URL}/start_capture"
        try:
            r = requests.post(notify_url, timeout=5)
            print(f"通知客户端启动采集成功，响应状态码: {r.status_code}")
        except Exception as e:
            print(f"通知客户端启动采集失败: {e}")

        return jsonify({"status": "success", "message": "Target updated and system ready to start capture"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------
# 原 /upload_target 接口可以保留或删除，方案二已统一由 /set_target 处理 target
# ---------------------------

@app.route('/api/process', methods=['POST'])
def process_image():
    try:
        image_file = request.files.get('image')
        if not image_file:
            return jsonify({"error": "No image received"}), 400
        
        nparr = np.frombuffer(image_file.read(), np.uint8)
        input_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if target_features is None:
            return jsonify({"error": "Target features not loaded"}), 500
        
        results = model.predict(source=input_image, device=0)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            if confidence > 0.5:
                cropped_image = input_image[y1:y2, x1:x2]
                person_features = extract_image_features(cropped_image, resnet)
                detections.append(((x1, y1, x2, y2), confidence, "person", person_features))
        
        target_box = None
        min_similarity = float('inf')
        for detection in detections:
            (x1, y1, x2, y2), confidence, label, features = detection
            similarity = cosine(features, target_features)
            if similarity < min_similarity:
                min_similarity = similarity
                target_box = (x1, y1, x2, y2)
        
        if target_box:
            x1, y1, x2, y2 = target_box
            cv2.rectangle(input_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            faces = blaze_detect(input_image)
            if len(faces) > 0:
                input_image = predict_expression(input_image, faces, target_box)
            input_image = process_frame(input_image, body=True, hands=True, target_box=target_box)
            car_controller = CarController()
            control_string = car_controller.control_car(x1, y1, x2, y2, input_image.shape)
        else:
            control_string = "stop"
        
        unique_id = str(uuid.uuid4())[:8]
        output_path = f"./processed_images/result_{unique_id}.jpg"
        cv2.imwrite(output_path, input_image)
        
        return jsonify({
            "status": "success",
            "control": control_string,
            "output_image": output_path
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500    

# ---------------------------
# 在启动服务器前加载初始 target 特征（可选）
# ---------------------------
def load_target_features():
    global target_features, target_user_id
    default_user = "default"
    target_path = os.path.join(TARGET_DIR, f"target_{default_user}.jpg")
    print(f"正在加载目标图像: {target_path}")
    if not os.path.exists(target_path):
        print("警告：目标图像不存在，等待 /set_target 上传目标图片")
        return True   # 返回 True 允许启动服务，但 target_features 留空
    try:
        target_image = cv2.imread(target_path)
        if target_image is None:
            print("错误：无法读取目标图像")
            return False
        target_features = extract_image_features(target_image, resnet)
        target_user_id = default_user
        print(f"目标特征加载成功，特征维度: {target_features.shape}")
        return True
    except Exception as e:
        print(f"加载目标特征失败: {str(e)}")
        return False


# ---------------------------
# 修改服务器启动代码
# ---------------------------
if __name__ == '__main__':
    os.makedirs("./processed_images", exist_ok=True)
    
    if not load_target_features():
        print("致命错误：目标特征加载失败，退出服务")
        exit(1)
    
    print("服务器已打开，等待连接...")
    app.run(host='0.0.0.0', port=5000)
