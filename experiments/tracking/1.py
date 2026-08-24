import copy
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
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
import os

app = Flask(__name__)

# 初始化 YOLO 模型
model = YOLO("yolo11x-pose.pt")

# 加载 ResNet 提取图像特征
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=True).to(device)
resnet.eval()

# 加载表情识别模型
def load_model():
    model = CNN3()
    model.load_weights('./model/cnn3_best_weights.h5')
    return model

# 初始化姿态识别
body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

def process_frame(frame, body=True, hands=True, target_box=None, expand=30):
    canvas = copy.deepcopy(frame)

    if target_box:
        # 获取target_box框住的区域
        x1, y1, x2, y2 = target_box

        # 扩大target_box范围，对目标附近的人物也进行相应的姿态识别，提高对婴儿的保护检测
        x1, y1, x2, y2 = x1 - expand, y1 - expand, x2 + expand, y2 + expand

        # 确保 target_box 在图像范围内
        if x1 < 0:
            x1 = 0  # 如果左边界超出图像，修正为图像的左边缘
        if y1 < 0:
            y1 = 0  # 如果上边界超出图像，修正为图像的上边缘
        if x2 > frame.shape[1]:
            x2 = frame.shape[1]  # 如果右边界超出图像，修正为图像的右边缘
        if y2 > frame.shape[0]:
            y2 = frame.shape[0]  # 如果下边界超出图像，修正为图像的下边缘

        # 确保修正后的框仍然有效
        if x2 <= x1 or y2 <= y1:
            print("Error: target_box is invalid after adjustment.")
            return frame  # 如果修正后框无效，返回原图像

        # 裁剪图像
        cropped_frame = frame[y1:y2, x1:x2]

        # 检查裁剪区域是否为空
        if cropped_frame.size == 0:
            print("Error: cropped_frame is empty.")
            return frame  # 如果裁剪结果为空，返回原图像

        # 在裁剪区域内进行姿态识别
        if body:
            candidate, subset = body_estimation(cropped_frame)

            # 如果没有检测到任何关键点，跳过绘制
            if candidate.shape[0] == 0:
                print("No body keypoints detected.")
            else:
                # 将关键点坐标从裁剪区域映射到原图
                candidate[:, 0] += x1
                candidate[:, 1] += y1
                canvas = util.draw_bodypose(canvas, candidate, subset)

        if hands:
            if candidate.shape[0] > 0:  # 只有在检测到人体时才进行手势识别
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

    iou = inter_area / union_area
    return iou

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
    resized_images = list()
    resized_images.append(face_img)
    resized_images.append(face_img[2:45, :])
    resized_images.append(face_img[1:47, :])
    resized_images.append(cv2.flip(face_img[:, :], 1))
    for i in range(len(resized_images)):
        resized_images[i] = cv2.resize(resized_images[i], (img_size, img_size))
        resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
    resized_images = np.array(resized_images)
    return resized_images

def predict_expression(frame, faces, target_box, iou_threshold=0.0802):
    if faces is None or len(faces) == 0:
        return frame
    model = load_model()
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
        results = model.predict(face_resized)
        result_sum = np.sum(results, axis=0).reshape(-1)
        label_index = np.argmax(result_sum, axis=0)
        emotion = index2emotion(label_index)
        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 0, 0), thickness=2)
        frame = cv2_img_add_text(frame, emotion, x + 30, y + 30, (255, 255, 255), 20)

    return frame

# 预处理图像
def extract_image_features(image, model):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image)
    return features.cpu().numpy().flatten()

class CarController:
    def control_car(self, x1, y1, x2, y2, frame_shape):
        """根据目标位置发送控制指令"""
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        img_center = (frame_shape[1] // 2, frame_shape[0] // 2)
        
        dx = cx - img_center[0]
        dy = cy - img_center[1]
        
        if abs(dx) > 50:
            if dx < 0:
                return 'left'
            else:
                return 'right'
        else:
            return 'forward'

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    target_image_path = data.get("target_image_path")
    input_image_path = data.get("input_image_path")
    
    print(f"Received target image path: {target_image_path}")
    print(f"Received input image path: {input_image_path}")
    
    if not os.path.exists(target_image_path) or not os.path.exists(input_image_path):
        return jsonify({"error": "Invalid image path"}), 400
    
    target_image = cv2.imread(target_image_path)
    input_image = cv2.imread(input_image_path)
    
    # 提取目标图像特征
    target_features = extract_image_features(target_image, resnet)
    
    detections = []
    results = model.predict(source=input_image, device=0)  # YOLO 目标检测
    
    # 遍历 YOLO 目标检测结果
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        confidence = float(box.conf[0].cpu().numpy())
        if confidence > 0.5:
            cropped_image = input_image[y1:y2, x1:x2]  # 裁剪出人体区域
            person_features = extract_image_features(cropped_image, resnet)  # 提取特征
            detections.append(((x1, y1, x2, y2), confidence, "person", person_features))
    
    print(f"YOLO detections: {len(detections)} targets found")
    
    # **目标匹配**
    target_box = None
    min_similarity = float('inf')
    
    for detection in detections:
        (x1, y1, x2, y2), confidence, label, features = detection
        similarity = cosine(features, target_features)
        if similarity < min_similarity:  # 找到最相似的人
            min_similarity = similarity
            target_box = (x1, y1, x2, y2)
    
    if target_box:
        x1, y1, x2, y2 = target_box
        cv2.rectangle(input_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(input_image, f"Target", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        faces = blaze_detect(input_image)
        if faces:
            print(f"Faces detected: {len(faces)}")
        
        # 处理姿态和表情
        input_image = process_frame(input_image, body=True, hands=True, target_box=(x1, y1, x2, y2))
        input_image = predict_expression(input_image, faces, (x1, y1, x2, y2))
        
        # 控制汽车移动
        car_controller = CarController()
        control_string = car_controller.control_car(x1, y1, x2, y2, input_image.shape)  # 返回控制指令
    
    # 保存处理后的图像
    success = cv2.imwrite(input_image_path, input_image)
    print(f"Image saved: {success}, Path: {input_image_path}")
    
    print(f"Control command: {control_string}")
    
    return jsonify({"message": "Processed successfully", "control": control_string})
