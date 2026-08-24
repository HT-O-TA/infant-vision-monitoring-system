import copy
import cv2
import numpy as np
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.models import efficientnet_b7
from PIL import Image
from ultralytics import YOLO
from scipy.spatial.distance import cosine
from src.blazeface import blaze_detect
from src.emotion_model import EmotionModel
from src.body import Body
from src.hand import Hand
from src import util
from flask import Flask, request, jsonify
import os
import uuid
import requests

app = Flask(__name__)
# ---------------------------
# 全局变量配置
# ---------------------------
# 默认目标图像目录（用户 target 文件在该目录下命名为 target_{user_id}.jpg）
TARGET_DIR = "target_img"
target_features = None
target_user_id = None  # 当前 target 对应的用户 id

# 全局变量存储token
global_token = None

# 全局变量存储情绪属性
strait = 0

# 新增：客户端地址（主动通知的目标地址）
JavaApi = "http://192.168.100.100:8081"  # 修改为实际客户端地址和端口

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_map = {0: 'non-negative', 1: 'negative'} 

class EfficientNetB7FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = efficientnet_b7(pretrained=False)
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d(1)  # 全局池化
        self.out_channels = base_model.classifier[1].in_features  # 输出通道数

    def forward(self, x):
        x = self.features(x)      # shape: [B, C, H, W]
        x = self.pool(x)          # shape: [B, C, 1, 1]
        x = x.view(x.size(0), -1) # shape: [B, C]
        return x

# ---------------------------
# 初始化模型等全局资源
# ---------------------------
num_classes = 2

model = YOLO("model/yolo11x.pt")

expression_model = EmotionModel(num_classes=num_classes).to(device)
expression_model.load_state_dict(torch.load("model/best_emotion_model.pt", map_location=device))
expression_model.eval()

efficientnet = EfficientNetB7FeatureExtractor().to(device)
efficientnet.load_state_dict(torch.load("model/efficientnet_b7.pth"), strict=False)
efficientnet.eval()

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')


def process_frame(frame, body=True, hands=True, target_box=None, expand=30):
    canvas = copy.deepcopy(frame)
    candidate, subset = None, None
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
        resized_images[i] = np.expand_dims(resized_images[i], axis=0)  # (1, H, W)
    return np.array(resized_images)


val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_expression(input_image, faces, target_box):
    global strait
    x1, y1, x2, y2 = target_box
    
    target_face = None
    
    for (fx, fy, fw, fh) in faces:
        face_cx = fx + fw // 2
        face_cy = fy + fh // 2
        if x1 <= face_cx <= x2 and y1 <= face_cy <= y2:
            target_face = input_image[fy:fy+fh, fx:fx+fw]
            break
    
    if target_face is None:
        return input_image  # 找不到目标人物的人脸，跳过识别

    try:
        face_pil = Image.fromarray(cv2.cvtColor(target_face, cv2.COLOR_BGR2RGB))
        face_tensor = val_transform(face_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = expression_model(face_tensor)
            prob = F.softmax(output, dim=1)
            pred_label = torch.argmax(prob, dim=1).item()
            strait = pred_label
            label_name = label_map[pred_label]
        
        # 在图像上写出情绪文字
        cv2.putText(input_image, label_name, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
    except Exception as e:
        print(f"[Emotion] Error: {e}")
    
    return input_image


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


@app.route('/set_target', methods=['POST'])
def set_target():
    global target_features, target_user_id, global_token

    print("\n===== 收到/set_target请求 =====")

    # 获取token并存储
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        global_token = auth_header.split(" ")[1]

    try:
        user_id = request.form.get("user_id")
        if not user_id:
            return jsonify({"error": "No user_id provided"}), 400
        target_path = TARGET_DIR + f"/target_{user_id}.jpg"
        print(f"正在加载目标图像: {target_path}")
        if not os.path.exists(target_path):
            print("错误：目标图像不存在!")
            return jsonify({"error": f"Target image for user {user_id} does not exist"}), 404
        target_image = cv2.imread(target_path)
        if target_image is None:
            print("错误：无法读取目标图像")
            return jsonify({"error": "Cannot read target image"}), 500
        target_features = extract_image_features(target_image, efficientnet)
        target_user_id = user_id
        print(f"目标特征加载成功，特征维度: {target_features.shape}")
        return jsonify({"status": "success", "message": "Target updated and system ready to start capture"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def is_fallen(keypoints):
    """
    判断是否摔倒
    :param keypoints: 人体关键点
    :return: 如果摔倒，返回True；否则返回False
    """
    # 检查关键点是否有效
    shoulder_left = keypoints[5]  # 左肩
    shoulder_right = keypoints[2]  # 右肩
    hip_left = keypoints[11]  # 左臀部
    hip_right = keypoints[8]  # 右臀部
    knee_left = keypoints[13]  # 左膝
    knee_right = keypoints[10]  # 右膝

    # 提取有效的坐标
    shoulder_left = np.array(shoulder_left[:2]) if shoulder_left[2] > 0 else None
    shoulder_right = np.array(shoulder_right[:2]) if shoulder_right[2] > 0 else None
    hip_left = np.array(hip_left[:2]) if hip_left[2] > 0 else None
    hip_right = np.array(hip_right[:2]) if hip_right[2] > 0 else None
    knee_left = np.array(knee_left[:2]) if knee_left[2] > 0 else None
    knee_right = np.array(knee_right[:2]) if knee_right[2] > 0 else None

    # 如果有任何关键点缺失，则无法判断是否摔倒
    if None in [shoulder_left, shoulder_right, hip_left, hip_right, knee_left, knee_right]:
        return False

    # 判断肩膀与臀部之间的角度关系
    body_angle = calculate_angle(shoulder_left, shoulder_right, hip_left, hip_right)
    
    # 如果肩膀和臀部之间的角度过大，可能意味着摔倒
    if body_angle > 90:  # 如果大于90度，认为有摔倒的可能
        return True

    # 判断膝盖的高度关系，如果膝盖在臀部下方，且没有直立，说明可能摔倒
    if knee_left[1] > hip_left[1] and knee_right[1] > hip_right[1]:  # 膝盖比臀部低
        return True

    return False


def calculate_angle(point1, point2, point3, point4):
    """
    计算两个关节之间的角度
    :param point1: 关节1的坐标
    :param point2: 关节2的坐标
    :param point3: 关节3的坐标
    :param point4: 关节4的坐标
    :return: 角度值
    """
    vector1 = point2 - point1  # 计算向量
    vector2 = point4 - point3
    
    dot_product = np.dot(vector1, vector2)  # 点积
    norm1 = np.linalg.norm(vector1)  # 向量1的长度
    norm2 = np.linalg.norm(vector2)  # 向量2的长度
    
    cos_theta = dot_product / (norm1 * norm2)  # 余弦值
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 反余弦求角度
    return np.degrees(angle)  # 返回角度值



@app.route('/api/process', methods=['POST'])
def process_image():
    global strait
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
                person_features = extract_image_features(cropped_image, efficientnet)
                detections.append(((x1, y1, x2, y2), confidence, "person", person_features))
        
        target_box = None
        strait = 0  # “未知”情绪
        min_similarity = 0.4
        for detection in detections:
            (x1, y1, x2, y2), confidence, label, features = detection
            similarity = cosine(features, target_features)
            if similarity < min_similarity:
                min_similarity = similarity
                target_box = (x1, y1, x2, y2)
        
        if target_box:
            # 获取人体关键点进行摔倒检测
            candidate, subset = body_estimation(input_image)
            is_falling_flag = is_fallen(candidate)  # 调用摔倒检测函数
            
            box_color = (0, 0, 255) if is_falling_flag else (0, 255, 0)
            cv2.rectangle(input_image, (x1, y1), (x2, y2), box_color, 2)
            if is_falling_flag:
                cv2.putText(input_image, "摔倒", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

            rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            faces = blaze_detect(rgb_image)
            if len(faces) > 0:
                input_image = predict_expression(input_image, faces, target_box)
            input_image = process_frame(input_image, body=True, hands=True, target_box=target_box)
            car_controller = CarController()
            control_string = car_controller.control_car(x1, y1, x2, y2, input_image.shape)
        else:
            control_string = "stop"
        
        unique_id = str(uuid.uuid4())[:8]
        output_path = f"F:/Scientific Rearch/infant new/recog_img/result_{unique_id}.jpg"
        cv2.imwrite(output_path, input_image)

        # 新增：将图片名称发送给Java后端
        try:
            java_api_url = JavaApi + "/photo/saveImagePath"
            payload = {
                "imagePath": output_path,
                "strait": strait
            }
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {global_token}'  # 回传token
            }
            response = requests.post(java_api_url, json=payload, headers=headers, timeout=5)
            if response.status_code != 200:
                print(f"Failed to send image info to Java backend: {response.text}")
        except Exception as e:
            print(f"Error sending image info to Java backend: {str(e)}")
        
        return jsonify({
            "status": "success",
            "control": control_string,
            "output_image": output_path
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500    


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
        target_features = extract_image_features(target_image, efficientnet)
        target_user_id = default_user
        print(f"目标特征加载成功，特征维度: {target_features.shape}")
        return True
    except Exception as e:
        print(f"加载目标特征失败: {str(e)}")
        return False


if __name__ == '__main__':
    os.makedirs("./processed_images", exist_ok=True)

    if not load_target_features():
        print("致命错误：目标特征加载失败，退出服务")
        exit(1)
    
    print("服务器已打开，等待连接...")
    app.run(host='0.0.0.0', port=5000)
    # gunicorn app:app --bind 0.0.0.0:5000
