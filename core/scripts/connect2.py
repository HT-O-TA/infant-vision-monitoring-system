from flask import Flask, request, jsonify
from insightface.app import FaceAnalysis
from src.emotion_model import EmotionModel
import torchvision.transforms as transforms
from scipy.spatial.distance import cosine
from src.blazeface import blaze_detect
import torch.nn.functional as F
from ultralytics import YOLO
from src.body import Body
from src.hand import Hand
from PIL import Image
from src import util
import numpy as np
import requests
import torch
import uuid
import copy
import cv2
import os

app = Flask(__name__)
TARGET_DIR = "F:\\project\\infant_new\\target"
target_features = None
target_user_id = None  # 当前 target 对应的用户 id
global_token = None
is_fallen = False
strait = 0

# 新增：客户端地址（主动通知的目标地址）
JavaApi = "http://192.168.100.100:8081"  # 修改为实际客户端地址和端口

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_map = {1: 'positive', 2: 'negative'}
num_classes = 2

# ---------------------------
model = YOLO("model/yolo11x.pt")

expression_model = EmotionModel(num_classes=num_classes).to(device)
expression_model.load_state_dict(torch.load("model/best_emotion_model.pt", map_location=device))
expression_model.eval()

face_app = FaceAnalysis(name='antelopev2') 
face_app.prepare(ctx_id=0, det_size=(640, 640))

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')
# ---------------------------

def get_face_embedding(img):
    if img is None:
        print(f"无法读取图像：")
        return None

    # 人脸检测与特征提取
    faces = face_app.get(img)
    if not faces:
        print(f"未检测到人脸：")
        return None

    # 提取第一个人脸的特征向量
    return faces[0].embedding

def detect_fall_from_estimation(frame, candidate, subset, hand_peaks):
    
    global is_fallen

    if len(subset) == 0:
        is_fallen = False
        return "No person detected"

    person = subset[0]
    pose = {}
    for i, index in enumerate(person[:18]):
        if index == -1:
            pose[i] = None
        else:
            pose[i] = candidate[int(index)][:2]  # x, y

    KEY = {
        "nose": 0, "neck": 1, "r_shoulder": 2, "r_elbow": 3, "r_wrist": 4,
        "l_shoulder": 5, "l_elbow": 6, "l_wrist": 7, "r_hip": 8, "r_knee": 9,
        "r_ankle": 10, "l_hip": 11, "l_knee": 12, "l_ankle": 13,
        "eye_r": 14, "eye_l": 15, "ear_r": 16, "ear_l": 17
    }

    def get_point(name):
        idx = KEY.get(name)
        return pose.get(idx)

    lower_body = [get_point("r_hip"), get_point("l_hip"),
                get_point("r_knee"), get_point("l_knee"),
                get_point("r_ankle"), get_point("l_ankle")]
    lower_y = [pt[1] for pt in lower_body if pt is not None]

    upper_body = [get_point("neck"), get_point("nose")]
    upper_y = [pt[1] for pt in upper_body if pt is not None]

    if not lower_y or not upper_y:
        is_fallen = False
        return "Not enough keypoints"

    avg_lower_y = np.mean(lower_y)
    avg_upper_y = np.mean(upper_y)
    vertical_distance = abs(avg_upper_y - avg_lower_y)
    img_height = frame.shape[0]
    height_ratio = vertical_distance / img_height

    hand_y = []
    for peaks in hand_peaks:
        for pt in peaks:
            if pt[0] > 0 and pt[1] > 0:
                hand_y.append(pt[1])
    hands_down = len(hand_y) > 0 and np.mean(hand_y) > 0.8 * img_height

    if height_ratio < 0.25 and hands_down:
        is_fallen = True
        return "Body flattened and hands down"
    elif height_ratio < 0.2:
        is_fallen = True
        return "Body very low to ground"
    else:
        is_fallen = False
        return "Normal posture"

def process_frame(frame, body=True, hands=True, target_box=None, expand=30):
    
    global is_fallen
    
    print("======开始姿态识别======")
    
    canvas = copy.deepcopy(frame)
    candidate, subset = None, None
    all_hand_peaks = []

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

        if hands and candidate is not None and candidate.shape[0] > 0:
            hands_list = util.handDetect(candidate, subset, cropped_frame)
            for x, y, w, is_left in hands_list:
                peaks = hand_estimation(frame[y:y + w, x:x + w, :])
                peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
                peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
                all_hand_peaks.append(peaks)
            canvas = util.draw_handpose(canvas, all_hand_peaks)

        # 🔍 调用摔倒检测
        if candidate is not None and subset is not None:
            reason = detect_fall_from_estimation(frame, candidate, subset, all_hand_peaks)
            print(f"Fall detection result: {is_fallen} | Reason: {reason}")

    return canvas

def predict_expression(input_image, faces, target_box):
    
    global strait, is_fallen
    
    val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
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

            # pred_label = torch.argmax(prob, dim=1).item()
            prob_np = prob.detach().cpu().numpy()[0]
            if prob_np[0] > 0.7:
                pred_label = 2
            else:
                pred_label = 1

            strait = pred_label
            label_name = label_map[pred_label]

        # 文字位置偏移处理
        text_offset = 10
        if is_fallen:
            text_offset += 30  # 避开 "FALLEN" 文字
            
        text_y = max(y1 - text_offset, 10) 

        # 标注预测标签（避免与FALLEN重叠）
        cv2.putText(input_image, label_name, (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
        
    except Exception as e:
        print(f"[Emotion] Error: {e}")
    
    return input_image

class CarController:
    def control_car(self, x1, y1, x2, y2, frame_shape):
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        img_width, img_height = frame_shape[1], frame_shape[0]
        img_center = (img_width // 2, img_height // 2)
        dx = cx - img_center[0]
        dy = cy - img_center[1]
        width = x2 - x1
        height = y2 - y1
        area = width * height

        small_area_threshold = int(img_width * img_height * 0.05)
        large_area_threshold = int(img_width * img_height * 0.2)

        # 再判断前后距离
        if area > large_area_threshold:
            return 'backward'  # 目标太近，后退
        elif area < small_area_threshold:
            return 'forward'   # 目标太远，前进

        # 先判断方向偏移
        if abs(dx) > int(img_width * 0.2):
            return 'spin_left' if dx < 0 else 'spin_right'
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
        yolo_result = model.predict(source=target_image, device=0)[0]

        # 提取最高置信度的人类目标（cls==0）
        person_boxes = [box for box in yolo_result.boxes if int(box.cls[0]) == 0]
        if not person_boxes:
            return jsonify({"error": "No person found in target image"}), 404

        # 取最高置信度的目标
        best_box = max(person_boxes, key=lambda b: b.conf[0])
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
        cropped_target = target_image[y1:y2, x1:x2]
        target_features = get_face_embedding(cropped_target)
        target_user_id = user_id
        print(f"目标特征加载成功，特征维度: {target_features.shape}")
        
        return jsonify({"status": "success", "message": "Target updated and system ready to start capture"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/process', methods=['POST'])
def process_image():
    global strait, is_fallen
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
        person_boxes = [box for box in results[0].boxes if int(box.cls[0]) == 0]  # 0是person类
        for box in person_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            print("x1、y1、x2、y2为：" + str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2))
            print("confidence为：" + str(confidence))
            if confidence > 0.6:
                cropped_image = input_image[y1:y2, x1:x2]
                person_features = get_face_embedding(cropped_image)
                detections.append(((x1, y1, x2, y2), confidence, "person", person_features))
        
        target_box = None
        strait = 0  # “未知”情绪
        min_similarity = 0.5
        for detection in detections:
            (x1, y1, x2, y2), confidence, label, features = detection    
            similarity = 1 - cosine(features, target_features)
            print("similarity为：" + str(similarity))
            if similarity > min_similarity:
                min_similarity = similarity
                target_box = (x1, y1, x2, y2)
        
        if target_box:
            print("======检测到目标======")
            x1, y1, x2, y2 = target_box

            # 判断摔倒状态，画框 + 标注文字
            if is_fallen:
                cv2.rectangle(input_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色框
                cv2.putText(input_image, "FALLEN", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.rectangle(input_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框

            # 进行表情识别
            faces = blaze_detect(input_image)
            if len(faces) > 0:
                input_image = predict_expression(input_image, faces, target_box)
                print("======表情已识别======")

            # 姿态识别
            input_image = process_frame(input_image, body=True, hands=True, target_box=target_box)

            # 控制小车
            car_controller = CarController()
            control_string = car_controller.control_car(x1, y1, x2, y2, input_image.shape)
        else:
            control_string = "stop"

        print("小车指令为：" + control_string)
        
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
        # 建议新增方法用于检测目标图片中的人物区域
        results = model.predict(source=target_image, device=0)
        person_boxes = [box for box in results[0].boxes if int(box.cls[0]) == 0]
        if person_boxes:
            x1, y1, x2, y2 = map(int, person_boxes[0].xyxy[0].cpu().numpy())
            target_crop = target_image[y1:y2, x1:x2]
            target_features = get_face_embedding(target_crop)
            
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

