from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
from scipy.spatial.distance import cosine
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models

class ObjectTracking:
    def __init__(self, yolo_model_path, target_image_path, device="cuda"):
        # 初始化 YOLO 模型
        self.model = YOLO(yolo_model_path)
        
        # 初始化 DeepSORT 跟踪器
        self.tracker = DeepSort(max_age=20, n_init=4, nn_budget=150, max_iou_distance=0.5)
        
        # 加载预训练的 ResNet 模型来提取图像特征
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.resnet = models.resnet50(pretrained=True).to(self.device)
        self.resnet.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 提取目标人物的特征
        target_image = cv2.imread(target_image_path)
        self.target_features = self.extract_image_features(target_image, self.resnet)
        if self.target_features is None:
            raise ValueError("无法提取目标人物的特征，请检查目标图片！")
        self.target_track_id = None

    def extract_image_features(self, image, model):
        image = self.transform(image).unsqueeze(0).to(self.device)  # 预处理并转到GPU
        with torch.no_grad():
            features = model(image)
        return features.cpu().numpy().flatten()

    def process_frame(self, frame):
        # 使用 YOLO 检测目标
        results = self.model.predict(source=frame, device=0)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            
            if confidence > 0.5:  # 假设只考虑高置信度的检测
                cropped_image = frame[y1:y2, x1:x2]  # 获取检测框内的图像
                person_features = self.extract_image_features(cropped_image, self.resnet)
                detections.append(((x1, y1, x2, y2), confidence, "person", person_features))

        # 使用 DeepSORT 跟踪
        tracks = self.tracker.update_tracks(detections, frame=frame)

        # 检查并锁定目标人物
        target_coordinates = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            if self.target_track_id is None:  # 尚未锁定目标
                min_similarity = float('inf')
                best_match_id = None

                # 遍历所有检测，计算相似度
                for detection in detections:
                    # 获取检测框的坐标
                    x1, y1, x2, y2 = detection[0]

                    # 裁剪出检测框内的图像
                    cropped_image = frame[y1:y2, x1:x2]

                    # 使用 ResNet 提取图像特征
                    detection_features = self.extract_image_features(cropped_image, self.resnet)
                    if detection_features is not None:
                        # 计算余弦相似度
                        similarity = cosine(detection_features, self.target_features)
                        if similarity < min_similarity:  # 更新最小相似度
                            min_similarity = similarity
                            best_match_id = track_id

                # 如果相似度满足阈值，则锁定目标
                if best_match_id is not None and min_similarity < 0.5:  # 阈值可以调整
                    self.target_track_id = best_match_id

            # 只绘制目标人物的跟踪框
            if track_id == self.target_track_id:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                target_coordinates.append((x1, y1, x2, y2))

        return frame, target_coordinates


# ------------------------------------------------------------------------

# 初始化 ObjectTracking 对象
tracking = ObjectTracking(
    yolo_model_path="F:/project/infant/yolov8/yolo11x-pose.pt",
    target_image_path="F:/project/infant/yolov8/infant3.png"
)

# 读取视频
cap = cv2.VideoCapture("F:/project/infant/yolov8/infant3.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 处理每一帧
    output_frame, target_coords = tracking.process_frame(frame)

    # 输出目标框的坐标
    print(target_coords)

    # 显示处理后的帧
    cv2.imshow("Output", output_frame)
    
    # 如果按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
