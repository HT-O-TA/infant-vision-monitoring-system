from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
from scipy.spatial.distance import cosine
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models

# 初始化 YOLO 模型
model = YOLO("F:/project/infant/yolov8/yolo11x-pose.pt")

# 初始化 DeepSORT 跟踪器
tracker = DeepSort(max_age=20, n_init=4, nn_budget=150, max_iou_distance=0.5)

# 设置视频源和输出路径
video_source = "F:/project/infant/yolov8/infant3.mp4"
output_path = "F:/project/infant/yolov8/output_track_target.mp4"
target_image_path = "F:/project/infant/yolov8/infant3.png"

cap = cv2.VideoCapture(video_source)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 初始化视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 加载预训练的ResNet模型来提取图像特征
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=True).to(device)
resnet.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 提取图像特征
def extract_image_features(image, model):
    image = transform(image).unsqueeze(0).to(device)  # 预处理并转到GPU
    with torch.no_grad():
        features = model(image)
    return features.cpu().numpy().flatten()

# 提取目标人物的特征
target_image = cv2.imread(target_image_path)
target_features = extract_image_features(target_image, resnet)

if target_features is None:
    print("无法提取目标人物的特征，请检查目标图片！")
    exit()

# 用于存储目标人物的跟踪ID
target_track_id = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用 YOLO 检测目标
    results = model.predict(source=frame, device=0)
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        confidence = float(box.conf[0].cpu().numpy())
        
        if confidence > 0.5:  # 假设只考虑高置信度的检测
            cropped_image = frame[y1:y2, x1:x2]  # 获取检测框内的图像
            person_features = extract_image_features(cropped_image, resnet)
            detections.append(((x1, y1, x2, y2), confidence, "person", person_features))

    # 使用 DeepSORT 跟踪
    tracks = tracker.update_tracks(detections, frame=frame)

    # 检查并锁定目标人物
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        if target_track_id is None:  # 尚未锁定目标
            min_similarity = float('inf')
            best_match_id = None

            # 遍历所有检测，计算相似度
            for detection in detections:
                # 获取检测框的坐标
                x1, y1, x2, y2 = detection[0]

                # 裁剪出检测框内的图像
                cropped_image = frame[y1:y2, x1:x2]

                # 使用 ResNet 提取图像特征
                detection_features = extract_image_features(cropped_image, resnet)
                
                # print(detection_features)
                
                if detection_features is not None:
                    # 计算余弦相似度
                    similarity = cosine(detection_features, target_features)
                    print(similarity, best_match_id)
                    if similarity < min_similarity:  # 更新最小相似度
                        min_similarity = similarity
                        best_match_id = track_id
                        print("更新")

            # 如果相似度满足阈值，则锁定目标
            if best_match_id is not None and min_similarity < 0.5:  # 阈值可以调整
                target_track_id = best_match_id
                print(f"目标锁定: ID {target_track_id}")

        # 只绘制目标人物的跟踪框
        if track_id == target_track_id:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 写入结果到视频
    out.write(frame)

    # 显示实时画面（可选）
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
