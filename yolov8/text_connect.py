from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
from scipy.spatial.distance import cosine

# 初始化 YOLO 模型
model = YOLO("F:/project/infant/yolov8/yolo11x-pose.pt")  # 替换为你的 YOLO 模型路径

# 初始化 DeepSORT 跟踪器
tracker = DeepSort(max_age=20, n_init=4, nn_budget=150, max_iou_distance=0.5)

# 设置视频源和输出路径
video_source = "F:/project/infant/yolov8/infant3.mp4"  # 输入视频路径
output_path = "F:/project/infant/yolov8/output_track_target.mp4"  # 输出结果路径

cap = cv2.VideoCapture(video_source)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 初始化视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 提取目标人物的特征
def extract_features(image, model):
    results = model.predict(source=image, device=0, task='pose')  # 检测关键点
    if len(results[0].keypoints) > 0:
        keypoints = results[0].keypoints[0].xy.cpu().numpy()
        return keypoints.flatten()  # 转为一维数组
    return None

# 加载目标图片并提取其特征
target_image_path = "F:/project/infant/yolov8/infant3.png"  # 目标人物图像路径
target_image = cv2.imread(target_image_path)
target_features = extract_features(target_image, model)

if target_features is None:
    print("无法提取目标人物的特征，请检查目标图片！")
    exit()

# 用于存储目标人物的跟踪ID（一旦检测到就锁定）
target_track_id = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用 YOLO 检测目标
    results = model.predict(source=frame, device=0)

    detections = []
    # 提取检测到的目标的关键点
    for idx, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        confidence = float(box.conf[0].cpu().numpy())

        if confidence > 0.5:  # 假设只考虑高置信度的检测
            person_keypoints = results[0].keypoints[idx].xy.cpu().numpy().flatten()  # 使用索引而非track_id
            detections.append(((x1, y1, x2, y2), confidence, "person", person_keypoints))

    # 使用 DeepSORT 跟踪
    tracks = tracker.update_tracks(detections, frame=frame)
    
    # 用于存储目标人物的索引（一旦检测到就锁定，注意这里改为了索引而不是track_id，因为我们需要用它来访问keypoints）
    target_index = None
    
    for idx, track in enumerate(tracks):
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
    
        if target_index is None:  # 尚未锁定目标

            detection_keypoints = None
            if idx < len(results[0].keypoints):  # 确保索引有效
                detection_keypoints = results[0].keypoints[idx].xy.cpu().numpy().flatten()
    
            if detection_keypoints is not None:
                similarity = cosine(detection_keypoints, target_features)
                if similarity < 0.5:  # 相似度小于阈值，认为是目标人物
                    target_index = idx  # 锁定目标人物的索引
                    print(f"目标已锁定，目标索引: {target_index}")
                    target_track_id = track_id
                    break
    
    # 绘制跟踪框和关键点（只针对目标人物）
    for idx, track in enumerate(tracks):
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
    
        if target_index is not None and track_id == target_track_id:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # print((results[0].keypoints))
    
    # 提取并打印目标人物的关键点坐标
            if len(results[0].keypoints) > target_index:  # 确保目标索引有效
                keypoints = results[0].keypoints[target_index].xy.cpu().numpy().reshape(-1, 2)
                
                for kp in keypoints:  # 每个keypoint是一个二维数组，表示 (x, y)
                    # print("1")
                    if isinstance(kp, np.ndarray) and len(kp) == 2:  # 检查每个关键点
                        x, y = int(kp[0]), int(kp[1])
                        # print("2")
                        if x != 0 and y != 0:
                            # print("3")
                            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)


    # 写入结果到视频
    out.write(frame)

    # 显示实时画面（可选）
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
