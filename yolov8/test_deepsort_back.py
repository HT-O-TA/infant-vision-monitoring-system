from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
from scipy.spatial.distance import cosine
import numpy as np


def target_index_to_detection_index(target_index, detections, tracks):
    target_track = tracks[target_index]
    target_ltrb = target_track.to_ltrb()
    target_x1, target_y1, target_x2, target_y2 = map(int, target_ltrb)

    best_match_idx = -1
    best_iou = 0

    for idx, detection in enumerate(detections):
        x1, y1, x2, y2 = detection[0]
        # 计算 IOU（交并比）
        iou = compute_iou((x1, y1, x2, y2), (target_x1, target_y1, target_x2, target_y2))
        if iou > best_iou:
            best_iou = iou
            best_match_idx = idx

    return best_match_idx

def compute_iou(boxA, boxB):
    xA, yA, xB, yB = boxA
    xC, yC, xD, yD = boxB

    # 计算交集
    interX1 = max(xA, xC)
    interY1 = max(yA, yC)
    interX2 = min(xB, xD)
    interY2 = min(yB, yD)
    
    # 计算交集的面积
    interArea = max(0, interX2 - interX1) * max(0, interY2 - interY1)
    
    # 计算两个框的面积
    boxAArea = (xB - xA) * (yB - yA)
    boxBArea = (xD - xC) * (yD - yC)

    # 计算IOU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# 初始化 YOLO 模型
model = YOLO("F:/project/infant/yolov8/yolo11x-pose.pt")  # 替换为你的 YOLO 模型路径

# 初始化 DeepSORT 跟踪器
tracker = DeepSort(max_age=20, n_init=4, nn_budget=150, max_iou_distance=0.5)

# 设置视频源和输出路径
video_source = "F:/project/infant/yolov8/infant2.mp4"  # 输入视频路径
output_path = "F:/project/infant/yolov8/output_track_target.mp4"  # 输出结果路径
target_image_path = "F:/project/infant/yolov8/infant2.png"  # 目标人物图像路径

cap = cv2.VideoCapture(video_source)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 初始化视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 提取目标人物的特征
def extract_features(image, model):
    results = model.predict(source=image, devicfrom ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
from scipy.spatial.distance import cosine
import numpy as np

# 初始化 YOLO 模型
model = YOLO("F:/project/infant/yolov8/yolo11x-pose.pt")  # 替换为你的 YOLO 模型路径

# 初始化 DeepSORT 跟踪器
tracker = DeepSort(max_age=20, n_init=4, nn_budget=150, max_iou_distance=0.5)

# 设置视频源和输出路径
video_source = "F:/project/infant/yolov8/infant3.mp4"  # 输入视频路径
output_path = "F:/project/infant/yolov8/output_track_target.mp4"  # 输出结果路径
target_image_path = "F:/project/infant/yolov8/infant3.png"  # 目标人物图像路径

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

target_image = cv2.imread(target_image_path)
target_features = extract_features(target_image, model)
print(target_features)

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
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        confidence = float(box.conf[0].cpu().numpy())
        
        if confidence > 0.5:  # 假设只考虑高置信度的检测
            person_keypoints = results[0].keypoints[0].xy.cpu().numpy().flatten()
            detections.append(((x1, y1, x2, y2), confidence, "person", person_keypoints))

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
            # 初始化最小相似度和对应Track ID
            min_similarity = float('inf')
            best_match_id = None
            print("1")
            print(detections)

            # 遍历所有检测，计算相似度
            for detection in detections:
                detection_keypoints = detection[3]
                print("2")
                if detection_keypoints is not None:
                    print("3")
                    print(detection_keypoints)
                    similarity = cosine(detection_keypoints, target_features)
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

            # # 绘制Pose关键点
            # keypoints = results[0].keypoints[track_id].xy.cpu().numpy()  # 获取当前帧的关键点
            # for kp in keypoints:
            #     cv2.circle(frame, (int(kp[0]), int(kp[1])), 5, (0, 0, 255), -1)  # 绘制关键点

            # # 绘制骨架连接线（如果关键点足够的话）
            # skeleton_pairs = [
            #     [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], 
            #     [1, 9], [9, 10], [10, 11], [5, 12], [12, 13], [13, 14], [6, 15], [15, 16]
            # ]
            # for pair in skeleton_pairs:
            #     start_point = keypoints[pair[0]]
            #     end_point = keypoints[pair[1]]
            #     cv2.line(frame, (int(start_point[0]), int(start_point[1])), 
            #             (int(end_point[0]), int(end_point[1])), (255, 0, 0), 2)  # 绘制骨架连线

    # 写入结果到视频
    out.write(frame)

    # 显示实时画面（可选）
    # cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
e=0, task='pose')  # 检测关键点
    if len(results[0].keypoints) > 0:
        keypoints = results[0].keypoints[0].xy.cpu().numpy()
        return keypoints.flatten()  # 转为一维数组
    return None

target_image = cv2.imread(target_image_path)
target_features = extract_features(target_image, model)

if target_features is None:
    print("无法提取目标人物的特征，请检查目标图片！")
    exit()

# 用于存储目标人物的索引（一旦检测到就锁定）
target_index = None

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

    # 检查并锁定目标人物
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
                    print(f"Target locked: Index {target_index}")
                    break

    # 绘制跟踪框和关键点（只针对目标人物）
    for idx, track in enumerate(tracks):
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        if target_index is not None and idx == target_index_to_detection_index(target_index, detections, tracks):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 提取并绘制关键点
            keypoints = results[0].keypoints[target_index].xy.cpu().numpy()  # 这里使用target_index作为示例
            for kp in keypoints:
                if isinstance(kp, np.ndarray) and len(kp) == 2:
                    x, y = int(kp[0]), int(kp[1])
                    if x != 0 and y != 0:
                        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    # 写入结果到视频
    out.write(frame)

    # 显示实时画面（可选）
    # cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
