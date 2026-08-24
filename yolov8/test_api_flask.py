from flask import Flask, request, jsonify
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models

app = Flask(__name__)

class ObjectTracking:
    def __init__(self, yolo_model_path, device="cuda"):
        self.model = YOLO(yolo_model_path)
        self.tracker = DeepSort(max_age=20, n_init=4, nn_budget=150, max_iou_distance=0.5)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.resnet = models.resnet50(pretrained=True).to(self.device)
        self.resnet.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.target_features = None
        self.target_track_id = None

    def set_target_image(self, target_image_path):
        target_image = cv2.imread(target_image_path)
        self.target_features = self.extract_image_features(target_image)
        if self.target_features is None:
            raise ValueError("无法提取目标人物的特征，请检查目标图片！")

    def extract_image_features(self, image):
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.resnet(image)
        return features.cpu().numpy().flatten()

    def process_frame(self, frame):
        results = self.model.predict(source=frame, device=0)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            if confidence > 0.5:
                cropped_image = frame[y1:y2, x1:x2]
                person_features = self.extract_image_features(cropped_image)
                detections.append(((x1, y1, x2, y2), confidence, "person", person_features))
        tracks = self.tracker.update_tracks(detections, frame=frame)
        target_coordinates = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            if track_id == self.target_track_id:
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                target_coordinates.append((x1, y1, x2, y2))
        return target_coordinates

tracking = ObjectTracking("F:/project/infant/yolov8/yolo11x-pose.pt")

@app.route('/set_target', methods=['POST'])
def set_target():
    data = request.json
    target_image_path = data['target_image_path']
    try:
        tracking.set_target_image(target_image_path)
        return jsonify({"message": "Target image set successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/process_video', methods=['POST'])
def process_video():
    data = request.json
    video_path = data['video_path']
    results = []
    try:
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            target_coords = tracking.process_frame(frame)
            results.append(target_coords)
        cap.release()
        return jsonify({"results": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
