from ultralytics import YOLO
import ssl

# 加载官方或自定义模型
model = YOLO("yolov8n.pt")  # Load an official Detect model

ssl._create_default_https_context = ssl._create_unverified_context

# Perform tracking with the model
results = model.track("https://youtu.be/LNwODJXcvt4", show=True)  # Tracking with default tracker

