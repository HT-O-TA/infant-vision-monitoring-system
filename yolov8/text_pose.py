# # 访问检测结果
# for result in results:
#     boxes = result.boxes.xywh  # 获取每个边界框的中心点、宽度和高度
#     classes = result.boxes.cls  # 获取每个目标的类别
#     for box, cls in zip(boxes, classes):
#         print(f"Class: {cls}, Box: {box}")

from ultralytics import YOLO

# 加载预训练的 YOLOv8 模型（n=Nano, s=Small, m=Medium, l=Large, x=Extra Large）
model = YOLO("F:/project/infant/yolov8/yolo11x-pose.pt")  # 使用预训练模型 

# 推理设置
source = "F:/project/infant/yolov8/infant1.mp4"  # 输入文件，可以是图像、视频或目录
results = model.predict(source=source, device=0, save=True, task='pose')  # 指定 GPU (device=0) 并保存结果

# 打印推理结果
for result in results:
    print(result.boxes)  # 输出检测到的边界框

