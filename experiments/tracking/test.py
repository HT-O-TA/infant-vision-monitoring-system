import cv2
import numpy as np
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis

# 初始化人脸分析器
app = FaceAnalysis(name='buffalo_l')  # 可根据需要选择模型，如 'buffalo_l'、'antelopev2' 等
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 表示使用 GPU，若无 GPU 可设置为 -1

def get_face_embedding(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像：{image_path}")
        return None

    # 人脸检测与特征提取
    faces = app.get(img)
    if not faces:
        print(f"未检测到人脸：{image_path}")
        return None

    # 提取第一个人脸的特征向量
    return faces[0].embedding

# 示例：计算两张人脸图像的相似度
embedding1 = get_face_embedding('crop_0.jpg')
embedding2 = get_face_embedding('target\\target_default.jpg')

if embedding1 is not None and embedding2 is not None:
    similarity = 1 - cosine(embedding1, embedding2)
    print(f"两人之间的相似度为: {similarity:.4f}")
else:
    print("无法计算相似度，特征提取失败。")
