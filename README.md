# 婴幼儿视觉识别与看护实验

面向看护场景的视觉算法实践，包含情绪识别 API，以及姿态估计、目标检测、摔倒判断和跟踪实验代码。

## 当前主线

`app/` 提供 FastAPI 图像情绪识别服务：

```text
上传图像 → RGB 转换与缩放 → PyTorch 推理 → 情绪类别与置信度
```

当前类别为 `positive` 与 `negative`。分类结果用于算法实验，不代表对儿童情绪或健康状态的专业判断。

## 能力与代码范围

| 路径 | 内容 |
| --- | --- |
| [app](app/) | 当前情绪识别 API 与模型定义 |
| [models](models/) | 模型文件放置位置，权重需另行准备 |
| [core](core/) | 综合推理与模型相关实验代码 |
| [track](track/) | DeepSORT 等跟踪实验 |
| [yolov8](yolov8/) | YOLO 与姿态相关实验 |
| [experiments](experiments/) | 实验材料 |
| [docs](docs/) | 补充说明 |

姿态、摔倒、跟踪和控制脚本属于独立实验，不是已贯通的统一看护系统；部分历史依赖与权重未包含，不保证这些脚本直接运行。

## 本地运行

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

准备与模型定义匹配的权重，默认位置为：

```text
models/best_emotion_model.pt
```

也可通过环境变量 `INFANT_EMOTION_MODEL` 指定路径。缺少权重时健康状态为 `degraded`，识别接口返回503，不会返回伪造结果。

## 接口

| 接口 | 用途 |
| --- | --- |
| `GET /health` | 模型加载状态与计算设备 |
| `POST /api/v1/emotion` | 图像情绪分类，上传字段为 `file` |
| `GET /docs` | 交互式 API 文档 |

请求示例：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/emotion \
  -F "file=@sample.jpg"
```

## 数据与结果

仓库不附带原始人脸数据、摄像头录像和模型权重。当前没有统一评测集上的准确率结论；使用授权的测试图像，并将单张预测与完整场景评测区分。

