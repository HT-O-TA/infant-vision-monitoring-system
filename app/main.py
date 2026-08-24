from __future__ import annotations

import io
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from app.model import EmotionModel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "best_emotion_model.pt"
MODEL_PATH = Path(os.getenv("INFANT_EMOTION_MODEL", DEFAULT_MODEL_PATH))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["negative", "positive"]

app = FastAPI(
    title="Infant Vision Service",
    description="婴幼儿看护场景的情绪识别实验 API",
    version="0.1.0",
)

model: EmotionModel | None = None
model_error: str | None = None

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def load_model() -> None:
    global model, model_error
    if not MODEL_PATH.exists():
        model_error = f"模型文件不存在: {MODEL_PATH}"
        return
    try:
        loaded = EmotionModel(num_classes=len(LABELS)).to(DEVICE)
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        loaded.load_state_dict(state)
        loaded.eval()
        model = loaded
        model_error = None
    except Exception as exc:  # 模型文件可能来自不同历史版本
        model = None
        model_error = str(exc)


load_model()


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "Infant Vision Service", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ready" if model is not None else "degraded",
        "device": str(DEVICE),
        "emotion_model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "detail": model_error,
    }


@app.post("/api/v1/emotion")
async def recognize_emotion(file: UploadFile = File(...)) -> dict[str, object]:
    if model is None:
        raise HTTPException(status_code=503, detail=model_error or "模型未加载")

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="无法解析上传的图片") from exc

    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    index = int(torch.argmax(probabilities).item())
    return {
        "emotion": LABELS[index],
        "confidence": round(float(probabilities[index].item()), 6),
    }
