from __future__ import annotations

from io import BytesIO

from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image

from bamboo_recognition.pipeline.predictor import BambooPredictor, PredictionRequest

app = FastAPI(title="Bamboo Recognition API", version="0.1.0")
predictor = BambooPredictor()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    retrieval_mode: str = Form("image_to_text"),
    top_k: int = Form(5),
    rhizome_type: str | None = Form(None),
    branch_type: str | None = Form(None),
) -> dict:
    image_bytes = await image.read()
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    request = PredictionRequest(
        image=pil_image,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        rhizome_type=rhizome_type,
        branch_type=branch_type,
    )
    return predictor.predict(request)
