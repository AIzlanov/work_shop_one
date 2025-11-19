from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import shutil
import os
from datetime import datetime, timezone

from model.model_wrapper import ModelWrapper

app = FastAPI(title="Heart Risk Prediction API")

MODEL_PATH = "best_model.pkl"
THRESHOLD_PATH = "best_threshold.json"

mw = ModelWrapper(
    model_path=MODEL_PATH,
    threshold_path=THRESHOLD_PATH
)

class PathRequest(BaseModel):
    csv_path: str
    out_dir: Optional[str] = "predictions"


@app.post("/predict_from_path")
def predict_from_path(req: PathRequest):
    if not os.path.exists(req.csv_path):
        raise HTTPException(status_code=400, detail="File not found")

    df = pd.read_csv(req.csv_path)

    preds = mw.predict_from_df(df)
    ids = df["id"].astype(int).tolist()

    os.makedirs(req.out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = os.path.join(req.out_dir, f"predictions_{ts}.csv")

    pd.DataFrame({"id": ids, "prediction": preds}).to_csv(out_path, index=False)

    return {"out_path": out_path, "n_rows": len(preds)}


@app.post("/predict_upload")
def predict_upload(file: UploadFile = File(...), out_dir: str = "predictions"):
    os.makedirs("tmp_uploads", exist_ok=True)
    tmp_path = os.path.join("tmp_uploads", file.filename)

    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    df = pd.read_csv(tmp_path)
    os.remove(tmp_path)

    # правильное распаковка результата
    ids, preds = mw.predict_from_df(df)

    # preds гарантированно привести к списку
    if hasattr(preds, "ravel"):
        preds = preds.ravel().tolist()
    else:
        preds = list(preds)

    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = os.path.join(out_dir, f"predictions_upload_{ts}.csv")

    pd.DataFrame({
        "id": ids,
        "prediction": preds
    }).to_csv(out_path, index=False)

    return {"out_path": out_path, "n_rows": len(ids)}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"status": "ok", "message": "Heart Risk API is running", "description": "go to: http://127.0.0.1:8000/docs"}