from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from fastapi_model_loader_template import (
    startup_load_models,
    predict_best_model,
    predict_knn,
    predict_kmeans_with_label_map,
)


app = FastAPI(title="Model Inference API")


class PredictRequest(BaseModel):
    rows: list[list[float]]


@app.on_event("startup")
def load_artifacts() -> None:
    startup_load_models()


@app.post("/predict/best")
def predict_best(payload: PredictRequest) -> dict:
    preds = predict_best_model(payload.rows)
    return {"predictions": preds}


@app.post("/predict/knn")
def predict_knn_endpoint(payload: PredictRequest) -> dict:
    preds = predict_knn(payload.rows)
    return {"predictions": preds}


@app.post("/predict/kmeans")
def predict_kmeans_endpoint(payload: PredictRequest) -> dict:
    preds = predict_kmeans_with_label_map(payload.rows)
    return {"predictions": preds}
