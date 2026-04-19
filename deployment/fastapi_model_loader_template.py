"""FastAPI-side model loading template.

Copy this file into your FastAPI repo and adjust path constants.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np


BUNDLE_ROOT = Path("model_bundle")
MANIFEST_PATH = BUNDLE_ROOT / "manifest.json"


class ModelRegistry:
    def __init__(self, bundle_root: Path = BUNDLE_ROOT) -> None:
        self.bundle_root = bundle_root
        self.manifest = self._read_manifest()
        self.models: Dict[str, Any] = {}
        self.preprocessing: Dict[str, Any] = {}

    def _read_manifest(self) -> Dict[str, Any]:
        manifest_path = self.bundle_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest.json at {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def load(self) -> None:
        model_map: Dict[str, str] = self.manifest.get("models", {})
        for model_name, rel_path in model_map.items():
            model_path = self.bundle_root / rel_path
            if model_path.exists():
                self.models[model_name] = joblib.load(model_path)

        preprocessing_bundle_rel = self.manifest.get("preprocessing", {}).get("bundle")
        if preprocessing_bundle_rel:
            preprocessing_path = self.bundle_root / preprocessing_bundle_rel
            if preprocessing_path.exists():
                self.preprocessing = joblib.load(preprocessing_path)

    def engineer_features(self, x: np.ndarray) -> np.ndarray:
        # Must match training logic: append mean, std, min, max per row.
        return np.hstack(
            (
                x,
                np.mean(x, axis=1, keepdims=True),
                np.std(x, axis=1, keepdims=True),
                np.min(x, axis=1, keepdims=True),
                np.max(x, axis=1, keepdims=True),
            )
        )

    def transform_for_pca_models(self, x: np.ndarray) -> np.ndarray:
        if not self.preprocessing:
            raise RuntimeError("Preprocessing bundle not loaded")

        scaler = self.preprocessing["scaler"]
        pca = self.preprocessing["pca"]

        engineered = self.engineer_features(x)
        x_scaled = scaler.transform(engineered)
        return pca.transform(x_scaled)


registry = ModelRegistry()


def startup_load_models() -> None:
    registry.load()


def predict_best_model(x_rows: list[list[float]]) -> list[int]:
    x = np.asarray(x_rows, dtype=np.float32)
    engineered = registry.engineer_features(x)

    # Prefer best_model if present, fallback to random forest.
    model = registry.models.get("best_model") or registry.models.get("random_forest")
    if model is None:
        raise RuntimeError("No compatible best model found in bundle")

    preds = model.predict(engineered)
    return [int(v) for v in preds]


def predict_knn(x_rows: list[list[float]]) -> list[int]:
    x = np.asarray(x_rows, dtype=np.float32)
    model = registry.models.get("knn_pca")
    if model is None:
        raise RuntimeError("knn_pca model not loaded")

    x_pca = registry.transform_for_pca_models(x)
    preds = model.predict(x_pca)
    return [int(v) for v in preds]


def predict_kmeans_with_label_map(x_rows: list[list[float]]) -> list[int]:
    x = np.asarray(x_rows, dtype=np.float32)
    model = registry.models.get("kmeans_pca")
    if model is None:
        raise RuntimeError("kmeans_pca model not loaded")

    x_pca = registry.transform_for_pca_models(x)
    cluster_ids = model.predict(x_pca)

    mapping_file = registry.bundle_root / registry.manifest["preprocessing"]["kmeans_label_map"]
    with mapping_file.open("r", encoding="utf-8") as f:
        mapping = json.load(f)["mapping"]

    return [int(mapping.get(str(cid), 0)) for cid in cluster_ids]
