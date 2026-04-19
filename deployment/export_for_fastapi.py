#!/usr/bin/env python3
"""Create a portable model bundle for FastAPI deployment.

Copies currently available artifacts from pipeline_output/training into
pipeline_output/fastapi_bundle and writes a manifest for server-side loading.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List


def copy_if_exists(src: Path, dst: Path, copied: List[str]) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(str(dst))


def build_manifest(bundle_dir: Path, copied_files: List[str]) -> Dict[str, object]:
    rel = [str(Path(p).relative_to(bundle_dir)) for p in copied_files]

    models = {
        "best_model": "models/best_model.pkl",
        "svm_rbf": "models/svm_rbf.pkl",
        "random_forest": "models/random_forest.pkl",
        "mlp": "models/mlp.pkl",
        "knn_pca": "models/knn_pca.pkl",
        "kmeans_pca": "models/kmeans_pca.pkl",
    }

    available_models = {k: v for k, v in models.items() if (bundle_dir / v).exists()}

    return {
        "bundle_version": 1,
        "files": sorted(rel),
        "models": available_models,
        "preprocessing": {
            "bundle": "models/preprocessing_bundle.pkl",
            "kmeans_label_map": "models/kmeans_cluster_to_label.json",
        },
        "metrics": {
            "summary": "metrics.json",
            "kmeans": "kmeans_metrics.json",
            "knn": "knn_metrics.json",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create FastAPI model bundle")
    parser.add_argument("--training-dir", type=Path, default=Path("pipeline_output/training"))
    parser.add_argument("--bundle-dir", type=Path, default=Path("pipeline_output/fastapi_bundle"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_dir = args.training_dir.resolve()
    bundle_dir = args.bundle_dir.resolve()

    if not training_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {training_dir}")

    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "models").mkdir(parents=True, exist_ok=True)

    copied: List[str] = []

    # Core training files
    copy_if_exists(training_dir / "best_model.pkl", bundle_dir / "models" / "best_model.pkl", copied)
    copy_if_exists(training_dir / "best_model.joblib", bundle_dir / "models" / "best_model.joblib", copied)
    copy_if_exists(training_dir / "metrics.json", bundle_dir / "metrics.json", copied)
    copy_if_exists(training_dir / "kmeans_metrics.json", bundle_dir / "kmeans_metrics.json", copied)
    copy_if_exists(training_dir / "knn_metrics.json", bundle_dir / "knn_metrics.json", copied)

    # Additional model artifacts
    model_dir = training_dir / "models"
    copy_if_exists(model_dir / "svm_rbf.pkl", bundle_dir / "models" / "svm_rbf.pkl", copied)
    copy_if_exists(model_dir / "random_forest.pkl", bundle_dir / "models" / "random_forest.pkl", copied)
    copy_if_exists(model_dir / "mlp.pkl", bundle_dir / "models" / "mlp.pkl", copied)
    copy_if_exists(model_dir / "knn_pca.pkl", bundle_dir / "models" / "knn_pca.pkl", copied)
    copy_if_exists(model_dir / "kmeans_pca.pkl", bundle_dir / "models" / "kmeans_pca.pkl", copied)
    copy_if_exists(model_dir / "preprocessing_bundle.pkl", bundle_dir / "models" / "preprocessing_bundle.pkl", copied)
    copy_if_exists(model_dir / "kmeans_cluster_to_label.json", bundle_dir / "models" / "kmeans_cluster_to_label.json", copied)

    manifest = build_manifest(bundle_dir, copied)
    manifest_path = bundle_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Bundle created at: {bundle_dir}")
    print(f"Files copied: {len(copied)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
