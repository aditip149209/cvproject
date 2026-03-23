#!/usr/bin/env python3
"""Train and evaluate models from precomputed arrays.

Input:
  pipeline_output/arrays/X_train.npy
  pipeline_output/arrays/y_train.npy
  pipeline_output/arrays/X_test.npy
  pipeline_output/arrays/y_test.npy

Output:
  pipeline_output/training/
    metrics.json
    report.txt
    best_model.joblib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def load_arrays(arrays_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"]
    for name in required:
        path = arrays_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing required array file: {path}")

    X_train = np.load(arrays_dir / "X_train.npy")
    y_train = np.load(arrays_dir / "y_train.npy")
    X_test = np.load(arrays_dir / "X_test.npy")
    y_test = np.load(arrays_dir / "y_test.npy")

    return X_train, y_train, X_test, y_test


def evaluate_model(model, X_train, y_train, X_test, y_test) -> Dict[str, object]:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, pred))

    return {
        "accuracy": acc,
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models from pipeline_output arrays")
    parser.add_argument("--arrays-dir", type=Path, default=Path("pipeline_output/arrays"), help="Input arrays folder")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pipeline_output/training"),
        help="Folder for training artifacts",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arrays_dir = args.arrays_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_arrays(arrays_dir)

    models = {
        "svm_rbf": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=args.random_seed),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=args.random_seed,
            n_jobs=-1,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            max_iter=500,
            random_state=args.random_seed,
        ),
    }

    metrics: Dict[str, object] = {
        "data_shapes": {
            "X_train": list(X_train.shape),
            "y_train": list(y_train.shape),
            "X_test": list(X_test.shape),
            "y_test": list(y_test.shape),
        },
        "models": {},
    }

    best_name = None
    best_acc = -1.0

    for model_name, model in models.items():
        result = evaluate_model(model, X_train, y_train, X_test, y_test)
        metrics["models"][model_name] = result
        if result["accuracy"] > best_acc:
            best_acc = result["accuracy"]
            best_name = model_name

    assert best_name is not None

    best_model = models[best_name]
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, output_dir / "best_model.joblib")

    metrics["best_model"] = best_name
    metrics["best_accuracy"] = float(best_acc)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report_lines = [
        "Training Pipeline Report",
        "========================",
        "",
        f"Arrays dir: {arrays_dir}",
        f"Output dir: {output_dir}",
        f"Best model: {best_name}",
        f"Best accuracy: {best_acc:.4f}",
        "",
        "Model Accuracies:",
    ]

    for model_name, result in metrics["models"].items():
        report_lines.append(f"- {model_name}: {result['accuracy']:.4f}")

    with (output_dir / "report.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print("Training pipeline completed.")
    print(f"Best model: {best_name}")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
