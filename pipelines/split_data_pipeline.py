#!/usr/bin/env python3
"""Build features from split_data without modifying the split itself.

Input:
  split_data/<split>/<class>/*

Output:
  pipeline_output/
    processed_images/<split>/<class>/*.png
    arrays/X_train.npy, y_train.npy, X_test.npy, y_test.npy
    metadata/records.csv
    metadata/summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def moving_average(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    if signal.size < window_size:
        return signal
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    return np.convolve(signal, kernel, mode="valid")


def preprocess_image(image_path: Path, img_size: int, ma_window: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None

    img = cv2.resize(img, (img_size, img_size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)

    h, w = equalized.shape
    cropped = equalized[int(h * 0.1) : int(h * 0.9), int(w * 0.1) : int(w * 0.9)]

    signal = cropped.flatten().astype(np.float32)
    filtered_signal = moving_average(signal, window_size=ma_window)
    return cropped, filtered_signal


def spectral_features(signal: np.ndarray) -> np.ndarray:
    fft_vals = np.abs(np.fft.fft(signal))
    fft_sum = np.sum(fft_vals)

    if fft_sum <= 0:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    mean_freq = float(np.mean(fft_vals))
    indices = np.arange(len(fft_vals), dtype=np.float32)
    spectral_centroid = float(np.sum(indices * fft_vals) / fft_sum)
    bandwidth = float(np.sqrt(np.sum(((indices - spectral_centroid) ** 2) * fft_vals) / fft_sum))
    flatness = float(np.exp(np.mean(np.log(fft_vals + 1e-10))) / (mean_freq + 1e-10))

    return np.array([mean_freq, spectral_centroid, bandwidth, flatness], dtype=np.float32)


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def zscore_mask(X: np.ndarray, threshold: float) -> np.ndarray:
    if len(X) == 0:
        return np.array([], dtype=bool)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    z = np.abs((X - mean) / std)
    return (z < threshold).all(axis=1)


def collect_records(
    input_dir: Path,
    output_dir: Path,
    img_size: int,
    ma_window: int,
    limit_per_class: int,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []

    splits = [p for p in sorted(input_dir.iterdir()) if p.is_dir()]
    if not splits:
        raise ValueError(f"No split folders found in {input_dir}")

    for split_dir in splits:
        split_name = split_dir.name
        class_dirs = [p for p in sorted(split_dir.iterdir()) if p.is_dir()]

        for class_dir in class_dirs:
            class_name = class_dir.name
            images = [p for p in sorted(class_dir.iterdir()) if is_image(p)]
            if limit_per_class > 0:
                images = images[:limit_per_class]

            for image_path in tqdm(images, desc=f"{split_name}/{class_name}", leave=False):
                processed_image, signal = preprocess_image(image_path, img_size=img_size, ma_window=ma_window)
                if processed_image is None or signal is None:
                    continue

                rel = image_path.relative_to(input_dir)
                output_img_path = output_dir / "processed_images" / rel
                output_img_path = output_img_path.with_suffix(".png")
                output_img_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_img_path), processed_image)

                feats = spectral_features(signal)

                records.append(
                    {
                        "split": split_name,
                        "label": class_name,
                        "input_image": str(image_path),
                        "processed_image": str(output_img_path),
                        "f0_mean_freq": float(feats[0]),
                        "f1_spectral_centroid": float(feats[1]),
                        "f2_bandwidth": float(feats[2]),
                        "f3_flatness": float(feats[3]),
                    }
                )

    return records


def save_records_csv(records: List[Dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        csv_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(records[0].keys()) + ["kept"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({**row, "kept": row.get("kept", False)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline from split_data to pipeline_output")
    parser.add_argument("--input-dir", type=Path, default=Path("split_data"), help="Input split folder")
    parser.add_argument("--output-dir", type=Path, default=Path("pipeline_output"), help="Output folder")
    parser.add_argument("--img-size", type=int, default=128, help="Image resize dimension")
    parser.add_argument("--ma-window", type=int, default=5, help="Moving-average window")
    parser.add_argument("--outlier-threshold", type=float, default=3.0, help="Z-score outlier threshold")
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="Optional debug limit per class (0 means all images)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    records = collect_records(
        input_dir=input_dir,
        output_dir=output_dir,
        img_size=args.img_size,
        ma_window=args.ma_window,
        limit_per_class=args.limit_per_class,
    )

    if not records:
        raise RuntimeError("No valid images processed. Check input paths and file types.")

    train_records = [r for r in records if str(r["split"]).lower() == "train"]
    test_records = [r for r in records if str(r["split"]).lower() == "test"]

    if not train_records or not test_records:
        raise RuntimeError("Both 'train' and 'test' split folders are required in split_data.")

    X_train_raw = np.array(
        [[r["f0_mean_freq"], r["f1_spectral_centroid"], r["f2_bandwidth"], r["f3_flatness"]] for r in train_records],
        dtype=np.float32,
    )
    X_test_raw = np.array(
        [[r["f0_mean_freq"], r["f1_spectral_centroid"], r["f2_bandwidth"], r["f3_flatness"]] for r in test_records],
        dtype=np.float32,
    )

    train_keep_mask = zscore_mask(X_train_raw, threshold=args.outlier_threshold)

    filtered_train_records: List[Dict[str, object]] = []
    for r, keep in zip(train_records, train_keep_mask):
        r["kept"] = bool(keep)
        if keep:
            filtered_train_records.append(r)

    for r in test_records:
        r["kept"] = True

    X_train_filtered = X_train_raw[train_keep_mask]
    y_train_labels = np.array([r["label"] for r in filtered_train_records])
    y_test_labels = np.array([r["label"] for r in test_records])

    if len(X_train_filtered) == 0:
        raise RuntimeError("All training samples were removed as outliers. Increase --outlier-threshold.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_filtered)
    X_test = scaler.transform(X_test_raw)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_labels)

    unseen_test = set(np.unique(y_test_labels)) - set(encoder.classes_)
    if unseen_test:
        raise RuntimeError(f"Test split has unseen labels not in train split: {sorted(unseen_test)}")
    y_test = encoder.transform(y_test_labels)

    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    np.save(arrays_dir / "X_train.npy", X_train)
    np.save(arrays_dir / "y_train.npy", y_train)
    np.save(arrays_dir / "X_test.npy", X_test)
    np.save(arrays_dir / "y_test.npy", y_test)

    metadata_dir = output_dir / "metadata"
    save_records_csv(train_records + test_records, metadata_dir / "records.csv")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "img_size": args.img_size,
        "ma_window": args.ma_window,
        "outlier_threshold": args.outlier_threshold,
        "num_total_records": len(records),
        "num_train_before_outlier": len(train_records),
        "num_train_after_outlier": int(len(filtered_train_records)),
        "num_test": len(test_records),
        "class_names": list(encoder.classes_),
        "arrays": {
            "X_train": [int(v) for v in X_train.shape],
            "y_train": [int(v) for v in y_train.shape],
            "X_test": [int(v) for v in X_test.shape],
            "y_test": [int(v) for v in y_test.shape],
        },
    }

    with (metadata_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (metadata_dir / "label_mapping.json").open("w", encoding="utf-8") as f:
        json.dump({"classes": list(encoder.classes_)}, f, indent=2)

    print("Pipeline completed.")
    print(f"Input split used: {input_dir}")
    print(f"Output written to: {output_dir}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


if __name__ == "__main__":
    main()
