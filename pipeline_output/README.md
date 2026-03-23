# Pipeline Output Folder

This folder stores artifacts produced by `pipelines/split_data_pipeline.py`.

Input is always read from `split_data/`.
The `split_data/` folder is never modified.

Expected output structure after running the pipeline:

- `processed_images/<split>/<class>/*.png`
- `arrays/X_train.npy`
- `arrays/y_train.npy`
- `arrays/X_test.npy`
- `arrays/y_test.npy`
- `metadata/records.csv`
- `metadata/summary.json`
- `metadata/label_mapping.json`

Additional output after running `pipelines/train_from_arrays.py`:

- `training/metrics.json`
- `training/report.txt`
- `training/best_model.joblib`
