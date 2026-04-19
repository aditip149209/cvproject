import torch
import numpy as np
from pathlib import Path
import json
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_dir = Path("pipeline_output/training/models")
models_dir.mkdir(parents=True, exist_ok=True)

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset
train_dataset = datasets.ImageFolder("split_data/train", transform=transform)
test_dataset = datasets.ImageFolder("split_data/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pretrained ResNet (feature extractor)
model = models.resnet18(pretrained=True)

# Remove final layer
model = torch.nn.Sequential(*list(model.children())[:-1])
model = model.to(device)
model.eval()

# 🔥 Feature extraction function
def extract_features(loader):
    features = []
    labels = []

    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)

            outputs = model(images)  # shape: (batch, 512, 1, 1)
            outputs = outputs.view(outputs.size(0), -1)  # flatten → (batch, 512)

            features.append(outputs.cpu().numpy())
            labels.append(lbls.numpy())

    return np.vstack(features), np.hstack(labels)

print("Extracting train features...")
X_train, y_train = extract_features(train_loader)

print("Extracting test features...")
X_test, y_test = extract_features(test_loader)

print("Feature shape:", X_train.shape)

# 🌲 Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n✅ Random Forest Accuracy:", acc)
print("\n📊 Classification Report:\n", report)
print("\n🧩 Confusion Matrix:\n", cm)

rf_path = models_dir / "rf_on_resnet_features.pkl"
joblib.dump(rf, rf_path)

feature_extractor_path = models_dir / "resnet18_feature_extractor_state_dict.pth"
torch.save(model.state_dict(), feature_extractor_path)

with (models_dir / "rf_on_resnet_metadata.json").open("w", encoding="utf-8") as f:
    json.dump(
        {
            "classes": train_dataset.classes,
            "accuracy": float(acc),
            "transform": {
                "resize": [224, 224],
                "to_tensor": True,
            },
            "feature_extractor": "resnet18_without_fc",
        },
        f,
        indent=2,
    )

print(f"✅ Random Forest model saved: {rf_path}")
print(f"✅ Feature extractor weights saved: {feature_extractor_path}")