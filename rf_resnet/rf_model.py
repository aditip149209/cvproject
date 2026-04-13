import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
joblib.dump(rf, "models/random_forest.pkl")

print("✅ Random Forest model saved!")