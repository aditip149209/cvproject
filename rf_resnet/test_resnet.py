import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
from utils import evaluate_model

# Config
data_dir = "split_data/test"
batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset
test_dataset = datasets.ImageFolder(data_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("models/resnet18.pth"))
model = model.to(device)

# Evaluation
acc, report, cm = evaluate_model(model, test_loader, device)

print("\n✅ Test Accuracy:", acc)
print("\n📊 Classification Report:\n", report)
print("\n🧩 Confusion Matrix:\n", cm)