import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Config
data_dir = "split_data/train"
batch_size = 32
epochs = 5
lr = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_dir = Path("pipeline_output/training/models")
models_dir.mkdir(parents=True, exist_ok=True)

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# Dataset
train_dataset = datasets.ImageFolder(data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("Classes:", train_dataset.classes)

# Model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)
model = model.to(device)

# Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training
for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}")

# Save model
weights_path = models_dir / "resnet18_state_dict.pth"
torch.save(model.state_dict(), weights_path)

torchscript_model = torch.jit.script(model.cpu())
torchscript_path = models_dir / "resnet18_torchscript.pt"
torchscript_model.save(str(torchscript_path))

with (models_dir / "resnet18_classes.json").open("w", encoding="utf-8") as f:
    json.dump({"classes": train_dataset.classes}, f, indent=2)

print(f"✅ Saved PyTorch state_dict: {weights_path}")
print(f"✅ Saved TorchScript model: {torchscript_path}")