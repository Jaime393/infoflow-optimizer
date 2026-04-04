# ==================== BENCHMARK CIFAR-10 - InfoFlow v5.0 MAXIMUM ====================
import sys
import os
# FIX para Colab + git clone (encuentra el módulo aunque se llame infoflow.py)
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
from infoflow import InfoFlow   # ← Nombre correcto del archivo infoflow.py

# Modelo CNN simple pero efectivo
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 InfoFlow v5.0 MAXIMUM corriendo en {device}")
criterion = nn.CrossEntropyLoss()

# === Adam baseline ===
model_adam = SimpleCNN().to(device)
optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)

print("\n=== Adam (lr=0.001) - CIFAR-10 ===")
for epoch in range(5):
    start = time.time()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_adam.zero_grad()
        outputs = model_adam(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_adam.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(trainloader)
    total_time = (time.time() - start) * 1000
    print(f"Epoch {epoch+1}: {epoch_loss:.4f}  (total: {total_time:.1f})")

# === InfoFlow v5.0 MAXIMUM ===
model_infoflow = SimpleCNN().to(device)
optimizer_infoflow = InfoFlow(model_infoflow.parameters())

print("\n=== InfoFlow v5.0 MAXIMUM - CIFAR-10 ===")
for epoch in range(5):
    start = time.time()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_infoflow.zero_grad()
        outputs = model_infoflow(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_infoflow.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(trainloader)
    total_time = (time.time() - start) * 1000
    print(f"Epoch {epoch+1}: {epoch_loss:.4f}  (total: {total_time:.1f})")

print("\n✅ BENCHMARK TERMINADO - InfoFlow v5.0 MAXIMUM (versión definitiva)")
