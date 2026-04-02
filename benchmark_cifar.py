import sys
sys.path.insert(0, '/content/infoflow-optimizer')
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from infoflow import InfoFlow

# ==================== CIFAR-10 + Simple CNN ====================
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128*8*8, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.net(x)

def train_benchmark(optimizer_class, name, lr, epochs=5):
    model = SimpleCNN()
    optimizer = optimizer_class(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    print(f"\n=== {name} (lr={lr}) - CIFAR-10 ===")
    for epoch in range(epochs):
        total_loss = 0.0
        for data, target in trainloader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(trainloader)
        print(f"Epoch {epoch}: {avg:.4f}")

# ====================== EJECUCIÓN ======================
print("=== Adam ===")
train_benchmark(optim.Adam, "Adam", lr=0.001, epochs=5)

print("\n=== InfoFlow v2.0 ===")
train_benchmark(InfoFlow, "InfoFlow", lr=0.08, epochs=5)
