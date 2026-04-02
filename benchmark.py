import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from infoflow import InfoFlow

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=64, shuffle=True)

def train(optimizer_class, name, lr=0.001):
    model = Net()
    optimizer = optimizer_class(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\n=== {name} (lr={lr}) ===")
    for epoch in range(3):
        total_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: {avg_loss:.4f} (total: {total_loss:.1f})")

# ====================== EJECUCIÓN ======================
print("=== Adam ===")
train(optim.Adam, "Adam", lr=0.001)

print("\n=== InfoFlow ===")
train(InfoFlow, "InfoFlow", lr=0.1)   # ← valor óptimo actual
