import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from infoflow import InfoFlow

# Modelo simple
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

# Dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=64, shuffle=True)

def train(optimizer_class, name):
    model = Net()
    optimizer = optimizer_class(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\n=== {name} ===")

    for epoch in range(3):
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"{name} Epoch {epoch}: {total_loss:.4f}")

train(lambda p, lr: optim.Adam(p, lr=lr), "Adam")
train(InfoFlow, "InfoFlow")
