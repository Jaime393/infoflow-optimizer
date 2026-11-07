"""Benchmark MNIST con el montaje que publica el README.

Parametros tomados solo del README del repo (commit fa659a5):
MLP con 128 hidden units, 3 epocas (0, 1, 2), Adam con lr=0.001
e InfoFlow con lr=0.1. La perdida impresa es la media por epoca
(CrossEntropyLoss, media por lote), como en `benchmark_cifar.py`.

Este archivo NO afirma reproducir los valores de la tabla del
README. Esos valores no se pueden reproducir desde el repo: no hay
ningun script MNIST en el repo, y la tabla es un dato del README,
no un resultado verificado aqui. Lo que este script hace es correr
el montaje descrito, con esos parametros, e imprimir su propia
perdida por epoca.

El resto de elecciones (batch 128, datos en ./data, cuda si hay)
sigue el mismo patron de `benchmark_cifar.py` y se marca como
eleccion del script, no como dato del README.
"""

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from infoflow import InfoFlow


class MLP128(nn.Module):
    """MLP con 128 hidden units, segun el README."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def run_one(name, make_optimizer, trainloader, device, criterion):
    model = MLP128().to(device)
    optimizer = make_optimizer(model.parameters())
    print(f"\n=== {name} ===")
    for epoch in range(3):
        start = time.time()
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(trainloader)
        total_time = (time.time() - start) * 1000
        print(f"Epoch {epoch}: {epoch_loss:.4f}  (total: {total_time:.1f})")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"InfoFlow v6.0 corriendo en {device}")

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    run_one(
        "Adam (lr=0.001)",
        lambda params: optim.Adam(params, lr=0.001),
        trainloader,
        device,
        criterion,
    )
    run_one(
        "InfoFlow (lr=0.1)",
        lambda params: InfoFlow(params, lr=0.1),
        trainloader,
        device,
        criterion,
    )
    print("\nBENCHMARK MNIST TERMINADO")


if __name__ == "__main__":
    main()
