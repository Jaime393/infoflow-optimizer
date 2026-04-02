import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# ==================== MODELO ====================
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

# ==================== DATASET ====================
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=64, shuffle=True)

# ==================== INFOFLOW CORREGIDO ====================
class InfoFlow(optim.Optimizer):
    def __init__(self, params, lr=0.5, eps=1e-8, beta=0.9):
        defaults = dict(lr=lr, eps=eps, beta=beta)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            beta = group['beta']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Normalización InfoFlow
                grad_mean_abs = grad.abs().mean() + eps
                info_grad = grad / grad_mean_abs

                # Momentum (estado guardado por parámetro)
                state = self.state[p]
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p.data)

                momentum = state['momentum']
                momentum.mul_(beta).add_(info_grad, alpha=1 - beta)

                # Actualización correcta (sin warning)
                p.data.add_(momentum, alpha=-lr)


# ==================== FUNCIÓN DE ENTRENAMIENTO ====================
def train(optimizer_class, name, lr=0.001):
    model = Net().cuda() if torch.cuda.is_available() else Net()
    optimizer = optimizer_class(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\n=== {name} (lr={lr}) ===")
    for epoch in range(3):
        total_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(model.fc[0].weight.device), target.to(model.fc[0].weight.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: {avg_loss:.4f}")

# ==================== EJECUCIÓN ====================
print("=== Adam ===")
train(optim.Adam, "Adam", lr=0.001)

print("\n=== InfoFlow (versión corregida) ===")
train(InfoFlow, "InfoFlow", lr=0.5)   # ← lr alto es clave aquí
