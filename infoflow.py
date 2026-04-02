import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.optimizer import Optimizer

# ====================== INFOFLOW v2.0 FINAL (AVANZADO) ======================
class InfoFlow(Optimizer):
    """
    InfoFlow Optimizer v2.0 FINAL
    Inspirado en Information Field Theory (Juan Diego Vicente Gabancho, 2026)
    """
    def __init__(self, params, lr=0.08, eps=1e-8, beta=0.9, weight_decay=1e-4, fisher_scale=0.1):
        defaults = dict(lr=lr, eps=eps, beta=beta, weight_decay=weight_decay, fisher_scale=fisher_scale)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            beta = group['beta']
            weight_decay = group['weight_decay']
            fisher_scale = group['fisher_scale']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # 🔥 CORE: Information Flow Normalization (∇log ρ)
                grad_mean_abs = grad.abs().mean() + eps
                info_grad = grad / grad_mean_abs

                # Fisher diagonal approximation
                fisher_diag = grad.abs() + eps
                info_grad = info_grad / (fisher_diag ** fisher_scale)

                # Protección anti-explosion
                info_grad = torch.clamp(info_grad, -5.0, 5.0)

                # Momentum
                state = self.state[p]
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data)
                momentum = state['momentum']
                momentum.mul_(beta).add_(info_grad, alpha=1 - beta)

                # Update
                p.data.add_(momentum, alpha=-lr)

        return loss


# ====================== BENCHMARK CIFAR-10 (NIVEL PRO) ======================
print("🚀 Descargando CIFAR-10...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

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

def run_benchmark(OptimizerClass, name, lr=0.001, epochs=5):
    model = SimpleCNN()
    optimizer = OptimizerClass(model.parameters(), lr=lr)
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
        print(f"Epoch {epoch+1}: {avg:.4f}  (total loss: {total_loss:.1f})")

# ====================== EJECUCIÓN FINAL ======================
print("🚀 INICIANDO BENCHMARK NIVEL FINAL...\n")

print("=== Adam ===")
run_benchmark(optim.Adam, "Adam", lr=0.001, epochs=5)

print("\n=== InfoFlow v2.0 (Information Field Theory) ===")
run_benchmark(InfoFlow, "InfoFlow", lr=0.08, epochs=5)

print("\n✅ ¡BENCHMARK TERMINADO!")
