import torch
from torch.optim.optimizer import Optimizer

class InfoFlow(Optimizer):
    def __init__(self, params, lr=0.1, eps=1e-8, beta=0.9):
        defaults = dict(lr=lr, eps=eps, beta=beta)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            beta = group['beta']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data)

                momentum = state['momentum']

                # 🔥 InfoFlow core (normalización de "flujo de información")
                info_grad = grad / (grad.abs().mean() + eps)

                # Momentum estándar (con factor 1-beta)
                momentum.mul_(beta).add_(info_grad, alpha=1 - beta)

                # Actualización moderna (sin warning de deprecación)
                p.data.add_(momentum, alpha=-lr)

        return loss
