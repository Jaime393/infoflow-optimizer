import torch
from torch.optim.optimizer import Optimizer

class InfoFlow(Optimizer):
    """InfoFlow Optimizer - Normaliza el gradiente por su "flujo de información"."""
    def __init__(self, params, lr=0.1, eps=1e-8, beta=0.9, weight_decay=1e-4):
        defaults = dict(lr=lr, eps=eps, beta=beta, weight_decay=weight_decay)
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

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Weight decay (L2 regularization)
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # 🔥 CORE: Information Flow normalization
                info_grad = grad / (grad.abs().mean() + eps)

                # Momentum (estándar como en Adam/SGD)
                state = self.state[p]
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data)

                momentum = state['momentum']
                momentum.mul_(beta).add_(info_grad, alpha=1 - beta)

                # Actualización moderna (sin warnings)
                p.data.add_(momentum, alpha=-lr)

        return loss
