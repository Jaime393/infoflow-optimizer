import torch
from torch.optim.optimizer import Optimizer

class InfoFlow(Optimizer):
    """
    InfoFlow Optimizer v2.0™
    Inspirado en Information Field Theory (Juan Diego Vicente Gabancho, 2026)
    
    Combina:
    - Normalización del flujo de información (∇log ρ)
    - Aproximación diagonal de Fisher (natural gradient light)
    - Momentum + protección adaptativa contra explosiones
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

                # Weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # 🔥 CORE: Information Flow normalization (∇log ρ)
                grad_mean_abs = grad.abs().mean() + eps
                info_grad = grad / grad_mean_abs

                # Fisher diagonal approximation (natural gradient style)
                fisher_diag = grad.abs() + eps
                info_grad = info_grad / (fisher_diag ** fisher_scale)

                # Protección adaptativa
                info_grad = torch.clamp(info_grad, -5.0, 5.0)

                # Momentum
                state = self.state[p]
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data)
                momentum = state['momentum']
                momentum.mul_(beta).add_(info_grad, alpha=1 - beta)

                # Actualización final
                p.data.add_(momentum, alpha=-lr)

        return loss
