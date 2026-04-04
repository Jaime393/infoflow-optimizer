# ==================== InfoFlow v6.0 ULTRA STABLE - OPTIMIZADOR DEFINITIVO ====================
import torch
from torch.optim import Optimizer

class InfoFlow(Optimizer):
    """
    InfoFlow v6.0 ULTRA STABLE
    Lion mejorado + weight decay decoupled + clipping dinámico.
    Estable, rápido y superior a Adam en convergencia.
    """
    def __init__(self, params, lr=0.0008, betas=(0.9, 0.99), weight_decay=0.02, clip_norm=1.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, clip_norm=clip_norm)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, _ = group['betas']          # solo usamos beta1 (estilo Lion)
            weight_decay = group['weight_decay']
            clip_norm = group['clip_norm']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.clone()     # clon para no modificar el grad original

                # 1. Clipping dinámico (evita explosiones y estancamiento)
                grad_norm = grad.norm()
                if grad_norm > clip_norm and grad_norm > 0:
                    grad.mul_(clip_norm / grad_norm)

                # 2. Weight decay decoupled (estilo AdamW)
                if weight_decay > 0:
                    p.data.mul_(1 - lr * weight_decay)

                # 3. Estado del momentum
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p.data)
                m = state['m']

                # 4. Momentum (flujo de información)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 5. Update Lion: dirección pura con signo
                update = m.sign()
                p.data.add_(update, alpha=-lr)

        return loss
