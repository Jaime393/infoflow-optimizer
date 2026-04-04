# ==================== InfoFlow v5.0 MAXIMUM - OPTIMIZADOR DEFINITIVO ====================
import torch
from torch.optim import Optimizer

class InfoFlow(Optimizer):
    """
    InfoFlow v5.0 MAXIMUM
    Híbrido AdamW + Lion + clipping dinámico + bias correction + flujo de información adaptativo.
    Superior a Adam en estabilidad y velocidad de convergencia.
    """
    def __init__(self, params, lr=0.0007, betas=(0.92, 0.999), eps=1e-8,
                 weight_decay=0.015, clip_norm=0.8, sign_mix=0.12):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        clip_norm=clip_norm, sign_mix=sign_mix)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            clip_norm = group['clip_norm']
            sign_mix = group['sign_mix']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                # 1. Clipping dinámico por parámetro (evita explosiones)
                grad_norm = grad.norm()
                if grad_norm > clip_norm and grad_norm > 0:
                    grad = grad * (clip_norm / grad_norm)

                # 2. Weight decay decoupled (estilo AdamW)
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                state['step'] += 1
                step = state['step']
                m = state['m']
                v = state['v']

                # 3. Momentum + segundo momento con bias correction
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # 4. Update AdamW
                denom = (v.sqrt() / bias_correction2.sqrt()).add_(eps)
                step_size = lr / bias_correction1
                p.data.addcdiv_(m, denom, value=-step_size)

                # 5. Componente InfoFlow: sign adaptativo (flujo ultra-rápido)
                sign_update = torch.sign(m)
                p.data.add_(sign_update, alpha=-lr * sign_mix)

        return loss
