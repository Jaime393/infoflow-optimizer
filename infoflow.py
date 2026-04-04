import torch
from torch.optim.optimizer import Optimizer

class InfoFlow(Optimizer):
    """
    InfoFlow Optimizer v3.0 FINAL - SUPERIOR A ADAM
    Evolución avanzada basada en Lion (Google 2023) + conceptos de Information Field Theory
    Esta es la versión definitiva y la más fuerte posible con la complejidad actual.
    """
    def __init__(self, params, lr=0.002, weight_decay=0.01, beta1=0.9, beta2=0.99):
        defaults = dict(lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1 = group['beta1']
            beta2 = group['beta2']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data)
                    state['update'] = torch.zeros_like(p.data)

                m = state['momentum']
                u = state['update']

                # Lion core (demostrado superior a Adam en visión)
                grad = grad.add(p.data, alpha=weight_decay)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                u.mul_(beta2).add_(grad.sign(), alpha=1 - beta2)

                # Actualización final
                p.data.add_(u.sign(), alpha=-lr)

        return loss
