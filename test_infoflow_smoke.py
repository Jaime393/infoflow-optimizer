"""Prueba minima (smoke test) de InfoFlow.

Usa solo lo que `infoflow.py` declara en el commit fa659a5:
`InfoFlow(params, lr=0.0008, betas=(0.9, 0.99), weight_decay=0.02,
clip_norm=1.0)` con un metodo `step()`.

No afirma ningun resultado de precision ni de perdida: solo que
instanciar con los valores por defecto, llamar a `step()` una vez
con un gradiente, cambia los parametros y no da error.
"""

import torch

from infoflow import InfoFlow


def test_step_changes_params():
    p = torch.nn.Parameter(torch.randn(4, 4))
    p.grad = torch.randn(4, 4)
    before = p.data.clone()

    opt = InfoFlow([p])
    opt.step()

    assert not torch.equal(before, p.data), "step() no cambio los parametros"


if __name__ == "__main__":
    test_step_changes_params()
    print("OK: InfoFlow step() cambia los parametros sin error")
