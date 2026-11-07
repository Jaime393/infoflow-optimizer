"""Punto de entrada fino para el benchmark existente.

El README del repo indica ejecutar `python benchmark.py`, pero en el
commit fa659a5 no existe ningun archivo con ese nombre. Si existe
`benchmark_cifar.py`, que corre el benchmark CIFAR-10 con Adam
(lr=0.001) e InfoFlow (valores por defecto) durante 5 epocas cada uno.

Este archivo no anade ningun benchmark nuevo: ejecuta tal cual
`benchmark_cifar.py` como programa principal, desde el directorio
del repo.
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

if HERE not in sys.path:
    sys.path.insert(0, HERE)

if __name__ == "__main__":
    runpy.run_path(
        os.path.join(HERE, "benchmark_cifar.py"),
        run_name="__main__",
    )
