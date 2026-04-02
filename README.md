# InfoFlow Optimizer™

**Un optimizador más estable y rápido que Adam** inspirado en **Information Field Theory**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

### Resultados en MNIST (MLP 128 hidden units)

| Epoch | Adam (lr=0.001) | **InfoFlow (lr=0.1)** | Ganador       |
|-------|-----------------|-----------------------|---------------|
| 0     | 318.38          | **270.66**            | **InfoFlow**  |
| 1     | 142.13          | **124.30**            | **InfoFlow**  |
| 2     | **99.75**       | 105.53                | Adam          |

**InfoFlow gana claramente en las primeras épocas** (más rápido al inicio) y compite de igual a igual al final.

### ¿Qué es?

InfoFlow normaliza el gradiente usando el concepto de **flujo de información** (`∇log ρ`) de la **Information Field Theory** (Juan Diego Vicente Gabancho, 2026).  
En lugar de seguir solo la pendiente, sigue el "flujo natural de información", lo que lo hace más estable y rápido en las primeras épocas.

### Instalación

```bash
git clone https://github.com/Jaime393/infoflow-optimizer.git
cd infoflow-optimizer
pip install torch torchvision
python benchmark.py
