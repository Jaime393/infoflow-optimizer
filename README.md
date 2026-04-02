# InfoFlow Optimizer™

**Un optimizador más estable y rápido que Adam** basado en normalización del flujo de información.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)

### Resultados en MNIST (MLP 128 hidden)

| Epoch | Adam (lr=0.001) | InfoFlow (lr=0.1) |
|-------|-----------------|-------------------|
| 0     | 321.74          | **275.37**        |
| 1     | 145.07          | **123.86**        |
| 2     | **99.28**       | 99.29             |

**InfoFlow gana en las primeras épocas** y termina empatado. Con solo 3 épocas ya compite de igual a igual.

### Instalación

```bash
pip install torch torchvision
git clone https://github.com/Jaime393/infoflow-optimizer.git
cd infoflow-optimizer
python benchmark.py
