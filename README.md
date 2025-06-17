# Proyecto BreastMNIST

Este repositorio contiene un ejemplo inicial para clasificar las im\u00e1genes del dataset **BreastMNIST** utilizando redes neuronales convolucionales y una arquitectura de Mezcla de Expertos (MoE).

## Estructura
- `main.ipynb`: Notebook inicial con la carga del dataset y análisis exploratorio.
- `proyecto_breastmnist.ipynb`: Ejemplo extendido con la definición de modelos.
- `models/cnn.py`: Definición de la clase `BasicCNN`.
- `models/moe.py`: Definición de la clase `MixtureOfExperts`.
- `utils/metrics.py`: Funciones auxiliares para calcular métricas.

## Uso r\u00e1pido
1. Clonar este repositorio en Google Colab o tu entorno local.
2. Abrir `main.ipynb` y ejecutar las celdas. El notebook instalar\u00e1 `medmnist` y descargar\u00e1 autom\u00e1ticamente el dataset.

Este punto de partida est\u00e1 listo para ampliar el entrenamiento de modelos y experimentar con t\u00e9cnicas de balanceo de clases y data augmentation.
