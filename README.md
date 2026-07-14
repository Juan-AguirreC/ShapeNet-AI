# GeoVision AI - Clasificador de Figuras Geométricas

## Descripción del proyecto

GeoVision AI es un sistema de visión artificial capaz de identificar figuras geométricas mediante una cámara en tiempo real.

El modelo clasifica tres tipos de figuras:
- Circle (Círculo)
- Square (Cuadrado)
- Triangle (Triángulo)

Utiliza OpenCV para procesamiento de imágenes y TensorFlow/Keras para la clasificación mediante inteligencia artificial.

## Dataset utilizado

Para el entrenamiento del modelo se utilizó el dataset:

PyTorch Geometric Shapes Dataset - Kaggle

https://www.kaggle.com/datasets/smeschke/pytorch-geometric-shapes

Clases utilizadas:
- Circle
- Square
- Triangle

## Procesamiento de imágenes

Las imágenes fueron procesadas mediante:
1. Conversión a escala de grises.
2. Redimensionamiento.
3. Binarización mediante Otsu.
4. Normalización entre 0 y 1.
5. Flatten para convertir la imagen en vector.

Tamaño utilizado:
64x64 píxeles.

## Arquitectura del modelo

Red neuronal utilizada:

Input (4096)
|
Dense 128 - ReLU
|
Dense 64 - ReLU
|
Dense 3 - Softmax

## Entrenamiento

División:
- 80% entrenamiento
- 20% prueba

Parámetros:
- Optimizador: Adam
- Loss: Sparse Categorical Crossentropy
- Batch size: 32
- Epochs: 17

## Modelo generado

El modelo entrenado se guarda como:

modelo_figuras.h5

## Detección en tiempo real

El sistema captura imágenes desde una cámara y realiza:

Cámara -> Escala de grises -> ROI -> Redimensionamiento -> Binarización -> Modelo IA -> Figura detectada

## Tecnologías utilizadas

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- PIL
- Scikit-learn