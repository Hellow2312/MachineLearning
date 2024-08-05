# Repositorio de Modelos de Machine Learning

Este repositorio contiene una colección de scripts de Python que implementan varios modelos y técnicas de aprendizaje automático. Cada script está diseñado para demostrar un concepto específico y puede ser utilizado como punto de partida para proyectos más avanzados o con fines educativos.

## Contenido

1. **Aprendizaje Federado** (`federated_learning.py`)
   - Implementa un modelo de regresión logística utilizando aprendizaje federado simulado.
   - Demuestra cómo entrenar un modelo en datos distribuidos sin centralizar información sensible.

2. **Aprendizaje Activo** (`active_learning.py`)
   - Utiliza SVM para clasificación con selección iterativa de muestras.
   - Muestra cómo mejorar el rendimiento del modelo seleccionando muestras informativas para etiquetar.

3. **Aprendizaje Multitarea** (`multitask_learning.py`)
   - Implementa una red neuronal convolucional para aprendizaje multitarea usando TensorFlow.
   - Clasifica imágenes de CIFAR-10 mientras realiza estimaciones simultáneas de género y edad.

4. **Random Forest Mejorado** (`random_forest_improved.py`)
   - Implementación avanzada de Random Forest utilizando el conjunto de datos Iris.
   - Incluye validación cruzada, métricas detalladas de rendimiento y visualización de la importancia de características.

5. **GAN (Red Generativa Adversaria)** (`gan.py`)
   - Implementa una GAN simple para generar dígitos similares a MNIST.
   - Demuestra el entrenamiento adversario entre redes generadoras y discriminadoras.

## Requisitos

- Python 3.7+
- TensorFlow 2.x
- scikit-learn
- NumPy
- Matplotlib

Puedes instalar las dependencias necesarias con:

```
pip install -r requirements.txt
```

## Uso

Cada script puede ejecutarse de forma independiente. Por ejemplo:

```
python federated_learning.py
```

Asegúrate de leer los comentarios dentro de cada script para entender los parámetros y el funcionamiento específico.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir cambios importantes antes de hacer un pull request.

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.
