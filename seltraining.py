from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import numpy as np

# Generar un conjunto de datos sintético
X, y = make_classification(n_samples=300, n_features=20, n_informative=15, n_clusters_per_class=1, random_state=42)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Supongamos que solo tenemos etiquetas para una pequeña fracción de los datos de entrenamiento
num_labeled = 30  # Solo 30 ejemplos etiquetados
y_train_labeled = np.copy(y_train)
y_train_labeled[num_labeled:] = -1  # Etiquetamos los demás datos como -1 (no etiquetados)

# Crear el modelo base SVM
base_model = SVC(probability=True)

# Crear y entrenar el modelo de Self-Training
model = SelfTrainingClassifier(base_model)
model.fit(X_train, y_train_labeled)

# Evaluar el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Self-Training SVM Accuracy: {accuracy:.2f}')
