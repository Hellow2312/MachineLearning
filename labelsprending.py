import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score

# Generar un conjunto de datos sintético
X, y = make_classification(n_samples=300, n_features=20, n_informative=15, n_clusters_per_class=1, random_state=42)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Solo etiquetamos una pequeña fracción de los datos de entrenamiento
num_labeled = 30
y_train_labeled = np.copy(y_train)
y_train_labeled[num_labeled:] = -1

# Crear y entrenar el modelo de LabelSpreading
model = LabelSpreading(kernel='knn', n_neighbors=5, alpha=0.2)
model.fit(X_train, y_train_labeled)

# Evaluar el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'LabelSpreading Accuracy: {accuracy:.2f}')
