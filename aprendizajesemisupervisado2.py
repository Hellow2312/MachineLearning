import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Generar un conjunto de datos sintético
X, y = make_classification(n_samples=300, n_features=20, n_informative=15, n_clusters_per_class=1, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Supongamos que solo tenemos etiquetas para una pequeña fracción de los datos de entrenamiento
num_labeled = 50  # Incrementar el número de ejemplos etiquetados
y_train_labeled = np.copy(y_train)
y_train_labeled[num_labeled:] = -1  # Etiquetamos los demás datos como -1 (no etiquetados)

# Crear un modelo de LabelSpreading
model = LabelSpreading(kernel='knn', n_neighbors=5, alpha=0.2)

# Entrenar el modelo con los datos parcialmente etiquetados
model.fit(X_train, y_train_labeled)

# Hacer predicciones en los datos de prueba
y_pred = model.predict(X_test)

# Evaluar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo con aprendizaje semi-supervisado: {accuracy:.2f}')
