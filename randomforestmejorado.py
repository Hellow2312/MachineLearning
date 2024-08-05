from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Cargar el conjunto de datos Iris
iris = load_iris()
X, y = iris.data, iris.target

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Realizar validación cruzada
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Puntuaciones de validación cruzada: {cv_scores}")
print(f"Precisión media de validación cruzada: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo Random Forest en el conjunto de prueba: {accuracy:.2f}')

# Imprimir informe de clasificación
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Imprimir matriz de confusión
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# Visualizar importancia de características
feature_importance = model.feature_importances_
feature_names = iris.feature_names
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(pos, feature_importance[sorted_idx], align='center')
ax.set_yticks(pos)
ax.set_yticklabels(np.array(feature_names)[sorted_idx])
ax.set_xlabel('Importancia de Características')
ax.set_title('Importancia de Características en Random Forest')
plt.tight_layout()
plt.show()