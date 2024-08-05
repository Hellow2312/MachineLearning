import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Generar un conjunto de datos de juguete
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_clusters_per_class=2, random_state=42)

# Dividir los datos en conjuntos etiquetados y no etiquetados
X_train, X_pool, y_train, y_pool = train_test_split(X, y, test_size=0.95, random_state=42)

# Inicializar el clasificador SVM
model = SVC(probability=True)

# Entrenar el modelo inicialmente con un pequeño conjunto de datos etiquetados
model.fit(X_train, y_train)
initial_accuracy = accuracy_score(y_train, model.predict(X_train))
print(f"Initial accuracy with small labeled set: {initial_accuracy:.2f}")

# Función para seleccionar muestras más inciertas
def select_uncertain_samples(model, X_pool, n_samples=10):
    probs = model.predict_proba(X_pool)
    uncertainty = 1 - np.max(probs, axis=1)  # Cuanto más bajo es el valor, más incierta es la predicción
    uncertain_samples = np.argsort(uncertainty)[-n_samples:]  # Seleccionar las muestras más inciertas
    return uncertain_samples

# Iterar en rondas de aprendizaje activo
n_iterations = 5
n_samples_per_round = 10

for i in range(n_iterations):
    # Seleccionar muestras inciertas para etiquetar
    uncertain_samples = select_uncertain_samples(model, X_pool, n_samples=n_samples_per_round)
    
    # "Etiquetar" estas muestras y agregarlas al conjunto de entrenamiento
    X_train = np.vstack((X_train, X_pool[uncertain_samples]))
    y_train = np.concatenate((y_train, y_pool[uncertain_samples]))

    # Eliminar las muestras etiquetadas del conjunto de pool
    X_pool = np.delete(X_pool, uncertain_samples, axis=0)
    y_pool = np.delete(y_pool, uncertain_samples, axis=0)

    # Reentrenar el modelo con el conjunto de datos expandido
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_train, model.predict(X_train))
    print(f"Iteration {i+1}, accuracy: {accuracy:.2f}")

# Evaluar el modelo final en un conjunto de datos de prueba separado
X_test, y_test = make_classification(n_samples=200, n_features=20, n_informative=15, n_clusters_per_class=2, random_state=42)
final_accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Final accuracy on test set: {final_accuracy:.2f}")
