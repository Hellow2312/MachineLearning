import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def select_diverse_uncertain_samples(model, X_pool, n_samples=10):
    probs = model.predict_proba(X_pool)
    uncertainty = 1 - np.max(probs, axis=1)
    
    # Seleccionar el doble de muestras inciertas
    uncertain_indices = np.argsort(uncertainty)[-n_samples*2:]
    uncertain_samples = X_pool[uncertain_indices]
    
    # Calcular la diversidad (usando distancia euclidiana como ejemplo)
    distances = np.sum((uncertain_samples[:, np.newaxis, :] - uncertain_samples[np.newaxis, :, :]) ** 2, axis=-1)
    diversity_scores = np.sum(distances, axis=1)
    
    # Seleccionar las muestras más diversas entre las inciertas
    diverse_indices = np.argsort(diversity_scores)[-n_samples:]
    return uncertain_indices[diverse_indices]

# Generar un conjunto de datos de juguete
X, y = make_classification(n_samples=2000, n_features=20, n_informative=15, n_clusters_per_class=2, random_state=42)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir los datos
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.8, random_state=42)
X_pool, X_val, y_pool, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)
initial_accuracy = accuracy_score(y_val, model.predict(X_val))
print(f"Initial accuracy on validation set: {initial_accuracy:.2f}")

n_iterations = 20
n_samples_per_round = 10

for i in range(n_iterations):
    uncertain_samples = select_diverse_uncertain_samples(model, X_pool, n_samples=n_samples_per_round)
    
    X_train = np.vstack((X_train, X_pool[uncertain_samples]))
    y_train = np.concatenate((y_train, y_pool[uncertain_samples]))
    
    X_pool = np.delete(X_pool, uncertain_samples, axis=0)
    y_pool = np.delete(y_pool, uncertain_samples, axis=0)
    
    model.fit(X_train, y_train)
    val_accuracy = accuracy_score(y_val, model.predict(X_val))
    print(f"Iteration {i+1}, validation accuracy: {val_accuracy:.2f}")

X_test, y_test = make_classification(n_samples=500, n_features=20, n_informative=15, n_clusters_per_class=2, random_state=43)
X_test = scaler.transform(X_test)
final_accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Final accuracy on test set: {final_accuracy:.2f}")