import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Función para generar datos simulados para cada cliente
def generate_client_data(n_samples, n_features, n_informative, n_redundant, n_classes):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                               n_informative=n_informative, n_redundant=n_redundant, 
                               n_classes=n_classes, random_state=42)
    return X, y

# Función sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

# Modelo de regresión logística
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = sigmoid(linear_model)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]

# Simulación de aprendizaje federado
def federated_learning(clients_data, X_test, y_test, num_rounds=20):
    num_clients = len(clients_data)
    client_models = [LogisticRegression(learning_rate=0.001, num_iterations=1000) for _ in range(num_clients)]
    global_model = LogisticRegression()
    
    for round in range(num_rounds):
        print(f"Round {round + 1}")
        
        # Entrenamiento local en cada cliente
        for i, (X, y) in enumerate(clients_data):
            client_models[i].fit(X, y)
        
        # Agregación de modelos (mediana de pesos)
        global_weights = np.median([model.weights for model in client_models], axis=0)
        global_bias = np.median([model.bias for model in client_models])
        
        # Actualización del modelo global
        global_model.weights = global_weights
        global_model.bias = global_bias
        
        # Distribución del modelo global a los clientes
        for model in client_models:
            model.weights = global_model.weights
            model.bias = global_model.bias
        
        # Evaluar el modelo global después de cada ronda
        predictions = global_model.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        print(f"Accuracy after round {round + 1}: {accuracy:.4f}")
    
    return global_model

# Generar datos para tres clientes
client1_data = generate_client_data(1000, 20, 10, 5, 2)
client2_data = generate_client_data(1000, 20, 10, 5, 2)
client3_data = generate_client_data(1000, 20, 10, 5, 2)

# Generar datos de prueba
X_test, y_test = generate_client_data(500, 20, 10, 5, 2)

# Escalar los datos
scaler = StandardScaler()
client1_data = (scaler.fit_transform(client1_data[0]), client1_data[1])
client2_data = (scaler.fit_transform(client2_data[0]), client2_data[1])
client3_data = (scaler.fit_transform(client3_data[0]), client3_data[1])
X_test = scaler.fit_transform(X_test)

# Realizar aprendizaje federado
federated_model = federated_learning([client1_data, client2_data, client3_data], X_test, y_test)

# Evaluar el modelo final
final_predictions = federated_model.predict(X_test)
final_accuracy = np.mean(final_predictions == y_test)
print(f"\nAccuracy final del modelo federado: {final_accuracy:.4f}")