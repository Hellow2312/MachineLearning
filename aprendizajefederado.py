import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Función para generar datos simulados para cada cliente
def generate_client_data(n_samples, n_features, n_informative, n_redundant, n_classes):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                               n_informative=n_informative, n_redundant=n_redundant, 
                               n_classes=n_classes, random_state=42)
    return X, y

# Función sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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
def federated_learning(clients_data, num_rounds=5):
    num_clients = len(clients_data)
    client_models = [LogisticRegression() for _ in range(num_clients)]
    global_model = LogisticRegression()
    
    for round in range(num_rounds):
        print(f"Round {round + 1}")
        
        # Entrenamiento local en cada cliente
        for i, (X, y) in enumerate(clients_data):
            client_models[i].fit(X, y)
        
        # Agregación de modelos (promedio simple de pesos)
        global_weights = np.mean([model.weights for model in client_models], axis=0)
        global_bias = np.mean([model.bias for model in client_models])
        
        # Actualización del modelo global
        global_model.weights = global_weights
        global_model.bias = global_bias
        
        # Distribución del modelo global a los clientes
        for model in client_models:
            model.weights = global_model.weights
            model.bias = global_model.bias
    
    return global_model

# Generar datos para dos clientes
client1_data = generate_client_data(1000, 20, 10, 5, 2)
client2_data = generate_client_data(1000, 20, 10, 5, 2)

# Realizar aprendizaje federado
federated_model = federated_learning([client1_data, client2_data])

# Generar datos de prueba
X_test, y_test = generate_client_data(500, 20, 10, 5, 2)

# Evaluar el modelo
predictions = federated_model.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy del modelo federado: {accuracy:.4f}")