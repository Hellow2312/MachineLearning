import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generar datos más realistas
np.random.seed(42)
tamaño = np.random.randint(1000, 5000, 100)
precio = 100 + 0.1 * tamaño + np.random.normal(0, 50, 100)

# Dividir los datos en conjuntos de entrenamiento y prueba
tamaño_entrenamiento, tamaño_prueba, precio_entrenamiento, precio_prueba = train_test_split(tamaño, precio, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(tamaño_entrenamiento.reshape(-1, 1), precio_entrenamiento)

# Hacer predicciones con los datos de prueba
precio_predicción = modelo.predict(tamaño_prueba.reshape(-1, 1))

# Evaluar el modelo
mse = mean_squared_error(precio_prueba, precio_predicción)
r2 = r2_score(precio_prueba, precio_predicción)

print(f"Error cuadrático medio: {mse:.2f}")
print(f"Coeficiente de determinación (R²): {r2:.2f}")

# Visualizar los resultados
plt.figure(figsize=(10, 6))
plt.scatter(tamaño, precio, color='blue', alpha=0.5, label='Datos reales')
plt.plot(tamaño_prueba, precio_predicción, color='red', label='Predicción del modelo')
plt.xlabel('Tamaño de la vivienda (pies cuadrados)')
plt.ylabel('Precio (en miles de dólares)')
plt.title('Regresión Lineal: Tamaño de la vivienda vs Precio')
plt.legend()
plt.savefig('regresion_lineal_viviendas.png')
plt.show()

print("\nEjecución del modelo completada.")