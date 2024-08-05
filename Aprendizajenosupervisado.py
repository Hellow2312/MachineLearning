import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generar datos ficticios
n_samples = 300
n_features = 2
n_clusters = 4
random_state = 42

# Asegúrate de que esta línea esté presente y se ejecute
X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)

# Crear el modelo K-Means con el número de clusters deseado
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

# Entrenar el modelo (ajustar el modelo a los datos)
kmeans.fit(X)

# Predecir a qué cluster pertenecen los puntos
y_kmeans = kmeans.predict(X)

# Visualizar los clusters
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', marker='o')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroides')
plt.title('Clusters formados por K-Means')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()

# Guardar la figura
plt.savefig('kmeans_clusters.png')
print("Gráfica guardada como 'kmeans_clusters.png'")

# Mostrar la gráfica
plt.show()

print("Ejecución completada.")