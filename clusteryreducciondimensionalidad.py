import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

# Cargar el conjunto de datos de dígitos
digits = load_digits()
X, y = digits.data, digits.target

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA para reducción de dimensionalidad
pca = PCA(n_components=0.95)  # Conservar el 95% de la varianza
X_pca = pca.fit_transform(X_scaled)

print(f"Dimensiones originales: {X.shape[1]}")
print(f"Dimensiones después de PCA: {X_pca.shape[1]}")

# Determinar el número óptimo de clusters usando el método del codo
inertias = []
silhouette_scores = []
k_range = range(2, 15)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_pca, kmeans.labels_))

# Visualizar el método del codo y el score de silueta
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(k_range, inertias, 'bx-')
ax1.set_xlabel('k')
ax1.set_ylabel('Inertia')
ax1.set_title('Método del Codo')

ax2.plot(k_range, silhouette_scores, 'rx-')
ax2.set_xlabel('k')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score vs. k')

plt.tight_layout()
plt.show()

# Seleccionar el número óptimo de clusters (en este caso, usaremos 10 ya que sabemos que hay 10 dígitos)
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_pca)

# Aplicar t-SNE para visualización en 2D
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_pca)

# Visualizar los clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap='viridis')
plt.colorbar(scatter)
plt.title('Visualización de Clusters usando t-SNE')
plt.show()

# Visualizar algunos ejemplos de cada cluster
fig, axes = plt.subplots(n_clusters, 10, figsize=(20, 20))
fig.suptitle('Muestras de Imágenes por Cluster', fontsize=16)

for i in range(n_clusters):
    cluster_examples = X[cluster_labels == i]
    for j in range(10):
        if j < len(cluster_examples):
            axes[i, j].imshow(cluster_examples[j].reshape(8, 8), cmap='binary')
        axes[i, j].axis('off')

plt.tight_layout()
plt.show()

# Evaluar la pureza de los clusters
def cluster_purity(labels_true, labels_pred):
    contingency_matrix = np.zeros((n_clusters, 10))
    for i in range(len(labels_true)):
        contingency_matrix[labels_pred[i], labels_true[i]] += 1
    return np.sum(np.max(contingency_matrix, axis=1)) / len(labels_true)

purity = cluster_purity(y, cluster_labels)
print(f"Pureza de los clusters: {purity:.2f}")

print("Análisis completado.")