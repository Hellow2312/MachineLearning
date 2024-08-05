import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el conjunto de datos de vinos
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
column_names = ['Class', 'Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of Ash', 'Magnesium', 
                'Total Phenols', 'Flavanoids', 'Nonflavanoid Phenols', 'Proanthocyanins', 
                'Color Intensity', 'Hue', 'OD280/OD315 of Diluted Wines', 'Proline']
data = pd.read_csv(url, names=column_names)

# Separar características y etiquetas
X = data.drop('Class', axis=1)
y = data['Class']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear pipelines para diferentes modelos
pipelines = {
    'rf': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('rf', RandomForestClassifier(random_state=42))
    ]),
    'svm': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('svm', SVC(random_state=42))
    ]),
    'knn': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('knn', KNeighborsClassifier())
    ])
}

# Definir parámetros para la búsqueda en cuadrícula
param_grids = {
    'rf': {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [None, 5, 10]
    },
    'svm': {
        'svm__C': [0.1, 1, 10],
        'svm__kernel': ['rbf', 'poly']
    },
    'knn': {
        'knn__n_neighbors': [3, 5, 7],
        'knn__weights': ['uniform', 'distance']
    }
}

# Realizar búsqueda en cuadrícula para cada modelo
best_models = {}
for name, pipeline in pipelines.items():
    grid_search = GridSearchCV(pipeline, param_grids[name], cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_

# Evaluar y comparar modelos
results = {}
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    results[name] = {
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

# Imprimir resultados
for name, result in results.items():
    print(f"\nResultados para {name}:")
    print(result['classification_report'])

# Visualizar matrices de confusión
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for i, (name, result) in enumerate(results.items()):
    sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', ax=axes[i])
    axes[i].set_title(f'Matriz de Confusión - {name}')
    axes[i].set_xlabel('Predicción')
    axes[i].set_ylabel('Valor Real')

plt.tight_layout()
plt.show()

# Visualizar importancia de características para Random Forest
rf_model = best_models['rf']
feature_importance = rf_model.named_steps['rf'].feature_importances_
pca = rf_model.named_steps['pca']
original_feature_importance = np.dot(pca.components_.T, feature_importance)
feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': original_feature_importance})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Importancia de Características - Random Forest')
plt.show()
# Al final del script
print("\nEjecución del modelo completada.")