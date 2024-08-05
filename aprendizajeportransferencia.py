import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Cargar los datos CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Convertir las etiquetas a categorías one-hot
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Preprocesar los datos de entrada
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Cargar el modelo ResNet50 preentrenado, excluyendo la capa superior
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Añadir capas personalizadas al final para la nueva tarea
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# Crear el modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# Congelar las capas del modelo base (ResNet50) para evitar que se actualicen durante el entrenamiento
for layer in base_model.layers:
    layer.trainable = False

# Compilar el modelo
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)

# Evaluar el modelo en los datos de prueba
score = model.evaluate(x_test, y_test, verbose=0)
print(f'Precisión del modelo en los datos de prueba: {score[1]:.2f}')
