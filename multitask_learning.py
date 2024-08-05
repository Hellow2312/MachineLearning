import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.datasets import cifar10
import numpy as np

# Configurar TensorFlow para usar solo CPU (opcional, elimina si quieres usar GPU)
tf.config.set_visible_devices([], 'GPU')

# Cargar y preprocesar los datos
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Simular etiquetas adicionales
y_gender_train = np.random.randint(2, size=(y_train.shape[0], 1))
y_age_train = np.random.randint(10, size=(y_train.shape[0], 1))
y_gender_test = np.random.randint(2, size=(y_test.shape[0], 1))
y_age_test = np.random.randint(10, size=(y_test.shape[0], 1))

# Construir el modelo
input_layer = Input(shape=(32, 32, 3))
x = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
gender_output = layers.Dense(1, activation='sigmoid', name='gender_output')(x)
age_output = layers.Dense(1, activation='linear', name='age_output')(x)
model = models.Model(inputs=input_layer, outputs=[gender_output, age_output])

# Compilar el modelo
model.compile(optimizer='adam',
              loss={'gender_output': 'binary_crossentropy', 'age_output': 'mse'},
              metrics={'gender_output': 'accuracy', 'age_output': 'mae'})

# Clase de callback personalizada para mostrar el progreso
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoca {epoch+1}/10 completada")
        print(f"Perdida: {logs['loss']:.4f}")
        print(f"Precision de genero: {logs['gender_output_accuracy']:.4f}")
        print(f"Error absoluto medio de edad: {logs['age_output_mae']:.4f}")
        print("--------------------")

# Entrenar el modelo
history = model.fit(
    x_train, 
    {'gender_output': y_gender_train, 'age_output': y_age_train},
    epochs=10, 
    batch_size=64, 
    validation_split=0.2, 
    verbose=0,
    callbacks=[CustomCallback()]
)

# Evaluar el modelo
print("Evaluando el modelo...")
evaluation = model.evaluate(x_test, {'gender_output': y_gender_test, 'age_output': y_age_test}, verbose=0)
print("Evaluacion completada")
print(f"Perdida total: {evaluation[0]:.4f}")
print(f"Perdida de genero: {evaluation[1]:.4f}")
print(f"Perdida de edad: {evaluation[2]:.4f}")
print(f"Precision de genero: {evaluation[3]:.4f}")
print(f"Error absoluto medio de edad: {evaluation[4]:.4f}")