# -*- coding: utf-8 -*-

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Dimensiones de la entrada para la red generadora
random_dim = 100

# Cargar el conjunto de datos MNIST
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5  # Normalizar a [-1, 1]

# Crear la red generadora
generator = Sequential([
    Dense(256, input_dim=random_dim),
    LeakyReLU(0.2),
    Dense(512),
    LeakyReLU(0.2),
    Dense(1024),
    LeakyReLU(0.2),
    Dense(784, activation='tanh'),
    Reshape((28, 28))  # Regresar a la forma (28, 28)
])

# Crear la red discriminadora
discriminator = Sequential([
    Flatten(input_shape=(28, 28)),  # Aplanar la entrada
    Dense(1024),
    LeakyReLU(0.2),
    Dense(512),
    LeakyReLU(0.2),
    Dense(256),
    LeakyReLU(0.2),
    Dense(1, activation='sigmoid')
])

# Compilar la red discriminadora
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Crear la GAN combinando generador y discriminador
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(random_dim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

# Entrenar la GAN
def train_gan(epochs=10000, batch_size=128):
    for epoch in range(epochs):
        # Seleccionar imágenes reales aleatoriamente
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_images = x_train[idx]
        real_labels = np.ones((batch_size, 1))

        # Generar imágenes falsas y sus etiquetas
        noise = np.random.normal(0, 1, (batch_size, random_dim))
        generated_images = generator.predict(noise)
        fake_labels = np.zeros((batch_size, 1))
        valid_labels = np.ones((batch_size, 1))

        # Entrenar el discriminador
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Entrenar el generador (a través de la GAN)
        g_loss = gan.train_on_batch(noise, valid_labels)

        # Mostrar progreso
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, D loss: {d_loss[0]}, G loss: {g_loss}")
            plot_generated_images(epoch)

# Función para visualizar las imágenes generadas
def plot_generated_images(epoch, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, (examples, random_dim))
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Entrenar la GAN
train_gan(epochs=10000, batch_size=128)