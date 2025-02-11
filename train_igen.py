import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras.models import Sequential
from keras.layers import Dense, Reshape, BatchNormalization, LeakyReLU, UpSampling2D, Conv2D, ReLU, Flatten, Dropout
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np

def normalize_image(image, mean=0.5, std=0.5):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = (image - mean) / std
    return image

def preprocess_data(image, label):
    image = normalize_image(image)
    return image, label

dataset = tfds.load('FashionMNIST', split='train', as_supervised=True, batch_size=32)

ds1 = dataset.map(preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Generator
latent_dim = 100
generator = Sequential()
    
generator.add(Dense(128 * 7 * 7, input_dim=latent_dim))
generator.add(ReLU())
generator.add(Reshape((7, 7, 128)))
    
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
generator.add(BatchNormalization(momentum=0.78))
generator.add(ReLU())
    
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
generator.add(BatchNormalization(momentum=0.78))
generator.add(ReLU())
    
generator.add(Conv2D(1, kernel_size=(3, 3), activation='tanh', padding='same'))

generator.summary()

# Discrimator
discriminator = Sequential()
    
discriminator.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', input_shape=(28, 28, 1)))
discriminator.add(LeakyReLU(alpha=0.2))
    
discriminator.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same'))
discriminator.add(BatchNormalization(momentum=0.82))
discriminator.add(LeakyReLU(alpha=0.2))
    
discriminator.add(Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same'))
discriminator.add(BatchNormalization(momentum=0.82))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.25))

discriminator.add(Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same'))
discriminator.add(BatchNormalization(momentum=0.8))
discriminator.add(LeakyReLU(alpha=0.25))
discriminator.add(Dropout(0.25))
    
discriminator.add(Flatten())    
discriminator.add(Dense(1, activation='sigmoid'))

discriminator.summary()

# Training
epochs = 10
batch_size = 32

optimizer_G = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
optimizer_D = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

cross_entropy = tf.keras.losses.BinaryCrossentropy()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return (real_loss + fake_loss) / 2

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

real_labels = np.ones((batch_size, 1))
fake_labels = np.zeros((batch_size, 1))

for epoch in range(epochs):
  i=0
  for batch in ds1:
    z = np.random.normal(0,1,(batch_size, latent_dim))
    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
      fake_images = generator(z, training=True)

      real = discriminator(batch[0], training=True)
      fake = discriminator(fake_images, training=True)

      d_loss = discriminator_loss(real, fake)
      g_loss = generator_loss(fake)

    d_grad = d_tape.gradient(d_loss, discriminator.trainable_variables)
    g_grad = g_tape.gradient(g_loss, generator.trainable_variables)

    optimizer_D.apply_gradients(zip(d_grad, discriminator.trainable_variables))
    optimizer_G.apply_gradients(zip(g_grad, generator.trainable_variables))

    if (i + 1) % 100 == 0:
      print(f"Epoch [{epoch+1}/{epochs}]\
      Batch [{i+1}/{len(ds1)}] DLoss: {d_loss.numpy():.4f} GLoss: {g_loss.numpy():.4f}")
      generator.save('models/igen-'+str(i)+'.keras')
    i+=1

z = tf.random.normal((9, latent_dim))
generated = generator.predict(z)
print(generated.shape)
grid = torchvision.utils.make_grid(np.transpose(torch.tensor(generated), (0, 3, 2, 1)),nrow=3, normalize=True)
print(grid.shape)

generator.save('models/igen-final.keras')
  
plt.imshow(np.transpose(grid, (2, 1, 0)))
plt.axis("off")
plt.show()
