import tensorflow as tf
import keras
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

model = keras.models.load_model('models/igen-1199.keras')
latent_dim = 100

z = tf.random.normal((9, latent_dim))
generated = model.predict(z)

print(generated.shape)
grid = torchvision.utils.make_grid(np.transpose(torch.tensor(generated), (0, 3, 2, 1)),nrow=3, normalize=True)
print(grid.shape)

plt.imshow(np.transpose(grid, (2, 1, 0)))
plt.axis("off")
plt.show()