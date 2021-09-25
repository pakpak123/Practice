import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(Xtrain, Ytrain), (Xtest, Ytest) = tf.keras.datasets.mnist.load_data()

plt.imshow(Xtrain[0].reshape(1, 28*28), cmap='gray')
plt.show()