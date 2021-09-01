import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(Xtrain,Ytrain),(Xtest,Ytest) =tf.keras.datasets.mnits.load_data()
#print(Xtrain.shape)
plt.imshow(Xtrain[0],cmap='gray')
plt.show()