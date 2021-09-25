from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import tensorflow as tf
import numpy as np
model = keras.Sequential()

model.add(Conv2D(6, (5, 5), input_shape=(32, 32, 1), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(16, (5, 5), activation='relu'))
model.add(MaxPool2D())

model.add(Flatten())

model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
              optimizer=keras.optimizers.SGD())

(Xtrain, Ytrain), (Xtest, Ytest) = keras.datasets.mnist.load_data()
Xtrain = Xtrain[:, :, :, None] / 255.  # 28 x 28 x 1
Xtest = Xtest[:, :, :, None] / 255.   # 28 x 28 x 1
Xtrain = tf.image.resize(Xtrain, (32, 32))  # 32 x 32 x 1
Xtest = tf.image.resize(Xtest, (32, 32))  # 32 x 32 x 1

model.fit(Xtrain, Ytrain, epochs=50)

Ztest = model.predict(Xtest)

acc = np.sum(Ztest.argmax(axis=1) == Ytest) / len(Ztest) * 100
print(f'accuracy rate = {acc}%')