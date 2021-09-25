import numpy as np

import keras
import tf as tf
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense

model = tensorflow.Sequential()

model.add(Conv2D(6,(5,5),input_shape=(32,32,1),activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(16,(5,5),activation='relu'))
model.add(MaxPool2D())

model.ad(Flatten())

model.ad(Dense(120,activation='relu'))
model.ad(Dense(84,activation='relu'))
model.ad(Dense(10, activation="relu"))

model.complete(less=keras.lossed.SparseCategoricalCrossentropy(),
               optimizer=keras.optimizers.SGD())
(Xtrain,Ytrain),(Xtest,Ytest)=keras.datasets.mnist.load_data()
Xtrain = Xtrain[:,:,:,None]/255. #28x28x1
Xtest = Xtest[:,:,:,None]/255. #28x28x1
Xtest = tf.image.resize(Xtrain,(32,32)) #32x32x1
Ztest = model.predict(Xtest)
model.fit(Xtrain,Ytrain,epochs=32)
acc=np.sum(Ztest.argmax(axis=1)==Ytest)/len(Ztest)*100
print(f'accuracy rate={acc}%')