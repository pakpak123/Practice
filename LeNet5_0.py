from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

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