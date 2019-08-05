import pandas as pd
import numpy as np

import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
from keras import backend as K

training = pd.read_csv("../input/train.csv")
testing = pd.read_csv("../input/test.csv")

y_train = training["label"].values
X_train = training.drop("label", axis=1).values
X_test = testing.values

# Reshape image, Standardize, One-hot labels
rows = 28
columns = 28
input_shape = (rows, columns, 1) # Depth of 1
classes = len(set(y_train))

X_train = X_train.reshape(X_train.shape[0], rows, columns, 1).astype("float32")/255
X_test = X_test.reshape(X_test.shape[0], rows, columns, 1).astype("float32")/255

y_train = keras.utils.to_categorical(y_train, classes)

# Model Parameters
BATCH_SIZE = 128
EPOCHS = 30
VERBOSE = 1

# Model Training
model = Sequential()

model.add(Conv2D(16, (3, 3), padding="same", activation="relu", input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(classes, activation="softmax"))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=["accuracy"])

model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE)

# Predict

predictions = model.predict_classes(X_test, verbose=VERBOSE)

predictions = np.column_stack((np.arange(1, 28001), predictions))
np.savetxt("predict.csv", predictions, fmt="%i", delimiter=",", header="ImageId,Label", comments="")
