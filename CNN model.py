import numpy as np
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
import matplotlib as mpl
import matplotlib.pyplot as plt 


#load, reshape, normalize and split the dataset

fashion = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion.load_data()

x_train_full = x_train_full.reshape((60000, 28, 28, 1))  #Add another dimenstion for color channel
x_test = x_test.reshape((10000, 28, 28, 1))

x_train_n = x_train_full / 255.
x_test_n = x_test / 255.

x_train, x_valid = x_train_n[0:50000], x_train_n[50000:]
y_train, y_valid = y_train_full[:50000], y_train_full[50000:]
x_test = x_test_n

class_names = ["Shirt/Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


#Building CNN model

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), strides=1, padding='valid',
                              activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
model.summary()


#Training CNN model
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

model_history = model.fit(x_train, y_train, epochs=70,batch_size= 64, validation_data=(x_valid, y_valid))



#CNN Training Phase Visualization

model_history.history.pop("val_loss")
model_history.history.pop("val_accuracy")
pd.DataFrame(model_history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.gca().set_xlim(0)
plt.show()


#CNN model evaluation

model.evaluate(x_test, y_test)

