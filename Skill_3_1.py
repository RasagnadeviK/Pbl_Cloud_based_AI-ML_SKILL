import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

(x_train,y_train,),(x_test,y_test)=keras.datasets.mnist.load_data()
print(len(x_train))
print(len(x_test))
print(x_train[1].shape)
plt.figure()
plt.imshow(x_train[1])  # Displaying the second image in grayscale
plt.colorbar()
plt.grid(False)
plt.show()

x_train=x_train/255
x_test=x_test/255

x_train_flatter=x_train.reshape(len(x_train),28*28)
x_test_flatter=x_test.reshape(len(x_test),28*28)

model=keras.Sequential([keras.layers.Dense(10,input_shape=(784,),activation='sigmoid')])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train_flatter,y_train,epochs=5)
result=model.evaluate(x_test_flatter,y_test)
print("Loss function and Accuracy",result)
model.save("mnist_model.h5")