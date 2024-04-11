from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

n_samples = 1000
x, y = make_circles(n_samples, noise=0.03, random_state=42)
circles = pd.DataFrame({"X0":x[:,0],"X1":x[:,1], "label":y})
circles.head()
circles.label.value_counts()

plt.scatter(x[:,0], x[:,1], c=y, cmap=plt.cm.RdYlBu)
plt.show()


print('MODEL1')
tf.random.set_seed(42)
model1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

model1.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.SGD(),metrics=['accuracy'])
model1.fit(x, y, epochs=5)
model1.evaluate(x,y)

print('MODEL2')
tf.random.set_seed(42)
model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(1)
])

model2.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.SGD(),metrics=['accuracy'])
model2.fit(x, y, epochs=5)
model2.evaluate(x,y)


print('MODEL3')
tf.random.set_seed(42)
model3 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(1)
])

model3.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.SGD(),metrics=['accuracy'])
model3.fit(x, y, epochs=25)
model3.evaluate(x,y)
model3.save('neuralClassification.h5')