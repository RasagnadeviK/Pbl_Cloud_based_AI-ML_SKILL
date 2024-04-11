import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
X = np.array([-1.0,0.0,1.0,2.0,3.0,4.0], dtype = float)
y = np.array([0.5,6.0,7.0,8.0,9.0,10.0], dtype = float)
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")
def linear_regression(x):
    return W * x + b
def mean_squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

optimizer = tf.optimizers.SGD(learning_rate=0.01)

def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = linear_regression(x)
        loss = mean_squared_error(y, predictions)
    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))
    return loss


epochs = 100
for epoch in range(epochs):
    loss = train_step(X, y)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.numpy()}")
new_X = np.array([11, 12, 13])  # New data for prediction
predictions = linear_regression(new_X)
print("Predictions:", predictions.numpy())
plt.scatter(X, y)
plt.plot(X, linear_regression(X), color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression using TensorFlow')
plt.show()
predictions.save("Linear_Regression_2.h5")