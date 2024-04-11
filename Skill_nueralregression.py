import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Existing data
X = np.arange(-100, 100, 4)
y = np.arange(-90, 110, 4)

# Splitting the data into training and testing sets (80% training, 20% testing)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshaping input data
X_train = X_train.reshape(-1, 1)  # Reshape to (samples, features)
X_test = X_test.reshape(-1, 1)    # Reshape to (samples, features)

# Plot original data
# Create model with sigmoid activation
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

# Compile model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

# Fit model on training data
history = model.fit(X_train, y_train, epochs=100, verbose=0)

# Predictions for training and testing data
predictions_train = model.predict(X_train)
predictions_test = model.predict(X_test)

# Plot predictions on training and testing data
plt.scatter(X_train, predictions_train, label='Training Predictions', color='red')
plt.scatter(X_test, predictions_test, label='Testing Predictions', color='green')



plt.legend()
plt.title('Original Data and Model Predictions')
plt.show()
model.save("nnregression.h5")