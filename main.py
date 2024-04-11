import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def solution_model():
    xs = np.array([-1.0,0.0,1.0,2.0,3.0,4.0], dtype = float)
    ys = np.array([0.5,6.0,7.0,8.0,9.0,10.0], dtype = float)


    model = tf.keras.Sequential([tf.keras.layers.Dense(units =1,input_shape =[1])])
    model.compile(optimizer='sgd',loss='mean_squared_error')
    model.fit(xs,ys,epochs =500)

    return model


model = solution_model()
model.save("Linear_Regression.h5")
print(model.predict([10.0]))