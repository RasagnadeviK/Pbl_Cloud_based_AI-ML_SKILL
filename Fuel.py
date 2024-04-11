import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers
d=pd.read_csv('Data/auto-mpg.csv')
print(d)
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight','Acceleration', 'Model Year', 'Origin']
d.isna().sum()
d=d.dropna()
d['Origin'] = d['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})