# Importing libraries

import pandas as pd
import numpy as np
import tensorflow as tf

# Importing dataset

dataset = pd.read_csv('Assets/Churn_Modelling.csv')

x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Data preprocessing

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

#

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

# Splitting the dataset

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Scaling the dataset

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Creating the ANN

ANN = tf.keras.models.Sequential()

ANN.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
ANN.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
ANN.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

# Compiling the ANN

ANN.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN

ANN.fit(x_train, y_train, batch_size = 32, epochs = 123)

# Predicting

ANN.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))

# Accuracy of the ANN

y_pred = ANN.predict(x_test)
y_pred = (y_pred > 0.5)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test), 1)), 1))