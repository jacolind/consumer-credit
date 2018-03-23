## Import necessary modules

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
import pandas as pd
import numpy as np

from sklearn import neighbors, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

## Import data

raw_data = pd.read_csv('dataset.csv', sep=";")
raw_data.isnull().sum() # many NA
# drop all rows where some columns is NA
df = raw_data.dropna(how='any')    #to drop if any value in the row has a nan
df.info() ## qq need to change dtypes to categorical later

## select y

# y: 'survived' is binary (n x 1) so create a (n x 2) matrix
y = to_categorical(df['default'])
y.shape
0 == y.shape[0] - np.nansum(y) # True => No NA! .nansum() return 0 for NAs

## select x

# we only select some variables - numerical ones!

# import csv that I copy pasted from pdf
vardescr = pd.read_csv('variabledescr.csv')
# select variables with type=numeric. save.
numerical_variables = vardescr[vardescr.Type == 'numeric'].Variable
x_df = df[numerical_variables]
# checks:
x_df.info() # qq is int65 and float64 - must I convert them?
x_df.isnull().sum().sum() == 0 #True
# convert to matrix - keras wants a matrix
x = x_df.as_matrix()

# x: drop y column from df and save as a matrix


## Specify

model = Sequential()
n_cols = x.shape[1]
shape = (n_cols,)
np.round(n_cols * 2/3, 0) # nodes in layer 1
np.round(nodes_1 / 2, 0) # nodes i layer 2
model.add(Dense(17, activation='relu', input_shape=shape))
#model.add(Dense(8, activation='relu', input_shape=shape))
model.add(Dense(2, activation='softmax'))

## Compile

# qq optimizer sgd or adam
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Early stopping
early_stopping_monitor = EarlyStopping(patience = 3)

## Fit

model.fit(X_train, Y_train, epochs = 25,
          batch_size = 64
          callbacks=[early_stopping_monitor])


## Predict
