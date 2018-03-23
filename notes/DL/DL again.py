
# keras for neural networks:
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical


# coding: utf-8

# In[19]:

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from numpy import*
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping


X_train, X_test, Y_train, Y_test= train_test_split(Xm, Ym, random_state=4)


# In[22]:

# create model
model = Sequential()
model.add(Dense(16, input_dim=25, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[30]:

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[31]:

# # Early stopping
# early_stopping_monitor = EarlyStopping(patience = 3)


# In[32]:

# Fit the model
model.fit(X_train, Y_train, epochs=50, batch_size=64)


# In[33]:

model.summary()


# In[34]:

# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:
