# Header 1 = ###. Header 2 = ##. So search for "## " to cycle through this code.

### Import libraries
################################################################################

# basics :
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import*

# sklearn for KNN and logreg. and train/test split:
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split

# keras for neural networks:
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

### Read data
################################################################################

raw_data = pd.read_csv('dataset.csv', sep=";")
vardescr = pd.read_csv('variabledescr.csv')

### Clean data
################################################################################

# create dfp: all rows that have deafult=NA. this df is used for prediction.
dfp = raw_data[pd.isnull(raw_data.default)]
# crate dfm: all rows that have deafult=1 or 0. this df is used for modeling.
dfm = raw_data[pd.notnull(raw_data.default)]

## choice of X's and how to handle NA

# how many NA?
total_na = dfm.isnull().sum().sum()
total_cells =  dfm.count().sum()
total_na / total_cells * 100

# handle NA in dfm :
dfm = dfm.fillna(0) # fill NA with zero

# Select y for prediction and modeling
yp = dfp['default'].as_matrix()
ym = dfm['default'].as_matrix()
ym.mean() #concl: 0.014 so very few deafults

## Select X variables

# .info() reveals these are dtype = object so exclude them to save time
exclude = ['default', 'uuid', 'merchant_category', 'merchant_group', 'name_in_email']
Xm = dfm.drop(exclude, axis=1).as_matrix()
Xp = dfp.drop(exclude, axis=1).fillna(0).as_matrix()
ym.shape[0] == Xm.shape[0]

## Split into train and test

# inside dfm we split into X_train and X_test  (75% and 25%) :
X_train, X_test, y_train, y_test = train_test_split(Xm, ym, random_state=9)

# check shapes look ok.
X_train.shape, X_test.shape, Xp.shape
y_train.shape, y_test.shape, yp.shape
# concl: size is enough for NN and KNN
(X_train.shape[0] + X_test.shape[0]) / len(raw_data)
# concl: only 9% so maybe the data with NA is different from the data with non-NA.
# this would decrease the prediction quality of the model. iff I have time, adress this.

## standardize X

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
Xp = scaler.transform(Xp)


### fit and predict
################################################################################

## Models to fit & predict

# We will fit: KNN, LogisticRegression, neural networks.
# Fit models using X_train. Predict y_hat using X_test.
# The model with the best prediction will be chosen.
# That model will take Xp as input and deliver outputs y_p to the company.
# Syntax: X_test predicts y_hat_modelname. Xp predicts y_p_modelname

## KNN

# KNN is slow to run

# Specify
knn = KNeighborsClassifier(n_neighbors=5)
# Fit
knn.fit(X_train, y_train)
# Predict
y_hat_knn = knn.predict(X_test)

## Logistic Regression

# Specify
logreg = LogisticRegression()
# Fit
logreg.fit(X_train, y_train)
# Predict
y_hat_reg = logreg.predict(X_test)

## Neural network

# Specify
nnet = Sequential()
n_cols = X_train.shape[1]
shape = (n_cols,)
np.round(n_cols * 2/3, 0) # standard choice for nodes in layer 1.
nnet.add(Dense(17, activation='relu', input_shape=shape))
nnet.add(Dense(2, activation='softmax'))
nnet.summary()

# Compile
nnet.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit
nnet.fit(X_train, to_categorical(y_train), epochs = 10, batch_size=128)

#Predict
loss_and_metrics = nnet.evaluate(X_test, to_categorical(y_test), batch_size=128)
y_hat_nnet_vec = nnet.predict(X_test, batch_size=128)
# select y=1 column and convert to 0 and 1
y_hat_nnet = (y_hat_nnet_vec[:, 1] > 0.5).astype(float)

### evaluate predictions and choose a winning model
################################################################################

## put predictions into a list

modelnames = ["logreg", "knn", "nnet"]
y_hatlist = [y_hat_reg, y_hat_knn, y_hat_nnet]

## create confusion matrix and classification report

print("Confusion matrix: Format is (actual \ predicted) \n e.g. right column is y_hat=1.")
print("\n Confusion matrix & classification report")
for model in range(0, 3):
    print("\n", modelnames[model])
    print(metrics.confusion_matrix(y_test, y_hatlist[model]))
    print(classification_report(y_test, y_hatlist[model]))
    print("Mean of predictions: ", y_hatlist[model].mean())

print("Mean of y_test", y_test.mean())
y_test.mean() / y_hatlist[0].mean() < 4 # false so we underpredict heavily

### Model winner on Xp to predict final predicitons y_p
################################################################################

# KNN on Xp. takes time to run.
y_p_knn = knn.predict(Xp)
y_p_knn_prob = knn.predict_proba(Xp)
y_p_knn_pd = y_p_knn_prob[:, 1]

# check threshold is 50%
y_p_knn.sum() == (y_p_knn_pd > .5).sum()

# check pd histogram
plt.hist(y_p_knn_pd)
plt.show()

# check if we suspect it underpredicts
y_p_knn.sum()
y_hat_knn.mean() / y_test.mean()
y_p_knn.mean() / y_test.mean()
# We think it underpredicts.

# reshape the predictions and send that to company.
np.savetxt("final_classes.txt", y_p_knn)
np.savetxt("final_pd.txt", y_p_knn_pd)
