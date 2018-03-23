## Import libraries

# basics :
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import*

# sklearn :
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score

from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

# keras for neural networks:
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


## Read data

raw_data = pd.read_csv('dataset.csv', sep=";")
vardescr = pd.read_csv('variabledescr.csv')

## Models to fit

# We will fit: KNN, LogisticRegression, neural networks.
# Fit models using X_train. Predict y_hat using X_test.
# The model with the best prediction will be chosen.
# That model will take Xp as input and deliver outputs y_p to the company.
# Syntax: X_test predicts y_hat_modelname. Xp predicts y_p_modelname


## Split dataset into rain, test, predict

# split X and Y into train, test, p - the last one for prediction.
# use X_train&y_train to fit the model. use X_test to predict Y and then compare those predictions to Y_test in order to see accuracy between different models. the model that is best in this step will be used for submitting our predictions. lastly, take Xp to predict y_p - the company will then use my yp against their true values of y to see how good my predictions were.
# the reason for _train, _test, _p is to remebles train, cv, test.

# create dfp: all rows that have deafult=NA.
dfp = raw_data[pd.isnull(raw_data.default)]
0 == dfp.default.notnull().sum() #True
Yp = dfp['default'].as_matrix() # Y for prediction
Xp = dfp.drop('default', axis=1) # X for prediction

# crate dfm: all rows that have deafult=1 or 0. this df is used for modeling.
dfm = raw_data[pd.notnull(raw_data.default)]
0 == dfm.default.isnull().sum() #True: no NA in y
0 < dfm.isnull().sum().sum() #true: many NA in our X variables


## choice of X's and how to handle NA

# see na-choice.txt qq include in text

## Drop NA rows

# drop rows with NA. used for fitting :
dfm = dfm.dropna()
# set NA to 0 for X used in the prediction :
Xp = Xp.fillna(0)

# Select Y for out model "Ym"
Ym = dfm['default'].as_matrix()
Ym.mean() #concl: 0.014 so very few deafults

## Select X variables

# Select numerical variables
numerical_variables = vardescr[vardescr.Type == 'numeric'].Variable
Xm = dfm[numerical_variables].as_matrix()
Xp = dfp[numerical_variables].fillna(0).as_matrix()
Ym.shape[0] == Xm.shape[0]  #check number of rows are the same

# qq maybe increase to also include categorical

## Split into train and test

# inside dfm we split into X_train and X_test  (70% and 30%) :
X_train, X_test, y_train, y_test = train_test_split(Xm, Ym, random_state=9)

# check shapes look ok. concl: size is enough for NN and KNN :
X_train.shape, X_test.shape, Xp.shape
y_train.shape, y_test.shape, Yp.shape
(X_train.shape[0] + X_test.shape[0]) / len(raw_data) # concl: only 9% so maybe the data with NA is different from the data with non-NA. this would decrease the prediction quality of the model. if I have time, adress this.

# Our data is now divided into: X_train, X_test, Xp and y_train, Y_test, Yp

## standardize numerical X

# qq do it later I get errors
# scaler = StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# Xp = scaler.transform(Xp)
# qq ska train o test o p skalas med samma scaler som X_train?


## qq delete this section?

# df_notna = raw_data.loc[raw_data['default'].isnull()]
# x_notna = raw_data[raw_data.default]
# Y = df['default'].as_matrix()


## KNN

# Specify
knn = KNeighborsClassifier(n_neighbors=5)
# Fit
knn.fit(X_train, y_train)
# Predict y_hat
y_hat_knn = knn.predict(X_test)
# y_hat accuracy_score
print(metrics.accuracy_score(y_test, y_hat_knn))
# y_hat nr of defaults vs training sampel
print("No of predicted defaults:" , y_hat_knn.sum())
#print(y_hat_knn.sum() / len(y_hat_knn))
#print(y_train.sum(), len(y_train))

## KNN on Xp

# just to make sure I will be able to predict y_p, assume KNN won. now we have to use it on the Xp data.
y_p_knn = knn.predict(Xp) # qq vrf fuckar det up?
y_p_knn.sum() #No of predicted defaults

## Logistic Regression

# Specify
logreg = LogisticRegression()

# Fit
logreg.fit(X_train, y_train)

# Predict y_hat
y_hat_reg = logreg.predict(X_test)
print("No of predicted defaults:" , y_hat_reg.sum())

# y_hat accuracy_score
print(metrics.accuracy_score(y_test,y_hat_reg))
# qq describe accuracy_score

## Logistic regression on Xp (assume it won)

y_p_reg = logreg.predict(Xpp)
# compare % predicted defaults vs % default in dataset
y_p_reg.sum()
# compare fraction of y=1 predictions vs in dfm
y_p_reg.sum() / len(y_p_reg), dfm['default'].mean()
# qq why is it so different from KNN prediction?


# ## KNN 500-fold cross-validation
#
# knn = KNeighborsClassifier(n_neighbors=5)
# scores = cross_val_score(knn, X, Y, cv=500, scoring='accuracy')
# print(scores.mean())
#
# ## KNN vary K value
#
# k_range = list(range(1, 8))
# k_scores = []
# # using k=1,...,8 we fit a knn.
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(knn, X, Y, cv=500, scoring='accuracy')
#     k_scores.append(scores.mean())
# print(k_range, np.round(k_scores, 4))


## Neural network - sigmoid and binary_crossentropy

# 1. Specify
# 2. Compile
# 3. Fit
# 4. Predict

# 1. Specify
model = Sequential()
n_cols = X_train.shape[1]
shape = (n_cols,)
np.round(n_cols * 2/3, 0) # nodes in layer 1
# nodes in layer 2 = half of those in layer 1
model.add(Dense(17, activation='relu', input_shape=shape))
#model.add(Dense(8, activation='relu', input_shape=shape))
model.add(Dense(1, activation='sigmoid'))
#model.add(Dense(2, activation='softmax'))
print(model.summary())

# 2. Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 3. Fit :
model.fit(X_train, y_train, epochs = 10, batch_size=128)
# Accuracy = (TP + TN) / (P+N).    T: True. P: positives. N: negatives.

# 5. Predict
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)
y_hat_nnet = model.predict(X_test, batch_size=128)
y_hat_nnet.sum()

## X categorical : one hot encoding?

# encode class values as integers
x1 = dfm['account_status']
encoder = LabelEncoder()
encoder.fit(x1)
encoded_x1 = encoder.transform(x1)

# If there is time you could implement kfold as seen in https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/ but that might be overkill


# qq select winner and apply it to Xp

## Selection of winning model

# Usually: Fraudulent transaction detector (positive class is "fraud"): Optimize for sensitivity because false positives (normal transactions that are flagged as possible fraud) are more acceptable than false negatives (fraudulent transactions that are not detected)
# Although this might not be true for this company. Sure they loose money but one very important selling factor is that people use it to pay later - it increases conversion. optimizing for sensitivity (high recall)

# put predictions into a list :
y_hatlist = [y_hat_reg, y_hat_knn, y_hat_nnet]
print("number of actaul defaults:", y_test.sum())
print("Number of predicted defaults in test set:")
print(y_hatlist[0].sum())
print(y_hatlist[1].sum())
print(y_hatlist[2].sum())

# Classification Accuracy: Overall, how often is the classifier correct? .accuracy_score

# Recall (aka Sensitivity): When the actual value is 1, how often is the prediction correct?  .recall_score

# Precision: When a positive value is predicted, how often is the prediction correct? .precision_score

# qq this should be done in a function.
modelnames = ["logreg", "knn", "nnets"]

print("models and their evaluation metrics:")
for model in range(0, 3):
    print("\n", modelnames[model])
    print("accuracy", metrics.accuracy_score(y_test, y_hatlist[model]))
    print("precision", metrics.recall_score(y_test, y_hatlist[model]))
    print("recall", metrics.precision_score(y_test, y_hatlist[model]))

print("Confusion matrix: Format is (actual \ predicted) \n e.g. right column is y_hat=1.")
for model in range(0, 3):
    print("\n", modelnames[model])
    print(metrics.confusion_matrix(y_test, y_hatlist[model]))


## Model winner on Xp to predict y_p

## reshape the predictions from Xp and concat with uuid: send that to company.
dfm['uuid'].head()
