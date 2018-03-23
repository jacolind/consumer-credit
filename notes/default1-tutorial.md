
# Description of problem

Binary classificaiton problem. Predict y=1 for default.

Variable names are hard to understand => use black box algorithm.

we use auc because...

# Import libraries


```python
# basics :
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import*
import datetime

# scale data
from sklearn.preprocessing import StandardScaler
# models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
# metrics
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# save models
from sklearn.externals import joblib

# keras for neural networks:
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical


```

# Read data


```python
raw_data = pd.read_csv('dataset.csv', sep=";")
vardescr = pd.read_csv('variabledescr.csv')
```

# Clean data

## Method A: fill NA with zeroes


```python
# create dfp: all rows that have deafult=NA. this df is used for prediction.
dfp = raw_data[pd.isnull(raw_data.default)]
# crate dfm: all rows that have deafult=1 or 0. this df is used for modeling.
dfm = raw_data[pd.notnull(raw_data.default)]

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

# Select X variables
exclude = ['default', 'uuid', 'merchant_category', 'merchant_group', 'name_in_email']
# .info() reveals these are dtype = object so exclude them to save time
Xm = dfm.drop(exclude, axis=1).as_matrix()
Xp = dfp.drop(exclude, axis=1).fillna(0).as_matrix()
ym.shape[0] == Xm.shape[0]

# standardize X
scaler = StandardScaler().fit(Xm)
Xm = scaler.transform(Xm)
Xp = scaler.transform(Xp)
```

## Method B: fill NA with column mean


```python
# qq write this code
```

# Fit and predict all models

We the following methods:

- Logistic regression
- K Nearest neighbour
- Decision tree


```python
# select scoring metric
scoring = 'roc_auc'
# number of crossvalidation folds:
cv = 5
```

## Logistic regression (reg)

param_grid_reLike the alpha parameter of lasso and ridge regularization  that you saw earlier, logistic regression also has a regularization parameter: CC. CC controls the inverse of the regularization strength, and this is what you will tune in this exercise. A large CC can lead to an overfit model, while a small CC can lead to an underfit model.g = {'C': c_space, 'penalty': ['l1', 'l2']}



```python
# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid_reg = {'C': c_space, 'penalty': ['l1', 'l2']}
# Instantiate the logistic regression classifier: logreg
reg = LogisticRegression()
# Instantiate the GridSearchCV object
reg_cv = GridSearchCV(reg, param_grid_reg, cv=cv, scoring=scoring)

# first time you run script, below should be false.
# Fit it to the training data
load_reg = True
# load model or fit
if load_reg == True:
    reg_cv = joblib.load("reg_cv.pkl")
else:
    t1 = datetime.datetime.now()
    reg_cv.fit(Xm, ym)
    t2 = datetime.datetime.now()
    reg_td = t2-t1
    print("Fitting time H:MM:SS ", reg_td)
    # save model
    joblib.dump(reg_cv, "reg_cv.pkl")

# Print the optimal parameters and best score
print("reg")
print("Tuned Parameter: {}".format(reg_cv.best_params_))
print("Tuned Accuracy: {}".format(reg_cv.best_score_))
# params C 0.44, penatly l2. score 0.876
```

## Decision tree (tree)


```python
# Setup the parameters
param_dist = {"max_depth": [None, 10, 20, 30], # 30-50% av nr features
              "max_features": [5, 10, 20, 30, Xm.shape[1]],
              "min_samples_leaf": [1, 10, 20, 30, Xm.shape[1]],
              "criterion": ["gini", "entropy"]}
# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()
# Instantiate the GridSearchCV() object: tree_cv
tree_cv = GridSearchCV(tree, param_dist, cv=cv, scoring=scoring, n_jobs = -1)

# Fit it to the data
load_tree = False
if load_tree == True:
    tree_cv = joblib.load('tree_cv.pkl')
else:
    t1 = datetime.datetime.now()
    tree_cv.fit(Xm, ym)
    t2 = datetime.datetime.now()
    tree_td = t2-t1
    print("Fitting time H:MM:SS ", tree_td)
    # save model
    joblib.dump(tree_cv, "tree_cv.pkl")

# Print the tuned parameters and score
print("tree")
print("Tuned Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))
# output: 'criterion': 'gini', 'max_depth': 10, 'max_features': 10, 'min_samples_leaf': 38}
# auc score 0.84
```

## K Nearest neighbour (knn)


```python
# set up parameters
k_range = list(range(4, 8))
param_grid = dict(n_neighbors = k_range)
# instantiate
knn = KNeighborsClassifier()
# knn = KNeighborsClassifier(n_neighbors=5)
knn_cv = GridSearchCV(knn, param_grid, cv=cv, scoring=scoring, n_jobs = -1)
# fit
load_knn = False
if load_knn == True:
    knn_cv = joblib.load('knn_cv.pkl')
else:
    t1 = datetime.datetime.now()
    knn_cv.fit(Xm, ym)
    t2 = datetime.datetime.now()
    knn_td = t2-t1
    print("Fitting time H:MM:SS ", knn_td)
# examine the best model
print("Tuned parameters: {}".format(knn_cv.best_params_))
print("Best score is {}".format(knn_cv.best_score_))
print(knn_cv.best_estimator_)
```

Another method is a for loop. GridSearchCV should be faster but for some reason it takes a lot of time to compute.


```python
for k in [2,4,8,12]:
    # Instantiate a Decision knn classifier: knn
    knn = KNeighborsClassifier(n_neighbors = k)
    # calc scores
    scores = cross_val_score(knn, Xm, ym, cv=5, scoring=scoring)
    # Print the tuned parameters and score
    print("knn", k, scores.mean())
```

# Compare models

We define best model as highest AUC.


```python
print("reg, tree, knn")
print(reg_cv.best_score_,
      tree_cv.best_score_
      knn_cv.best_score_
     )
print("Winning model is:", "...")
```

# Final predictions

We take the model with the highest AUC above and use that to make prediction on `Xp`.


```python
# instantiate model with optimal parameters
logreg = LogisticRegression(C = 0.44, penalty = 'l2')
# fit on entire modeling dataframe
logreg.fit(Xm, ym)
# predict
predictions = logreg.predict_proba(Xp)[:,1]
# save predictions and IDs to csv

```
de
