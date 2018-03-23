### Import libraries
################################################################################

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

deep = False
if deep == True:
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

## standardize X

scaler = StandardScaler().fit(Xm)
Xm = scaler.transform(Xm)
Xp = scaler.transform(Xp)

### fit and predict
################################################################################

scoring = 'roc_auc'
cv = 5

## i) reg

# param_grid_reLike the alpha parameter of lasso and ridge regularization  that you saw earlier, logistic regression also has a regularization parameter: CC. CC controls the inverse of the regularization strength, and this is what you will tune in this exercise. A large CC can lead to an overfit model, while a small CC can lead to an underfit model.g = {'C': c_space, 'penalty': ['l1', 'l2']}

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

## ii) decision trees

# Setup the parameters
param_dist = {"max_depth": [None, 10, 20, 30], # 30-50% of #features
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

## iii) knn

# set up parameters
k_range = list(range(4, 8))
param_grid = dict(n_neighbors = k_range)
# instantiate
# knn = KNeighborsClassifier(n_neighbors=5)
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=cv, scoring=scoring, n_jobs = -1)
# fit
load_knn = True
if load_knn == True:
    knn_cv = joblib.load('knn_cv.pkl')
else:
    t1 = datetime.datetime.now()
    knn_cv.fit(Xm, ym)
    t2 = datetime.datetime.now()
    knn_td = t2-t1
    print("Fitting time H:MM:SS ", knn_td)
    joblib.dump(knn_cv, 'knn_cv.pkl')
# examine the best model
print("Tuned parameters: {}".format(knn_cv.best_params_))
print("Best score is {}".format(knn_cv.best_score_))
print(knn_cv.best_estimator_)
# tog 1h att köra. k=7 var bäst. med score=0.685


for k in [2,4,8,12]:
    # Instantiate a Decision knn classifier: knn
    knn = KNeighborsClassifier(n_neighbors = k)
    # calc scores
    scores = cross_val_score(knn, Xm, ym, cv=5, scoring=scoring)
    # Print the tuned parameters and score
    print("knn", k, scores.mean())

## compare scores

print("reg, tree, knn")
print(reg_cv.best_score_, tree_cv.best_score_)

## take the selected model on Xp dataset
print(reg_cv.best_score_, reg_cv.best_params_)
# qq write code
logreg = LogisticRegression(C = 0.44, penalty = 'l2')
logreg.fit(Xm, ym)
predictions = logreg.predict_proba(Xp)[:,1]

### todo

# qq läs på om C och penalty i teorin. försök att implemetera det på min data kanske? ev 1) välj typ av modell mha dataset1 2) välj hyperparams såsom penalty&C (logreg) elr neigbors (knn) elr depth&features&leafs (DecisionTree) mha dataset2 som ger dig finalmodel 3) använd finalmodel för att göra finalpredictions. eller varför göra dom två stegen? använd concat(dataset1,2 dvs Xm) för att låta massa modeller tävla mot varandra, den som performs best i Xm får predicta på Xp - simpelt.
