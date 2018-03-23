### Description of the problem

'''
Binary classificaiton problem. Predict y=1 for default.
Variable names are hard to understand => use black box algorithm.
We use auc because it is a good overeall measure of a model
'''


### Import libraries

# basics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import*
import datetime

# custom plot
import itertools

# scale data
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
# models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score
from sklearn.metrics import roc_curve, precision_score, recall_score, classification_report
from sklearn.metrics import classification_report
# save models
from sklearn.externals import joblib


### Read data

raw_data = pd.read_csv('dataset/dataset.csv', sep=";")
vardescr = pd.read_csv('dataset/variabledescr.csv')

### Clean data

## Method A: fill NA with zeroes and standardize

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
yp = dfp['default']
ym = dfm['default']
ym.mean() #concl: 0.014 so high class imbalance

# Select X variables
exclude = ['default', 'uuid', 'merchant_category',
            'merchant_group', 'name_in_email']
# .info() reveals these are dtype = object so exclude them to save time
Xm = dfm.drop(exclude, axis=1)
Xp = dfp.drop(exclude, axis=1).fillna(0)
ym.shape[0] == Xm.shape[0]

# standardize X
scaler = StandardScaler().fit(Xm)
Xm = scaler.transform(Xm)
Xp = scaler.transform(Xp)

## Method B: fill NA with column mean

# We impute the missing value (of variable j) for a certain observation (i,j)
# with the column mean (mean of j).

methodb = False
if methodb == True:
    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    Xm = imp.fit_transform(Xm)
    Xp = imp.transform(Xp)


## Method C: fill NA with clustered column mean

'''
Firstly we divide the data into clusters using k-means clustering.
Then we impute the missing value for a certain row's column **j**
simply by looking at which group this observation falls into
and then replace the missing value with the that group's mean value for **j**.
'''

# todo write this code


### Fit and predict all models

'''
We use the following methods:

- Logistic regression
- K Nearest neighbour
- Decision tree

Using 5-fold crossvalidation, we see which model has the highest mean score.

Which scoring should we use? Either roc_auc or recall.

Argument for recall: Fraudulent transaction detector (positive class is "fraud"): Optimize for sensitivity because false positives (normal transactions that are flagged as possible fraud) are more acceptable than false negatives (fraudulent transactions that are not detected)

Argument for roc_auc: it is the standard method and selects a flexible model. If the AUC for one model is higher we can adjust the threshold in going form proba to classes...
'''


# split into fit and tune
Xf, Xt, yf, yt = train_test_split(Xm, ym, test_size=0.20, random_state=9)

# number of crossvalidation folds:
cv = 5

# select scoring metric
scoring = 'roc_auc'

## Logistic regression (reg)

'''
Explanation of what the hyperparameters measures:

* Like the alpha parameter of lasso and ridge regularization, logistic regression also has a regularization parameter: C. C controls the inverse of the regularization strength, and this is what you will tune in this exercise. A large C can lead to an overfit model, while a small C can lead to an underfit model. `param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}`
* penalty... todo write text.
'''

# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid_reg = {'C': c_space, 'penalty': ['l1', 'l2']}
# Instantiate the logistic regression classifier: logreg
reg = LogisticRegression()
# Instantiate the GridSearchCV object
reg_cv = GridSearchCV(reg, param_grid_reg, cv=cv, scoring=scoring)

# Fit it to the training data
load_reg = True# iff you run script for the first time, it should be False.
# load model or fit
if load_reg == True:
    reg_cv = joblib.load("output/reg_cv.pkl")
else:
    t1 = datetime.datetime.now()
    reg_cv.fit(Xf, yf)
    t2 = datetime.datetime.now()
    reg_td = t2-t1
    print("Fitting time H:MM:SS ", reg_td)
    # save model
    joblib.dump(reg_cv, "output/reg_cv.pkl")

# Print the optimal parameters and best score
print("reg")
print("Tuned Parameter: {}".format(reg_cv.best_params_))
print("Tuned Accuracy: {}".format(reg_cv.best_score_))
# output:
# params C = 0.44, penatly = 'l2' (with dfm)
# params C = 0.4394, penalty = 'l1' (with dfm split into Xf Xt)
# score 0.88


## Decision tree (tree)

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
load_tree = True
if load_tree == True:
    tree_cv = joblib.load('output/tree_cv.pkl')
else:
    t1 = datetime.datetime.now()
    tree_cv.fit(Xf, yf)
    t2 = datetime.datetime.now()
    tree_td = t2-t1
    print("Fitting time H:MM:SS ", tree_td)
    # save model
    joblib.dump(tree_cv, "output/tree_cv.pkl")

# Print the tuned parameters and score
print("tree")
print("Tuned Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))
# output:
# params are 'criterion': 'gini', 'max_depth': 10, 'max_features': 10, 'min_samples_leaf': 38}
# score 0.84


## K Nearest neighbour (knn)

# i wont even use knn because i happen to know a priori it take slong time and is bad, so i dont want to waste my time right now. maybe later when presenting a nice script.

# set up parameters
k_range = list(range(4, 8))
param_grid = dict(n_neighbors = k_range)
# instantiate
knn = KNeighborsClassifier()
# knn = KNeighborsClassifier(n_neighbors=5)
knn_cv = GridSearchCV(knn, param_grid, cv=cv, scoring=scoring, n_jobs = -1)
# fit
load_knn = True
if load_knn == True:
    knn_cv = joblib.load('output/knn_cv.pkl')
else:
    t1 = datetime.datetime.now()
    knn_cv.fit(Xm, ym)  # took a long time to run - 1 hour
    t2 = datetime.datetime.now()
    knn_td = t2-t1
    print("Fitting time H:MM:SS ", knn_td)
    # save model
    joblib.dump(tree_cv, "output/knn_cv.pkl")
# examine the best model
print("Tuned parameters: {}".format(knn_cv.best_params_))
print("Best score is {}".format(knn_cv.best_score_))
print(knn_cv.best_estimator_)


### Compare models

# We define best model as highest AUC.
# Rule of thumb says an AUC > 0.80 is to be considered very good.

print("reg", reg_cv.best_score_)
print("tree", tree_cv.best_score_)
print("knn", knn_cv.best_score_)
print("Winning model is: reg")
print("reg details:", reg_cv.best_estimator_)

# calculate AUC
penalty = 'l1' # TODO was l1 the most efficitent?
C = 0.4394
reg = LogisticRegression(C = C, penalty = penalty)
reg.fit(Xf, yf)
fpr, tpr, threshold = roc_curve(yt, reg.predict_proba(Xt)[:,1]) # om den ej funkar, skriv metric.roc_curve, om den funkar, radera kommentar
roc_auc = auc(fpr, tpr)  # om den ej funkar, skriv metric.auc, om den funkar, radera kommentar

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


### Threshold

'''
We now have a winning model. Even though the task was to present probabilities, the task in real life is to predict 0 or 1 and act based on that. So we will create a model that is best for this purpose. Given this purpose, we will change the threshold for predict class=1 based on the predicted proba. By default threshold=0.5 but what if being more suspoicious against users and setting a lower threshold of say 0.4 makes the number of defaults lower?

We will take the winning (highest auc) model from the "fit set" (Xf, yf) and apply it to the "tune set" (Xt, yt). Based on the Xt our model predicts proba. These proba are turned into classifications via a threshold that we vary.  We plot one confusion matrix per threshold. Based on the confusion matrix we chose the threshold that balance FP and FN in a way we believe is optimal for the business.
'''

### Fit, predict, and create confusion matrices

thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1,0.2,0.3,0.4]
recall_list = []
precision_list = []
for thresh in thresholds:
    # predict y=1 if proba > threshold
    y_pred_class = (thresh < reg.predict_proba(Xt)[:,1])
    #print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
    recall = np.round(recall_score(yt, y_pred_class), 3)
    precision = np.round(precision_score(yt, y_pred_class), 3)
    recall_list.append(recall)
    precision_list.append(precision)
    print("Recall & precision: ", recall, " & ", precision)
    # plot matrix
    print(confusion_matrix(yt, y_pred_class))

# plot recall & precision
plt.plot(recall_list, precision_list)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

# plot recall & threshold
plt.plot(recall_list, thresholds)
plt.xlabel("Recall")
plt.ylabel("Threshold")
plt.show()

### Final predictions

# We take the model with the highest AUC above and use that to make prediction on `Xp`.

# fit on entire modeling dataframe
reg.fit(Xm, ym)
# predict
predictions = reg.predict_proba(Xp)[:,1]
# save predictions and IDs to csv
assert len(predictions) == len(dfp.uuid)
pd.DataFrame({'ID':dfp['uuid'].values,
              'Probability':predictions}
            ).set_index('ID').to_csv('predictions/submission.csv')

#############################################################################

### pipeline


from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder

cols_cat = ['merchant_group']
cols_num = ['account_amount_added_12_24m', 'account_days_in_dc_12_24m',
                'account_days_in_rem_12_24m', 'account_days_in_term_12_24m',
                'account_incoming_debt_vs_paid_0_24m']

get_categorical_cols = FunctionTransformer(lambda x: x[cols_cat], validate=False)
get_numerical_cols = FunctionTransformer(lambda x: x[cols_num], validate=False)

numeric_pipeline = Pipeline([
    ('selector', get_numerical_cols),
    ('imputer', Imputer()),
    ('scale', StandardScaler())
    ])

categorical_pipeline = Pipeline([
    ('selector', get_categorical_cols),
    ('imputer', Imputer(strategy='most_frequent'),
    ('onehot', OneHotEnconder()))
    ])

union = FeatureUnion([
    ('numeric', numeric_pipeline),
    ('categorical', categorical_pipeline)
    ])

pipe = Pipeline([
    ('union', FeatureUnion([
      ('numeric', numeric_pipeline),
      ('categorical', categorical_pipeline)
      ])),
    ('reg', (LogisticRegression()))
    ])
# nu kan jag inte ändra classifiers särskilt smidigt

pipe.fit(dfm)

params = {reg__C = c_space,
        reg__penalty = ['l1', 'l2']
        }

cv = GridSearchCV(pipe, param_grid=params)
cv.fit(X_train, y_train)
print(cv.best_params_)
y_pred_probas = cv.predict_proba(X_test)
