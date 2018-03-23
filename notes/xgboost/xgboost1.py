## source
# https://jessesw.com/XG-Boost/

import xgboost as xgb
import datetime

## handle Categoricals

# Stack them vertically - Xm and Xp are pd dataframes
combined_set = pd.concat([Xm, Xp], axis = 0)
# Loop through "object" cols  & replace strings with int
for feature in combined_set.columns:
    if combined_set[feature].dtype == 'object':
        combined_set[feature] = pd.Categorical(combined_set[feature]).codes
Xm = combined_set[:Xm.shape[0]]
Xp = combined_set[Xm.shape[0]:]

### Initial Model Setup and Grid Search

scoring = 'roc_auc'
cv=5

## a) vary max_depth and min_child_weight

# specify
cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0,
              'subsample': 0.8, 'colsample_bytree': 0.8,
              'objective': 'binary:logistic'}
# fit
optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), cv_params,
                             scoring = scoring, cv=5, n_jobs = -1)
t1 = datetime.datetime.now()
optimized_GBM.fit(Xm, ym)
t2 = datetime.datetime.now()
gbm_td = t2-t1
# fit evaluate
optimized_GBM.grid_scores_
# select best params
optimized_GBM.best_params_ # qq funkar det? om ej, så syns det i output ovan
max_depth = 3
min_child_weight = 1

## b) vary subsample and learning_rate

# "Let's try optimizing some other hyperparameters now to see if we can beat a mean of 86.78% accuracy. This time, we will play around with subsampling along with lowering the learning rate to see if that helps.""

# specify
cv_params = {'learning_rate': [0.1, 0.01], 'subsample': [0.7,0.8,0.9]}
ind_params = {'n_estimators': 1000, 'seed':0, 'colsample_bytree': 0.8,
             'objective': 'binary:logistic',
             'max_depth': max_depth, 'min_child_weight': min_child_weight}
# fit
optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), cv_params,
                             scoring = scoring, cv=cv, n_jobs = -1)
print(datetime.datetime.now().time())
optimized_GBM.fit(Xm, ym)
print(datetime.datetime.now().time())
# fit evaluate
optimized_GBM.grid_scores_
# select best params
optimized_GBM.best_params_ # qq funkar det? om ej, så syns det i output ovan
subsample = 0.8
learning_rate = 0.1


## a + b in one step

# jag skrev koven nedan baserat på ovan. funkar dne?
# om det ovan tog tid, kör ej det nedean. att dela i två går snabbare (färre kombinationer)
# så om params i a) är ortogonala mot params i b) då går a sen b snabbare än a+b.

# specify
cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5],
            'learning_rate': [0.1, 0.01], 'subsample': [0.07, 0.8, 0.09]}
ind_params = {'colsample_bytree': 0.8, 'n_estimators': 1000, 'seed':0,
              'objective': 'binary:logistic'}
# fit
optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), cv_params,
                             scoring = scoring, cv=5, n_jobs = -1)
print(datetime.datetime.now().time())
optimized_GBM.fit(Xm, ym)
print(datetime.datetime.now().time())
# fit evaluate
optimized_GBM.grid_scores_
# select best params
optimized_GBM.best_params_ # qq funkar det? om ej, så syns det i output ovan


### Early stopping CV

## specify

# Create our DMatrix to make XGBoost more efficient
xgdmat = xgb.DMatrix(final_train, y_train)
# select params - based on presiously done CV
our_params = {'eta': learning_rate, 'subsample': subsample,  # params chosen in b)
              'max_depth':max_depth, 'min_child_weight':min_child_weight, # from a)
              'colsample_bytree': 0.8, 'objective': 'binary:logistic', 'seed':0}
# select Grid Search CV settings
earlystop=100 # lets be aggressive - we don't want the accuracy to improve for at least 100 new trees.
maxrounds=3000

## fit

# cv to find num_boost_round
cv_xgb = xgb.cv(params = our_params, dtrain = xgdmat,  nfold = cv,
                num_boost_round = maxrounds,
                # qq kan jag välja 'roc_auc' eller metrics=scoring? 1-error=accuracy tydligen
                # Look for early stopping that minimizes error
                metrics = ['error'],
                early_stopping_rounds = earlystop)

# look at the 10 best (it's a pd dataframe )
cv_xgb.tail(10)
# select the num_boost_round that our early_stopping_rounds chose:
rounds = cv_xgb.shape[1]
# fit tha data with that number of rounds (not CV)
print(datetime.datetime.now().time())
final_gb = xgb.train(our_params, xgdmat, num_boost_round = rounds)
print(datetime.datetime.now().time())


## fit evaluate
importances = final_gb.get_fscore()
importance_frame = pd.DataFrame({'Importance': list(importances.values()),
                                 'Feature': list(importances.keys())})
importance_frame.plot(kind = 'barh', x = 'Feature', figsize = (8,8))

## predict

# create a DMatrix
testdmat = xgb.DMatrix(Xp)
# Predict using our testdmat
# predict proba
y_pred = final_gb.predict(testdmat)
y_pred_proba = y_pred
# predict classes
threshold = 0.5
y_pred[y_pred > threshold] = 1
y_pred[y_pred <= threshold] = 0
# accuracy score
print("accuracy score:", accuracy_score(y_pred, y_test))
# confusion matrix

# qq jag fattar inte: om det går snabbare med xgb.cv vrf inte göra så även i a och b?
# jag tror steg 1 är: kör koden ovan och fatta den. steg 2: ändra a och b så att dom är xgd.cv oxå, ifall du har speed problems.
