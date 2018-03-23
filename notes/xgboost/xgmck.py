clf = GradientBoostingClassifier()

param_grid = {
    #'max_features':(None, 6, 9),
    'max_depth':(None, 16),
    #'min_samples_split':(2,4,8),
    #'min_samples_leaf':(4, 12, 16)
}

cv = GridSearchCV(clf, param_grid, cv=10)
cv.fit(X_train, y_train)
cv.best_score_
cv.best_params_

## xgb

# fit and predict

# read in data
dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test)
# specify parameters via map
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)
