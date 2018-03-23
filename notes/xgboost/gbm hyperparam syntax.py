# hyperparams
cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'objective': 'binary:logistic'}
# grid search
optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                            cv_params, scoring = scoring, cv=5, n_jobs = -1)
# fit
t1 = datetime.datetime.now()
optimized_GBM.fit(Xm, ym)
t2 = datetime.datetime.now()
gbm_td = t2-t1
print("Fitting time H:MM:SS ", gbm_td)
# fit output
print(optimized_GBM.grid_scores_)
# sen forts han... e can see that the first hyperparameter combination performed
