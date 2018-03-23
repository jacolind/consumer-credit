### Method 2 -
################################################################################

## step 0 - clean data, by...
# i) run same code as before to import data. then change it a bit:
# ii) data should be in traintest, tuneparam, prediction (tt, tp, pr)
#########################################

# i) run some of the code above (if you have not done that already)

# split Xm into Xtt and Xtp

# inside dfm we split into X_train and X_test
Xtt, Xtp, yytt, ytp = train_test_split(Xm, ym, random_state=8, test_size = 0.20)
# check shapes:
Xtp.shape, ytp.shape
Xtt.shape, ytt.shape
# qq sample randmoly - how? https://stackoverflow.com/questions/14262654/numpy-get-random-set-of-rows-from-2d-array
# Xtt = np.concatenate((X_train, X_test), axis=0)
# ytt = np.concatenate((y_train, y_test), axis=0)

## step 1 - choose model by...
# i) specify models.
# ii) calc AUC using k-fold cv (k=10 good according to research).
#########################################

# i) models

# specify knn
knn = KNeighborsClassifier(n_neighbors=5)
# Fit
knn.fit(X_train, y_train)
# Predict y_hat
y_hat_knn = knn.predict(Xtp)
y_hat_knn.mean() * 100

# specify logreg
logreg = LogisticRegression()
# fit
logreg.fit(Xtt, ytt)
# predict
y_hat_reg = logreg.predict(Xtp)
y_hat_reg.mean() * 100


# specify neural nets

# 1. Specify
nnet = Sequential()
n_cols = Xtt.shape[1]
shape = (n_cols,)
nnet.add(Dense(17, activation='relu', input_shape=shape))
nnet.add(Dense(8, activation='relu', input_shape=shape))
nnet.add(Dense(1, activation='sigmoid'))
# 2. Compile
nnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 3. Fit :
nnet.fit(Xtt, ytt, epochs = 10, batch_size=128)
# 5. Predict
loss_and_metrics = nnet.evaluate(Xtp, ytp, batch_size=128)
y_hat_nnet = nnet.predict(Xtt, batch_size=128)
y_hat_nnet.sum() != 0 #false => stupid model

# ii) evalaute AUC scores

cv_logreg = cross_val_score(logreg, Xtt, ytt, cv=10, scoring='roc_auc').mean()
cv_knn = cross_val_score(knn, Xtt, ytt, cv=10, scoring='roc_auc').mean()
cv_logreg > cv_knn #true => chose logreg
# cross cal score cannot be computed for nnet (its from Keras)
# knn always predicts zero so its bad.

# qq also make sure it does not make stupid prediction "always zero"


## step 2: fine tune threshold parameter, by...
# i) calc the predicted prob and class, from the model chosen above.
# ii) draw the AUC curve
# iii) use eyes+function to evaluate what model we want. sensitivity is most important.
#########################################

# i) probabilies and classes

y_pred_prob = logreg.predict_proba(Xtp)[:, 1]
# predict deafult if the predicted probability is greater than threshold
from sklearn.preprocessing import binarize
thresh = 0.3
y_pred_class = binarize([y_pred_prob], thresh)[0]

#inaktuell kod:
#print(metrics.roc_auc_score(ytp, y_hat_prob_reg))

# ii) ROC curve

# tabulate the confusion matrix
metrics.confusion_matrix(ytp, y_pred_class)
# there are many actual=1 that we miss by predicting 0, so lower the threshold!
# which threshold to choose? se ROC curve and evaluation function below

# first argument is true values, second argument is predicted probabilities
fpr, tpr, thresholds = metrics.roc_curve(ytp, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
# concl: looking at this curve we search for Sensitivity=0.8
# iii) find optimal threshold

print(metrics.roc_auc_score(ytp, y_pred_prob))
# define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
  print(threshold)
  print('Sensitivity:', tpr[thresholds > threshold][-1])
  print('Specificity:', 1 - fpr[thresholds > threshold][-1])

# Look at the best threshold
for t in [0.0001, 0.001, 0.01, 0.015, 0.01999, 0.02, 0.05, 0.1, 0.2]:
	print(evaluate_threshold(t))

thresh_opt = 0.015 # somewehre between here and 0.02

# 0.0001
# Sensitivity: 1.0
# Specificity: 0.289110005528
# None
# 0.001
# Sensitivity: 0.785714285714
# Specificity: 0.846876727474
# None
# 0.01
# Sensitivity: 0.785714285714
# Specificity: 0.846876727474
# None
# 0.015
# Sensitivity: 0.785714285714
# Specificity: 0.846876727474
# None
# 0.02
# Sensitivity: 0.714285714286
# Specificity: 0.857379767828
# None
# 0.05
# Sensitivity: 0.428571428571
# Specificity: 0.942509673853
# None
# 0.1
# Sensitivity: 0.142857142857
# Specificity: 0.991155334439
# None
# 0.2
# Sensitivity: 0.0714285714286
# Specificity: 1.0



## step 3: use the chosen model and threshold for final predictions
#########################################

# now use chosen model and chrosen threshold on Xpr to submit predictions
Xpr = Xp
ypr = yp
y_pred_prob = logreg.predict_proba(Xpr)[:, 1]
y_pred_class = binarize([y_pred_prob])[0]
y_pred_class.mean()
y_pred_class_withID = # concat it with ID. qq
np.savetxt('method2_finalpredictions.csv', y_pred_class_withID, deliminer=",")
