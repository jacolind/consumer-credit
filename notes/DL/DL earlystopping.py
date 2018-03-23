# remove NA in numpy
x = x[np.logical_not(np.isnan(x))]

dat.dropna(how='any')    #to drop if any value in the row has a nan
dat.dropna(how='all')    #to drop if all values in the row are na


## to have a integer y (not categorical) do this:
# Dense(1 instead of Dense(2
model.add(Dense(1, activation='sigmoid')) # predicts 0 or 1
# loss=sparse_blabla instead of loss=blabla
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# accuracy in keras is: the proportion of true results (both true positives and true negatives) among the total number of cases examined

raw_data = pd.read_csv('dataset.csv', sep=",")
df = raw_data.dropna(how='any')    #to drop if any value in the row has a nan
y = df['default'].as_matrix()
vardescr = pd.read_csv('variabledescr.csv')
# select variables with type=numeric
numerical_variables = vardescr[vardescr.Type == 'numeric'].Variable
x_df = df[numerical_variables]
x = x_df.as_matrix() #keras wants a matrix

#now y,x as matrix / array

## early stopping
from keras.callbacks import EarlyStopping

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Early stopping
early_stopping_monitor = EarlyStopping(patience = 3)

## Fit

model.fit(X_train, Y_train, epochs = 25,
          batch_size = 64
          callbacks=[early_stopping_monitor])


scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
scaler = StandardScaler().fit(X_test)
X_test = scaler.transform(X_test)

from sklearn.preprocessing import StandardScaler
