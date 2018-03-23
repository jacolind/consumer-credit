## Neural network - using to_categorical

# use 1 layer not 2 because of dataset size is small
# how many nodes? one standard:
n_cols * 2/3       # nodes in layer 1
n_cols * 2/3 * 0.5 # nodes in layer 2
# but we have limits:  set nodes such that total_params * 30 < sample_size (according to https://stats.stackexchange.com/questions/78289/neural-network-modeling-sample-size) so we have 10 nodes and 1 layer. <- that reduced prediction accuracy so dont do that. qq.

# 1. Specify
model = Sequential()
n_cols = X_train.shape[1]
shape = (n_cols,)
model.add(Dense(17, activation='relu', input_shape=shape))
model.add(Dense(7, activation='relu', input_shape=shape))
model.add(Dense(2, activation='softmax'))
print(model.summary())

# 2. Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 3. Fit :
model.fit(X_train, to_categorical(y_train), epochs = 10)

# 5. Predict
loss_and_metrics = model.evaluate(X_test, to_categorical(y_test))
print(np.round(loss_and_metrics, 3)) # 0.0489 0.989
y_hat_nnet_pred = model.predict(X_test)
np.round(y_hat_nnet_pred, 3) #here we see it can be [0.995, 0.005]
# final predictions are the rounded version:
y_hat_nnet = np.round(y_hat_nnet_pred, 0) #round >0.5 to 1
y_hat_nnet.sum()
# qq that needs to be converted to
