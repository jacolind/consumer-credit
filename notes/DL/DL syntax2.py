

## Specify

model = Sequential()
n_cols = x.shape[1]
shape = (n_cols,)
np.round(n_cols * 2/3, 0) # nodes in layer 1
np.round(nodes_1 / 2, 0) # nodes i layer 2
model.add(Dense(17, activation='relu', input_shape=shape))
#model.add(Dense(8, activation='relu', input_shape=shape))
model.add(Dense(2, activation='softmax'))

## Compile

# qq optimizer sgd or adam
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Early stopping
early_stopping_monitor = EarlyStopping(patience = 3)

## Fit

model.fit(X_train, Y_train, epochs = 25,
          batch_size = 64
          callbacks=[early_stopping_monitor])


## Predict
