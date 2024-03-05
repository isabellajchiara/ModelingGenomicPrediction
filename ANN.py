# Define hyperparameters
nEpoch = 1000
batchSize = 50
valSplit = 0.2
learning_rate = 0.001  # You can adjust this learning rate as needed
adam_optimizer = Adam(learning_rate=learning_rate)

# Add layers
model = Sequential([
    Dense(units=xTrain.shape[1], activation='relu', input_shape=(xTrain.shape[1],)),
    Dropout(rate=0.05),
    Dense(units=xTrain.shape[1], activation='relu'),
    Dropout(rate=0.05),
    Dense(units=1)
])

# Create and compile model
model.compile(
    optimizer=adam_optimizer,
    loss='mean_squared_error',
    metrics=['mean_squared_error']
)

# Define early stopping
early_stopping = EarlyStopping(
    monitor='val_mean_squared_error',
    min_delta=0.1,
    patience=10,
    mode='auto'
)

# Fit the model
model = model.fit(
    x_train, y_train,
    batch_size=batchSize,
    epochs=nEpoch,
    verbose=0,
    validation_data=(x_train, y_train),
    callbacks=[early_stopping])
