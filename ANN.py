import dependencies.py
import formatDataANN.py

# Define hyperparameters
nEpoch = 100
batchSize = 50
valSplit = 0.2
learning_rate = 0.01  # You can adjust this learning rate as needed
adam_optimizer = Adam(learning_rate=learning_rate)

# Add layers
model = Sequential([
    Dense(units=xTrain.shape[1], activation='relu', input_shape=(xTrain.shape[1],)),
    Dropout(rate=0.05),
    Dense(units=geno.shape[1], activation='relu'),
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
    xTrain, yTrain,
    batch_size=batchSize,
    epochs=nEpoch,
    verbose=0,
    validation_data=(xValid, yValid),
    callbacks=[early_stopping])

scores = cross_val_score(model, X, Y, cv=5)
scores = pd.DataFrame(scores)
scores.to_csv("baseANNperf_SY.csv")
