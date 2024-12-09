

def OptimizeTransformer(params):
    d_model, num_heads, num_layers, d_ff, dropout, lr = params

    accuracies = []  # Store accuracies for each k-fold
    kf = KFold(n_splits=5, shuffle=True, random_state=100)

    # 5-fold cross-validation
    for train_index, test_index in kf.split(X):
        xTrain, xTest = X.iloc[train_index], X.iloc[test_index]
        yTrain, yTest = y.iloc[train_index], y.iloc[test_index]

        xTrain = torch.tensor(xTrain.values, dtype=torch.int64)
        yTrain = torch.tensor(yTrain.values, dtype=torch.float32)
        xTest = torch.tensor(xTest.values, dtype=torch.int64)
        yTest = torch.tensor(yTest.values, dtype=torch.float32)

        train_dataset = TensorDataset(xTrain, yTrain)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        src_vocab_size = int(unique)
        max_seq_length = X.shape[1]

        # Initialize the model
        transformer = Transformer(
            src_vocab_size, 1, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout
        )
        transformer.apply(initialize_attention_weights)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(transformer.parameters(), lr=lr)

        # Train the model
        for epoch in range(100):
            transformer.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                estimations = transformer(batch_x)
                loss = criterion(estimations, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate
        transformer.eval()
        with torch.no_grad():
            prediction = transformer(xTest)
        prediction_np = prediction.numpy()
        yTest_np = yTest.numpy()
        accuracy, _ = pearsonr(prediction_np, yTest_np)
        accuracies.append(accuracy)

    # Average accuracy across folds
    result = sum(accuracies) / len(accuracies)
    return params, result

# Run in parallel
results = []
with ProcessPoolExecutor() as executor:
    results = list(executor.map(train_and_evaluate, param_grid))

# Find the best parameter combination
best_params, best_accuracy = max(results, key=lambda x: x[1])

# Save results
results_csv_file = "grid_search_results.csv"
with open(results_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["d_model", "num_heads", "num_layers", "d_ff", "dropout", "lr", "accuracy"])
    for params, accuracy in results:
        writer.writerow(list(params) + [accuracy])

print(f"Best parameters: {best_params}, Accuracy: {best_accuracy}")
print(f"Results saved to {results_csv_file}")
