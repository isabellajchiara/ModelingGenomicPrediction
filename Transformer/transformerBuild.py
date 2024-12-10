class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout,weights):
        super(Transformer, self).__init__()
        self.register_buffer('weights', weights)  # Register weights as a buffer

        #define layers
        
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        for layer in self.encoder_layers:
            src_embedded = layer(src_embedded)
        output = self.fc(src_embedded)
        output = output.view(src.shape[0], -1)
        weighted_output = output * self.weights.unsqueeze(0)
        pred = torch.mean(weighted_output, dim=1)
        
        return pred


def trainTest5Fold(X,y):
    accuracies = []  # Store accuracies for each k-fold
    loss_values = []  # Store loss values across all epochs

    kf = KFold(n_splits=5, shuffle=True, random_state=100)  # Cross-validation

    # 5-fold cross-validation
    for train_index, test_index in kf.split(X):

        # train rest split for k fold
        xTrain, xTest = X.iloc[train_index], X.iloc[test_index]
        yTrain, yTest = y.iloc[train_index], y.iloc[test_index]

        xTrain = torch.tensor(xTrain.values, dtype=torch.int64)
        yTrain = torch.tensor(yTrain.values, dtype=torch.float32)
        xTest = torch.tensor(xTest.values, dtype=torch.int64)
        yTest = torch.tensor(yTest.values, dtype=torch.float32)    

        train_dataset = TensorDataset(xTrain, yTrain)

        batch_size = 32  # Choose an appropriate batch size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        src_vocab_size = int(unique)
        tgt_vocab_size = 1
        d_model = d_model
        num_heads = num_heads
        num_layers = num_layers
        d_ff = d_ff
        max_seq_length = X.shape[1]
        dropout = dropout

        transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
        transformer.apply(initialize_attention_weights)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)

        for epoch in range(100):
            transformer.train()
            epoch_loss = 0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                estimations = transformer(batch_x)
                loss = criterion(estimations, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            loss_values.append(avg_epoch_loss)
            print(f"Epoch: {epoch + 1}, Loss: {avg_epoch_loss}")

        torch.save(transformer, "transformerDF.pth")
        print(f"transformer trained and saved")
        return transformer
        
        transformer.eval()
        print(f"transformer in eval mode")
        with torch.no_grad():
            prediction = transformer(xTest)

        prediction_np = prediction.numpy()
        yTest_np = yTest.numpy()
        accuracy, _ = pearsonr(prediction_np, yTest_np)
        accuracies.append(accuracy)

    result = sum(accuracies) / len(accuracies)
    print("Final Accuracy is", result)
    return result
    

    loss_csv_file = "loss_values.csv"
    with open(loss_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Loss"])  # Add headers
        for i, loss in enumerate(loss_values):
            writer.writerow([i + 1, loss])

    print(f"Loss values saved to {loss_csv_file}")
