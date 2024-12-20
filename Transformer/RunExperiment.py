import csv
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch
import pandas as pd
from scipy.stats import pearsonr

# Load dependencies
exec(open("dependencies.py").read())
exec(open("transformerBlocks.py").read())
exec(open("transformerBuild.py").read())

# Load data
data = pd.read_csv("DF_Data.csv")

X = data.drop(['Unnamed: 0', 'IDS', 'Days to flowering'], axis=1)
y = data['Days to flowering']

# Find vocab size
stacked = X.stack().unique()
unique = stacked.shape[0]

accuracies = []  # Store accuracies for each k-fold
kf = KFold(n_splits=5, shuffle=True, random_state=100)

loss_values = []  # Store training loss across all epochs

# Early stopping setup
early_stopping_patience = 10  # Stop training after 10 epochs of no improvement
early_stopping_min_delta = 0.001

# 5-fold cross-validation
for train_index, test_index in kf.split(X):
    xTrain, xTest = X.iloc[train_index], X.iloc[test_index]
    yTrain, yTest = y.iloc[train_index], y.iloc[test_index]

    xTrain = torch.tensor(xTrain.values, dtype=torch.int64)
    yTrain = torch.tensor(yTrain.values, dtype=torch.float32)
    xTest = torch.tensor(xTest.values, dtype=torch.int64)
    yTest = torch.tensor(yTest.values, dtype=torch.float32)

    train_dataset = TensorDataset(xTrain, yTrain)

    batch_size = 5
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Transformer parameters
    src_vocab_size = int(unique)
    tgt_vocab_size = 1
    d_model = 300
    num_heads = 2
    num_layers = 1
    d_ff = 50
    max_seq_length = X.shape[1]
    dropout = 0.01

    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    transformer.apply(initialize_attention_weights)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)

    for epoch in range(100):
        transformer.train()
        train_loss = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            estimations = transformer(batch_x)
            loss = criterion(estimations, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        loss_values.append(avg_train_loss)

    torch.save(transformer, "transformerDF.pth")

    transformer.eval()
    with torch.no_grad():
        prediction = transformer(xTest)

    prediction_np = prediction.numpy()
    yTest_np = yTest.numpy()
    accuracy, _ = pearsonr(prediction_np, yTest_np)
    accuracies.append(accuracy)
    print(f"Fold Accuracy: {accuracy:.4f}")

result = sum(accuracies) / len(accuracies)
print(f"Final Mean Accuracy: {result:.4f}")

loss_csv_file = f"loss_values_optim1.csv"
with open(loss_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Loss"])
    for i, loss in enumerate(loss_values):
        writer.writerow([i + 1, loss])

print(f"Loss values saved to {loss_csv_file}")
Results_df = pd.DataFrame({"Mean Accuracy": [result]})
Results_df.to_csv("DF_Optim1_result.csv", index=False)
