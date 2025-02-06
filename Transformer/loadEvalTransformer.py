
exec(open("dependencies.py").read())
exec(open("transformerBlocks_Test15.py").read())

data = pd.read_csv("fileredTrainingSY.csv")

X = data.drop(['Unnamed: 0', "2", "3"], axis=1)
y = data['3']

stacked = X.stack().unique()
unique = stacked.shape[0]

#load weights
weights = pd.read_csv("effectsSY.csv")
weights = weights['0']
feature_weights = torch.tensor(weights, dtype = torch.float32)

# Find vocab size
unique = X.stack().nunique()

# Convert to tensors
X = torch.tensor(X.values, dtype=torch.long)
y = torch.tensor(y.values, dtype=torch.float32)

test_dataset = TensorDataset(X, y)
batch_size = 200
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


src_vocab_size = int(unique)
tgt_vocab_size = 1
d_model = 100
num_heads = 5
num_layers = 2
d_ff = 50
max_seq_length = xTrain.shape[1]
dropout = 0.2


#create model
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout,feature_weights)

#load the pretrained weights 
transformer = torch.load("transformerSY_Test.pth")

#put in eval mode
transformer.eval()



# Evaluate on test set


transformer.eval()
with torch.no_grad():
    preds = transformer(X

    
test_accuracy, _ = pearsonr(preds, y)

test_accuracy

