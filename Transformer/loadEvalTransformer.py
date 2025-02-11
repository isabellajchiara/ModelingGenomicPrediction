
exec(open("dependencies.py").read())
exec(open("transformerBlocks_Test15.py").read())

data = pd.read_csv("fullDatasetSY_blups.csv")

'''
isolate X and Y
remove any low variance features
they will not contribute to the model
'''
X = data.drop(["Unnamed: 0","X2","X3"],axis=1)
threshold = 0.01
X = X.drop(X.std()[X.std() < threshold].index.values, axis=1)

'''
identify number of unique tokens
'''
unique = X.stack().nunique()
y = data["X3"]
X = torch.tensor(X.values, dtype=torch.long)
y = torch.tensor(y.values, dtype=torch.float32)
IDS = data["X2"]

'''load weights
weights were previouly calculated
from RRBLUP
'''
weights = pd.read_csv("SY_Effects_Blup.csv")
weights = weights['0']
feature_weights = torch.tensor(weights, dtype = torch.float32)

# Find vocab size
unique = X.stack().nunique()

# Convert to tensors
X = torch.tensor(X.values, dtype=torch.long)
y = torch.tensor(y.values, dtype=torch.float32)

test_dataset = TensorDataset(X, y)
test_loader = DataLoader(test_dataset, shuffle=False,drop_last=False)

src_vocab_size = int(unique)
tgt_vocab_size = 1
d_model = 200
num_heads = 5
num_layers = 2
d_ff = 100
max_seq_length = X.shape[1]
dropout = 0.2


#create model
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout,feature_weights)

#load the pretrained weights 
transformer = torch.load("transformerSY_Test_fold3.pth")

#put in eval mode
transformer.eval()

# Evaluate on test set

transformer.eval()
with torch.no_grad():
    for batch in test_loader:
        preds = transformer(X)

    
test_accuracy, _ = pearsonr(preds, y)

pred = pd.DataFrame(preds)
true = pd.DataFrame(y)
IDs = 

test_accuracy

