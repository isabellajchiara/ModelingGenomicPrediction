device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import math
from torch.distributions.normal import Normal

genoTrain, genoTest, phenoTrian, phenoTest = train_test_split(Geno, pheno, train_size=0.7, shuffle=True)

inputSize = nSNPs
outputSize = nTraits


# define your Bayesian Transformer model
class BayesianTransformerModel(nn.Module):
    def __init__(self, inputSize, outputSize, numLayers=6, hiddenSize=((nSNPs+nTraits)/2), numHeads=8, dropout=0.1):
        super(BayesianTransformerModel, self).__init__()
        self.input_size = inputSize
        self.output_size = outputSize
        self.hidden_size = hiddenSize
        self.embedding = nn.Linear(inputSize, hiddenSize) # store embeddings
        self.pos_encoding = PositionalEncoding(hidden_size, dropout=dropout) #accounts for marker loc
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads) #self attention and feedforward
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) #stack of encoders
        self.mean_decoder = nn.Linear(hidden_size, output_size)
        self.log_var_decoder = nn.Linear(hidden_size, output_size)
        
    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.hidden_size)
        src = self.pos_encoding(src)
        output = self.transformer_encoder(src)
        mean_output = self.mean_decoder(output)
        log_var_output = self.log_var_decoder(output)
        return mean_output, log_var_output

# positional encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# parameters
num_layers = 6
hidden_size = 512
num_heads = 8
dropout = 0.1
lr = 0.001
batch_size = 32
num_epochs = 10

# create model, optimizer, loss function, and data loader
model = BayesianTransformerModel(input_size, output_size, num_layers, hidden_size, num_heads, dropout)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
dataset = genoTrain, phenoTrain, num_samples=100)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        mean_output, log_var_output = model(inputs)
        # Using the log variance as a measure of uncertainty
        loss = criterion(mean_output, targets) + torch.mean(0.5 * torch.exp(log_var_output))  # Total loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")



# define a function to make predictions
def predict(model, data):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(data).float()  # Assuming data is a numpy array or a list
        outputs = model(inputs)
        predictions = outputs.cpu().numpy()  # Convert predictions to numpy array
    return predictions

# test
new_data = genoTest  # Example new input data
predictions = predict(model, genoTest)
print(predictions)

