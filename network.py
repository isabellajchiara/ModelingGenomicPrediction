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
    def __init__(self, inputSize, outputSize, numLayers, hiddenSize, numHeads, dropout):
        super(BayesianTransformerModel, self).__init__()
        
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize = (nSNPs+nTraits)/2
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.dropoutProb = dropout

        # encoder layers
        self.encoderLayers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=self.inputSize, nhead=self.numHeads, dim_feedforward=self.hiddenSize, dropout=self.dropoutProb)
            for _ in range(self.numLayers)
        ])
        self.encoder = nn.TransformerEncoder(self.encoderLayers, numLayers=self.numLayers)

        # embed environment 

        nEnv = 270  #
        embeddingDim = 3
        embedding = nn.Embedding(vocab_size, embedding_dim)
        inputSequence = torch.tensor([geno1, geno2,...,genoN])
        embeddedSequence = embedding(inputSequence)


        # feedforward portion 
        self.bayesian_feedforward = nn.Linear(hiddenSize, hiddenSize)

        # output layer
        self.outputLayer = nn.Linear(self.outputDim, self.outputDim)
        
    def forward(self, inputData):
        
        encoderOutput = self.encoder(inputData)

        # Generate prior distribution for decoder input
        priorMean = torch.zeros_like(encoderOutput)
        priorStd = torch.ones_like(encoderOtput)
        priorDistribution = Normal(priorMean, priorStd)
        
        # Sample from the prior distribution
        input = priorDistribution.rsample()


        embedded = self.embedding(x)
        encoded = self.encoder_layer(embedded)
        
        # Bayesian feedforward
        bayesian_encoded = self.bayesian_feedforward(encoded)
        
        # Output layer
        output = self.output_layer(bayesian_encoded)
        return F.log_softmax(output, dim=-1)
        


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

