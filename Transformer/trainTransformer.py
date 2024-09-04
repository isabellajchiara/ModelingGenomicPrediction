exec(open("dependencies.py").read())
exec(open("transformerBlocks.py").read())
exec(open("transformerBuild.py").read())

data = pd.read_csv("SYdataTF.csv")

testing = data[data["1"]== "ID"]
training = data[data["1"] != "ID"]

testing = testing.drop(["1"],axis=1) #remove loc column
training = training.drop(["1"],axis=1)

nSNPs = len(data.columns) -3


xTrain = training.drop(["0","Unnamed: 0"], axis=1)
xTest = testing.drop(["0","Unnamed: 0"],axis=1)
xTrain = xTrain.replace({0.0:-1,int(1.0):0,2.0:int(1)})
xTest = xTest.replace({0.0:-1,int(1.0):0,2.0:int(1)})

yTrain = training["0"]
yTest = testing["0"]


xTrain = np.array(xTrain)
yTrain = np.array(yTrain)
xTest = np.array(xTest)
yTest = np.array(yTest)

xTrain = torch.from_numpy(xTrain)
xTest = torch.from_numpy(xTest)
yTrain = torch.from_numpy(yTrain)
yTest = torch.from_numpy(yTest)

src_vocab_size = nSNPs
tgt_vocab_size = 1
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 20
max_seq_length = 100
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# Convert src_data and tgt_data to LongTensor
src_data = xTrain.long() 
tgt_data = yTrain.unsqueeze(1).long() 
src_data = torch.clamp(xTrain.long(), 0, src_vocab_size - 1) # Clamp values to be within vocabulary range
tgt_data = torch.clamp(yTrain.unsqueeze(1).long(), 0, tgt_vocab_size - 1)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.1)

transformer.train()

for epoch in range(100):
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data)
    # Make sure tgt_data is of the correct shape and type
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 0].contiguous().view(-1).float()) 
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

print("Training Complete")

torch.save(transformer,"transformer.pth")

transformer.eval()

xTest = torch.clamp(xTest.long(), 0, src_vocab_size - 1) # Clamp values to be within vocabulary range
yTest = torch.clamp(yTest.unsqueeze(1).long(), 0, tgt_vocab_size - 1)

with torch.no_grad():

    val_output = transformer(xTest, yTest)
    val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), yTest.contiguous().view(-1))
    print(f"Validation Loss: {val_loss.item()}")
