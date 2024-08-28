exec(open("dependencies.py").read())
exec(open("transformerBlocks.py").read())
exec(open("transformerBuild.py").read())

data = pd.read_csv("SYdataTF.csv")
data = data.drop(list(data)[0:1], axis=1) #remove useless columns

nSNPs = len(data.columns) - 1
print("there are",len(data["1"].unique()), "locations")
print("there are", len(data["2"].unique()),"genotypes")
print("there are",(len(data.columns) -2), "SNP markers")
print("there are",len(data),"total observations across all environments")

X = data.drop(["1","2","3"], axis=1)
X = X.replace({0.0:-1,int(1.0):0,2.0:int(1)})

X = np.array(data.drop(["0"], axis=1))
y = np.array(data["0"])

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.33, shuffle=True)

xTrain = np.array(xTrain)
yTrain = np.array(yTrain)
xTest = np.array(xTest)
yTest = np.array(yTest)



dataset = CDBNDataset(X, y)
datasetTest = CDBNDataset(xTest,yTest)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1,drop_last=True)
testloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1,drop_last=True)

geno = data.drop(['0'],axis=1)


xTrain = torch.from_numpy(xTrain)
xTest = torch.from_numpy(xTest)
yTrain = torch.from_numpy(yTrain)
yTest = torch.from_numpy(yTest)

src_vocab_size = nSNPs
tgt_vocab_size = 1
d_model = 3
num_heads = 1
num_layers = 6
d_ff = 20
max_seq_length = 100
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# Generate random sample data
src_data = xTrain #must reshape to (1,nSNPs,(batch size, max_seq_lenght))
src_data = src_data.long()
tgt_data = yTrain.unsqueeze(1) # 1,1,(batch size, max_seq_lenght))


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.1)

transformer.train()

for epoch in range(100):
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data)
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

print("Training Complete")

torch.save(transformer)
