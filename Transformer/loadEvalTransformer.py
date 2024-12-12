
exec(open("dependencies.py").read())
exec(open("transformerBlocks.py").read())
exec(open("transformerBuild.py").read())

data = pd.read_csv("DF_Data.csv")

X = data.drop(['Unnamed: 0','IDS','Days to flowering'], axis=1)
y = data['Days to flowering']

stacked = X.stack().unique()
unique = stacked.shape[0]

src_vocab_size = int(unique)
tgt_vocab_size = 1
d_model = 250
num_heads = 1
num_layers = 2
d_ff = 100
max_seq_length = X.shape[1]
dropout = 0.05


#create model
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

#load the pretrained weights 
transformer = torch.load("transformerDF.pth")

#put in eval mode
transformer.eval()

#test data must be same format as train data
Xtensor= torch.tensor(X.values)
ytensor = torch.tensor(y.values)

with torch.no_grad():
    prediction = transformer(Xtensor)

accuracy = pearsonr(prediction,y)


