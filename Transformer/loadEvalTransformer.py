#First, redefine model shape 

data = pd.read_csv("tokenTrainingSY.csv")

X = data.drop(['Unnamed: 0','2','3'], axis=1)
y = data['3']

nSNPs = X.shape[1] 

src_vocab_size = nSNPs
tgt_vocab_size = 1
d_model = 100
num_heads = 2
num_layers = 3
d_ff = 200
max_seq_length = 100
dropout = 0.1

exec(open("dependencies.py").read())
exec(open("transformerBlocks.py").read())
exec(open("transformerBuild.py").read())

#create model
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

#load the pretrained weights 
transformer = torch.load("transformer.pth")

#put in eval mode
transformer.eval()

#test data must be same format as train data
Xtensor= torch.tensor(X.values)
ytensor = torch.tensor(y.values)

with torch.no_grad():
    prediction = transformer(Xtensor, ytensor)

prediction = prediction[:,0,0]

accuracy = pearsonr(prediction,y)


