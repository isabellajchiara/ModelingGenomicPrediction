
exec(open("dependencies.py").read())
exec(open("transformerBlocks.py").read())
exec(open("transformerBuild.py").read())
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("SY_Data.csv")

X = data.drop(['Unnamed: 0','IDS','Yield'], axis=1)
y = data['Yield']

# Scale and center the response variable
scaler = StandardScaler()
y = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
y = pd.DataFrame(y)

stacked = X.stack().unique()
unique = stacked.shape[0]

src_vocab_size = int(unique)
tgt_vocab_size = 1
d_model = 400
num_heads = 2
num_layers = 5
d_ff = 50
max_seq_length = X.shape[1]
dropout = 0.05


#create model
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

#load the pretrained weights 
transformer = torch.load("transformerSY_Final.pth")

#put in eval mode
transformer.eval()

#test data must be same format as train data
Xtensor= torch.tensor(X.values)
ytensor = torch.tensor(y.values)

with torch.no_grad():
    prediction = transformer(Xtensor)

accuracy = pearsonr(prediction,y)
accuracy


