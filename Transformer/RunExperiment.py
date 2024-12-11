''' Load dependencies '''
exec(open("dependencies.py").read())
exec(open("transformerBlocks.py").read())
exec(open("transformerBuild.py").read())

''' Load data '''
data = pd.read_csv("DF_Data.csv")

''' X Y split '''
X = data.drop(['Unnamed: 0', 'IDS', 'Days to flowering'], axis=1)
y = data['Days to flowering']

''' find the vocab size '''
stacked = X.stack().unique()
unique = stacked.shape[0]
weights = getWeights(data)

''' define params to test'''
d_modelList = [100, 300, 500]
num_headsList = [1, 2, 5]
num_layersList = [2, 3, 5]
d_ffList = [100, 300, 500]
dropoutList = [0.01, 0.05, 0.1]
lrList = [0.001, 0.01, 0.1]

'''test and select params'''
param_grid = list(itertools.product(d_modelList, num_headsList, num_layersList, d_ffList, dropoutList, lrList))
params = optimizeTransformer(params)
params

d_model = d_model
num_heads = num_heads
num_layers = num_layers
d_ff=  d_ff
dropout = dropout
lr = lr

'''train and test optimized model using 5 fold cv'''
model, accuracy = trainTest5Fold(X,y)

