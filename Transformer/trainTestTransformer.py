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

model, accuracy = trainTest5Fold(X,y)

