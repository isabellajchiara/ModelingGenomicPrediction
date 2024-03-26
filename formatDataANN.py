data = pd.read_csv("finalCleanedDataSY.csv")
Y = data['taxa']
X = data.drop(['Unnamed: 0','env','taxa','SY'])

for i in X.columns:
  try: 
    X[[i]] = X[[i]].astype(float).astype(int)
  except:
    pass
  

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.3)
xTest,xValid,yTest,yValid = train_test_split(xTest, yTest, test_size=0.5,shuffle=True)
