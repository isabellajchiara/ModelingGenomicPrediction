data = pd.read_csv("fullDatasetDM.csv")
data.columns = data.columns.str.strip()

envs = data["1"].unique()
data = data.drop(list(data)[0:2], axis=1) #remove useless columns

X = data.drop(["Unnamed: 0", "0","1","2","3"], axis=1)
X = X.replace({0.0:-1,int(1.0):0,2.0:int(1)})

X_columns = X.columns

y = np.array(data["3"])

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.33, shuffle=True)

xTrain = np.array(xTrain)
yTrain = np.array(yTrain)
xTest = np.array(xTest)
yTest = np.array(yTest)

folds = KFold(n_splits=5,shuffle=True)
params ={'alpha':[0.001,0.1,10,50,100]}

model = Ridge()

model_cv  = GridSearchCV(estimator=model,
                         param_grid=params,
                         scoring='neg_mean_squared_error',
                         cv=folds,
                         return_train_score=True,
                         verbose=1)
model_cv.fit(xTrain,yTrain)

print(model_cv.best_params_)
