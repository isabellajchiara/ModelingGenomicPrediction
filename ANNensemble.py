#Ensemble across environments

bigX = data.drop(['taxa','SY','DM','SW','Unnamed: 0','Seq_ID'],axis=1)
bigY = data['DM','env'] #response variable only

xTrain, xTest, yTrain, yTest = train_test_split(bigX, bigY, test_size=0.2,shuffle=True)


modelPerformance = []
modelPredictions = []
environments = pheno['envs'].unique().tolist()

#cycle through all environments for training
#predict test set for every environment and collect predictions 

for env in environments:

  location = bigX[bigX['env'] == env] #pull just one state
  x = location.drop(['env'], axis=1) #drop env and leave only genotypes

  y = bigY[bigY['env'] == env]
  y = y.drop(['env'])

  imp = SimpleImputer(missing_values=np.nan, strategy='mean')
  imp.fit(x)
  x = pd.DataFrame(imp.transform(x))
  xTrain, xTest, _, _ = train_test_split( x, y, test_size=0.3) #split to train/test

  model.fit(
    xTrain, yTrain,
    batch_size=batchSize,
    epochs=nEpoch,
    verbose=0,
    validation_data=(xTrain, yTrain),
    callbacks=[early_stopping])

    predictions = pd.DataFrame(model.predict(xTest))

    truth = yTest
    perf = predictions.corrwith(truth)

    modelPerformance.append(perf)
    modelPredictions.append(predictions)

allPredictions = pd.DataFrame(np.concatenate(modelPredictions)) #all envt predictions = one column
allPredictions['mean'] = df.mean(axis=1) #get rowmeans to arrive at final prediction
ensemblePred = allPredictions['mean']
finalPerf = ensemblePred.corrwith(yTest) #how does average prediction across envs compare to truth?

