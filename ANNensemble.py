# separate X and Y
X = data.drop(['taxa','SY','Unnamed: 0','Seq_ID'],axis=1) #genotypes only and keep env
Y = data['SY','env'] #response variable only and keep env

sel = VarianceThreshold(threshold=(.91 * (1 - .91)))  #remove cols with variance below 0.91
xGeno = X.drop['env']
xGeno = sel.fit_transform(xGeno)
env = X['env']

X = pd.concat([env,xGeno])

xTrain, xTest, yTrain, yTest = train_test_split( X, Y, test_size=0.3)

modelPerformance = []
modelPredictions = []
environments = pheno['envs'].unique().tolist()

#cycle through all environments for training
#predict test set for every environment and collect predictions 

for env in environments:

  locationGeno = xTrain[xTrain['env'] == env] #pull just one location for training
  xTrain = locationGeno.drop(['taxa','SY','Unnamed: 0','env','Seq_ID'], axis=1) #drop all vars except geno 

  locationPheno = yTrain[yTrain['env'] == env] #pull just one location of phenotypes for training 
  yTrain = y['SY']) #drop all vars except pheno

  model.fit(  #train model on our 1 loc 
    xTrain, yTrain,
    batch_size=batchSize,
    epochs=nEpoch,
    verbose=0,
    validation_data=(xTrain, yTrain),
    callbacks=[early_stopping])

    predictions = pd.DataFrame(model.predict(xTest))#predict on test

    truth = yTest
    perf = predictions.corrwith(truth)

    modelPerformance.append(perf)
    modelPredictions.append(predictions)

allPredictions = pd.DataFrame(np.concatenate(modelPredictions)) #all envt predictions = one column
allPredictions['mean'] = df.mean(axis=1) #get rowmeans to arrive at final prediction
ensemblePred = allPredictions['mean']
finalPerf = ensemblePred.corrwith(yTest) #how does average prediction across envs compare to truth?

