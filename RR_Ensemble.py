## BULK RR ##


data = pd.read_csv("fullDatasetDM.csv")
envs = data["1"].unique()
scores = []

#Baseline Performance for each test env with all locations included in ens
for env in envs:
  testData = data[data["1"] == env] #test data = one environment
  x_test = testData.drop(["Unnamed: 0","0","1","2","3"], axis=1) #only geno
  y_test = testData["3"] #only pheno
  trainData = data.drop(data[data['1'] == env].index) #remove test env from training
  X_train = trainData.drop(["Unnamed: 0","0","1","2","3"], axis=1)
  y_train = trainData["3"] #only pheno
  ridge = linear_model.Ridge(alpha=100)
  ridge.fit(X_train, y_train)
  predictions = ridge.predict(x_test)
  predictions = np.array(predictions)
  truth = np.array(y_test)
  accuracy = np.corrcoef(predictions,truth)
  accuracy = accuracy[0,1]
  scores.append(accuracy)



allAccuracies = pd.DataFrame(scores)
envs = pd.DataFrame(envs)
baselines = pd.concat([envs, allAccuracies], axis=1)

simpleRR = baselines.iloc[:,1]

### BULK ENSEMBLE ##


##################
#Ensemble across environments
data = pd.read_csv("fullDatasetDM.csv")
envs = data["1"].unique()

allAccuracies = []
baseline = {}

#Baseline Performance for each test env with all locations included in ens
for env in envs:
  testData = data[data["1"] == env] #test data = one environment
  x_test = testData.drop(["Unnamed: 0","0","1","2","3"], axis=1) #only geno
  y_test = testData["3"] #only pheno
  trainData = data.drop(data[data['1'] == env].index) #remove test env from training
  locations = trainData["1"].unique() #list of all locations in test set
  #
  predictions = pd.DataFrame()


  #Train the ensemble
  for loc in locations:
      ensData = trainData[trainData["1"] == loc] #take one environment for training
      x_Train = ensData.drop(["Unnamed: 0","0","1","2","3"], axis=1) #geno only
      y_Train = ensData["3"] #pheno only
      ridge = linear_model.Ridge(alpha=100)
      ridge.fit(x_Train, y_Train)
      prediction = pd.DataFrame(ridge.predict(x_test)) #prediction on test
      predictions = pd.concat([predictions,prediction],axis=1) #collect predictions from each location

  #Determine accuracy of ensemble
  truth = np.array(y_test) #re-name ground truth
  finalPred = predictions.mean(axis=1) #mean pred across all models in ens
  accuracy = np.corrcoef(truth,finalPred)
  accuracy = accuracy[0,1]
  baseline[env] = accuracy

simpleRRensemble = pd.DataFrame(baseline.values())


## LEAVE ONE OUT OPTIM ##

#Ensemble across environments
data = pd.read_csv("fullDatasetDM.csv")
envs = data["1"].unique()

allLocAccuracies = {}

for env in envs:
  testData = data[data["1"] == env] #test data = one environment
  x_test = testData.drop(["Unnamed: 0","0","1","2","3"], axis=1) #only geno
  y_test = testData["3"] #only pheno
  trainData = data.drop(data[data['1'] == env].index) #remove test env from training
  locations = trainData["1"].unique() #list of all locations in test set
  predictions = pd.DataFrame()
  allAccuracies = []

  #Train the ensemble leaving one environment out
  for loc in locations:
    ensTrainData = trainData.drop(trainData[trainData['1'] == loc].index) #remove test env from training
    trainingLocs = ensTrainData["1"].unique() #pull remaining location names
    for place in trainingLocs: #for remaining location names
      ensData = ensTrainData[ensTrainData["1"] == place] #take one environment for training
      x_Train = ensData.drop(["Unnamed: 0","0","1","2","3"], axis=1) #geno only
      y_Train = ensData["3"] #pheno only
      ridge = linear_model.Ridge(alpha=100)
      ridge.fit(x_Train, y_Train)
      prediction = pd.DataFrame(ridge.predict(x_test)) #prediction on test
      predictions = pd.concat([predictions,prediction],axis=1) #collect predictions from each location

    #Determine accuracy of ensemble without given loc
    truth = np.array(y_test) #re-name ground truth
    finalPred = predictions.mean(axis=1) #mean pred across all models in ens

    accuracy = np.corrcoef(truth,finalPred)
    allAccuracies.append(accuracy[0,1]) #collect acc for every left our location

  locations = pd.DataFrame(locations)
  allAccuracies = pd.DataFrame(allAccuracies)
  allAccuracies = pd.concat([locations,allAccuracies],axis=1)
  allLocAccuracies[env]= allAccuracies

# all loc accuracies contains the 17 ensemble accuracies (one with each environment left out of the ensemble) predicting on each of the test environments

Remove = {}
for env in envs:
  envBaseline = float(baseline[env])
  predictionAccuracy = allLocAccuracies[env]
  predictionAccuracy = pd.DataFrame(predictionAccuracy)
  predictionAccuracy.columns = ['loc','acc']
  remove = predictionAccuracy.loc[(predictionAccuracy['acc']-0.0001) >= envBaseline]
  remove = list(remove['loc'])
  remove.append(env)
  removeDict =remove
  Remove[env]= removeDict

### OPTIMIZED ENSEMBLE ###

#Ensemble across environments
data = pd.read_csv("fullDatasetDM.csv")
envs = data["1"].unique()

allAccuracies = []

#Baseline Performance with all locations included
val = 0
for env in envs:
  testData = data[data["1"] == env] #test data = one environment
  x_test = testData.drop(["Unnamed: 0","0","1","2","3"], axis=1) #only geno
  y_test = testData["3"] #only pheno
  trainData = data.drop(data[data['1'] == env].index) #remove test env from training
  locationsAll = trainData["1"].unique() #list of all locations in test set
  predictions = pd.DataFrame()
  withold = list(Remove[env])
  withold.append(env)
  locations = [item for item in locationsAll if item not in withold]

  #Train the ensemble
  for loc in locations:
      ensData = trainData[trainData["1"] == loc] #take one environment for training
      x_Train = ensData.drop(["Unnamed: 0","0","1","2","3"], axis=1) #geno only
      y_Train = ensData["3"] #pheno only
      ridge = linear_model.Ridge(alpha=100)
      ridge.fit(x_Train, y_Train)
      prediction = pd.DataFrame(ridge.predict(x_test)) #prediction on test
      predictions = pd.concat([predictions,prediction],axis=1) #collect predictions from each location

  #Determine accuracy of ensemble
  truth = np.array(y_test) #re-name ground truth
  finalPred = predictions.mean(axis=1) #mean pred across all models in ens
  accuracy = np.corrcoef(truth,finalPred)
  allAccuracies.append(accuracy[0,1]) #collect acc for env 1

overallAccuracy = mean(allAccuracies)

allAccuracies = pd.DataFrame(allAccuracies)
envs = pd.DataFrame(envs)
optimRRensemble = pd.concat([envs, allAccuracies], axis=1)


### PLOT ###

Result=pd.concat([optimRRensemble,simpleRR,simpleRRensemble],axis=1)
Result.columns= ["Location","optimRREnsemble","RR","simpleRRensemble"]
Result.to_csv("RR_Results_DM.csv")

# Set plot parameters
fig, ax = plt.subplots()
width = 0.4 # width of bar
x = np.arange(len(Result["Location"]))  # Use numerical x-positions for the bars

ax.bar(x, Result["optimRREnsemble"], width, color="blue",label = 'Optimized Ensemble')
ax.bar(x + width, Result["simpleRRensemble"], width, color='goldenrod', label='Bulk Ensemble')
ax.set_ylabel('Prediction Accuracy')
ax.set_xlabel('Locations')
ax.set_title('Ridge Regression Accuracy for Each Location (Seed Weight)')
ax.set_xticks(x + width / 2)  # Set x-tick positions in the middle of the grouped bars
ax.set_xticklabels(Result["Location"])  # Set x-tick labels to location names
ax.set_ylim(top=1.2)
ax.hlines(Result['RR'],xmin=x-0.2,xmax=x+0.6,colors="black",label ="Ridge Regression")
ax.legend()

plt.savefig('SWridgeRegressionNew.png')
