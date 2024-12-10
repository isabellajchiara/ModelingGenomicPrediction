data = pd.read_csv("2021_Training_Set_Tokenized.csv")
data = data.drop(['Unnamed: 0'], axis=1)


def createKmers(genotypes, nK):
  kMers = []
  starts = [i * nK for i in range(genotypes.shape[1] // nK)]  # Starting positions for each k-mer

  for seq in range(genotypes.shape[0]):  # Iterate over each sequence (row)
        geno = genotypes.iloc[seq, :]  # Take each genotype sequence
        for i in starts:
            K = tuple(geno.iloc[i:i + nK])  # Extract k-mer
            if K not in kMers:  #  add seq if not alread collected
                kMers.append(K)
  return kMers

def tokenize(genotypes, nK):
    val_list = kMers # Existing k-mer list
    newGenotypes = {}

    for seq in range(genotypes.shape[0]):  # Iterate over sequences
        geno = genotypes.iloc[seq, :]  # Take genotype sequence (row)
        tokenSeq = []  # List to hold tokenized sequence

        # Iterate over genotype in steps of nK to form k-mers
        for i in range(0, len(geno) - nK + 1, nK):  # Step by nK to get non-overlapping k-mers
            seg = tuple(geno.iloc[i:i + nK])  # Extract k-mer as tuple
            if seg in val_list:  # Check if k-mer is in list
                tokenSeq.append(val_list.index(seg))  # Append tokenized k-mer to new geno seq


        newGenotypes[seq] = tokenSeq  # Store the tokenized sequence

    # Convert the dictionary to a DataFrame
    newGenotypes = pd.DataFrame.from_dict(newGenotypes, orient='index')


    return newGenotypes


''' tokenize and save dataset'''

data = pd.read_csv("2021_Training_Set_Tokenized.csv")
data = data.drop(['Unnamed: 0'], axis=1)

def createTokenizedSet(data,traitName):

  tokenTrainings = {}

  '''separate genotype data from phenotype data'''

  genotypes = data.drop(['IDS',"100 Seed weight","Plot Weight","Moisture","Yield","Days to flowering"], axis=1)
  genotypes = genotypes.astype(int)

  '''determine best alpha value for RR'''

  folds = KFold(n_splits=5,shuffle=True)
  params ={'alpha':[10,100,1000,5000]}
  model = Ridge()
  model_cv  = GridSearchCV(estimator=model,
                         param_grid=params,
                         scoring='neg_mean_squared_error',
                         cv=folds,
                         return_train_score=True,
                         verbose=1)
  xTrain, xTest, yTrain, yTest = train_test_split(genotypes, data[trait], test_size=0.33, shuffle=True)
  model_cv.fit(xTrain,yTrain)

  alphaVal = model_cv.best_params_['alpha']


  '''test different K mer sizes on RR accuracy using 5 fold CV'''

  kAccuracies = {} #Store accuracies for eac k-mer size
  kf = KFold(n_splits=5, shuffle=True, random_state=100) #cross validate to evaluate k values
  allEffects = {}

  for kVal in [2,3,4,5,6,7,8,9,10]:
      kMers = createKmers(genotypes, kVal) #create list of kmers
      newGenotypes = tokenize(genotypes, kVal) #tokenize genotypes

      tokenTrainingSet = pd.concat([data["IDS"], data[trait], newGenotypes], axis=1)  # 2 = IDs and 3 = Yield
      tokenTrainings[kVal] = tokenTrainingSet

      X = tokenTrainingSet.drop(['IDS', trait], axis=1)
      y = tokenTrainingSet[trait]

      accuracies = []  # To store accuracy for each fold
      effects = []

      # 5-fold cross-validation
      for train_index, test_index in kf.split(X):
          xTrain, xTest = X.iloc[train_index], X.iloc[test_index]
          yTrain, yTest = y.iloc[train_index], y.iloc[test_index]

          LR = linear_model.Ridge(alphaVal)
          LR.fit(xTrain, yTrain)

          coeffs = LR.coef_
          effects.append(coeffs)

          yPred = LR.predict(xTest)
          yTest = np.array(yTest)

          accuracy = np.corrcoef(yPred, yTest)[0, 1]
          accuracies.append(accuracy)

          aveEffects = pd.DataFrame(np.mean(effects, axis=0))

      allEffects[kVal] = aveEffects
      kAccuracies[kVal] = np.mean(accuracies)  # Pull mean accuracy for this kVal


  '''record the best k mer size and corresponding accuracy for each trait'''
  bestK = max(kAccuracies, key=kAccuracies.get)
  results[trait] = bestK, kAccuracies[bestK]

  '''save effects from best k as information for transformer'''

  bestEffects = allEffects[bestK]
  bestEffects.to_csv(trait + 'effects.csv')
  
  '''save best K dataset for future use'''

  best_tokenTrainingSet = tokenTrainings[bestK]
  best_tokenTrainingSet.to_csv(trait + 'tokenTrainingSet.csv')

  print(f"finished {trait}")

print(results)

resultsDF = pd.DataFrame(results)
resultsDF = resultsDF.transpose()
resultsDF = resultsDF.rename(columns={0: 'Best_kMerSize', 1: 'accuracy'})
resultsDF['Best_kMerSize'] = resultsDF['Best_kMerSize'].astype(int)

resultsDF.to_csv("resultsRR.csv")
return resultsDF


def getWeights(data,traitName):

  tokenTrainings = {}

  '''separate genotype data from phenotype data'''

  genotypes = data.drop(['IDS',"100 Seed weight","Plot Weight","Moisture","Yield","Days to flowering"], axis=1)
  genotypes = genotypes.astype(int)

  '''determine best alpha value for RR'''

  folds = KFold(n_splits=5,shuffle=True)
  params ={'alpha':[10,100,1000,5000]}
  model = Ridge()
  model_cv  = GridSearchCV(estimator=model,
                         param_grid=params,
                         scoring='neg_mean_squared_error',
                         cv=folds,
                         return_train_score=True,
                         verbose=1)
  xTrain, xTest, yTrain, yTest = train_test_split(genotypes, data[trait], test_size=0.33, shuffle=True)
  model_cv.fit(xTrain,yTrain)

  alphaVal = model_cv.best_params_['alpha']


  '''test different K mer sizes on RR accuracy using 5 fold CV'''

  kAccuracies = {} #Store accuracies for eac k-mer size
  kf = KFold(n_splits=5, shuffle=True, random_state=100) #cross validate to evaluate k values
  allEffects = {}

  for kVal in [2,3,4,5,6,7,8,9,10]:
      kMers = createKmers(genotypes, kVal) #create list of kmers
      newGenotypes = tokenize(genotypes, kVal) #tokenize genotypes

      tokenTrainingSet = pd.concat([data["IDS"], data[trait], newGenotypes], axis=1)  # 2 = IDs and 3 = Yield
      tokenTrainings[kVal] = tokenTrainingSet

      X = tokenTrainingSet.drop(['IDS', trait], axis=1)
      y = tokenTrainingSet[trait]

      accuracies = []  # To store accuracy for each fold
      effects = []

      # 5-fold cross-validation
      for train_index, test_index in kf.split(X):
          xTrain, xTest = X.iloc[train_index], X.iloc[test_index]
          yTrain, yTest = y.iloc[train_index], y.iloc[test_index]

          LR = linear_model.Ridge(alphaVal)
          LR.fit(xTrain, yTrain)

          coeffs = LR.coef_
          effects.append(coeffs)

          yPred = LR.predict(xTest)
          yTest = np.array(yTest)

          accuracy = np.corrcoef(yPred, yTest)[0, 1]
          accuracies.append(accuracy)

          aveEffects = pd.DataFrame(np.mean(effects, axis=0))

      allEffects[kVal] = aveEffects
      kAccuracies[kVal] = np.mean(accuracies)  # Pull mean accuracy for this kVal


  '''record the best k mer size and corresponding accuracy for each trait'''
  bestK = max(kAccuracies, key=kAccuracies.get)
  results[trait] = bestK, kAccuracies[bestK]

  '''save effects from best k as information for transformer'''

  bestEffects = allEffects[bestK]
  return bestEffects
  





