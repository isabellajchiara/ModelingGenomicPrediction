import pandas as pd
import numpy as np
import os
import io
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn import decomposition, datasets
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

pheno = pd.read_csv("SYphenoData.csv") #load data
#envs = pheno.groupby("env") #list containing one array for each env

pheno['env'] = pheno['env'].str[0:2] #removes years to just look at locs
#location = pheno[pheno['env'].str.contains("NE")] #pul
val = 1
chr = 1
genotypes = []

os.chdir('/home/ich/projects/def-haricots/ich/CDBN/genotypes')

for file in os.listdir('/home/ich/projects/def-haricots/ich/CDBN/genotypes'):
    file_name, file_ext = os.path.splitext(file)
    with open(f'chr{chr}.txt', 'r') as geno:
      geno = geno.read()
      array = pd.read_csv(io.StringIO(geno), sep='\s+')
      chr +=1
      if chr == 12:
          break
    genotypes.append(array)
    val += 1

os.chdir('/home/ich/projects/def-haricots/ich/CDBN')

print("loaded geno data")

geno = pd.concat(genotypes) #huge DF containing all SNPs for all genotypes
geno['taxa'] = geno['taxa'].str[0:8] #make geno entry names match pheno entry n>
entryNames = geno['taxa']
geno = geno.drop(['taxa'],axis=1)


imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp.fit(geno)
geno = pd.DataFrame(imp.transform(geno))

selector=VarianceThreshold(threshold=0.15)
selector.fit_transform(geno)

geno = pd.DataFrame(geno)
entryNames = pd.DataFrame(entryNames)
geno.reset_index(drop=True,inplace=True)
entryNames.reset_index(drop=True,inplace=True)
#genoPloidy = np.stack([genotypes], axis=2) #for more complex input in the futu>
#params = tf.convert_to_tensor(genoPloidy) #for more complex input in the future
geno = pd.concat([entryNames,geno],axis=1)

geno.to_csv("preprocessedGenoData.csv")

print("labeled geno data")

envs = pheno["env"].unique().tolist()


xTrainAll = list()
yTrainAll = list()

xEvalAll = list()
yEvalAll = list()

for env in envs:
   location = pheno[pheno['env']==env] #pull just one state
   location = pd.DataFrame(location)
   if len(location.index) > 30:
      location=location.rename(columns = {'Seq_ID':'taxa'})
      final = pd.merge(location,geno, on=['taxa'],how='inner')
      final.drop_duplicates(subset=['taxa'], keep='first', inplace=True, ignore>
      y = final['SY']
      x = final.drop(['Unnamed: 0','env','taxa','SY'])
      xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.33, ran>
      xTrainAll.append(xTrain)
      yTrainAll.append(yTrain)
      xEvalAll.append(xTest)
      yEvalAll.append(yTest)

xTrainAll = np.concatenate(xTrainAll)
yTrainAll = np.concatenate(yTainAll)
xEvalAll = np.concatenate(xEvalAll)
yEvalAll = np.concatenate(yEvalAll)

xTrainAll = pd.DataFrame(xTrainAll)
yTrainAll = pd.DataFrame(yTrainAll)
xEvalAll = pd.DataFrame(xEvalAll)
yEvalAll = pd.DataFrame(yEvalAll)

xTrainAll.to_csv("xTrainSY.csv")
yTrainAll.to_csv("yTrainSY.csv")
xEvalAll.to_csv("xTestSY.csv")
yEvalAll.to_csv("yTestSY.csv")





