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
from sklearn.model_selection import train_test_split

pheno = pd.read_csv("SWphenoData.csv") #load data
pheno['env'] = pheno['env'].str[0:2] #removes years to just look at locs
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
featureNames = list(geno.columns)
geno.columns = featureNames

pca = PCA(n_components=3000,svd_solver='full')
model = pca.fit(geno)
genoReduced = model.transform(geno)
genoReduced  = pd.DataFrame(genoReduced)
n_pcs= model.components_.shape[0]

most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]
most_important_names = [featureNames[most_important[i]] for i in range(n_pcs)]
subset = list(most_important_names)
geno = geno[geno.columns.intersection(subset)]
geno = pd.DataFrame(geno)
entryNames = pd.DataFrame(entryNames)
geno.reset_index(drop=True,inplace=True)
entryNames.reset_index(drop=True,inplace=True)


geno = pd.concat([entryNames,geno],axis=1)
geno.to_csv("preprocessedGenoData.csv")
print("labeled geno data")


envs = pheno["env"].unique().tolist()

fullDataset= list()

for env in envs:
   location = pheno[pheno['env']==env] #pull just one state
   location = pd.DataFrame(location)
   if len(location.index) > 30:
      location=location.rename(columns = {'Seq_ID':'taxa'})
      final = pd.merge(location,geno, on=['taxa'],how='inner')
      final.drop_duplicates(subset=['taxa'], keep='first', inplace=True, ignore_index=True)
      fullDataset.append(final)
      y = final['SW']
      x = final.drop(['Unnamed: 0','env','taxa','DF'],axis=1)
      xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.33, random_state=42)
      xTrainAll.append(xTrain)
      yTrainAll.append(yTrain)
      xEvalAll.append(xTest)
      yEvalAll.append(yTest)


fullDataset = np.concatenate(fullDataset)
fullDataset = pd.DataFrame(fullDataset)
fullDataset.to_csv("fullDatasetSW.csv")
