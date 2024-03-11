pheno = pd.read_csv("SYphenoData.csv") #load data
#envs = pheno.groupby("env") #list containing one array for each env

pheno['env'] = pheno['env'].str[0:3] #removes years to just look at locs
#location = pheno[pheno['env'].str.contains("NE")] #pull just one state

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
    genotypes.append(array)
    val += 1

os.chdir('/home/ich/projects/def-haricots/ich/CDBN')
 
geno = pd.concat(genotypes) #huge DF containing all SNPs for all genotypes
geno['taxa'] = geno['taxa'].str[0:8] #make geno entry names match pheno entry names 

from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.91 * (1 - .9)))    
sel.fit_transform(geno)

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(geno)
geno = pd.DataFrame(imp.transform(geno))

#genoPloidy = np.stack([genotypes], axis=2) #for more complex input in the future
#params = tf.convert_to_tensor(genoPloidy) #for more complex input in the future

data = pheno.merge(geno, left_on='Seq_ID', right_on='taxa') #merge data to pull genotypes present in a given envt

data.to_csv("cleanedTrainingSet.csv")

# separate X and Y
X = data.drop(['taxa','SY','Unnamed: 0','env','Seq_ID'],axis=1) #genotypes only
Y = data['SY'] #response variable only


sel = VarianceThreshold(threshold=(.91 * (1 - .91)))  #remove cols with variance below 0.91
X = sel.fit_transform(X)
print(X.shape)



xTrain, xTest, yTrain, yTest = train_test_split( X, Y, test_size=0.3)
xTest,xValid,yTest,yValid = train_test_split(xTest, yTest, test_size=0.5,shuffle=True)



