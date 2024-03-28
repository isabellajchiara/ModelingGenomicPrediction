def center_markers(m):
    m = m - m.min()
    m = m / m.max()
    return m - 0.5
  
def normalize(y):
    if len(y.shape) > 0:
        return (y - y.mean()) / y.std()
    else:
        return y

def preProcessGeno(geno):
  imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
  imp.fit(geno)
  geno = pd.DataFrame(imp.transform(geno))
  pca = PCA(n_components=25)
  pca.fit(geno).fit_transform(geno)
  geno = pd.DataFrame(geno)
  print("processed geno data")
  return geno
      

pheno = pd.read_csv("SWphenoData.csv") #load data
#envs = pheno.groupby("env") #list containing one array for each env
  
pheno['env'] = pheno['env'].str[0:2] #removes years to just look at locs
#location = pheno[pheno['env'].str.contains("NE")]

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
geno['taxa'] = geno['taxa'].str[0:8] #make geno entry names match pheno entry names 
entryNames = geno['taxa']
geno = geno.drop(['taxa'],axis=1)
  
geno = preProcessGeno(geno)

#re attach entry names for aligning with corresponding phenotypes 
entryNames = pd.DataFrame(entryNames)
geno.reset_index(drop=True,inplace=True)
entryNames.reset_index(drop=True,inplace=True)
#genoPloidy = np.stack([genotypes], axis=2) #for more complex input in the future
#params = tf.convert_to_tensor(genoPloidy) #for more complex input in the future
geno = pd.concat([entryNames,geno],axis=1)
  
print("labeled geno data")

def priorDescribe(pheno,geno):
  
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
        final.drop_duplicates(subset=['taxa'], keep='first', inplace=True, ignore_index = True)
        y = final['SW']
        x = final.drop(['Unnamed: 0','env','taxa','SW'])
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.33, random_state=42)
        xTrainAll.append(xTrain)
        yTrainAll.append(yTrain)
        xEvalAll.append(xTest)
        yEvalAll.append(yTest)
  
  
  xTrainAll = np.concatenate(xTrainAll)
  yTrainAll = np.concatenate(yTainAll)
  xEvalAll = np.concatenate(xEvalAll)
  yEvalAll = np.concatenate(yEvalAll)
  
  xs = np.concatenate((xTrainAll, xEvalAll), axis=0).astype(np.float32)
  ys = np.concatenate((yTrainAll, yEvalAll)).astype(np.float32)

  
  
  xs = center_markers(xs)
  ys = normalize(ys)
  
  xs = pd.DataFrame(xs)
  ys = pd.FataFrame(ys)
  
  return xs, ys

