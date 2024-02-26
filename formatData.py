pheno = pd.read_csv("phenoEnvironmentDataClean.csv") #load data
envs = pheno.groupby("env") #list containing one array for each env

location = pheno[pheno['env'].str.contains("NE")]

genotypes = []
chr = 1
while chr < 12 :
  for file in os.listdir(f'/projects/def-haricots/ich/CDBN/genotypes'):
    with open(f'chr{chr}.txt', 'r') as geno:
      geno = geno.read()
      geno = pd.read_csv(io.StringIO(geno), sep='\s+') #produces a DF with one col per locus
    genotypes.append(geno)
    chr +=1 
geno = pd.concat(genotypes)

#genoPloidy = np.stack([genotypes], axis=2) #for more complex input in the future
#params = tf.convert_to_tensor(genoPloidy) #for more complex input in the future

data = location.merge(geno, left_on='Seq_ID', right_on='taxa') #merge data to pull genotypes present in a given envt


# separate X and Y
X = data.drop(['taxa','SY','DM','SW','Unnamed: 0','env','Seq_ID'],axis=1) #genotypes only
Y = data['DM'] #response variable only

print(Y.head())

xTrain, xTest, yTrain, yTest = train_test_split( X, Y, test_size=0.2)


