geno = pd.read_csv("simHaplo.csv")
geno = pd.DataFrame(geno.drop('Unnamed: 0', axis=1))

resp = pd.read_csv("simYieldData.csv")
resp = resp.drop('Unnamed: 0', axis=1)

nLoci = int(len(geno.columns))
nInd = int(len(geno)/2)
ploidy = 2

chr2 = np.empty([nInd, nLoci], dtype=int)
chr1 = np.empty([nInd, nLoci], dtype=int)
x = 1

while x < (len(geno)-2):
    c1 = geno.iloc[x:(x+1),:] # pull row 1
    c2 = geno.iloc[(x+1):(x+2),:] #pull row 2
    chr1[x,]  = c1
    chr2[x,] = c2
    x+=2

genoPloidy = np.stack([chr1,chr2], axis=2)


params = tf.convert_to_tensor(genoPloidy)




