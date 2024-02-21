#one file per chromosome

#source: https://github.com/Alice-MacQueen/CDBNgenomics/tree/master/data-raw

# unzip
#path = "./genoData""
#dir = os.listdir(path )

#import os
#ath = "./genoData"

#with zipfile.ZipFile(("chr" + str(11)+".txt.zip"),"r") as zip_ref:
    zip_ref.extractall("/workspaces/ModelingGenomicPrediction/genoData")

# read and turn to DF
with open('Numerical_format_GD_CDBN_001_359_pedigree_fillin_chr1.txt', 'r') as f2:
    data = f2.read()

chr = pd.read_csv(io.StringIO(data), sep='\s+') #produces a DF with one col per locus

