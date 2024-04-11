These models are intended to perform genomic prediction on dense SNP data.

Data for these experiments can be found in genoData and phenoData folders. 

createTrainTest.py will transform the geno and pheno folders into a format suitable for modeling.

Three models are tested:
- An artificial neural network(single and ensemble). These models can be found in the ANN folder. 
- A transformer encoder model.  This model can be found in the Transformer folder.
- A PFN adapted from Ubbens, 2023. This model can be found in the adaptPFN folder. 
