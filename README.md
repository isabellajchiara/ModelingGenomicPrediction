These models are intended to perform genomic prediction on dense SNP data.

Data for these experiments can be found in genoData and phenoData folders. 

createTrainTest.py will transform the geno and pheno folders into a format suitable for modeling.

Three models are tested:
- Artificial neural networks (single and ensemble). These models can be found in the ANN folder. 
- Transformer encoder model.  This model can be found in the Transformer folder.
- PDFN adapted from Ubbens, 2023. This model can be found in the adaptPFN folder. 

The data used for these experiments comes from the Cooperative Dry Bean Nursery. 
metadata.csv contains more detail about each of the lines genotyped
and phenotyped in the genoData and phenoData files. 
