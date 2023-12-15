device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



genoTrain, genoTest, phenoTrian, phenoTest = train_test_split(Geno, pheno, train_size=0.7, shuffle=True)


class GenoModel(nn.Model):

    def __init__(self):
        super(GenoModel, self).__init__()

        self.conv1 = torch.nn.Conv3d(nSNP,1,100) #input channel #output channel, 100X100 convolution )
        self.linear1 = torch.nn.Linear(nSNP, nInd,Ploidy)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(nSNP, nInd,Ploidy)
        self.activation = torch.nn.Activation()

    def forward(self, x):

        x = F.max_pool3d(F.relu(self.conv1(x)), (nSNP, nPloidy))
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        return x

GenoModel = GenoModel()