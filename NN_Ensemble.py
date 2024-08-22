warnings.simplefilter('ignore')

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

print("Imported modules successfully")

# Constants
MAX_NUM_EPOCHS = 50
NUM_H_PARAMS_IN_SEARCH = 300
CROSS_VALIDATION_K = 3
NON_SNP_COLUMNS = ['1', '2', '3']
ATTRIBUTE_COLUMN = '3'


class CDBNDataset(torch.utils.data.Dataset):
    '''
    Prepare the CDBN dataset for regression
    '''

    def __init__(self, X, y, scale_data=True):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            # Apply scaling if necessary
            if scale_data:
                X = StandardScaler().fit_transform(X)
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class MLP(nn.Module):
    '''
      Multilayer Perceptron.
    '''

    def __init__(self, X, y, input_dim, hidden_dims=[5], lr=0.1, batch_size=16, output_dim=1):
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size

        layer_list = self.__constructLayerList(input_dim, hidden_dims, output_dim)
        self.layers = nn.Sequential(*layer_list)

        X = np.array(X)
        y = np.array(y)
        dataset = CDBNDataset(X, y)
        self.trainloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                                       drop_last=True)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def trainOneEpoch(self):
        '''train for a single epoch'''
        # Perform backward propagation one batch at a time
        for i, data_batch in enumerate(self.trainloader, 0):
            # Get and prepare inputs
            inputs, targets = data_batch
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))

            # Reset optimiser
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self(inputs)
            loss = self.loss_fn(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

    def __constructLayerList(self, input_dim, hidden_dims, output_dim):
        '''
        Helper method to construct list of MLP layers during initialisation
        '''
        layer_list = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU()
        ]
        for i in range(len(hidden_dims) - 1):
            layer_list.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layer_list.append(nn.ReLU())
        layer_list.append(nn.Linear(hidden_dims[-1], output_dim))
        return layer_list

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

    def predict(self, X0):
        '''predict values from matrix X0'''
        X0 = torch.from_numpy(np.array(X0)).float()
        return self(X0)


class Ensemble:
    '''An ensemble of neural networks'''

    def __init__(self, training_data, method="l1o", lr=0.1, batch_size=16, hidden_dims=[5], max_num_epochs=50,
                 compute_loss=True, n_weak_learners=21):
        self.data = training_data
        self.method = method
        self.lr = lr
        self.batch_size = batch_size
        self.hidden_dims = hidden_dims
        self.max_num_epochs = max_num_epochs
        self.models = []
        self.subsets = []
        self.loss = []
        self.unique_groups = self.data['1'].unique()
        self.compute_loss = compute_loss

        # Only used for bagging
        self.n_weak_learners = n_weak_learners
        self.n_groups_per_learner = len(self.unique_groups)

        if self.compute_loss:
            self.response = np.array(self.data[ATTRIBUTE_COLUMN])
            self.matrix = self.data.drop(NON_SNP_COLUMNS, axis=1)

        self.__initialiseModels()

    def __initialiseModels(self):
        '''helper method for constructor'''
        if self.method == "bagging":
            self.__bagging()
        elif self.method == "l1o":
            self.__leaveOneOut()
        elif self.method == "mlp":
            self.__oneModel()
        elif self.method == "1by1":
            self.__1by1()
        else:
            raise ValueError("Invalid method")

        for i in range(len(self.subsets)):
            X = self.subsets[i].drop(NON_SNP_COLUMNS, axis=1)
            y = self.subsets[i][ATTRIBUTE_COLUMN]
            mlp = MLP(X, y, input_dim=X.shape[1], hidden_dims=self.hidden_dims, output_dim=1, lr=self.lr,
                      batch_size=self.batch_size)
            self.models.append(mlp)

    def __bagging(self):
        '''
        Create a bootstrap aggregation ensemble.
        '''
        for i in range(self.n_weak_learners):
            # Randomly sample locations with replacement
            bootstrapped_groups = np.random.choice(self.unique_groups, size=self.n_groups_per_learner, replace=True)
            sample = pd.DataFrame(columns=self.data.columns)

            for group in bootstrapped_groups:
                sample = pd.concat([sample, self.data[self.data['1'] == group]])

            self.subsets.append(sample)

    def __leaveOneOut(self):
        '''
        Create a leave-one-out ensemble.
        '''
        for env in self.unique_groups:
            without_env = self.data[self.data['1'] != env]
            self.subsets.append(without_env)

    def __oneModel(self):
        '''
        Create a single neural network.
        '''
        self.subsets.append(self.data)

    def __1by1(self):
        '''
        Create an ensemble with one NN per environment.
        '''
        for env in self.unique_groups:
            only_env = self.data[self.data['1'] == env]
            self.subsets.append(only_env)

    def computeLoss(self):
        '''
        Compute loss of entire ensemble.
        '''
        predictions = np.array(self.predict(self.matrix).detach().numpy())
        mse = np.mean((predictions - self.response) ** 2)
        self.loss.append(mse)

    def train(self):
        '''
        Train all models.
        '''
        if self.compute_loss:
            # Compute initial loss
            self.computeLoss()
        for i in range(self.max_num_epochs):
            # print(f"Epoch {i+1}")

            for model in self.models:
                model.trainOneEpoch()

            if self.compute_loss:
                self.computeLoss()
                # If loss curve has 'bottomed out', then exit early
                if i > 0 and (self.loss[-1] - self.loss[-2]) > (self.loss[-2] - self.loss[-3]) and self.loss[-1] / \
                        self.loss[-2] > 0.95:
                    break

    def predict(self, X0):
        '''
        Predict values from matrix X0.
        Computes average of all model predictions.
        '''
        predictions = sum([model.predict(X0) for model in self.models]) / len(self.models)
        return predictions


class HParamTuner:
    '''Class for performing nested cross validation on ensemble neural networks for a CDBN dataset'''

    def __init__(self, k=5, max_num_epochs=50):
        '''k is the number of cross-validation folds to use during hyperparameter tuning'''
        self.k = k
        self.max_num_epochs = max_num_epochs
        self.subsets = []
        self.current_train_set = None

    def pearsonCorrCoef(self, x1, x2):
        '''x1, x2 column tensors'''
        x1_df = pd.DataFrame(x1.detach().numpy())
        x2_df = pd.DataFrame(x2.detach().numpy())
        return x1_df.corrwith(x2_df)[0]

    def testModel(self, model, test_data):
        '''''
        Test model on test data.
        Returns Pearson correlation coefficient of predictions with true values (testdata).

        model must have predict method
        test_data must be in format: env, genome, attr, snp1, snp2, ...
        '''
        Xtest = test_data.drop(NON_SNP_COLUMNS, axis=1)
        ytest = test_data[ATTRIBUTE_COLUMN]
        predictions = model.predict(Xtest)
        truth = torch.from_numpy(np.array(ytest))  # Ensure that ytest is tensor
        accuracy = self.pearsonCorrCoef(predictions, truth)
        if np.isnan(accuracy): # Almost certainly due to model always predicting the same value
            print("Nan accuracy found")
        return np.nan_to_num(accuracy)

    def crossValidateParams(self, params, k=None):
        '''
        Score a set of hyperparameters on the current training set
        '''
        if k is None:
            k = self.k

        shuffled = self.current_train_set.sample(frac=1)
        subsets = np.array_split(shuffled, k)

        scores = []
        for i in range(k):
            test = subsets[i]
            train = pd.concat(subsets[:i] + subsets[i + 1:])
            score = self.fitAndScoreModel(params, train, test)
            scores.append(score)

        return sum(scores) / k

    def createRandomParameterGrid(self, n=100, method="l1o"):
        '''Create a random list of n combinations of hyperparameters'''
        param_grid = []

        for i in range(n):
            lr = np.random.choice(np.logspace(-4, 0))
            batch_size = int(np.random.choice([4, 8, 16, 32, 64, 128]))
            depth = np.random.randint(6) + 1
            hidden_dims = []
            for j in range(depth):
                width = np.random.randint(100) + 3
                hidden_dims.append(width)

            # Only affects bagging model
            n_weak_learners = np.random.randint(2, 30)

            param_grid.append(
                {
                    'method': method,
                    'lr': lr,
                    'batch_size': batch_size,
                    'hidden_dims': hidden_dims,
                    'max_num_epochs': self.max_num_epochs,
                    'n_weak_learners': n_weak_learners  # Only used for bagging
                }
            )
        return param_grid

    def remove5WorstEnvs(self, best_params, train, test):
        '''
          remove the 5 worst offending environments
        '''
        scores = []
        envs = self.current_train_set['1'].unique()

        for env in envs:
            train_subset = train[train != env]
            score = self.fitAndScoreModel(best_params, train_subset, test)
            scores.append(score)

        # Get envs which (when removed) correspond to top 5 scores
        envs_to_remove = [env for i, (score, env) in enumerate(sorted(zip(scores, envs), reverse=True)) if i < 5]
        return envs_to_remove

    def getEnvsToRemove(self, best_params, train, test, critical_point):
        '''
          determine if removing certain environments from the training set improves the model.
          Returns optimised training set
        '''
        envs_to_remove = []
        envs = self.current_train_set['1'].unique()

        for env in envs:
            train_subset = train[train['1'] != env]
            score = self.fitAndScoreModel(best_params, train_subset, test)
            if score > critical_point:
                envs_to_remove.append(env)
                print("Removing env with score:", score)
        print("envs to remove: ")
        print(envs_to_remove)
        return envs_to_remove

    def computeTunedAccuracies(self, data, num_params=50, method="l1o", env_index=None, destination="./"):
        '''
        Compute the accuracy of the bulk and optimised ensemble models on the chosen environment test set
        '''
        # Validate env_index
        envs = data['1'].unique()
        if env_index >= len(envs):
            print("Out of env range")
            exit(0)

        # Create train/test split
        test = data[data['1'] == envs[env_index]]
        self.current_train_set = data[data['1'] != envs[env_index]]

        # Tune hyperparameters
        param_grid = self.createRandomParameterGrid(n=num_params, method=method)
        print(param_grid)
        best_params = self.findBestHParams(destination, env_index, param_grid)

        # Compute score of bulk and optimised ensemble
        bulk_score = self.fitAndScoreModel(best_params, self.current_train_set, test)
        envs_to_remove, optimised_train_set = self.optimseTrainingSet(best_params)
        optim_score = self.fitAndScoreModel(best_params, optimised_train_set, test)

        # Save the accuracies to csv
        output = pd.DataFrame(
            {f'{method}_score': [bulk_score], 'optim_score': [optim_score], 'env': [envs[env_index]],
             'envs_removed_in_optim': [envs_to_remove]})
        output.to_csv(f'{destination}/env_score_{env_index}')
        return output

    def findBestHParams(self, destination, env_index, param_grid):
        '''Find the best combination of hyperparameters out of the ones provided'''
        # Cross validation of training set
        with Pool() as pool:
            tuning_scores = pool.map(
                self.crossValidateParams, param_grid
            )
        print("Completed cross validation")
        self.saveDetailedResults(destination, env_index, param_grid, tuning_scores)

        # Return the best combination of hyperparameters
        best_score = max(tuning_scores)
        index = tuning_scores.index(best_score)
        best_params = param_grid[index]
        return best_params

    def fitAndScoreModel(self, params, train, test):
        '''Fit a model with provided hyperparameters and training data. Return score when tested on test'''
        model = Ensemble(train, **params)
        model.train()
        optim_score = self.testModel(model, test)
        return optim_score

    def optimseTrainingSet(self, best_params):
        '''remove certain environments from training set that might lower the quality of the model'''
        # Create validation sets
        train_validation, test_validation = train_test_split(self.current_train_set, test_size=0.2)

        # Estimate baseline accuracy
        scores = []
        for i in range(31):  # > 30 so central limit theorem applies and odd so median is easy
            score = self.fitAndScoreModel(best_params, train_validation, test_validation)
            scores.append(score)
        scores = sorted(scores)
        print("baseline scores:", scores)
        # Take median instead of mean in case some models run into issues and output 0 -> skews mean
        baseline = scores[int(len(scores) / 2)]

        # Determine which environments to remove from training set
        threshold = 0.01
        envs_to_remove = self.getEnvsToRemove(best_params, train_validation, test_validation, baseline + threshold)

        # If all environments are marked to be removed, only remove worst 5
        if len(envs_to_remove) == len(self.current_train_set['1'].unique()):
            envs_to_remove = self.remove5WorstEnvs(best_params, train_validation, test_validation)

        # Construct optimised training set
        optimised_train_set = self.current_train_set[~self.current_train_set['1'].isin(envs_to_remove)]
        return envs_to_remove, optimised_train_set

    def saveDetailedResults(self, destination, env_index, param_grid, tuning_scores):
        '''Save scores of each combination of hyperparameters - helps with debugging'''
        results = pd.DataFrame(param_grid)
        results['score'] = tuning_scores
        results.to_csv(f'{destination}/detailed_env_score_{env_index}.csv')


if __name__ == "__main__":
    # Validate command line arguments
    args = sys.argv[1:]
    if len(args) != 4:
        raise ValueError(f"This program requires 4 positional arguments, {len(args)} arguments were provided")
    method, datapath, destination, fold = args
    if method not in ['l1o', 'bagging', 'mlp', '1by1']:
        raise ValueError(f"The method {method} does not exist, please choose from: l1o, bagging, mlp, 1by1")
    if not os.path.exists(datapath):
        raise ValueError(f"The provided dataset: {datapath} does not exist")
    try:
        fold = int(fold)
    except:
        raise ValueError(f"The provided fold: {fold} is not an integer")

    print(f"Ensemble method chosen: {method}")

    # Ensure destination folder has been created
    if os.path.isdir(destination):
        print(f"WARNING: Folder /{destination} already exists. Contents may be overridden.")
    else:
        print(f"Creating folder /{destination}")
        os.mkdir(destination)

    # Read in data
    data = pd.read_csv(datapath)
    data = data.drop(list(data)[0:2], axis=1)  # remove useless columns

    # Print summary to output
    nSNPs = len(data.columns) - len(NON_SNP_COLUMNS)
    print("There are", len(data["1"].unique()), "locations")
    print("There are", len(data["2"].unique()), "genotypes")
    print("There are", nSNPs, "SNP markers")
    print("There are", len(data), "total observations across all environments")

    # Perform nested cross validation
    tuner = HParamTuner(k=CROSS_VALIDATION_K, max_num_epochs=MAX_NUM_EPOCHS)
    result = tuner.computeTunedAccuracies(data, num_params=NUM_H_PARAMS_IN_SEARCH, method=method, destination=destination, env_index=fold)
    print(result)
