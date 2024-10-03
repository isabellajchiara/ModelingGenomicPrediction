exec(open("dependencies.py").read())
exec(open("transformerBlocks.py").read())
exec(open("transformerBuild.py").read())

data = pd.read_csv("tokenTrainingSY.csv")

X = data.drop(['Unnamed: 0','2','3'], axis=1)
y = data['3']

nSNPs = X.shape[1] 

accuracies = {} #Store accuracies for eac k fold
kf = KFold(n_splits=5, shuffle=True, random_state=100) #cross validate to evaluate k values


# 5-fold cross-validation
for train_index, test_index in kf.split(X):

    # train rest split for k fold
    xTrain, xTest = X.iloc[train_index], X.iloc[test_index]
    yTrain, yTest = y.iloc[train_index], y.iloc[test_index]

    xTrain = torch.tensor(xTrain.values)
    yTrain = torch.tensor(yTrain.values)
    xTest = torch.tensor(xTest.values)
    yTest = torch.tensor(yTest.values)  

    src_vocab_size = nSNPs
    tgt_vocab_size = 1
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = X.shape[1]
    dropout = 0.1

    #define model
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

    #define loss and optimizer 
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.01)

    # train mode
    transformer.train()

    #train loop
    for epoch in range(100):

        optimizer.zero_grad()
        output = transformer(xTrain, yTrain.long().unsqueeze(1)) 

        loss = criterion(output.contiguous().view(-1, pheno_size), yTrain[:, 0].contiguous().view(-1).int())
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    print("Training Complete")

    #save model
    torch.save(transformer,"transformer.pth")

    #eval mode
    transformer.eval()

    #determine accuracy for k fold
    with torch.no_grad():

        prediction = transformer(xTest, yTest)
        print(f"Validation Loss: {val_loss.item()}")


    prediction = prediction.detach().numpy()
    df_val_output = pd.DataFrame(val_output_np, columns=['val_output'])
    correlation = yTest['0'].corr(df_val_output['val_output'])


    accuracy = np.corrcoef(yPred, yTest)[0, 1]
    accuracies.append(correlation)

result = accuracies.mean()

result.to_csv('SYresults.csv', index=False)
#
