import dependencies.py
import transformerBlocks.py
import transformerBuild.py
import transformerEncoderDecoder.py
import formatDataANN.py
import 

nSNP = X.shape[1]
targetVar = 1
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

transformer = Transformer(nSNP,targetVar, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

for epoch in range(100):
    optimizer.zero_grad()
    output = transformer(xTrain, yTrain)
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

transformer.eval()

# Generate random sample validation data

scores = cross_val_score(transformer, X, Y, cv=5)
scores = pd.DataFrame(scores)
scores.to_csv("baseTransformerperf_SY.csv")

