class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        global attn_scores

        attn_output = self.scaled_dot_product_attention(Q, K, V)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

def initialize_attention_weights(module):
    if isinstance(module, MultiHeadAttention):
        nn.init.xavier_uniform_(module.W_q.weight)
        nn.init.xavier_uniform_(module.W_k.weight)
        nn.init.xavier_uniform_(module.W_v.weight)
        nn.init.xavier_uniform_(module.W_o.weight)


def OptimizeTransformer(params):
    d_model, num_heads, num_layers, d_ff, dropout, lr = params

    accuracies = []  # Store accuracies for each k-fold
    kf = KFold(n_splits=5, shuffle=True, random_state=100)

    # 5-fold cross-validation
    for train_index, test_index in kf.split(X):
        xTrain, xTest = X.iloc[train_index], X.iloc[test_index]
        yTrain, yTest = y.iloc[train_index], y.iloc[test_index]

        xTrain = torch.tensor(xTrain.values, dtype=torch.int64)
        yTrain = torch.tensor(yTrain.values, dtype=torch.float32)
        xTest = torch.tensor(xTest.values, dtype=torch.int64)
        yTest = torch.tensor(yTest.values, dtype=torch.float32)

        train_dataset = TensorDataset(xTrain, yTrain)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        src_vocab_size = int(unique)
        max_seq_length = X.shape[1]

        # Initialize the model
        transformer = Transformer(
            src_vocab_size, 1, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout
        )
        transformer.apply(initialize_attention_weights)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(transformer.parameters(), lr=lr)

        # Train the model
        for epoch in range(100):
            transformer.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                estimations = transformer(batch_x)
                loss = criterion(estimations, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate
        transformer.eval()
        with torch.no_grad():
            prediction = transformer(xTest)
        prediction_np = prediction.numpy()
        yTest_np = yTest.numpy()
        accuracy, _ = pearsonr(prediction_np, yTest_np)
        accuracies.append(accuracy)

    # Average accuracy across folds
    result = sum(accuracies) / len(accuracies)
    return params, result


