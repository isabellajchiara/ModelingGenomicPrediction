class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()

        #define layers
        
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        for layer in self.encoder_layers:
            src_embedded = layer(src_embedded)
        output = self.fc(src_embedded)
        #weights = SNP weights
        #output = torch.sum(model_output * weights, dim=1)
        outputReshape = output.view(src.shape[0], -1)
        pred, _ = torch.max(outputReshape, dim=1)#collapse 2nd dimension of output 
        return pred

