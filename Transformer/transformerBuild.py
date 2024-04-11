class Transformer(nn.Module):
    def __init__(self, nSNP, targetVar, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()

        self.encoder_embedding = nn.Embedding(nSNP, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]) #calls AttentionHead
        self.fc = nn.Linear(nSNP)
        self.fc =nn.Linear(1)
        self.dropout = nn.Dropout(dropout)


    def forward(self, SNPs):
        SNPs = genoTrain
        geno_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(SNPs)))
        enc_output = geno_embedded
        output = self.fc(enc_output)
        return output
