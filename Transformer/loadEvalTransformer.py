#First, redefine model shape 

src_vocab_size = nSNPs
tgt_vocab_size = 1
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 20
max_seq_length = 100
dropout = 0.1

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, src, tgt):

        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output)

        output = self.fc(enc_output)
        return output

#create model
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

#load the pretrained weights 
transformer.load_state_dict(torch.load("transformer.pth"))

#put in eval mode
transformer.eval()

#test data must be same format as train data
xTest = torch.clamp(xTest.long(), 0, src_vocab_size - 1) # Clamp values to be within vocabulary range
yTest = torch.clamp(yTest.unsqueeze(1).long(), 0, tgt_vocab_size - 1)

#make predictions and evaluate
with torch.no_grad():

    val_output = transformer(xTest, yTest)
    val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), yTest.contiguous().view(-1))
    print(f"Validation Loss: {val_loss.item()}")
