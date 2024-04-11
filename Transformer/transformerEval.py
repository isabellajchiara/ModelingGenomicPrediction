transformer.eval()

# Generate random sample validation data

with torch.no_grad():

    val_output = transformer(genoValid, phenoValid)
    val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))
    print(f"Validation Loss: {val_loss.item()}")
