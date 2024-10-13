#import os
import sys
# torch packages
import torch
import torch.nn as nn
from torch import Tensor
# embedding modules
from features.embedding import Embeddings, build_vocab, tokenize, pad_seq, make_src_mask, make_trg_mask
from features.pos_encoding import PositionalEncoding
# attention modules
#from models.attention import MultiHeadAttention
#from models.feed_forward import PositionwiseFeedForward
from utils.visualize import display_attention, display_mask
from models.encoder import Encoder
from models.decoder import Decoder
from models.transformer import make_model

#print(os.getcwd())
#print(sys.path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 0
torch.manual_seed(seed)

# build the vocab
de_corpus = "Hallo! Dies ist ein Beispiel fuer einen Absatz, der in seine Grundkomponenten aufgeteilt wurde. Ich frage mich, was als naechstes kommt! Irgendwelche Ideen?"
en_corpus = "Hello! This is an example of a paragraph that has been split into its basic components. I wonder what will come next! Any guesses?"

# build the vocab
de_vocab = build_vocab(de_corpus)
en_vocab = build_vocab(en_corpus)
print(f"German vocabulary size: {len(de_vocab)}")
print(f"English vocabulary size: {len(en_vocab)}")

# build integer-to-string decoder for the vocab
de_itos = {v:k for k,v in de_vocab.items()}
en_itos = {v:k for k,v in en_vocab.items()}


if __name__ == '__main__':

    torch.set_printoptions(precision=2, sci_mode=False)
    torch.manual_seed(seed)

    # settings
    de_vocab_size = len(de_vocab)
    en_vocab_size = len(en_vocab)
    d_model = 12 # dimension of embeddings
    n_heads = 4 # number of heads
    n_layers = 4 # number of encoder layers
    d_ffn = d_model * 4  # hidden size of the feed-forward neural network
    dropout = 0.0 # probability of dropout occurring
    max_length = 9 # maximum length of sequences
    pad_idx_de = de_vocab['<pad>']
    pad_idx_en = en_vocab['<pad>']
    #print(f"pad_idx_de: {pad_idx_de}")
    #print(f"pad_idx_en: {pad_idx_en}")

    # list of sequences (3, )
    de_sequences = ["Ich frage mich, was als naechstes kommt!",
                    "Dies ist ein Beispiel fuer einen Absatz.",
                    "Hallo, was ist ein Grundkomponenten?"]

    en_sequences = ["I wonder what will come next!",
                    "This is a basic example paragraph.",
                    "Hello, what is a basic split?"]

    ###################################################
    ################ 1. Tokenization  #################
    ###################################################
    # tokenize the sequences
    de_tokenized_sequences = [tokenize(seq) for seq in de_sequences]
    en_tokenized_sequences = [tokenize(seq) for seq in en_sequences]
    # index the sequences 
    de_indexed_sequences = [[de_vocab[word] for word in seq] for seq in de_tokenized_sequences]
    en_indexed_sequences = [[en_vocab[word] for word in seq] for seq in en_tokenized_sequences]
    #print(de_tokenized_sequences)
    #print(en_tokenized_sequences)
    
    ###################################################
    ################### 2. Padding ####################
    ###################################################
    de_padded_seqs = []
    en_padded_seqs = []
    # pad each sequence
    for de_seq, en_seq in zip(de_indexed_sequences, en_indexed_sequences):
        de_padded_seqs.append(pad_seq(torch.Tensor(de_seq), max_length, pad_idx_de))
        en_padded_seqs.append(pad_seq(torch.Tensor(en_seq), max_length, pad_idx_en))

    # create a tensor from the padded sequences
    src = torch.stack(de_padded_seqs).long()
    trg = torch.stack(en_padded_seqs).long()
    print(f"trg.size(): {trg.size()}")

    #display_mask(trg[0].int().tolist(), trg_mask[0], en_itos=en_itos)
    
    ###################################################
    ################## 3. Initialize ##################
    ###################################################
    # initialize the model
    model = make_model(
        device=device, src_vocab=de_vocab, trg_vocab=en_vocab, 
        n_layers=n_layers, n_heads=n_heads, d_model=d_model, d_ffn=d_ffn, 
        max_length=max_length, src_pad_idx=pad_idx_de, trg_pad_idx=pad_idx_en
    )
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    # normalize the weights
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    ###################################################
    ################# 4. Forward Pass #################
    ###################################################
    epochs = 100
    learning_rate = 0.005
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx_en)

    model.train()

    # loop through each epoch
    for i in range(epochs):
        epoch_loss = 0

        # zero the gradients
        optimizer.zero_grad()

        # logits for each output
        logits = model(src, trg[:,:-1])

        # remove the first token
        expected_output = trg[:,1:]

        # calculate the loss
        loss = criterion(logits.contiguous().view(-1, logits.shape[-1]), 
                         expected_output.contiguous().view(-1))

        # backpropagation
        loss.backward()

        # clip the weights
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            
        # update the weights
        optimizer.step()

        # preview the predictions
        predictions = [[en_itos[tok] for tok in seq] for seq in logits.argmax(-1).tolist()]

        if i % 7 == 0:
            print("="*25)
            print(f"epoch: {i}")
            print(f"loss: {loss.item()}")
            print(f"predictions: {predictions}")
    
    # convert the indices to strings
    decoder_input = [en_itos[i] for i in trg[:,:-1][0].tolist()]
    print(decoder_input)
    display_attention(de_tokenized_sequences[0], decoder_input, model.decoder.attn_probs[0], n_heads, n_rows=2, n_cols=2)





