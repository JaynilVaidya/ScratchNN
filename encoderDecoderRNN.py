#%%
import torch 
import torch.nn as nn
from simpleRNN import RNN as simpleRNN
import random

class Encoder(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_size, num_layers):
        super().__init__()
        
        self.embed = nn.Embedding(input_size, embed_dim)
        self.rnn = simpleRNN(embed_dim, hidden_size, num_layers)
        
    def forward(self, input):
        """
        Args:
            input (tensor): (seq, batch_size)

        Returns:
            output (tensor): (seq, batch_size, hidden_size)
            hidden (tensor): (num_layers, batch_size, hidden_size)
        """
        embeddings = self.embed(input) # seq_len, batch_size, embed_dim
        output, hidden = self.rnn(embeddings)
        #print("input size: ", input.size(), "output size: ", output.size(), "hidden size", hidden.size())
        return output, hidden 
    
class Decoder(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_size, num_layers, output_size):
        super().__init__()
        
        self.embed = nn.Embedding(input_size, embed_dim)
        self.rnn = simpleRNN(embed_dim, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, target, context):
        """
        Args:
            target (tensor): (seq, batch_size)
            context (tensor): (batch_size, hidden_size)
        """
        embeddings = self.embed(target) # -> (seq, batch_size, embed_dim)
        output, hidden = self.rnn(embeddings, h0=context) 
        pred = self.fc(output) # (1 , batch_size, output_size)
        return pred, hidden

class Seq2Seq(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        
        self.encoder = enc
        self.decoder = dec
        
    def forward(self, input, target, teacher_force_ratio = 0.5):
        """
        Args:
            input (tensor): shape (seq_len, batch)
            target (tensor): shape (seq_len, batch)
        """
        output, hidden = self.encoder(input) ## output unused since no-attention mech
        
        preds = []
        trglen = target.shape[0]
        input_tok = target[0]
        
        for t in range(trglen):
            pred, hidden = self.decoder(input_tok.unsqueeze(0), hidden) ## directly using output(context) as h0 since RNNencoder archi == RNNdecoder archi else extra mapping layer 
            top1 = pred.argmax(2).squeeze(0) # -> (batch_size,)
            
            teacher_force = random.random() < teacher_force_ratio
            input_tok = target[t] if teacher_force else top1
            preds.append(pred)
        
        return torch.stack(preds, dim=0)
    



IN_VOCAB_SIZE = 500
OUT_VOCAB_SIZE = 700

ENC_HIDDEN_SIZE = 256
ENC_NUM_LAYERS = 2
ENC_EMB_DIM = 128

DEC_HIDDEN_SIZE = 256
DEC_NUM_LAYERS = 2
DEC_EMB_DIM = 128

        
enc = Encoder(IN_VOCAB_SIZE, ENC_EMB_DIM, ENC_HIDDEN_SIZE, ENC_NUM_LAYERS)
dec = Decoder(OUT_VOCAB_SIZE, DEC_EMB_DIM, DEC_HIDDEN_SIZE, DEC_NUM_LAYERS, OUT_VOCAB_SIZE)
model = Seq2Seq(enc, dec)

print("Encoder-Decoder Machine Translation RNN (no attention)\n", model)     
   
in_seq_len = 8
out_seq_len = 12
batch_size = 4

dummy_input = torch.randint(0, 200, (in_seq_len, batch_size))
dummy_output = torch.randint(0, 500, (out_seq_len, batch_size)) # 12, 4

output = model(dummy_input, dummy_output)
print("OP size: ", output.shape)