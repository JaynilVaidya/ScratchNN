#%%
import torch
import torch.nn as nn
from simpleRNN import RNN
import random

class Attention(nn.Module):
    """
    Additive Attention
    """
    def __init__(self, attention_hidden):
        super().__init__()
        
        self.Wk = nn.LazyLinear(attention_hidden)
        self.Wq = nn.LazyLinear(attention_hidden)
        self.Wv = nn.LazyLinear(1)
        self.tanh = nn.Tanh()        
        
    def forward(self, key, value, query):
        """
        Args:
            key (tensor): (seq_len, batch_size, hidden_size)
            value (tensor): (seq_len, batch_size, hidden_size)
            query (tensor): (1, batch_size, hidden_size)

        Returns:
            (batch_size, seq_len, hidden_size)
        """
        K = self.Wk(key) # -> (seq_len, batch_size, attention_hidden) 
        Q = self.Wq(query) # -> (1, batch_size, attention_hidden)
        V = torch.permute(value, (1,0,2)) # -> (batch_size, seq_len, hidden_size) 
        
        score = self.tanh(K + Q) # (same as K)
        alpha = self.Wv(score) # -> (seq_len, batch_size, 1) 
        alpha = torch.softmax(alpha.permute((1,2,0)), dim=2) # -> ( batch_size, 1, seq_len)
        
        result = torch.bmm(alpha, V) # -> (batch_size, 1, hidden_size)
        return result
        
        

class Encoder(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_size, num_layers):
        super().__init__()
        
        self.embed = nn.Embedding(input_size, embed_dim)
        self.rnn = RNN(embed_dim, hidden_size, num_layers)
        
    def forward(self, x):
        """
        Args:
            x (tensor): (seq_len, batch_size)
        """
        x = self.embed(x) # -> (seq_len, batch_size, embed_dim)
        output, hidden = self.rnn(x) # -> (seq_len, batch_size, hidden_size) & (num_layers, batch_size, hidden_size)
        return output, hidden
    
class AttentionDecoder(nn.Module):
    def __init__(self, encoder,  input_size, embed_dim, hidden_size, num_layers, output_size, attention_hidden=256, teacher_force_ratio = 0.5): 
        super().__init__()
        
        self.encoder = encoder
        self.embed = nn.Embedding(input_size, embed_dim)
        self.rnn = RNN(embed_dim + hidden_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.attention = Attention(attention_hidden)
        self.teacher_force_ratio = teacher_force_ratio
        
    def forward(self, x, target):
        """
        Args:
            x (tensor): (seq_len, batch_size)
            target (tensor): (seq_len, batch_size)
        """
        enc_output, enc_hidden = self.encoder(x) # -> (seq_len, batch_size, hidden_size) & (num_layers, batch_size, hidden_size)
        
        input_tok = target[0] # -> (batch_size)
        hidden = enc_hidden
        trglen = target.size()[0]
        preds = []
        
        for t in range(trglen):
            print("input",input_tok.size())
            input_tok = self.embed(input_tok).unsqueeze(1) # -> (batch_size, 1, embed_dim) 
                        
            ## in bahnadau attention (enc_output, enc_output, hidden[-1]) acts as (key, value, query) 
            attention_score = self.attention(enc_output, enc_output, hidden[-1].unsqueeze(0)) # -> (batch_size, 1, hidden_size)
            input_tok = torch.cat([input_tok, attention_score], dim=2) # -> (batch_size, 1, hidden_size + embed_dim)
            input_tok = torch.permute(input_tok, (1,0,2)) # -> (1, batch_size, hidden_size + embed_dim)
            
            dec_output, dec_hidden = self.rnn(input_tok, hidden) 
            pred = self.fc(dec_output) # (1 , batch_size, output_size)
            print("pred: ", pred.size())
            hidden = dec_hidden
            
            top1 = pred.argmax(2).squeeze(0) # (batch_size,)
            print("top1",top1.size(), "target[t]:", target[t].size())
            teacher_force = random.random() < self.teacher_force_ratio
            input_tok = target[t] if teacher_force else top1
            preds.append(pred) ## top1 if you only want answers
          
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
attention_dec = AttentionDecoder(enc, OUT_VOCAB_SIZE, DEC_EMB_DIM, DEC_HIDDEN_SIZE, DEC_NUM_LAYERS, OUT_VOCAB_SIZE)

print("Encoder-Decoder Machine Translation RNN (no attention)\n", attention_dec)     
 
in_seq_len = 8
out_seq_len = 12
batch_size = 4

dummy_input = torch.randint(0, 200, (in_seq_len, batch_size))
dummy_output = torch.randint(0, 500, (out_seq_len, batch_size)) # 12, 4

output = attention_dec(dummy_input, dummy_output)
print("OP size: ", output.shape)
            
            
            
        
        
        
        