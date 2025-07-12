#%%
import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
import math


class PositionalEmbedding(nn.Module):
    """
        Learned Positional Embedding
    """
    def __init__(self, max_seq_len, embed_dim):
        super().__init__()
        
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim) 
        
    def forward(self, X):
        return self.pos_embed(X) # -> (batch_size, seq, embed_dim)
 


class Encoder(nn.Module):
    def __init__(self, num_layers, MHA_hidden, MHA_num_heads, input_size, embed_dim, max_seq_len):
        super().__init__()
        
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.PE = PositionalEmbedding(max_seq_len, embed_dim)
        self.embed = nn.Embedding(input_size, embed_dim)
        
        self.MHA = nn.ModuleList([MultiHeadAttention(MHA_num_heads, MHA_hidden, embed_dim) for i in range(num_layers)])
        self.layerNorm1 = nn.ModuleList([nn.LayerNorm(embed_dim) for i in range(num_layers)])
        self.layerNorm2 = nn.ModuleList([nn.LayerNorm(embed_dim) for i in range(num_layers)])
        self.pfnn = nn.ModuleList([nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.ReLU(),
            nn.Linear(4*embed_dim, embed_dim),
        ) for i in range(num_layers)])
    
    def forward(self, X):
        """
        Args:
            X (tensor): (batch_size, seq)
        """
        batch_size, seq_len = X.shape
        pos = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)  
        X_emb = self.embed(X)*math.sqrt(self.embed_dim) + self.PE(pos)  # -> (batch_size, seq, embed_dim)
        
        for layer in range(self.num_layers):
            X_MHA = self.MHA[layer](X_emb, X_emb, X_emb) # -> (batch_size, seq, embed_dim)
            X_MHA = self.layerNorm1[layer](X_MHA +X_emb)
            
            X_pfnn = self.pfnn[layer](X_MHA)
            X_out = self.layerNorm2[layer](X_pfnn + X_MHA) # same
            X_emb = X_out
        
        return X_out    
        
        
class Decoder(nn.Module):
    def __init__(self, num_layers, MHA_hidden, MHA_num_heads, input_size, embed_dim, output_dim, max_seq_len):
        super().__init__()
        
        self.num_layers = num_layers
        self.MHA = nn.ModuleList([MultiHeadAttention(MHA_num_heads, MHA_hidden, embed_dim) for i in range(num_layers)])
        self.crossMHA = nn.ModuleList([MultiHeadAttention(MHA_num_heads, MHA_hidden, embed_dim) for i in range(num_layers)])
        self.PE = PositionalEmbedding(max_seq_len, embed_dim)
        
        self.embed = nn.Embedding(input_size, embed_dim)
        self.layerNorm1 = nn.ModuleList([nn.LayerNorm(embed_dim) for i in range(num_layers)])
        self.layerNorm2 = nn.ModuleList([nn.LayerNorm(embed_dim) for i in range(num_layers)])
        self.pfnn = nn.ModuleList([nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.ReLU(),
            nn.Linear(4*embed_dim, embed_dim),
            nn.Dropout(0.2)
        ) for i in range(num_layers)])
        self.ffnn = nn.Linear(embed_dim, output_dim, bias=False) 
        self.ffnn.weight = self.embed.weight # same reverse mapping as embed, reducing num params 
        
    def forward(self, Trg, Enc_out):
        """
        Args:
            Trg (tensor): (batch_size, seq)  # seq = t
            Enc_out (tensor): (batch_size, seq, enc_embed_dim)
        """   
        batch_size, seq_len = Trg.shape
        pos = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)  
        Trg_emb = self.embed(Trg) + self.PE(pos)  # -> (batch_size, seq, embed_dim)
        
        for layer in range(self.num_layers):
            Trg_MHA = self.MHA[layer](Trg_emb, Trg_emb, Trg_emb) # -> (batch_size, seq, embed_dim)
            Trg_MHA = self.layerNorm1[layer](Trg_MHA +Trg_emb)
            
            Trg_crossMHA = self.crossMHA[layer](Trg_MHA, Enc_out, Enc_out)
            Trg_pfn = self.pfnn[layer](self.layerNorm2[layer](Trg_crossMHA + Trg_MHA)) # -> (batch_size, seq, embed_dim)
            Trg_emb = Trg_pfn
            
        out = self.ffnn(Trg_pfn) # -> (batch_size, seq, output_dim)
        
        return out
    


MHA_hidden = 256
MHA_num_heads = 3
input_size = 5000
embed_dim = 256
seq_len = 24
max_seq_len = 128
batch_size = 8
ENC_NUM_LAYERS = 3
DEC_NUM_LAYERS = 2

encoder = Encoder(ENC_NUM_LAYERS, MHA_hidden, MHA_num_heads, input_size, embed_dim, max_seq_len)
decoder = Decoder(DEC_NUM_LAYERS, MHA_hidden, MHA_num_heads, input_size, embed_dim, input_size, max_seq_len)

input_text = torch.randint(0, 5000, (batch_size, seq_len))
num_token_gen = 100

enc_output = encoder(input_text)
print("Enc output", enc_output.size())
output_tok = torch.zeros((batch_size, 1), dtype=torch.int) # assume 0 is [BOS]
all_outputs = output_tok
preds=output_tok

for i in range(num_token_gen):
    dec_output = decoder(preds, enc_output) # -> (batch_size, seq, output_dim)
    output_tok = dec_output[:,-1,:].argmax(-1, keepdim = True) # -> (batch_size, 1)
    preds = torch.cat([preds, output_tok], dim=1)   

print("Preds: ",preds.size())



        


