#%%
import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention


class PositionalEmbeddin(nn.Module):
    """
        Learned Positional Embedding
    """
    def __init__(self, max_seq_len, embed_dim):
        super().__init__()
        
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim) 
        
    def forward(self, X):
        return self.pos_embed(X) # -> (batch_size, seq, embed_dim)
 


class Encoder(nn.Module):
    def __init__(self, MHA_hidden, MHA_num_heads, input_size, embed_dim, max_seq_len):
        super().__init__()
        
        self.MHA = MultiHeadAttention(MHA_num_heads, MHA_hidden, embed_dim)
        self.PE = PositionalEmbeddin(max_seq_len, embed_dim)
        
        self.embed = nn.Embedding(input_size, embed_dim)
        self.layerNorm = nn.LayerNorm(embed_dim)
        self.pfnn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, X):
        """
        Args:
            X (tensor): (batch_size, seq)
        """
        batch_size, seq_len = X.shape
        pos = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)  
        X_emb = self.embed(X) + self.PE(pos)  # -> (batch_size, seq, embed_dim)
        
        X_MHA = self.MHA(X_emb, X_emb, X_emb) # -> (batch_size, seq, embed_dim)
        X_MHA = self.layerNorm(X_MHA +X_emb)
        
        X_pfnn = self.pfnn(X_MHA)
        X_out = self.layerNorm(X_pfnn + X_MHA) # same
        
        return X_out    
        
        
class Decoder(nn.Module):
    def __init__(self, MHA_hidden, MHA_num_heads, input_size, embed_dim, output_dim, max_seq_len):
        super().__init__()
        
        self.MHA = MultiHeadAttention(MHA_num_heads, MHA_hidden, embed_dim)
        self.crossMHA = MultiHeadAttention(MHA_num_heads, MHA_hidden, embed_dim)
        self.PE = PositionalEmbeddin(max_seq_len, embed_dim)
        
        self.embed = nn.Embedding(input_size, embed_dim)
        self.layerNorm = nn.LayerNorm(embed_dim)
        self.pfnn = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.ReLU(),
            nn.Linear(4*embed_dim, embed_dim),
            nn.Dropout(0.2)
        )
        self.ffnn = nn.Linear(embed_dim, output_dim) 
        
    def forward(self, Trg, Enc_out):
        """
        Args:
            Trg (tensor): (batch_size, seq)  # seq = 1
            Enc_out (tensor): (batch_size, seq, enc_embed_dim)
        """   
        batch_size, seq_len = Trg.shape
        pos = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)  
        Trg_emb = self.embed(Trg) + self.PE(pos)  # -> (batch_size, seq, embed_dim)
        
        Trg_MHA = self.MHA(Trg_emb, Trg_emb, Trg_emb) # -> (batch_size, seq, embed_dim)
        Trg_MHA = self.layerNorm(Trg_MHA +Trg_emb)
        
        Trg_crossMHA = self.crossMHA(Trg_MHA, Enc_out, Enc_out)
        Trg_pfn = self.pfnn(self.layerNorm(Trg_crossMHA + Trg_MHA)) # -> (batch_size, seq, embed_dim)
        out = self.ffnn(Trg_pfn) # -> (batch_size, seq, output_dim)
        
        return out
    


MHA_hidden = 256
MHA_num_heads = 3
input_size = 5000
embed_dim = MHA_hidden*MHA_num_heads
seq_len = 24
max_seq_len = seq_len
batch_size = 8


encoder = Encoder(MHA_hidden, MHA_num_heads, input_size, embed_dim, max_seq_len)
decoder = Decoder(MHA_hidden, MHA_num_heads, input_size, embed_dim, input_size, max_seq_len)

input_text = torch.randint(0, 5000, (batch_size, seq_len))
num_token_gen = 100

enc_output = encoder(input_text)
print("Enc output", enc_output.size())
output_tok = torch.zeros((batch_size, 1), dtype=torch.int) # assume 0 is [BOS]
preds=[]

for i in range(num_token_gen):
    dec_output = decoder(output_tok, enc_output) # -> (batch_size, seq, output_dim)
    output_tok = dec_output.argmax(2) # -> (batch_size, 1)
    preds.append(output_tok)
 
preds = torch.cat(preds, dim=1)   
print("Preds: ",preds.size())



        


