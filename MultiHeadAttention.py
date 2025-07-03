#%%
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
        Scaled Dot Product Multi-Head Attention
    """
    def __init__(self, num_heads, attention_hidden, output_dim):
        super().__init__()
        
        self.num_heads = num_heads
        self.attention_hidden = attention_hidden
        self.rootD = attention_hidden**0.5
        
        self.Wk = nn.LazyLinear(num_heads*attention_hidden)
        self.Wq = nn.LazyLinear(num_heads*attention_hidden)
        self.Wv = nn.LazyLinear(num_heads*attention_hidden)
        self.Wo = nn.LazyLinear(output_dim)
        
        
    def forward(self, query, key, value):
        """
        Args:
            key (tensor): (seq_len, batch_size, hidden_size)
            value (tensor): (seq_len, batch_size, hidden_size)
            query (tensor): (1, batch_size, hidden_size)
        Returns:
            (seq_len, batch_size, output_dim)
        """
        Qseq_len, batch_size, _ = query.size() ## for supporting cross-attention where query seq_len is diff
        Kseq_len, batch_size, _ = key.size()
        
        Q = self.Wq(query) # -> (seq_len, batch_size, num_heads*attention_hidden)
        K = self.Wk(key) # same
        V = self.Wv(value) # same
        
        Q = Q.view(Qseq_len, batch_size, self.num_heads, self.attention_hidden).permute(1,2,0,3) # -> (batch_size, num_heads, Qseq_len, attention_hidden)
        K = K.view(Kseq_len, batch_size, self.num_heads, self.attention_hidden).permute(1,2,0,3) # same
        V = V.view(Kseq_len, batch_size, self.num_heads, self.attention_hidden).permute(1,2,0,3) #same
        
        Q = Q.view(batch_size*self.num_heads, Qseq_len, self.attention_hidden) # -> (batch_size*num_heads, Qseq_len, attention_hidden)
        K = K.view(batch_size*self.num_heads, Kseq_len, self.attention_hidden) # same
        V = V.view(batch_size*self.num_heads, Kseq_len, self.attention_hidden) # same
        
        # optional: If attention mask, then apply after bmm and before softmax w value -inf
        scaled_dotproduct = torch.softmax(torch.bmm(Q, K.transpose(1,2))/self.rootD, dim=2) # -> (batch_size*num_heads, Qseq_len, seq_len)
        result = torch.bmm(scaled_dotproduct, V) # -> (batch_size*num_heads, Qseq_len, attention_hidden)
        
        result = result.view(batch_size, self.num_heads, Qseq_len, self.attention_hidden).permute(2,0,1,3) # -> (Qseq_len, batch_size, num_heads, attention_hidden)
        result = result.reshape(Qseq_len, batch_size, self.num_heads*self.attention_hidden) # -> (Qseq_len, batch_size, num_heads*attention_hidden)
        
        
        result = self.Wo(result) # -> (Qseq_len, batch_size, output_dim)
        return result
        


NUM_HEADS = 3
ATTENTION_DIM = 128
OUTPUT_DIM = 256

MHA = MultiHeadAttention(NUM_HEADS, ATTENTION_DIM, OUTPUT_DIM) 

X = torch.randn((8, 4, 128))
output = MHA(X,X,X)
print("OP size: ", output.size())

        
        
        
        