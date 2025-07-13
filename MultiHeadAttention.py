#%%
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
        Scaled Dot Product Multi-Head Attention
    """
    def __init__(self, num_heads, embed_dim, output_dim=None):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.attention_hidden = embed_dim//num_heads
        self.rootD = self.attention_hidden**0.5
        self.output_dim = output_dim
        self.dropout = nn.Dropout(0.2)
        
        self.Wk = nn.LazyLinear(num_heads*self.attention_hidden)
        self.Wq = nn.LazyLinear(num_heads*self.attention_hidden)
        self.Wv = nn.LazyLinear(num_heads*self.attention_hidden)
        if output_dim!=None: self.Wo = nn.LazyLinear(output_dim)
        
        
    def forward(self, query, key, value, mask = None):
        """
        This is a batch-first implementation
        Args:
            key (tensor): (batch_size, seq_len, hidden_size)
            value (tensor): (batch_size, seq_len, hidden_size)
            query (tensor): (batch_size, Qseq_len, hidden_size)
            mask (tensor): (Qseq_len, hidden_size)
        Returns:
            (batch_size, Qseq_len, output_dim) or (batch_size, Qseq_len, embed_dim)
        """
        batch_size, Qseq_len, _ = query.size() ## for supporting cross-attention where query seq_len is diff
        batch_size, Kseq_len, _ = key.size()
        if mask==None: mask = torch.ones((Qseq_len, Kseq_len))
        mask = mask.unsqueeze(0).bool()
        
        Q = self.Wq(query) # -> (seq_len, batch_size, num_heads*attention_hidden) == (seq_len, batch_size, embed_dim)
        K = self.Wk(key) # same
        V = self.Wv(value) # same
        
        Q = Q.view(batch_size, Qseq_len, self.num_heads, self.attention_hidden).permute(0,2,1,3) # -> (batch_size, num_heads, Qseq_len, attention_hidden)
        K = K.view(batch_size, Kseq_len, self.num_heads, self.attention_hidden).permute(0,2,1,3) # same
        V = V.view(batch_size, Kseq_len, self.num_heads, self.attention_hidden).permute(0,2,1,3) #same
        
        Q = Q.reshape(batch_size*self.num_heads, Qseq_len, self.attention_hidden) # -> (batch_size*num_heads, Qseq_len, attention_hidden)
        K = K.reshape(batch_size*self.num_heads, Kseq_len, self.attention_hidden) # same
        V = V.reshape(batch_size*self.num_heads, Kseq_len, self.attention_hidden) # same
        
        # scaled-dot-product and masking
        scaled_dotproduct = torch.bmm(Q, K.transpose(1,2))/self.rootD # -> (batch_size*num_heads, Qseq_len, Kseq_len)
        masked_scaled_dotproduct = scaled_dotproduct.masked_fill(mask==0, -1e9)
        attention = self.dropout(torch.softmax(masked_scaled_dotproduct,dim=2)) # -> same
        result = torch.bmm(attention, V) # -> (batch_size*num_heads, Qseq_len, attention_hidden)
        
        result = result.view(batch_size, self.num_heads, Qseq_len, self.attention_hidden).permute(0,2,1,3) # -> (batch_size, Qseq_len, num_heads, attention_hidden)
        result = result.reshape(batch_size, Qseq_len, self.num_heads*self.attention_hidden) # -> (batch_size, Qseq_len, num_heads*attention_hidden)
        
        
        if self.output_dim!=None: result = self.Wo(result) # -> (batch_size, Qseq_len, output_dim)
        return result
        

if __name__ =="__main__":
    NUM_HEADS = 4
    EMBED_DIM = 256
    OUTPUT_DIM = 256

    MHA = MultiHeadAttention(NUM_HEADS, EMBED_DIM) 

    batch_size, seq_len, hidden_size = 8,4, 128
    X = torch.randn((batch_size, seq_len, hidden_size))
    output = MHA(X,X,X)
    print("OP size: ", output.size())

        
        
        
        