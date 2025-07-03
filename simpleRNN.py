#%%
import torch
import torch.nn as nn

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        
        self.Wx = nn.Linear(input_size, hidden_size)
        self.Wh = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        
    def forward(self, x, h_prev):
        return self.activation(self.Wx(x) + self.Wh(h_prev))


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList([RNNCell(input_size, hidden_size, dropout) if i==0 else RNNCell(hidden_size, hidden_size, dropout) for i in range(num_layers)])
        
    def forward(self, x, h0=None):
        """
        Args:
            x (tensor): shape (seq_len, batch_size, embedding_dim) ## since seq_len as rows because timestamp corresponds to sequence 
            h0 (tensor, optional): (num_layers, batch_size, hidden_size)

        Returns:
            outputs: Tensor (seq_len, batch_size, hidden_size) ## acts as a context-vector for encoder-decoder
            h_n: final hidden states (num_layers, batch_size, hidden_size)
        """
        seq_len, batch_size, vocab_size =  x.size()
        if h0==None: h0 = torch.stack([torch.zeros((batch_size, self.hidden_size)) for i in range(self.num_layers)]) # numlayers, layersize 
        
        outputs = []
        for t in range(seq_len):
            h_t = []
            inp = x[t]
            for layer_num, layer in enumerate(self.layers):
                h_curr = layer(inp, h0[layer_num])
                inp = h_curr
                h_t.append(inp)
            h0 = torch.stack(h_t)
            outputs.append(h0[-1])
            
        outputs = torch.stack(outputs)
        return outputs, h0
    

if __name__ == "main":
    input_size = 128 ## embedding dim
    hidden_size = 256
    num_layers = 2
    model = RNN(input_size, hidden_size, num_layers)
    print("Simple RNN: \n", model)

    dummy = torch.randn((8, 3, 128)) ## seq_len, batch, embedding_dim
    output, hidden = model(dummy) 
    print("OP size: ", output.shape, " Hidden Size", hidden.shape)

        
        