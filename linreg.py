#%%
import numpy as np
import torch

## y = 2x1 - 4x2 + 0.3

x = torch.rand((100,3))
x[:,0] = 1
y = 2*x[:,1] - 4*x[:,2] + 0.3

#%%

weights = torch.rand(3)
lr = 0.01

for epoch in range(200):
    y_pred = x@weights
    error = 0.5*torch.sum((y_pred - y)**2)
    
    weights[0] -= lr*torch.sum((y_pred - y)* x[:,0])
    weights[1] -= lr*torch.sum((y_pred - y)* x[:,1])
    weights[2] -= lr*torch.sum((y_pred - y)* x[:,2])
    
    print(epoch, error, weights,)
    
    
    