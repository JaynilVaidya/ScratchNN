#%%
import torch
import torch.nn as nn

class BlockType1(nn.Module):
    def __init__(self, in_ch):
        super(BlockType1, self).__init__()
        
        self.conv1 = nn.ReLU(nn.Conv2d(in_ch, in_ch, 3, 1, 0))
        self.conv2 = nn.ReLU(nn.Conv2d(in_ch, in_ch, 3, 1, 0))
        self.conv3 = nn.ReLU(nn.Conv2d(in_ch, in_ch, 3, 1, 0))
        self.pool1 = nn.MaxPool2d(2, 1) 
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        return x
    
class BlockType2(nn.Module):
    def __init__(self, in_ch):
        super(BlockType2, self).__init__()
        
        self.conv1 = nn.ReLU(nn.Conv2d(in_ch, in_ch, 3, 1, 0))
        self.conv2 = nn.ReLU(nn.Conv2d(in_ch, in_ch, 3, 1, 0))
        self.pool1 = nn.MaxPool2d(2, 1) 
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        return x

class VGG(nn.Module):
    def __init__(self):
        """
        Idea was to use more conv before downsampling (i.e. maxpool). 
        2 successive (3,3) kernals touches same pixels as 1 (5,5) kernal but the latter requires more #params.
        Deep and narrow models outperform shallow counterparts. 
        
        This is a type of n/w using blocks architecture which uses a VGG block.
        """    
        super(VGG, self).__init__()
            
        self.blocks = nn.Sequential(
            BlockType2(64),
            BlockType2(128),
            BlockType1(256),
            BlockType1(512),
            BlockType1(512),
        )
        self.flat = nn.Flatten()
        
        dummy = torch.ones((1,3,224,224))
        with torch.no_grad():
            x = self.blocks(dummy)
        flat_size = self.flat(x).shape[-1]        
        
        self.fc1 = nn.Linear(flat_size, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.op = nn.Linear(4096, 1000)
        
    def forward(self, img):
        assert img.ndim == 4 and img.size(1) == 3, "Expected image of size [N,3,H,W]"
        x = self.blocks(img)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.op(x)
        
        return x
                

model = VGG()
print('Model Architecture: \n', model)

dummy = torch.randn(1, 3, 224, 224)
output = model(dummy)

print("Output size: ", output.shape)

    
       