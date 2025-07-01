#%%
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    """
    AlexNet, which employed an 8-layer CNN, won ILSVRC 2012
    i/p size = (3, 227, 227)
    o/p = 1000 classes 
    """
    def __init__(self):
        super(AlexNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 96, 11, 4, 0) #55
        self.pool1 = nn.MaxPool2d(3, 2) #27
        
        self.conv2 = nn.Conv2d(96, 256, 5, 1, 2) #27
        self.pool2 = nn.MaxPool2d(3, 2) #13
        
        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1) 
        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1) 
        self.conv5 = nn.Conv2d(384, 256, 3, 1, 1) #13
        self.pool3 = nn.MaxPool2d(3, 2) #6,6    ,256
        
        self.fcnn1 = nn.Linear(9216, 4096)
        self.fcnn2 = nn.Linear(4096, 4096)
        self.fcnn3 = nn.Linear(4096, 1000)
        
        self.relu  = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        
    def forward(self, img):
        assert img.ndim == 4 and img.size(1) == 3, "Expected image of size [N,3,H,W]"
        
        x = self.pool1(self.relu(self.conv1(img)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.pool3(x)
        
        x = self.flatten(x)
        x = self.fcnn1(x)
        x = self.fcnn2(x)
        x = self.fcnn3(x)
        
        return x
    
    
if __name__ == "main":
    print("### ALEXNET ###")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlexNet()
    
    print("Model Architecture \n", model)
    
    dummy = torch.randn(1, 3, 227, 227).to(device)
    output = model(dummy)
    print("Output size: ", output.shape) 
            
        