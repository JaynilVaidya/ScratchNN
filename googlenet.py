#%%
import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out1x1, red3x3, out3x3, red5x5, out5x5, proj1x1):
        super(InceptionBlock, self).__init__()
        
        self.track1 = nn.Sequential(
            nn.Conv2d(in_ch, out1x1, 1, 1, 0),
            nn.ReLU()
        )
        
        self.track2 = nn.Sequential(
            nn.Conv2d(in_ch, red3x3, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(red3x3, out3x3, 3, 1, 1),
            nn.ReLU()
        )
        
        self.track3 = nn.Sequential(
            nn.Conv2d(in_ch, red5x5, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(red5x5, out5x5, 5, 1, 2),
            nn.ReLU()
        )
        
        self.track4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(in_ch, proj1x1, 1, 1, 0),
            nn.ReLU()
        )
        
    def forward(self, x):
        op = torch.concat([self.track1(x), self.track2(x), self.track3(x), self.track4(x)], 1)
        return op
        

class GoogleNet(nn.Module):
    def __init__(self, in_ch):
        super(GoogleNet, self).__init__()
        
        self.module1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 7, 2, 3),
            nn.ReLU(),
            nn.MaxPool2d(3, 2), 
            nn.Conv2d(64, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 192, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 1, 1)
        )
        
        self.module2 = nn.Sequential(
            InceptionBlock(192, 64, 96, 128, 16, 32, 32), # 256 3a
            InceptionBlock(256, 128, 128, 192, 32, 96, 64), # 480 3b
            nn.MaxPool2d(3, 2)
        )
        
        self.module3 = nn.Sequential(
            InceptionBlock(480, 192, 96, 208, 16, 48, 64), # 512 4a
            InceptionBlock(512, 160, 112, 224, 24, 64, 64), # 512 4b
            InceptionBlock(512, 128, 128, 256, 24, 64, 64), # 512 4c
            InceptionBlock(512, 112, 144, 288, 32, 64, 64), # 528 4d
            InceptionBlock(528, 256, 160, 320, 32, 128, 128), # 832 4e  
            nn.MaxPool2d(3, 2)   
        )
        
        self.module4 = nn.Sequential(
            InceptionBlock(832, 256, 160, 320, 32, 128, 128), # 832 5a
            InceptionBlock(832, 384, 192, 384, 48, 128, 128), # 1024 5b
        )
        
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, 1000)
        
    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)
        x = self.GAP(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    

model = GoogleNet(3)
print(model)

dummy = torch.zeros((1, 3, 227, 227))
output = model(dummy)

print("Output size: ", output.shape) 
        
        
        
        