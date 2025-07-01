#%%
import torch
import torch.nn as nn

class convblock(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p):
        super(convblock, self).__init__()
        
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_ch)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, in_ch, first3x3out, out_ch, downsample = False):
        super(ResBlock, self).__init__()
        self.downsample = downsample
        
        if self.downsample: 
            self.conv1 = nn.Conv2d(in_ch, first3x3out, 3, 2, 1, bias=False)
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, 2, 0, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else: 
            self.conv1 = nn.Conv2d(in_ch, first3x3out, 3, 1, 1, bias=False)
            self.down = nn.Identity()
        
        self.bn1 = nn.BatchNorm2d(first3x3out)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(first3x3out, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        
        
    def forward(self, ip):
        x = self.relu(self.bn1(self.conv1(ip)))
        x = self.bn2(self.conv2(x))
        
        ip = self.down(ip)
        x = x+ip
        x = self.relu(x)
            
        return x


class ResNet(nn.Module):
    def __init__(self, in_ch):
        super(ResNet, self).__init__()
        
        self.conv1 = convblock(in_ch, 64, 7, 2, 3)
        self.pool1 = nn.MaxPool2d(3, 2)
        
        self.layer1 = nn.Sequential(
            ResBlock(64, 64, 64),
            ResBlock(64, 64, 64),
        )
        
        self.layer2 = nn.Sequential(
            ResBlock(64, 128, 128, True),
            ResBlock(128, 128, 128)
        )
        
        self.layer3 = nn.Sequential(
            ResBlock(128, 256, 256, True),
            ResBlock(256, 256, 256)
        )
        
        self.layer4 = nn.Sequential(
            ResBlock(256, 512, 512, True),
            ResBlock(512, 512, 512)
        )
        
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, 1000)
        
    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.GAP(x)
        x = torch.flatten(x)
        x = self.fc(x)
        return x
    
model = ResNet(3)
print(model)

dummy = torch.zeros((1, 3, 224, 224))
op = model(dummy)
print("OP size: ", op.shape)
