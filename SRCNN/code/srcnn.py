import torch
import torch.nn as nn
from math import sqrt

class Net(torch.nn.Module):
    def __init__(self, num_channels = 1, base_filter = 64):
        super(Net, self).__init__()
        self.extr = torch.nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=base_filter, kernel_size=9, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),)
        self.mapping = torch.nn.Sequential(    
            nn.Conv2d(in_channels=base_filter, out_channels=base_filter // 2, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),)
        self.recon = torch.nn.Sequential(
            nn.Conv2d(in_channels=base_filter // 2, out_channels=num_channels, kernel_size=5, stride=1, padding=0, bias=True),
        )
        for m in self.modules():
            normal_init(m)
            
    def forward(self, x):
        x = self.extr(x)
        x = self.mapping(x)
        x = self.recon(x)
        return x



def normal_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, sqrt(2. / n) )
        m.bias.data.zero_()
