import torch

from module import *


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.En1 = ConvBlock(3, 64, kernel=3, stride=1, padding=1)
        self.En2 = ConvBlock(64, 128, kernel=3, stride=1, padding=1)
        self.En3 = ConvBlock(128, 256, kernel=3, stride=1, padding=1)
        self.En4 = ConvBlock(256, 512, kernel=3, stride=1, padding=1)
        self.En5 = ConvBlock(512, 1024, kernel=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.Up1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.De1 = ConvBlock(1024, 512, kernel=3, stride=1, padding=1)
        self.Up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.De2 = ConvBlock(512, 256, kernel=3, stride=1, padding=1)
        self.Up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.De3 = ConvBlock(256, 128, kernel=3, stride=1, padding=1)
        self.Up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.De4 = ConvBlock(128, 64, kernel=3, stride=1, padding=1)

        self.final = nn.Conv2d(64,3,1,1,0)

    def forward(self, x):
        x1= self.En1(x)  #512
        x= self.pool(x1) #256
        x2= self.En2(x) #256
        x= self.pool(x2) #128
        x3= self.En3(x) #128
        x= self.pool(x3) #64
        x4= self.En4(x) #64
        x= self.pool(x4) #32
        x5= self.En5(x) #32

        x=self.Up1(x5) #
        x= torch.cat((x4, x),dim =1)
        x= self.De1(x)

        x=self.Up2(x)
        x= torch.cat((x3, x),dim =1)
        x= self.De2(x)

        x=self.Up3(x)
        x= torch.cat((x2, x),dim =1)
        x= self.De3(x)

        x=self.Up4(x)
        x= torch.cat((x1, x),dim =1)
        x= self.De4(x)
        out= self.final(x)

        return out
