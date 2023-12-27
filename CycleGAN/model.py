from module import *
from module import Conv2D, TConv2D, ResConv


class Generator(nn.Module):
    def __init__(self, indim=3, outdim=3, featdim=256, norm='instance', act='lrelu'):
        super().__init__()
        block = []
        block += [Conv2D(indim, 64, stride=2, norm=norm, act=act)]
        block += [Conv2D(64, 128, stride=2, norm=norm, act=act)]
        block += [Conv2D(128, featdim, stride=2, norm=norm, act=act)]

        for _ in range(8):
            block += [ResConv(featdim, featdim, norm=norm, act=act)]

        block += [TConv2D(featdim, 128, norm=norm, act=act)]
        block += [TConv2D(128, 64, norm=norm, act=act)]
        block += [TConv2D(64, outdim, norm=norm, act='tanh')]

        self.block = nn.ModuleList(block)

    def forward(self, x):
        for layer in self.block:
            x = layer(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, indim=3, outdim=1, featdim=256, norm=None, act='relu'):
        super().__init__()
        block = []
        block += [Conv2D(indim, 64, stride=2, norm=None, act=act)]
        block += [Conv2D(64, 128, stride=2, norm=norm, act=act)]
        block += [Conv2D(128, featdim, stride=1, norm=norm, act=act)]
        block += [Conv2D(featdim, featdim, stride=1, norm=norm, act=act)]
        block += [Conv2D(featdim, featdim, stride=1, norm=norm, act=act)]
        block += [Conv2D(featdim, outdim, stride=1, norm=None, act=None)]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)
