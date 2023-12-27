import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride=1, padding=1, norm=True, act=nn.LeakyReLU()):
        super().__init__()
        block = []
        block += [nn.Conv2d(in_dim, out_dim, kernel_size=kernel, stride=stride, padding=padding)]
        if norm:
            block += [nn.InstanceNorm2d(out_dim)]
        if act is not None:
            block += [act]
        self.conv = nn.Sequential(*block)

    def forward(self, x):
        return self.conv(x)


class DeConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=3, stride=1, padding=1, norm=True, act=nn.LeakyReLU()):
        super().__init__()
        block = []
        block += [nn.ConvTranspose2d(in_dim, out_dim, kernel_size=kernel, stride=stride, padding=padding)]
        if norm:
            block += [nn.InstanceNorm2d(out_dim)]
        if act is not None:
            block += [act]
        self.deconv = nn.Sequential(*block)

    def forward(self, x):
        return self.deconv(x)

