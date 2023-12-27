import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=3, stride=2, padding=1):
        super().__init__()

        self.cbr= nn.Sequential(
            nn.Conv2d(in_dim,out_dim, kernel_size=kernel,stride=stride, padding=padding,bias=False),   #32,3,128,128
            nn.BatchNorm2d(out_dim),
            nn.Tanh(),
            nn.Conv2d(out_dim,out_dim, kernel_size=kernel ,stride=stride, padding=padding,bias=False),   #32,3,128,128
            nn.BatchNorm2d(out_dim),
            nn.Tanh(),
            )

    def forward(self, x):
        f = self.cbr(x)
        return f
