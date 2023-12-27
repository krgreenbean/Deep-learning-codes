import torch.nn as nn


class resblock(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        super().__init__()

        self.residual_func = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
        )
        # shortcut skip connection using paper method 2: when no. of feature layers is different
        if stride != 1 or in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_dim)
            )
        # shortcut skip connection using paper method 1: identity mapping where feature layer num is same
        else:
            self.shortcut = nn.Sequential()  # == nn.Identity()
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.residual_func(x)+self.shortcut(x)
        x = self.ReLU(x)
        return x


class BottleNeck(nn.Module):
    def __init__(self, bottleneck, in_dim, out_dim, stride=1):
        super().__init__()
        self.residual_func = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),

            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),

            nn.Conv2d(out_dim, out_dim*bottleneck*4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim*bottleneck*4)
        )
        # shortcut skip connection
        if stride != 1 or in_dim != out_dim * bottleneck*4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim*bottleneck*4, kernel_size=3, stride=2,padding=1, bias=False),
                nn.BatchNorm2d(out_dim*bottleneck*4)
            )
        else:
            self.shortcut = nn.Sequential()
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.residual_func(x)+self.shortcut(x)
        x = self.ReLU(x)
        return x
