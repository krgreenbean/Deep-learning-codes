from functions import *


def get_norms(norm_type):
    if norm_type == 'batch':
        return nn.BatchNorm2d
    elif norm_type == 'instance':
        return nn.InstanceNorm2d
    else:
        return None


def get_act(act_type):
    if act_type == 'relu':
        return nn.ReLU
    elif act_type == 'lrelu':
        return nn.LeakyReLU
    elif act_type == 'tanh':
        return nn.Tanh
    else:
        return None


class Conv2D(nn.Module):
    def __init__(self, indim, outdim, ks=3, stride=1, pd=1, norm='batch', act='lrelu'):
        super().__init__()
        block = []
        block += [nn.Conv2d(indim, outdim, kernel_size=ks, stride=stride, padding=pd)]
        norm_ = get_norms(norm)
        if norm_ is not None:
            block += [norm_(outdim)]
        act_ = get_act(act)
        if act_ is not None:
            block += [act_()]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class TConv2D(nn.Module):
    def __init__(self, indim, outdim, ks=3, stride=1, pd=1, norm='batch', act='lrelu'):
        super().__init__()
        block = [nn.Upsample(scale_factor=2)]
        block += [nn.Conv2d(indim, outdim, kernel_size=3, stride=1, padding=1)]  # 3, 1, 1 고정
        norm_ = get_norms(norm)
        if norm_ is not None:
            block += [norm_(outdim)]
        act_ = get_act(act)
        if act_ is not None:
            block += [act_()]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class ResConv(nn.Module):
    def __init__(self, indim, outdim, ks=3, stride=1, pd=1, norm='batch', act='lrelu'):
        super().__init__()
        resblock = []
        self.mapper = None
        if indim != outdim:
            self.mapper = Conv2D(indim, outdim, ks=ks, stride=stride, pd=pd, norm=norm, act=act)
        resblock += [Conv2D(outdim, outdim, ks=ks, stride=stride, pd=pd, norm=norm, act=act)]
        resblock += [Conv2D(outdim, outdim, ks=ks, stride=stride, pd=pd, norm=norm, act=None)]

        self.resblock = nn.Sequential(*resblock)
        self.act = get_act(act)()

    def forward(self, x):
        if self.mapper is not None:
            x = self.mapper(x)
        x_ = self.resblock(x)
        return self.act(x_ + x)


