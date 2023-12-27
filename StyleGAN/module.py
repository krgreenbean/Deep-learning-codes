import torch
import torch.nn as nn


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


class Linear(nn.Module):
    def __init__(self, indim, outdim, act='lrelu'):
        super().__init__()
        block = []
        block += [nn.Linear(indim, outdim)]
        act_ = get_act(act)
        if act_ is not None:
            block += [act_()]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class PixelNormLayer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class InjectNoise(nn.Module):
    def __init__(self, fdim):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1, fdim, 1, 1))

    def forward(self, x):
        b, d, h, w = x.shape
        noise = torch.randn(b, 1, h, w).type_as(x)
        return x + noise * self.w


class ModulationAdaIN(nn.Module):
    def __init__(self, zdim, fdim):
        super().__init__()
        self.linear = nn.Linear(zdim, 2 * fdim, bias=False)
        self.norm = nn.InstanceNorm2d(fdim)

    def forward(self, x, z):
        b, d, h, w = x.shape
        parms = self.linear(z)
        mu, var = torch.split(parms, d, dim=1)
        return x * var.view(b, d, 1, 1) + mu.view(b, d, 1, 1)


class Constant(nn.Module):
    def __init__(self, fdim, size):
        super().__init__()
        self.constant = nn.Parameter(torch.randn((1, fdim,) + size))

    def forward(self, ):
        return self.constant


class StyleBlock(nn.Module):
    def __init__(self, indim, outdim, zdim, upsample=False, norm='instance', act='lrelu'):
        super().__init__()
        self.conv0 = Conv2D(indim, outdim, norm=norm, act=act)
        if upsample:
            self.conv1 = TConv2D(outdim, outdim, norm=norm, act=act)
        else:
            self.conv1 = Conv2D(outdim, outdim, norm=norm, act=act)

        self.noise0 = InjectNoise(indim)
        self.noise1 = InjectNoise(outdim)

        self.adain0 = ModulationAdaIN(zdim, indim)
        self.adain1 = ModulationAdaIN(zdim, outdim)

    def forward(self, x, s):
        x = self.noise0(x)
        x = self.adain0(x, s)
        x = self.conv0(x)

        x = self.noise1(x)
        x = self.adain1(x, s)
        x = self.conv1(x)
        return x






# def calc_mean_std(feat, eps=1e-5):
#     size = feat.size()
#     assert (len(size) == 4)
#     batch_size, feat_dim = size[:2]
#     feat_var = feat.view(batch_size, feat_dim, -1).var(dim=0) + eps
#     feat_std = feat_var.sqrt().view(batch_size, feat_dim, 1, 1)
#     feat_mean = feat.view(batch_size, feat_dim, -1).mean(dim=0).view(batch_size, feat_dim, 1, 1)
#     return feat_mean, feat_std
#
# class AdaIN(nn.Module):
#     def __init__(self, in_dim, f_dim):
#         super().__init__()
#         self.mu = nn.Linear(in_dim, f_dim)
#         self.var = nn.Linear(in_dim, f_dim)
#         self.norm = nn.InstanceNorm2d(in_dim)
#
#     def forward(self, x, style):
#         """
#         x = b f_dim, h, w
#         style = b in_dim
#         """
#         batch, dim, h, w = x.shape
#         mu_s = self.mu(style).view(batch, dim, 1, 1) # b f_dim
#         var_s = self.var(style).view(batch, dim, 1, 1)
#         return self.norm(x) * var_s + mu_s
#
#
# # def AdaIN(content_feat, style_feat):
# #     size = content_feat.size()
# #     content_mean, content_std = calc_mean_std(content_feat)
# #     style_mean, style_std = calc_mean_std(style_feat)
# #
# #     styled_image = ((content_feat - content_mean.expand(size)) / (content_std + 1e-6)) * style_std + style_mean
# #     return styled_image
#
#
# class ConstantInput(nn.Module):
#     def __init__(self, in_dim, batch_size, size=4):
#         super().__init__()
#
#         self.input = nn.Parameter(torch.randn(batch_size, in_dim, size, size))
#
#     def forward(self):
#         constant_input = self.input
#         return constant_input   #initial starting noise image, size [8,512,4,4]
#
#
# class InjectNoise(nn.Module):
#     def __init__(self, batch_size, in_dim, size):
#         super().__init__()
#
#         self.noise_weight = nn.Parameter(torch.randn(batch_size, in_dim, 1, 1))
#         self.noise = torch.randn(batch_size, in_dim, 1, 1)
#
#     def forward(self, x):
#         weight = self.noise_weight.to(DEVICE)
#         noise = self.noise.to(DEVICE)
#         # print("weight", weight.shape)
#         # print("noise", noise.shape)
#         # print('x', x.shape)
#         out = x + weight * noise
#         # print('out', out.shape)
#         return out
#