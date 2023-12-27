from module import *


class StyleEncoder(nn.Module):
    def __init__(self, indim=3, outdim=256, featdim=256):
        super().__init__()

        self.outdim = outdim
        fc = []
        fc += [PixelNormLayer()]
        fc += [Linear(indim, featdim * 2)]
        fc += [Linear(featdim * 2, featdim * 2)]
        fc += [Linear(featdim * 2, featdim * 2)]
        fc += [Linear(featdim * 2, featdim * 2)]
        fc += [Linear(featdim * 2, featdim * 2)]
        fc += [Linear(featdim * 2, outdim, act=None)]

        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        return self.fc(x)


class Generator(nn.Module):
    def __init__(self, zdim=256, norm='instance', act='lrelu'):
        super().__init__()
        self.const = Constant(512, (4, 4))

        head, block = [], []
        head += [StyleBlock(512, 512, zdim, upsample=True, norm=norm, act=act)]  # 8x8
        head += [StyleBlock(512, 512, zdim, upsample=True, norm=norm, act=act)]  # 16x16
        block += [StyleBlock(512, 256, zdim, upsample=True, norm=norm, act=act)]  # 32x32

        block += [StyleBlock(256, 128, zdim, upsample=True, norm=norm, act=act)]  # 64x64
        block += [StyleBlock(128, 64, zdim, upsample=True, norm=norm, act=act)]  # 128x128

        self.head = nn.ModuleList(head)
        self.block = nn.ModuleList(block)
        self.toRGB = Conv2D(64, 3, norm=norm, act='tanh')

    def forward(self, s):
        x = self.const()
        x = x.repeat(s.shape[0], 1, 1, 1)
        for layer in self.head:
            x = layer(x, s)
        for layer in self.block:
            x = layer(x, s)
        return [self.toRGB(x)]


class StyleGAN(nn.Module):
    def __init__(self, indim=3, zdim=256, norm='instance', act='lrelu'):
        super().__init__()
        self.zdim = zdim
        self.enc_s = StyleEncoder(indim, zdim)
        self.gen = Generator(zdim, norm=norm, act=act)

    def forward(self, x):
        z = self.enc_s(x)
        pred = self.gen(z)
        return pred

    def rand_gen(self, bs):
        self.gen.train(False)
        z = torch.randn(bs, self.zdim).cuda().requires_grad_(False)
        pred = self.gen(z)
        self.gen.train(True)
        return pred[-1]


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class Discriminator(nn.Module):
    """Discriminator. PatchGAN."""

    def __init__(self, input_dim=3, conv_dim=64, repeat_num=3, norm='SN'):
        super(Discriminator, self).__init__()
        layers, head, tail = [], [], []
        feat_dim = 192
        head.append(spectral_norm(nn.Conv2d(input_dim, conv_dim, kernel_size=4, stride=2, padding=1)))
        head.append(nn.LeakyReLU(0.01, inplace=True))
        head.append(spectral_norm(nn.Conv2d(conv_dim, feat_dim, kernel_size=4, stride=2, padding=1)))
        head.append(nn.LeakyReLU(0.01, inplace=True))
        head.append(spectral_norm(nn.Conv2d(feat_dim, feat_dim, kernel_size=4, stride=2, padding=1)))
        head.append(nn.LeakyReLU(0.01, inplace=True))
        head.append(spectral_norm(nn.Conv2d(feat_dim, feat_dim, kernel_size=4, stride=1, padding=1)))
        head.append(nn.LeakyReLU(0.01, inplace=True))
        tail.append(spectral_norm(nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=1, padding=1)))
        tail.append(nn.LeakyReLU(0.01, inplace=True))
        tail.append(spectral_norm(nn.Conv2d(feat_dim, 1, kernel_size=1, stride=1, padding=0)))
        tail.append(nn.ReLU())
        self.head = nn.Sequential(*head)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        feat = self.head(x)
        out = self.tail(feat)
        return out


class SpectralNorm(object):
    def __init__(self):
        self.name = "weight"
        self.power_iterations = 1

    def compute_weight(self, module):
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        w = getattr(module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        return w / sigma.expand_as(w)

    @staticmethod
    def apply(module):
        name = "weight"
        fn = SpectralNorm()

        try:
            u = getattr(module, name + "_u")
            v = getattr(module, name + "_v")
            w = getattr(module, name + "_bar")

        except AttributeError:
            w = getattr(module, name)
            height = w.data.shape[0]
            width = w.view(height, -1).data.shape[1]
            u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
            v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
            w_bar = nn.Parameter(w.data)

            # del module._parameters[name]

            module.register_parameter(name + "_u", u)
            module.register_parameter(name + "_v", v)
            module.register_parameter(name + "_bar", w_bar)

        # remove w from parameter list
        del module._parameters[name]

        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_u']
        del module._parameters[self.name + '_v']
        del module._parameters[self.name + '_bar']
        module.register_parameter(self.name, nn.Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))


def spectral_norm(module):
    SpectralNorm.apply(module)
    return module


def remove_spectral_norm(module):
    name = 'weight'
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("spectral_norm of '{}' not found in {}"
                     .format(name, module))
