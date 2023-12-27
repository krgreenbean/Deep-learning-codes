import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=3, stride=2, padding=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Tanh()
        )

    def forward(self, x):
        f = self.encoder(x)
        return f


class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=4, stride=2, padding=1):
        super().__init__()
        self.decoder= nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(out_dim),
            nn.Tanh()
        )

    def forward(self, x):
        f = self.decoder(x)
        return f
