from module import *


class Generator(nn.Module):
    def __init__(self, noise_size):
        super().__init__()
        self.fc = nn.Linear(noise_size, 3*16*16)
        block = []
        block += [
            Conv(3, 64, 3, 1, 1),
            Conv(64, 128, 3, 1, 1),
            Conv(128, 256, 3, 2, 1)]
        block +=[
            DeConv(256, 128, 4, 2, 1),
            DeConv(128, 64, 4, 2, 1),
            DeConv(64, 32, 4, 2, 1),
            DeConv(32, 3, 4, 2, 1, act=nn.Tanh())
        ]
        self.BLOCK = nn.Sequential(*block)

    def forward(self, x):
        x = self.fc(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, 3, 16, 16)
        x = self.BLOCK(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        block = []
        block += [
            Conv(3, 64, 5, 2, 1, False),
            Conv(64, 128, 5, 2, 1),
            Conv(128, 256, 5, 2, 1),
            Conv(256, 256, 5, 2, 1),
            Conv(256, 1, 5, 2, 1),
        ]
        self.BLOCK = nn.Sequential(*block)
        self.fc = nn.Linear(9, 1)

    def forward(self, x):
        x = self.BLOCK(x)
        s = x.shape[1] * x.shape[2] * x.shape[3]
        x = x.view(x.shape[0], s)
        x = self.fc(x)
        return x
