from module import *


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvAe_encoder = nn.Sequential(
            Encoder(3, 32, kernel=3, stride=2, padding=1),
            Encoder(32, 64, kernel=3, stride=2, padding=1),
            Encoder(64, 128, kernel=3, stride=2, padding=1),
            Encoder(128, 256, kernel=3, stride=2, padding=1),
            Encoder(256, 512, kernel=3, stride=2, padding=1),
        )
        self.ConvAe_decoder=nn.Sequential(
            Decoder(512, 256, kernel=4, stride=2, padding=1),
            Decoder(256, 128, kernel=4, stride=2, padding=1),
            Decoder(128, 64, kernel=4, stride=2, padding=1),
            Decoder(64, 32, kernel=4, stride=2, padding=1),
            Decoder(32, 3, kernel=4, stride=2, padding=1),
        )

    def forward(self, x):
        z = self.ConvAe_encoder(x)
        output = self.ConvAe_decoder(z)
        return z, output


class MixNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvAE = nn.Sequential(
            Decoder(512, 256, kernel=4, stride=2, padding=1),
            Decoder(256, 128, kernel=4, stride=2, padding=1),
            Decoder(128, 64, kernel=4, stride=2, padding=1),
            Decoder(64, 32, kernel=4, stride=2, padding=1),
            Decoder(32, 3, kernel=4, stride=2, padding=1),
        )

    def forward(self, x):
        out = self.ConvAE(x)
        return out
