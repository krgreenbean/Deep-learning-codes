from module import *


class ResNet(nn.Module):
    def __init__(self, blocktype, num_block):
        super().__init__()
        self.in_dim = 64
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        self.conv2 = self._make_layer(blocktype, 64, num_block[0], 1)
        self.conv3 = self._make_layer(blocktype, 128, num_block[1], 2)
        self.conv4 = self._make_layer(blocktype, 256, num_block[2], 2)
        self.conv5 = self._make_layer(blocktype, 512, num_block[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))        #gap
        self.fc = nn.Linear(512, 10)

    def _make_layer(self, blocktype, out_dim, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(blocktype(self.in_dim, out_dim, stride))
            self.in_dim = out_dim

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

