import torch
import torch.nn as nn


def batch_set_value(params, values):
    for p, v in zip(params, values):
        p.data.copy_(torch.from_numpy(v).data)


class ConvRelu(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3,3),
                 stride=1,
                 act = nn.ReLU(),
                 pad_fn = nn.ReflectionPad2d,
                 ):
        super().__init__()
        self.pad = pad_fn(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)
        self.relu = act

    def forward(self,  x):
        x = self.pad(x)
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class VGG19Normalized(nn.Sequential):
    def __init__(self, weight = 'weights/vgg_normalised.pth'):
        super().__init__()
        self.norm = nn.Conv2d(3,3,(1,1))
        self.conv1_1 = ConvRelu(3, 64)
        self.conv1_2 = ConvRelu(64, 64)
        self.pool1 = nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
        self.conv2_1 = ConvRelu(64, 128)
        self.conv2_2 = ConvRelu(128, 128)
        self.pool2 = nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
        self.conv3_1 = ConvRelu(128, 256)
        self.conv3_2 = ConvRelu(256, 256)
        self.conv3_3 = ConvRelu(256, 256)
        self.conv3_4 = ConvRelu(256, 256)
        self.pool3 = nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
        self.conv4_1 = ConvRelu(256, 512)
        self.conv4_2 = ConvRelu(512, 512)
        self.conv4_3 = ConvRelu(512, 512)
        self.conv4_4 = ConvRelu(512, 512)
        self.pool4 = nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
        self.conv5_1 = ConvRelu(512, 512)
        self.conv5_2 = ConvRelu(512, 512)
        self.conv5_3 = ConvRelu(512, 512)
        self.conv5_4 = ConvRelu(512, 512 )
        self.load_params(weight)
        for p in self.parameters():
            p.requires_grad_(False)

    def load_params(self, param_file):
        trained = torch.load(param_file).values()#[np.array(layer[1], 'float32') for layer in list(f.items())]
        weight_value_tuples = []
        for p, tp in zip(self.parameters(), trained):
            weight_value_tuples.append((p, tp.numpy()))
        batch_set_value(*zip(*(weight_value_tuples)))

    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)
        return x
    
    def get_features(self, x, feat_list):
        x = (x + 1) / 2
        outs = []
        for name, layer in self.named_children():
            x = layer(x)
            if name in feat_list:
                outs.append(x)
        return outs

