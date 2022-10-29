import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transform


PRINTLOG = False
POOLING = False


class Logs:

    def __init__(self, printlogs=False, pooling=False):
        global PRINTLOG, POOLING
        PRINTLOG = printlogs
        POOLING = pooling

    def __str__(self):
        return f"Printing Discriminator logs : {PRINTLOG}, Pooling logs : {POOLING}"


class Convblock(nn.Module):

    def __init__(self, input_channel, output_channel, kernal=4, stride=2, padding=1, activation=True, batchnorm=True):

        super().__init__()
        self.Batchnorm = None
        self.Activation = None
        self.conv = nn.Conv2d(
            input_channel, output_channel, kernal, stride, padding)
        if batchnorm:
            self.Batchnorm = nn.BatchNorm2d(output_channel)
        if activation:
            self.Activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        if self.Batchnorm:
            x = self.Batchnorm(x)
        if self.Activation:
            x = self.Activation(x)
        return x


class PathGan(nn.Module):
    def __init__(self, input_nc):
        super().__init__()
        self.conv1 = Convblock(input_nc, 64, batchnorm=False)
        self.conv2 = Convblock(64, 256)
        self.conv3 = Convblock(256, 512)

        self.final = nn.Conv2d(512, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        if PRINTLOG:
            print(f"D => Input : X {x.shape}, Y {y.shape} ")
        x = torch.cat([x, y], axis=1)
        if PRINTLOG:
            print(f"D => layer0: {x.shape}")
        x = self.conv1(x)
        if PRINTLOG:
            print(f"D => layer1: {x.shape}")
        x = self.conv2(x)
        if PRINTLOG:
            print(f"D => layer2: {x.shape}")
        x = self.conv3(x)
        if PRINTLOG:
            print(f"D => layer3: {x.shape}")
        x = self.final(x)
        x = self.sigmoid(x)
        if PRINTLOG:
            print(f"D => layer5: {x.shape}")
        return x
