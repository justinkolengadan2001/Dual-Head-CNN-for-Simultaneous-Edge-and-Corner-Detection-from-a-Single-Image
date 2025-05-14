import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# Simple residual block
class BasicBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.activation = Mish()

        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel)

        self.downsample = None
        if input_channel != output_channel or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(output_channel)
            )

    def forward(self, x):
        identity = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(identity)

        return self.activation(out + identity)

class ResNetLiteDualHeadSkip(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(input_channel, output_channel):
            return nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(output_channel),
                Mish(),
                nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(output_channel),
                Mish(),
            )

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            Mish(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = BasicBlock(64, 128, stride=2)
        self.layer2 = BasicBlock(128, 256, stride=2)
        self.layer3 = BasicBlock(256, 512, stride=2)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up5 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)

        self.dec1 = conv_block(256 + 256, 256)
        self.dec2 = conv_block(128 + 128, 128)
        self.dec3 = conv_block(64 + 64, 64)
        self.dec4 = conv_block(32, 32)
        self.dec5 = conv_block(16, 16)

        self.out_edge = nn.Conv2d(16, 1, kernel_size=1)
        self.out_corner = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        s = self.stem(x)
        l1 = self.layer1(s)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)

        x = self.up1(l3)
        x = self.dec1(torch.cat([x, l2], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, l1], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, s], dim=1))

        x = self.up4(x)
        x = self.dec4(x)

        x = self.up5(x)
        x = self.dec5(x)

        edge = torch.sigmoid(self.out_edge(x))
        corner = torch.sigmoid(self.out_corner(x))

        return torch.cat([edge, corner], dim = 1)

# ================================== UNet Dual Head Model with Mish Activation ========================================

class UNetDualHead(nn.Module):
    def __init__(self):
        super(UNetDualHead, self).__init__()

        def conv_block(input_channels, output_channels):
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(output_channels),
                Mish(),
                nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(output_channels),
                Mish(),
            )

        self.enc1 = conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(256, 512)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = conv_block(128, 64)

        self.out_edge = nn.Conv2d(64, 1, kernel_size=1)
        self.out_corner = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d1 = self.up1(b)
        d1 = self.dec1(torch.cat([d1, e3], dim=1))
        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d3 = self.up3(d2)
        d3 = self.dec3(torch.cat([d3, e1], dim=1))

        out_edge = torch.sigmoid(self.out_edge(d3))
        out_corner = torch.sigmoid(self.out_corner(d3))

        return torch.cat([out_edge, out_corner], dim=1)
