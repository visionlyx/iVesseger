from net.HCS_Net.attention_block import *

class Resi_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(Resi_Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=padding, groups=groups),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.layer(x)

class Single_Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel):
        super(Single_Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock_done(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock_done, self).__init__()
        self.layer = nn.Sequential(
            Resi_Conv(in_channels, in_channels//2, 1, 1),
            Resi_Conv(in_channels//2, out_channels//2, 3, 1, padding=1, groups=2),
            nn.Conv3d(out_channels//2, out_channels, 1, 1),
            nn.BatchNorm3d(out_channels)
        )
        self.conv = nn.Conv3d(in_channels, out_channels, 1, 1)
        self.active = nn.ReLU(True)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.layer(x)
        return self.active(x1 + x2)

class ResidualBlock_up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock_up, self).__init__()
        self.layer = nn.Sequential(
            Resi_Conv(in_channels, in_channels//2, 1, 1),
            Resi_Conv(in_channels//2, out_channels//2, 3, 1, padding=1, groups=2),
            nn.Conv3d(out_channels//2, out_channels, 1, 1),
            nn.BatchNorm3d(out_channels)
        )

        self.conv = nn.Conv3d(in_channels, out_channels, 1, 1)
        self.active = nn.ReLU(True)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.layer(x)
        return self.active(x1 + x2)

class HCS_Net(nn.Module):

    def __init__(self, in_channels=2, out_channels=1, image_size=128):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv0 = Single_Conv(self.in_channels, 16, 3)
        self.conv1 = ResidualBlock_done(16, 32)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = ResidualBlock_done(32, 64)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = ResidualBlock_done(64, 128)
        self.pool3 = nn.MaxPool3d(2)

        self.conv4 = ResidualBlock_done(128, 256)
        self. up_conv1 = nn.ConvTranspose3d(256, 128, 2, stride=2)

        self.conv5 = ResidualBlock_up(256, 128)
        self. up_conv2 = nn.ConvTranspose3d(128, 64, 2, stride=2)

        self.conv6 = ResidualBlock_up(128, 64)
        self. up_conv3 = nn.ConvTranspose3d(64, 32, 2, stride=2)

        self.conv7 = ResidualBlock_up(64, 32)

        self.channel0 = CBAMBlock(channel=16, reduction=4, size=image_size,  kernel_size=5)
        self.channel1 = CBAMBlock(channel=128, reduction=32, size=int(image_size/4),  kernel_size=5)
        self.channel2 = CBAMBlock(channel=64, reduction=16, size=int(image_size/2),  kernel_size=5)
        self.channel3 = CBAMBlock(channel=32, reduction=8, size=image_size,  kernel_size=5)

        self.up1 = nn.Upsample(mode="trilinear", scale_factor=2, align_corners=False)
        self.up2 = nn.Upsample(mode="trilinear", scale_factor=2, align_corners=False)

        self.mix0 = Single_Conv(128, 32, 3)
        self.mix1 = Single_Conv(96, 24, 3)
        self.mix2 = Single_Conv(56, 14, 3)

        self.conv_final = nn.Conv3d(14, self.out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        x0 = self.conv0(x0)

        x1 = self.channel0(x0)
        x1 = self.conv1(x1)
        x2 = self.pool1(x1)

        x2 = self.conv2(x2)
        x3 = self.pool2(x2)

        x3 = self.conv3(x3)
        x4 = self.pool3(x3)
        x4 = self.conv4(x4)
        x5 = self.up_conv1(x4)

        x3 = self.channel1(x3)
        x5 = torch.concat([x3, x5], dim=1)
        x5 = self.conv5(x5)
        x6 = self.up_conv2(x5)

        x2 = self.channel2(x2)
        x6 = torch.concat([x2, x6], dim=1)
        x6 = self.conv6(x6)
        x7 = self.up_conv3(x6)

        x1 = self.channel3(x1)
        x7 = torch.concat([x1, x7], dim=1)
        x7 = self.conv7(x7)

        x5 = self.mix0(x5)
        x5 = self.up1(x5)
        x8 = torch.concat([x5, x6], dim=1)
        x8 = self.mix1(x8)

        x8 = self.up2(x8)
        x9 = torch.concat([x7, x8], dim=1)
        x9 = self.mix2(x9)

        out = self.conv_final(x9)
        out = self.sigmoid(out)

        return out
