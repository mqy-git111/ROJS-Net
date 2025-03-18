import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, normal=True):
        super(DoubleConv, self).__init__()
        channels = int(out_channels / 2)
        if in_channels > out_channels:
            channels = int(in_channels / 2)

        layers = [
            # in_channels：输入通道数
            # channels：输出通道数
            # kernel_size：卷积核大小
            # stride：步长
            # padding：边缘填充
            nn.Conv3d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),

            nn.Conv3d(channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        ]
        if normal:  # 如果要添加BN层
            layers.insert(1, nn.InstanceNorm3d(channels))
            layers.insert(len(layers) - 1, nn.InstanceNorm3d(out_channels))

        # 构造序列器
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, normal=True):
        super(DownSampling, self).__init__()
        self.maxpool_to_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, normal)
        )

    def forward(self, x):
        return self.maxpool_to_conv(x)


class Unet(nn.Module):
    def __init__(self, in_channels, base_channel, normal=True):
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.normal = normal

        self.inputs = DoubleConv(in_channels, base_channel, self.normal)
        self.down_1 = DownSampling(base_channel, base_channel * 2, self.normal)
        self.down_2 = DownSampling(base_channel * 2, base_channel * 4, self.normal)
        self.down_3 = DownSampling(base_channel * 4, base_channel * 8, self.normal)

    def forward(self, x):
        # down 部分
        x1 = self.inputs(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)

        return x1, x2, x3, x4


if __name__ == '__main__':
    x = torch.rand([1, 1, 128, 128, 128])
    mask = torch.rand([1, 1, 128, 128, 128])
    model = Unet(in_channels=1, base_channel=32, normal=True)
    x1, x2, x3, x4 = model(x)
    print(x1.shape, x2.shape, x3.shape, x4.shape)
