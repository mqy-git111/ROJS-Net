import torch
from torch import nn

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels//reduction, bias=False),
            nn.LeakyReLU(inplace=False),
            nn.Linear(in_channels//reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size, stride=stride,
                      padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=False),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=k_size, stride=stride,
                      padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=False)
        )

        self.se = SEBlock(out_channels)

        self.shortcut = nn.Conv3d(in_channels=in_channels,  out_channels=out_channels, kernel_size=(1, 1, 1), stride=stride,
                                  bias=False)

        self.concat = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        x1 = self.layer(x)
        x1 = self.se(x1)
        x2 = self.shortcut(x)
        x = self.concat(x1 + x2)
        return x


class FinalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        super(FinalConvBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size, stride=stride,
                      padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=False),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=k_size, stride=stride,
                      padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        x = self.layer(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, root_feat_maps=16, pool_size=2, p=0.3):
        super(Encoder, self).__init__()

        self.first = ConvBlock(in_channels, root_feat_maps)
        self.down1 = nn.Sequential(
            nn.MaxPool3d(pool_size),
            ConvBlock(root_feat_maps, root_feat_maps * 2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool3d(pool_size),
            ConvBlock(root_feat_maps * 2, root_feat_maps * 4)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool3d(pool_size),
            ConvBlock(root_feat_maps * 4, root_feat_maps * 8)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool3d(pool_size),
            ConvBlock(root_feat_maps * 8, root_feat_maps * 16)
        )

        self.down5 = nn.Sequential(
            nn.MaxPool3d(pool_size),
            FinalConvBlock(root_feat_maps * 16, root_feat_maps * 32)
        )

        self.drop = nn.Dropout3d(p=p, inplace=True)

    def forward(self, x):
        out1 = self.first(x)
        out2 = self.down1(out1)
        out3 = self.down2(out2)
        out4 = self.down3(out3)
        out4 = self.drop(out4)
        out5 = self.down4(out4)
        out5 = self.drop(out5)
        out6 = self.down5(out5)
        out6 = self.drop(out6)

        res = [out1, out2, out3, out4, out5, out6]

        return res

class DecBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mode_upsample=0):
        super().__init__()
        self.mode = mode_upsample
        if mode_upsample == 0:
            self.down = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2, bias=False),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(inplace=False)
            )
            self.conv = ConvBlock(in_channels, out_channels)
        elif mode_upsample == 1:
            self.down = nn.Sequential(
                nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True),
            )
            self.conv = ConvBlock(in_channels + out_channels, out_channels)
        elif mode_upsample == 2:
            self.down = nn.Sequential(
                nn.Upsample(scale_factor=(2, 2, 2), mode='nearest'),
            )
            self.conv = ConvBlock(in_channels + out_channels, out_channels)

    def forward(self, x, y):
        x = self.down(x)
        x = torch.cat([y, x], dim=1)
        x = self.conv(x)
        return x

class Decoder(nn.Module):
    def __init__(self, root_feat_maps=16, mode_upsample=0):
        super(Decoder, self).__init__()

        self.up5 = DecBlock(root_feat_maps * 32, root_feat_maps * 16, mode_upsample)
        self.up4 = DecBlock(root_feat_maps * 16, root_feat_maps * 8, mode_upsample)
        self.up1 = DecBlock(root_feat_maps * 8, root_feat_maps * 4, mode_upsample)
        self.up2 = DecBlock(root_feat_maps * 4, root_feat_maps * 2, mode_upsample)
        self.up3 = DecBlock(root_feat_maps * 2, root_feat_maps, mode_upsample)

    def forward(self, res):
        out1 = res[0]
        out2 = res[1]
        out3 = res[2]
        out4 = res[3]
        out5 = res[4]
        out6 = res[5]
        out = self.up5(out6, out5)
        out = self.up4(out, out4)
        out = self.up1(out, out3)
        out = self.up2(out, out2)
        out = self.up3(out, out1)
        return out

class FinalConv(nn.Module):
    def __init__(self, out_channels, root_feat_maps=16):
        super(FinalConv, self).__init__()

        self.final = nn.Conv3d(root_feat_maps, out_channels, 1)

    def forward(self, x):
        x = self.final(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels,task_num, root_feat_maps=12, pool_size=2, training=True, p=0.3):
        super(UNet, self).__init__()

        self.training = training
        self.root_feat_maps = root_feat_maps
        self.encoder = Encoder(in_channels, self.root_feat_maps, pool_size=2, p=0.3)
        self.decoder = Decoder(self.root_feat_maps, mode_upsample=0)
        # self.finalconv = FinalConv(out_channels, root_feat_maps)
        self.finalorganconv = FinalConv(4, self.root_feat_maps)
        self.finaltumorconv = FinalConv(2, self.root_feat_maps)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = self.encoder(x)
        out = self.decoder(res)
        output = [self.finalorganconv(out),self.finaltumorconv(out)]
        return output

if __name__ == '__main__':
    from torchsummary import summary

    summary(UNet(1, 3).cuda(), (1, 96, 128, 128))