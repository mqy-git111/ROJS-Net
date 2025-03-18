import torch
import torch.nn as nn
from models.transunet3d.Transformer import TransformerModel
from models.transunet3d.Unet_skipconnection import Unet
from models.transunet3d.Embeddings import LearnedPositionalEncoding


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.norm(x))


class Decoder(nn.Module):
    def __init__(self, embedding_dim):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim

        self.DeUp4 = DeUp_Cat(in_channels=self.embedding_dim, out_channels=self.embedding_dim // 2)
        self.DeBlock4 = DeBlock(in_channels=self.embedding_dim // 2)

        self.DeUp3 = DeUp_Cat(in_channels=self.embedding_dim // 2, out_channels=self.embedding_dim // 4)
        self.DeBlock3 = DeBlock(in_channels=self.embedding_dim // 4)  # 32

        self.DeUp2 = DeUp_Cat(in_channels=self.embedding_dim // 4, out_channels=self.embedding_dim // 8)
        self.DeBlock2 = DeBlock(in_channels=self.embedding_dim // 8)  # 16

    def forward(self, x1_1, x2_1, x3_1, x8):
        return self.decode(x1_1, x2_1, x3_1, x8)

    def decode(self, x1_1, x2_1, x3_1, x8):
        y4 = self.DeUp4(x8, x3_1)  # (1, 64, 32, 32, 32)
        y3_1 = self.DeBlock4(y4)

        y3 = self.DeUp3(y3_1, x2_1)  # (1, 32, 64, 64, 64)
        y2_1 = self.DeBlock3(y3)

        y2 = self.DeUp2(y2_1, x1_1)  # (1, 16, 128, 128, 128)
        y2 = self.DeBlock2(y2)

        return y2


class TransUnetEncoder(nn.Module):
    def __init__(self, img_dim, in_channels, num_heads, num_layers,
                 dropout_rate=0.0, attn_dropout_rate=0.0, base_channel=12):
        super(TransUnetEncoder, self).__init__()
        self.img_dim = img_dim
        self.embedding_dim = base_channel * 8
        self.num_heads = num_heads
        self.patch_dim = (self.img_dim[0] // 8, self.img_dim[1] // 8, self.img_dim[2] // 8)
        self.num_channels = in_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.num_patches = self.patch_dim[0] * self.patch_dim[1] * self.patch_dim[2]
        self.position_encoding = LearnedPositionalEncoding(self.num_patches, self.embedding_dim)
        self.pe_dropout = nn.Dropout(p=self.dropout_rate)
        self.transformer = TransformerModel(
            base_channel * 8,
            num_layers,
            num_heads,
            base_channel * 8 * 4,
            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(base_channel * 8)
        self.Unet = Unet(in_channels=self.num_channels, base_channel=base_channel, normal=True)
        self.norm = nn.InstanceNorm3d(base_channel * 8)
        self.relu = nn.LeakyReLU(inplace=True)

    def encode(self, x):
        x1_1, x2_1, x3_1, x4_1 = self.Unet(x)
        x = self.norm(x4_1)
        x = self.relu(x)

        b, c, s, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(x.size(0), -1, self.embedding_dim)
        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # apply transformer
        x = self.transformer(x)
        encoder_output = self.pre_head_ln(x)
        x8 = self._reshape_output(encoder_output, b, c, s, h, w)

        return x1_1, x2_1, x3_1, x4_1, x8

    def forward(self, x):
        x1_1, x2_1, x3_1, x4_1, x8 = self.encode(x)
        return x1_1, x2_1, x3_1, x4_1, x8

    def get_last_shared_layer(self):
        return self.pre_head_ln

    @staticmethod
    def _reshape_output(x, b, c, s, h, w):
        x = x.view(b, s, h, w, c)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x

class FinalConv(nn.Module):
    def __init__(self, out_channels, root_feat_maps=16):
        super(FinalConv, self).__init__()

        self.final = nn.Conv3d(root_feat_maps, out_channels, 1)

    def forward(self, x):
        x = self.final(x)
        # a = self.final.weight
        return x
class TranUnet(TransUnetEncoder):
    def __init__(self, img_dim, in_channels, base_channel, num_heads, num_layers,
                 dropout_rate=0.0, attn_dropout_rate=0.0):
        super(TranUnet, self).__init__(
            img_dim=img_dim,
            in_channels=in_channels,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            base_channel=base_channel
        )
        self.decoder = Decoder(base_channel*8)
        self.seg = nn.Conv3d(base_channel, 2, 1)  #第二个参数就是最后输出的通道数
        self.finalorganconv = nn.Conv3d(base_channel, 4, 1)
        self.finaltumorconv = nn.Conv3d(base_channel, 2, 1)

    def forward(self, x):
        x1_1, x2_1, x3_1, x4_1, x8 = super(TranUnet, self).forward(x)
        y = self.decoder(x1_1, x2_1, x3_1, x8)
        # seg_y = self.seg(y)
        seg_y = [self.finalorganconv(y), self.finaltumorconv(y)]
        return seg_y


class DeUp_Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp_Cat, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y


class DeBlock(nn.Module):
    def __init__(self, in_channels):
        super(DeBlock, self).__init__()

        self.bn1 = nn.InstanceNorm3d(in_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm3d(in_channels)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = self.conv2(x1)
        x1 = x1 + x
        return x1


def make_trans_unet(img_dim):
    in_channels = 1
    model = TranUnet(img_dim, in_channels, base_channel=32, num_heads=8, num_layers=4,
                     dropout_rate=0.1, attn_dropout_rate=0.1)
    return model


if __name__ == '__main__':
    with torch.no_grad():
        import os

        # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        # cuda0 = torch.device('cuda:1')
        x = torch.rand((1, 1, 128, 128, 128))
        model = make_trans_unet((128, 128, 128))
        model.eval()
        # model.cuda()
        seg = model(x)
        print(seg.shape)
