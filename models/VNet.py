import torch
import torch.nn as nn
from torchviz import make_dot

# 输入头
class InputBlock(nn.Module):
    def __init__(self, out_channel):
        super(InputBlock, self).__init__()
        self.conv1 = nn.Conv3d(1, out_channel, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm3d(out_channel)
        self.activation1 = nn.LeakyReLU()
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=2, stride=2)
        self.activation2 = nn.LeakyReLU()
    def forward(self, x):
        c1 = self.conv1(x)
        c1 = self.norm(c1)
        c2 = self.activation1(c1)
        c3 = c1 + x
        c4 = self.conv2(c3)
        c4 = self.norm(c4)
        out = self.activation2(c4)
        return c2


# 压缩模块
class CompressionBlock(nn.Module):
    def __init__(self, in_channel, out_channel, layer_num):
        super(CompressionBlock, self).__init__()
        self.layer_num = layer_num
        self.norm = nn.InstanceNorm3d(out_channel)
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size=3, padding=1)
        self.activation1 = nn.LeakyReLU()
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=3, padding=1)
        self.activation2 = nn.LeakyReLU()
        if self.layer_num == 3:
            self.conv3 = nn.Conv3d(out_channel, out_channel, kernel_size=3, padding=1)
            self.activation3 = nn.LeakyReLU()
        self.conv4 = nn.Conv3d(out_channel, out_channel, kernel_size=2, stride=2)
        self.activation4 = nn.LeakyReLU()

    def forward(self, x):
        c1 = self.conv1(x)
        c1 = self.norm(c1)
        c1 = self.activation1(c1)
        out = self.conv2(c1)
        out = self.norm(out)
        out = self.activation2(out)
        if self.layer_num == 3:
            out = self.conv3(out)
            out = self.norm(out)
            out = self.activation4(out)
        out = out + c1
        out = self.conv4(out)
        out = self.norm(out)
        out = self.activation4(out)
        return out


# 解压缩模块
class DeCompressionBlock(nn.Module):
    def __init__(self, in_channel, out_channel, com_block_channel, layer_num):
        super(DeCompressionBlock, self).__init__()
        self.layer_num = layer_num
        self.norm = nn.InstanceNorm3d(out_channel)
        self.deconv1 = nn.ConvTranspose3d(in_channel, out_channel, kernel_size=2, stride=2)
        self.activation1 = nn.LeakyReLU()
        self.conv2 = nn.Conv3d(out_channel + com_block_channel, out_channel, kernel_size=3, padding=1)
        self.activation2 = nn.LeakyReLU()
        self.conv3 = nn.Conv3d(out_channel, out_channel, kernel_size=3, padding=1)
        self.activation3 = nn.LeakyReLU()
        if self.layer_num == 3:
            self.conv4 = nn.Conv3d(out_channel, out_channel, kernel_size=3, padding=1)
            self.activation4 = nn.LeakyReLU()

    def forward(self, x1, x2):
        dc1 = self.deconv1(x1)
        dc1 = self.norm(dc1)
        a1 = self.activation1(dc1)
        concat = torch.cat((a1, x2), axis=1)
        out = self.conv2(concat)
        out = self.norm(out)
        out = self.activation2(out)
        out = self.conv3(out)
        out = self.norm(out)
        out = self.activation3(out)
        if self.layer_num == 3:
            out = self.conv4(out)
            out = self.norm(out)
            out = self.activation4(out)
        out = out + a1
        return out


# 输出头
class OutputBlock_organ(nn.Module):
    def __init__(self, in_channel, out_channel, com_block_channel):
        super(OutputBlock_organ, self).__init__()
        self.norm = nn.InstanceNorm3d(out_channel)
        self.deconv1 = nn.ConvTranspose3d(in_channel, out_channel, kernel_size=2, stride=2)
        self.activation1 = nn.LeakyReLU()
        self.conv2 = nn.Conv3d(out_channel + com_block_channel, out_channel, kernel_size=3, padding=1)
        self.activation2 = nn.LeakyReLU()
        self.conv3 = nn.Conv3d(out_channel, 4, kernel_size=1, padding=0)
        self.activation3 = nn.Softmax(1)

    def forward(self, x1, x2):
        dc1 = self.deconv1(x1)
        dc1 = self.norm(dc1)
        a1 = self.activation1(dc1)
        concat = torch.cat((a1, x2), axis=1)
        out = self.conv2(concat)
        out = self.norm(out)
        out = self.activation2(out)
        out = out + a1
        out = self.conv3(out)
        # out = self.norm(out)
        # out = self.activation3(out)
        return out

class OutputBlock_tumor(nn.Module):
    def __init__(self, in_channel, out_channel, com_block_channel):
        super(OutputBlock_tumor, self).__init__()
        self.norm = nn.InstanceNorm3d(out_channel)
        self.deconv1 = nn.ConvTranspose3d(in_channel, out_channel, kernel_size=2, stride=2)
        self.activation1 = nn.LeakyReLU()
        self.conv2 = nn.Conv3d(out_channel + com_block_channel, out_channel, kernel_size=3, padding=1)
        # self.activation2 = nn.PReLU()
        self.activation2 = nn.LeakyReLU()
        self.conv3 = nn.Conv3d(out_channel, 2, kernel_size=1, padding=0)
        self.activation3 = nn.Softmax(1)

    def forward(self, x1, x2):
        dc1 = self.deconv1(x1)
        a1 = self.activation1(dc1)
        concat = torch.cat((a1, x2), axis=1)
        out = self.conv2(concat)
        out = self.activation2(out)
        out = out + a1
        out = self.conv3(out)
        # out = self.activation3(out)
        return out

class VNet(nn.Module):
    def __init__(self):
        super(VNet, self).__init__()
        self.input_block = InputBlock(out_channel=12)
        # self.cb1 = CompressionBlock(in_channel=16, out_channel=32, layer_num=2)
        # self.cb2 = CompressionBlock(in_channel=32, out_channel=64, layer_num=3)
        # self.cb3 = CompressionBlock(in_channel=64, out_channel=128, layer_num=3)
        # self.cb4 = CompressionBlock(in_channel=128, out_channel=256, layer_num=3)
        # self.dcb1 = DeCompressionBlock(in_channel=256, out_channel=256, com_block_channel=128, layer_num=3)
        # self.dcb2 = DeCompressionBlock(in_channel=256, out_channel=128, com_block_channel=64, layer_num=3)
        # self.dcb3 = DeCompressionBlock(in_channel=128, out_channel=64, com_block_channel=32, layer_num=2)
        # self.output_block_organ = OutputBlock_organ(in_channel=64, out_channel=32, com_block_channel=16)
        # self.output_block_tumor = OutputBlock_tumor(in_channel=64, out_channel=32, com_block_channel=16)



        self.cb1 = CompressionBlock(in_channel=12, out_channel=24, layer_num=2)
        self.cb2 = CompressionBlock(in_channel=24, out_channel=48, layer_num=3)
        self.cb3 = CompressionBlock(in_channel=48, out_channel=96, layer_num=3)
        self.cb4 = CompressionBlock(in_channel=96, out_channel=192, layer_num=3)
        self.dcb1 = DeCompressionBlock(in_channel=192, out_channel=192, com_block_channel=96, layer_num=3)
        self.dcb2 = DeCompressionBlock(in_channel=192, out_channel=96, com_block_channel=48, layer_num=3)
        self.dcb3 = DeCompressionBlock(in_channel=96, out_channel=48, com_block_channel=24, layer_num=2)
        self.output_block_organ = OutputBlock_organ(in_channel=48, out_channel=24, com_block_channel=12)
        self.output_block_tumor = OutputBlock_tumor(in_channel=48, out_channel=24, com_block_channel=12)
    def forward(self, x):
        i = self.input_block(x)
        c1 = self.cb1(i)
        c2 = self.cb2(c1)
        c3 = self.cb3(c2)
        c4 = self.cb4(c3)
        dc1 = self.dcb1(c4, c3)
        dc2 = self.dcb2(dc1, c2)
        dc3 = self.dcb3(dc2, c1)
        out = [self.output_block_organ(dc3, i),self.output_block_tumor(dc3, i)]
        return out
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VNet().to(device)
    print (model)
    pseudo_input = torch.randn(1,1,64,128,128).to(device) # BCDHW
    out = model(pseudo_input)
    print()