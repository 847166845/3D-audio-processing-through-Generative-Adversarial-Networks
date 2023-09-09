import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


# class ResidualDenseBlock_5C(nn.Module):
#     def __init__(self, nf=64, gc=32, bias=True):
#         super(ResidualDenseBlock_5C, self).__init__()
#         # gc: growth channel, i.e. intermediate channels
#         self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
#         self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
#         self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
#         self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
#         self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#
#         # initialization
#         # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
#
#     def forward(self, x):
#         x1 = self.lrelu(self.conv1(x))
#         x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
#         x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
#         x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
#         x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
#         return x5 * 0.2 + x
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv1d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv1d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv1d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv1d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv1d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x



class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

import torch.nn as nn
import torch.nn.functional as F

class Upsample(nn.Module):
    def __init__(self, scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')

class Downsample(nn.Module):
    def __init__(self, output_size):
        super(Downsample, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return F.adaptive_avg_pool1d(x, self.output_size)

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv1d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        self.upconv3 = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        self.upconv4 = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Upsampling and Downsampling layers
        self.up = Upsample(scale_factor=3)  # 784*3 = 2352
        self.down = Downsample(output_size=841)  # downsample to 841

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv3(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv4(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        # Apply Upsampling and Downsampling
        out = self.up(out)
        out = self.down(out)

        return out

# #laest version
# class RRDBNet(nn.Module):
#     def __init__(self, in_nc, out_nc, nf, nb, gc=32):
#         super(RRDBNet, self).__init__()
#         RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
#
#         self.conv_first = nn.Conv1d(in_nc, nf, 3, 1, 1, bias=True)
#         self.RRDB_trunk = make_layer(RRDB_block_f, nb)
#         self.trunk_conv = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
#         #### upsampling
#         # note: upsampling will be more complex in 1D
#         # and might require a different approach
#         self.upconv1 = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
#         self.upconv2 = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
#         self.upconv3 = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
#         self.upconv4 = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
#         # self.upconv5 = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
#         self.HRconv = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
#         self.conv_last = nn.Conv1d(nf, out_nc, 3, 1, 1, bias=True)
#
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#
#     def forward(self, x):
#         fea = self.conv_first(x)
#         trunk = self.trunk_conv(self.RRDB_trunk(fea))
#         fea = fea + trunk
#
#         # Note: Upsampling in 1D will be different than in 2D
#         # You might need a different approach
#         # The below is a placeholder and might not work as expected
#         fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
#         fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
#         fea = self.lrelu(self.upconv3(F.interpolate(fea, scale_factor=2, mode='nearest')))
#         fea = self.lrelu(self.upconv4(F.interpolate(fea, scale_factor=2, mode='nearest')))
#         # fea = self.lrelu(self.upconv5(F.interpolate(fea, scale_factor=2, mode='nearest')))
#         out = self.conv_last(self.lrelu(self.HRconv(fea)))
#
#         return out
#

# class RRDBNet(nn.Module):
#     def __init__(self, in_nc, out_nc, nf, nb, gc=32):
#         super(RRDBNet, self).__init__()
#         RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
#
#         self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
#         self.RRDB_trunk = make_layer(RRDB_block_f, nb)
#         self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         #### upsampling
#         self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
#
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#
#     def forward(self, x):
#         fea = self.conv_first(x)
#         trunk = self.trunk_conv(self.RRDB_trunk(fea))
#         fea = fea + trunk
#
#         fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
#         fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
#         out = self.conv_last(self.lrelu(self.HRconv(fea)))
#
#         return out


####################### discriminator
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from basicsr.utils.registry import ARCH_REGISTRY


# @ARCH_REGISTRY.register()
# class VGGStyleDiscriminator1D_v2(nn.Module):
#     """VGG style discriminator with input size 128 x 128 or 256 x 256.
#
#     It is used to train SRGAN, ESRGAN, and VideoGAN.
#
#     Args:
#         num_in_ch (int): Channel number of inputs. Default: 3.
#         num_feat (int): Channel number of base intermediate features.Default: 64.
#     """
#
#     def __init__(self, num_in_ch, num_feat, input_size=128):
#         super(VGGStyleDiscriminator1D_v2, self).__init__()
#         self.input_size = input_size
#         assert self.input_size in [128, 256], (
#             f'input size must be 128 or 256, but received {input_size}')
#
#         self.conv0_0 = nn.Conv1d(num_in_ch, num_feat, 3, 1, 1, bias=True)
#         self.conv0_1 = nn.Conv1d(num_feat, num_feat, 4, 2, 1, bias=False)
#         self.bn0_1 = nn.BatchNorm1d(num_feat, affine=True)
#
#         self.conv1_0 = nn.Conv1d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
#         self.bn1_0 = nn.BatchNorm1d(num_feat * 2, affine=True)
#         self.conv1_1 = nn.Conv1d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
#         self.bn1_1 = nn.BatchNorm1d(num_feat * 2, affine=True)
#
#         self.conv2_0 = nn.Conv1d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
#         self.bn2_0 = nn.BatchNorm1d(num_feat * 4, affine=True)
#         self.conv2_1 = nn.Conv1d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
#         self.bn2_1 = nn.BatchNorm1d(num_feat * 4, affine=True)
#
#         self.conv3_0 = nn.Conv1d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
#         self.bn3_0 = nn.BatchNorm1d(num_feat * 8, affine=True)
#         self.conv3_1 = nn.Conv1d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
#         self.bn3_1 = nn.BatchNorm1d(num_feat * 8, affine=True)
#
#         self.conv4_0 = nn.Conv1d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
#         self.bn4_0 = nn.BatchNorm1d(num_feat * 8, affine=True)
#         self.conv4_1 = nn.Conv1d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
#         self.bn4_1 = nn.BatchNorm1d(num_feat * 8, affine=True)
#
#         if self.input_size == 256:
#             self.conv5_0 = nn.Conv1d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
#             self.bn5_0 = nn.BatchNorm1d(num_feat * 8, affine=True)
#             self.conv5_1 = nn.Conv1d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
#             self.bn5_1 = nn.BatchNorm1d(num_feat * 8, affine=True)
#
#         self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
#         self.linear2 = nn.Linear(100, 1)
#
#         # activation function
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#
#     def forward(self, x):
#         assert x.size(2) == self.input_size, (f'Input size must be identical to input_size, but received {x.size()}.')
#
#         feat = self.lrelu(self.conv0_0(x))
#         feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))  # output spatial size: /2
#
#         feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
#         feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))  # output spatial size: /4
#
#         feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
#         feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))  # output spatial size: /8
#
#         feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
#         feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # output spatial size: /16
#
#         feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
#         feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # output spatial size: /32
#
#         if self.input_size == 256:
#             feat = self.lrelu(self.bn5_0(self.conv5_0(feat)))
#             feat = self.lrelu(self.bn5_1(self.conv5_1(feat)))  # output spatial size: / 64
#
#         # spatial size: (4, 4)
#         feat = feat.view(feat.size(0), -1)
#         feat = self.lrelu(self.linear1(feat))
#         out = self.linear2(feat)
#         return out

@ARCH_REGISTRY.register()
class VGGStyleDiscriminator1D(nn.Module):
    def __init__(self, num_in_ch, num_feat, input_size=49):
        super(VGGStyleDiscriminator1D, self).__init__()
        self.input_size = input_size

        self.conv0_0 = nn.Conv1d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv1d(num_feat, num_feat, 3, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm1d(num_feat, affine=True)

        self.conv1_0 = nn.Conv1d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm1d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv1d(num_feat * 2, num_feat * 2, 3, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm1d(num_feat * 2, affine=True)

        self.linear1 = nn.Linear(1664, 100)  # Adjusted input dimension
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        assert x.size(2) == self.input_size, (f'Input size must be identical to input_size, but received {x.size()}.')

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))

        # spatial size: (4, 4)
        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out


if __name__ == '__main__':
    # Generator
    # output = torch.randn(128, 841)
    # in-channel out-channel should be same size 841
    # model = RRDBNet(128, 128, 128, 23, gc=32)
    # # z = torch.randn(1,10)
    # # batch size, input size, number of coefficients
    # # [1 128 49], [1 128 100], [1 128 196] [1 128 400]
    # z = torch.randn(1, 128, 49)
    # out = model(z)
    # print(out.shape)

    # Discriminator
    # [128 or 256]
    model = VGGStyleDiscriminator1D(128, 64)
    # z = torch.randn(1,10)
    # batch size, input size, number of coefficients
    # [1 128 49], [1 128 100], [1 128 196] [1 128 400]
    z = torch.randn(1, 128, 49)
    out = model(z)
    print(out.shape)



