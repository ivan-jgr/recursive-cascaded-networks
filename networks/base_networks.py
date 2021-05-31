import torch
import torch.nn as nn
from torch.nn import ReLU, LeakyReLU
import torch.nn.functional as F


def conv():
    return nn.Conv2d


def trans_conv():
    return nn.ConvTranspose2d


def convolve(in_channels, out_channels, kernel_size, stride):
    return conv()(in_channels, out_channels, kernel_size, stride=stride, padding=1)


def convolveReLU(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(ReLU, convolve(in_channels, out_channels, kernel_size, stride))


def convolveLeakyReLU(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(LeakyReLU(0.1), convolve(in_channels, out_channels, kernel_size, stride))


def upconvolve(in_channels, out_channels, kernel_size, stride):
    return trans_conv()(in_channels, out_channels, kernel_size, stride, padding=1)


def upconvolveReLU(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(ReLU, upconvolve(in_channels, out_channels, kernel_size, stride))


def upconvolveLeakyReLU(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(LeakyReLU(0.1), upconvolve(in_channels, out_channels, kernel_size, stride))


class VTN(nn.Module):
    def __init__(self, input_channels=1, flow_multiplier=1., channels=16):
        super(VTN, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        dim = 2 if input_channels == 1 else 3

        if dim == 3:
            raise NotImplementedError

        # Network architecture
        # The first convolution's input is the concatenated image
        self.conv1 = convolveLeakyReLU(2 * input_channels, channels, 3, 2)
        self.conv2 = convolveLeakyReLU(channels, 2 * channels, 3, 2)
        self.conv3 = convolveLeakyReLU(2 * channels, 4 * channels, 3, 2)
        self.conv3_1 = convolveLeakyReLU(4 * channels, 4 * channels, 3, 1)
        self.conv4 = convolveLeakyReLU(4 * channels, 8 * channels, 3, 2)
        self.conv4_1 = convolveLeakyReLU(8 * channels, 8 * channels, 3, 1)
        self.conv5 = convolveLeakyReLU(8 * channels, 16 * channels, 3, 2)
        self.conv5_1 = convolveLeakyReLU(16 * channels, 16 * channels, 3, 1)
        self.conv6 = convolveLeakyReLU(16 * channels, 32 * channels, 3, 2)
        self.conv6_1 = convolveLeakyReLU(32 * channels, 32 * channels, 3, 1)

        self.pred6 = convolve(32 * channels, dim, 3, 1)
        self.upsamp6to5 = upconvolve(dim, dim, 4, 2)
        self.deconv5 = upconvolveLeakyReLU(32 * channels, 16 * channels, 4, 2)

        self.pred5 = convolve(32 * channels + 2, dim, 3, 1)  # 514 = 32 * channels + 1 + 1
        self.upsamp5to4 = upconvolve(dim, dim, 4, 2)
        self.deconv4 = upconvolveLeakyReLU(32 * channels + 2, 8 * channels, 4, 2)

        self.pred4 = convolve(16 * channels + 2, dim, 3, 1)  # 258 = 64 * channels + 1 + 1
        self.upsamp4to3 = upconvolve(dim, dim, 4, 2)
        self.deconv3 = upconvolveLeakyReLU(16 * channels + 2,  4 * channels, 4, 2)

        self.pred3 = convolve(8 * channels + 2, dim, 3, 1)
        self.upsamp3to2 = upconvolve(dim, dim, 4, 2)
        self.deconv2 = upconvolveLeakyReLU(8 * channels + 2, 2 * channels, 4, 2)

        self.pred2 = convolve(4 * channels + 2, dim, 3, 1)
        self.upsamp2to1 = upconvolve(dim, dim, 4, 2)
        self.deconv1 = upconvolveLeakyReLU(4 * channels + 2, channels, 4, 2)

        self.pred0 = upconvolve(2 * channels + 2, dim, 4, 2)

    def forward(self, fixed, moving):
        concat_image = torch.cat((fixed, moving), dim=1)  # 2 x 512 x 512
        x1 = self.conv1(concat_image)  # 16 x 256 x 256
        x2 = self.conv2(x1)  # 32 x 128 x 128
        x3 = self.conv3(x2)  # 1 x 64 x 64 x 64
        x3_1 = self.conv3_1(x3)  # 64 x 64 x 64
        x4 = self.conv4(x3_1)  # 128 x 32 x 32
        x4_1 = self.conv4_1(x4)  # 128 x 32 x 32
        x5 = self.conv5(x4_1)  # 256 x 16 x 16
        x5_1 = self.conv5_1(x5)  # 256 x 16 x 16
        x6 = self.conv6(x5_1)  # 512 x 8 x 8
        x6_1 = self.conv6_1(x6)  # 512 x 8 x 8

        pred6 = self.pred6(x6_1)  # 2 x 8 x 8
        upsamp6to5 = self.upsamp6to5(pred6)  # 2 x 16 x 16
        deconv5 = self.deconv5(x6_1)  # 256 x 16 x 16
        concat5 = torch.cat([x5_1, deconv5, upsamp6to5], dim=1)  # 514 x 16 x 16

        pred5 = self.pred5(concat5)  # 2 x 16 x 16
        upsamp5to4 = self.upsamp5to4(pred5)  # 2 x 32 x 32
        deconv4 = self.deconv4(concat5)  # 2 x 32 x 32
        concat4 = torch.cat([x4_1, deconv4, upsamp5to4], dim=1)  # 258 x 32 x 32

        pred4 = self.pred4(concat4)  # 2 x 32 x 32
        upsamp4to3 = self.upsamp4to3(pred4)  # 2 x 64 x 64
        deconv3 = self.deconv3(concat4)  # 64 x 64 x 64
        concat3 = torch.cat([x3_1, deconv3, upsamp4to3], dim=1)  # 130 x 64 x 64

        pred3 = self.pred3(concat3)  # 2 x 63 x 64
        upsamp3to2 = self.upsamp3to2(pred3)  # 2 x 128 x 128
        deconv2 = self.deconv2(concat3)  # 32 x 128 x 128
        concat2 = torch.cat([x2, deconv2, upsamp3to2], dim=1)  # 66 x 128 x 128

        pred2 = self.pred2(concat2)  # 2 x 128 x 128
        upsamp2to1 = self.upsamp2to1(pred2)  # 2 x 256 x 256
        deconv1 = self.deconv1(concat2)  # 16 x 256 x 256
        concat1 = torch.cat([x1, deconv1, upsamp2to1], dim=1)  # 34 x 256 x 256

        pred0 = self.pred0(concat1)  # 2 x 512 x 512

        return pred0 * 20 * self.flow_multiplier  # why the 20?


class VTNAffineStem(nn.Module):
    """
    This implementation is based on the pytorch version in https://github.com/voxelmorph/voxelmorph
    """
    def __init__(self, input_channels=1, channels=16, flow_multiplier=1.):
        super(VTNAffineStem, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        dim = 2 if input_channels == 1 else 3
        if dim == 3:
            raise NotImplementedError

        # Network architecture
        # The first convolution's input is the concatenated image
        self.conv1 = convolveLeakyReLU(2 * input_channels, channels, 3, 2)
        self.conv2 = convolveLeakyReLU(channels, 2 * channels, 3, 2)
        self.conv3 = convolveLeakyReLU(2 * channels, 4 * channels, 3, 2)
        self.conv3_1 = convolveLeakyReLU(4 * channels, 4 * channels, 3, 1)
        self.conv4 = convolveLeakyReLU(4 * channels, 8 * channels, 3, 2)
        self.conv4_1 = convolveLeakyReLU(8 * channels, 8 * channels, 3, 1)
        self.conv5 = convolveLeakyReLU(8 * channels, 16 * channels, 3, 2)
        self.conv5_1 = convolveLeakyReLU(16 * channels, 16 * channels, 3, 1)
        self.conv6 = convolveLeakyReLU(16 * channels, 32 * channels, 3, 2)
        self.conv6_1 = convolveLeakyReLU(32 * channels, 32 * channels, 3, 1)

        self.fc_loc = nn.Sequential(
            nn.Linear(512 * 8 * 8, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 6)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, fixed, moving):
        concat_image = torch.cat((fixed, moving), dim=1)  # 2 x 512 x 512
        x1 = self.conv1(concat_image)  # 16 x 256 x 256
        x2 = self.conv2(x1)  # 32 x 128 x 128
        x3 = self.conv3(x2)  # 1 x 64 x 64 x 64
        x3_1 = self.conv3_1(x3)  # 64 x 64 x 64
        x4 = self.conv4(x3_1)  # 128 x 32 x 32
        x4_1 = self.conv4_1(x4)  # 128 x 32 x 32
        x5 = self.conv5(x4_1)  # 256 x 16 x 16
        x5_1 = self.conv5_1(x5)  # 256 x 16 x 16
        x6 = self.conv6(x5_1)  # 512 x 8 x 8
        x6_1 = self.conv6_1(x6)  # 512 x 8 x 8

        # Affine transformation
        xs = x6_1.view(-1, 512 * 8 * 8)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        flow = F.affine_grid(theta, moving.size())  # batch x 512 x 512 x 2

        flow = flow.permute(0, 3, 1, 2)  # batch x 2 x 512 x 512
        W = theta[..., :2]
        b = theta[..., 2]
        batch_size = fixed.size()[0]
        det_loss = (W.det() - 1.0).sum() / batch_size

        return flow, W, b, det_loss