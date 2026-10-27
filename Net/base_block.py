# This block is used to build the base block for encoder and decoder.
import torch
import torch.nn as nn
from torch.amp import autocast
import torch.nn.functional as F
from Net.attention import CAM, SAM
from Net.utils import Norm
from Net.difference_conv import ConvHD, ConvVD, ConvCD


class GenDown(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.layer = nn.Sequential(nn.Conv2d(c_in, c_in * 2, 3, 2, 1, bias=True),
                                   Norm(c_in * 2),
                                   nn.LeakyReLU(0.2))

    def forward(self, x):
        out = self.layer(x)
        return out


class GenUp(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.layer = nn.Sequential(nn.ConvTranspose2d(c_in, c_in // 2, 4, 2, 1),
                                   Norm(c_in // 2),
                                   nn.LeakyReLU(0.2)
                                   )

    def forward(self, x):
        out = self.layer(x)
        return out


class DisDown(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.layer = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Conv2d(c_in, c_in * 2, 1, 1, bias=True),
                                   Norm(c_in * 2),
                                   nn.LeakyReLU(0.2))

    def forward(self, x):
        out = self.layer(x)
        return out


class DisUp(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.layer = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                   nn.Conv2d(c_in, c_in // 2, 1, 1, bias=True),
                                   Norm(c_in // 2),
                                   nn.LeakyReLU(0.2))

    def forward(self, x):
        out = self.layer(x)
        return out


class CCA(nn.Module):
    """
    CCA Block
    """

    def __init__(self, c_in):
        super().__init__()
        self.linear1 = nn.Linear(c_in, c_in)
        self.linear2 = nn.Linear(c_in, c_in)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)),
                                  stride=(x.size(2), x.size(3)))
        avg_pool_x = avg_pool_x.view(x.size(0), 1, -1)
        channel_att_x = self.linear1(avg_pool_x)
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)),
                                  stride=(g.size(2), g.size(3)))
        avg_pool_g = avg_pool_g.view(g.size(0), 1, -1)
        channel_att_g = self.linear2(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        channel_att_sum = channel_att_sum.view(x.size(0), -1, 1, 1)
        scale = torch.sigmoid(channel_att_sum)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


class ResBlock(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_in, 3, 1, 1)
        self.norm1 = Norm(c_in)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(c_in, c_in, 3, 1, 1)
        self.norm2 = Norm(c_in)
        self.act2 = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.act2(out + x)
        return out


class DisPerception(nn.Module):
    def __init__(self, c_in, num_block=2):
        super().__init__()
        self.blocks = nn.Sequential(*[ResBlock(c_in) for _ in range(num_block)])

    def forward(self, x):
        return self.blocks(x)


class GDEM(nn.Module):
    """
    Global detail-enhanced module
    """

    def __init__(self, c_in):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.layer1 = nn.Sequential(nn.Conv2d(c_in, c_in, 1, 1, 0),
                                    nn.LeakyReLU(0.2),
                                    )
        self.layer2 = nn.Sequential(nn.Conv2d(c_in, c_in, 3, 1, 1),
                                    nn.LeakyReLU(0.2),
                                    nn.Conv2d(c_in, c_in, 3, 1, 1)
                                    )
        # self.layer2 = ResBlock(c_in)
        self.conv1 = nn.Conv2d(c_in, c_in, 3, 1, 1)
        self.norm1 = Norm(c_in)
        self.act1 = nn.LeakyReLU(0.2)
        # self.res1 = ResBlock(c_in)
        self.conv2 = nn.Conv2d(c_in, c_in, 3, 1, 1)
        self.norm2 = Norm(c_in)
        self.act2 = nn.LeakyReLU(0.2)
        # self.res2 = ResBlock(c_in)

    def forward(self, x):
        x0 = self.act1(self.norm1(self.conv1(x)))
        # x0 = self.res1(x)
        _, _, h, w = x0.shape
        with autocast('cuda', enabled=False):
            x0 = x0.float()
            freq = torch.fft.rfft2(x0, norm="backward")
            mag = torch.abs(freq)
            pha = torch.angle(freq)
            mag = self.layer1(mag)
            pha = self.layer2(pha)
            pha = (pha % (2 * torch.pi)) - torch.pi

            real = mag * torch.cos(pha)
            imag = mag * torch.sin(pha)
            freq = torch.complex(real, imag)
            x1 = torch.fft.irfft2(freq, s=(h, w), norm="backward")
            x1 = self.act2(self.norm2(self.conv2(x1)))
            # x1 = self.res2(x1)
            out = x + x1 * self.scale
        out = out.to(x.dtype)
        return out


class LDEM(nn.Module):
    """
    Local detail-enhanced module
    """

    def __init__(self, c_in):
        super().__init__()
        self.hd = ConvHD(c_in)
        self.cd = ConvCD(c_in)
        self.vd = ConvVD(c_in)
        self.conv1 = nn.Conv2d(c_in, c_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(c_in, c_in, 3, 1, 1)
        self.conv3 = nn.Conv2d(c_in, c_in, 3, 1, 1)
        self.conv4 = nn.Conv2d(c_in * 3, c_in, 3, 1, 1)
        self.norm = Norm(c_in)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x1 = self.hd(x)
        x2 = self.cd(x)
        x3 = self.vd(x)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        output = torch.cat((x1, x2, x3), 1)
        output = self.act(self.norm(self.conv4(output)))
        return output


class DEM(nn.Module):
    """
    Perception Child Block of Generator --- original
    """

    def __init__(self, c_in):
        super().__init__()
        self.gdem = GDEM(c_in)
        self.ldem = LDEM(c_in)
        self.cam = CAM(c_in)
        self.sam = SAM()
        self.conv1 = nn.Conv2d(c_in, c_in, 3, 1, 1)
        self.norm1 = Norm(c_in)
        self.conv2 = nn.Conv2d(c_in, c_in, 3, 1, 1)
        self.norm2 = Norm(c_in)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(c_in * 4, c_in, 3, 1, 1)
        self.norm3 = Norm(c_in)
        self.act2 = nn.LeakyReLU(0.2)

    def forward(self, x):
        g = self.gdem(x)
        l = self.ldem(x)
        x1 = self.cam(x)
        x1 = self.norm1(self.conv1(x1))
        x2 = self.sam(x1)
        x2 = self.act1(self.norm2(self.conv2(x2)))
        x3 = torch.cat((g, l, x2, x), 1)
        out = self.act2(self.norm3(self.conv3(x3)))
        return out


class GenPerception(nn.Module):
    """
    Perception Block of Generator
    """

    def __init__(self, c_in, num_block=4):
        super().__init__()
        self.num_block = num_block
        self.blocks = nn.Sequential(*[DEM(c_in) for _ in range(self.num_block)])

    def forward(self, x):
        return self.blocks(x)

