import torch
import torch.nn as nn
from Net.utils import Norm



class Fusion(nn.Module):
    def __init__(self, c_in, dropout=0.1):
        super().__init__()
        self.res_block1 = nn.Sequential(
            nn.Conv2d(c_in, c_in*2, 3, 1, 1),
            Norm(c_in*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c_in*2, c_in*2, 3, 1, 1),
            Norm(c_in*2),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.2)
        )
        self.res_block2 = nn.Sequential(
            nn.Conv2d(c_in*2, c_in*4, 3, 1, 1),
            Norm(c_in*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c_in*4, c_in*4, 3, 1, 1),
            Norm(c_in*4),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.2)
        )
        self.res_block3 = nn.Sequential(
            nn.Conv2d(c_in, c_in*2, 3, 1, 1),
            Norm(c_in*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c_in*2, c_in*2, 3, 1, 1),
            Norm(c_in*2),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.2)
        )
        self.res_block4 = nn.Sequential(
            nn.Conv2d(c_in*2, c_in*4, 3, 1, 1),
            Norm(c_in*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c_in*4, c_in*4, 3, 1, 1),
            Norm(c_in*4),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.2)
        )
        # down sample
        self.down1 = nn.MaxPool2d(2, 2)
        self.down2 = nn.MaxPool2d(2, 2)
        self.down3 = nn.MaxPool2d(2, 2)
        self.down4 = nn.MaxPool2d(2, 2)

        self.reconstruction1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(c_in*4, c_in*2, 3, 1, 1),
            Norm(c_in*2),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.2)
        )
        self.reconstruction2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(c_in*2, c_in, 3, 1, 1),
            Norm(c_in),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.2)
        )
        self.anneal1 = nn.Conv2d(c_in*2, c_in*2, 1, 1)
        self.anneal2 = nn.Conv2d(c_in, c_in, 1, 1)
        self.anneal3 = nn.Conv2d(c_in*2, c_in*2, 1, 1)
        self.anneal4 = nn.Conv2d(c_in, c_in, 1, 1)
        self.act1 = nn.LeakyReLU(0.2)
        self.act2 = nn.LeakyReLU(0.2)

    def forward(self, x, y):
        x0 = x
        y0 = y
        x = self.res_block1(x)
        x = self.down1(x)
        x1 = self.res_block2(x)
        x1 = self.down2(x1)
        y = self.res_block3(y)
        y = self.down3(y)
        y1 = self.res_block4(y)
        y1 = self.down4(y1)
        out = torch.add(x1, y1)
        out = self.reconstruction1(out)
        out = out + self.anneal1(x) + self.anneal3(y)
        out = self.act1(out)
        out = self.reconstruction2(out)
        out = out + self.anneal2(x0) + self.anneal4(y0)
        out = self.act2(out)
        return out



