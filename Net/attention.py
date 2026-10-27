import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from Net.utils import Norm


class CAM(nn.Module):
    """
        channel attention
    """

    def __init__(self, c_in, rate=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Sequential(nn.Conv2d(c_in, c_in // rate, 1, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(c_in // rate, c_in, 1, 1)
                                 )
        self.fc2 = nn.Sequential(nn.Conv2d(c_in, c_in // rate, 1, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(c_in // rate, c_in, 1, 1)
                                 )

        self.act = nn.Sigmoid()

    def forward(self, x):
        x_avg = self.fc1(self.avg_pool(x))  # 平均池化分支
        x_max = self.fc2(self.max_pool(x))  # 最大池化分支
        y = self.act(x_avg + x_max)  # 融合两个分支
        return x * y


class SAM(nn.Module):
    """
        spatial attention
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 3, 1, 1, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # [B,1,H,W]
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # [B,1,H,W]
        cat_pool = torch.cat([max_pool, avg_pool], dim=1)  # [B,2,H,W]
        att_mask = self.conv(cat_pool)
        att_mask = self.act(att_mask)
        out = x * att_mask
        return out





class CrossAtt(nn.Module):
    """
        bottleneck: cross attention
    """

    def __init__(self, height, width, batch):
        super().__init__()
        self.batch = batch
        self.height = height
        self.width = width
        self.pos_encoding1 = nn.Parameter(torch.randn(self.batch, 1, self.height, self.width))  # 随机初始化
        self.pos_encoding2 = nn.Parameter(torch.randn(self.batch, 1, self.height, self.width))

        self.weight_q1 = nn.Conv2d(1, 1, 3, padding=1)
        self.weight_k1 = nn.Conv2d(1, 1, 3, padding=1)
        self.weight_v1 = nn.Conv2d(1, 1, 3, padding=1)
        self.weight_q2 = nn.Conv2d(1, 1, 3, padding=1)
        self.weight_k2 = nn.Conv2d(1, 1, 3, padding=1)
        self.weight_v2 = nn.Conv2d(1, 1, 3, padding=1)
        self.x_ff = nn.Conv2d(96, 96, 3, 1, 1)
        self.y_ff = nn.Conv2d(96, 96, 3, 1, 1)

    def forward(self, x, y):
        x0 = x
        y0 = y
        x = torch.mean(x, dim=1, keepdim=True) + self.pos_encoding1
        y = torch.mean(y, dim=1, keepdim=True) + self.pos_encoding2
        x_q = self.weight_q1(x)
        x_k = self.weight_k1(x)
        x_v = self.weight_v1(x)
        y_q = self.weight_q2(y)
        y_k = self.weight_k2(y)
        y_v = self.weight_v2(y)

        x_qk = torch.matmul(x_q, torch.transpose(x_k, -2, -1)) / (np.sqrt(self.width) + 0.0001)
        y1 = torch.matmul(x_qk, y_v)
        y_kq = torch.matmul(y_k, torch.transpose(y_q, -2, -1)) / (np.sqrt(self.width) + 0.0001)
        x1 = torch.matmul(y_kq, x_v)
        x2 = x0 + x1
        y2 = y0 + y1
        x2 = ((x2 - torch.mean(x2, dim=1, keepdim=True)) / (torch.std(x2, dim=1, keepdim=True) + 0.0001))
        y2 = ((y2 - torch.mean(y2, dim=1, keepdim=True)) / (torch.std(y2, dim=1, keepdim=True) + 0.0001))
        x3 = self.x_ff(x2)
        y3 = self.y_ff(y2)
        x3 += x2
        y3 += y2
        x3 = ((x3 - torch.mean(x3, dim=1, keepdim=True)) / (torch.std(x3, dim=1, keepdim=True) + 0.0001))
        y3 = ((y3 - torch.mean(y3, dim=1, keepdim=True)) / (torch.std(y3, dim=1, keepdim=True) + 0.0001))

        return x3, y3


class MultiScaleCrossConv(nn.Module):
    """
        bottleneck: cross attention
    """

    def __init__(self, c_1=24, c_2=48):
        super().__init__()
        self.c_1 = c_1
        self.c_2 = c_2
        self.fixed_dim = 16
        ## 自适应池化，将任意大小的特征图缩成固定的大小
        self.adaptive_pool1 = nn.AdaptiveAvgPool2d((self.fixed_dim, self.fixed_dim))
        self.adaptive_pool2 = nn.AdaptiveAvgPool2d((self.fixed_dim, self.fixed_dim))
        ## 通道数转换层
        self.conv1 = nn.Conv2d(self.c_1, self.c_2, 1)
        self.conv2 = nn.Conv2d(self.c_2, self.c_1, 1)
        ## 线性层
        # self.linear1_3 = nn.Linear(self.fixed_dim ** 2, 9)
        # self.linear1_5 = nn.Linear(self.fixed_dim ** 2, 25)
        # self.linear1_7 = nn.Linear(self.fixed_dim ** 2, 49)
        # self.linear2_3 = nn.Linear(self.fixed_dim ** 2, 9)
        # self.linear2_5 = nn.Linear(self.fixed_dim ** 2, 25)
        # self.linear2_7 = nn.Linear(self.fixed_dim ** 2, 49)

        self.linear1_3 = nn.Sequential(nn.Linear(self.fixed_dim ** 2, 128),
                                       nn.LayerNorm(128),
                                       nn.LeakyReLU(),
                                       nn.Linear(128, 9)
                                       )
        self.linear1_5 = nn.Sequential(nn.Linear(self.fixed_dim ** 2, 128),
                                       nn.LayerNorm(128),
                                       nn.LeakyReLU(),
                                       nn.Linear(128, 25)
                                       )
        self.linear1_7 = nn.Sequential(nn.Linear(self.fixed_dim ** 2, 128),
                                       nn.LayerNorm(128),
                                       nn.LeakyReLU(),
                                       nn.Linear(128, 49)
                                       )
        self.linear2_3 = nn.Sequential(nn.Linear(self.fixed_dim ** 2, 128),
                                       nn.LayerNorm(128),
                                       nn.LeakyReLU(),
                                       nn.Linear(128, 9)
                                       )
        self.linear2_5 = nn.Sequential(nn.Linear(self.fixed_dim ** 2, 128),
                                       nn.LayerNorm(128),
                                       nn.LeakyReLU(),
                                       nn.Linear(128, 25)
                                       )
        self.linear2_7 = nn.Sequential(nn.Linear(self.fixed_dim ** 2, 128),
                                       nn.LayerNorm(128),
                                       nn.LeakyReLU(),
                                       nn.Linear(128, 49)
                                       )

        ## 输出层
        self.layer1 = nn.Sequential(nn.Conv2d(self.c_1 + 3, self.c_1, 3, 1, 1),
                                    Norm(self.c_1),
                                    nn.LeakyReLU(0.2)
                                    )
        self.layer2 = nn.Sequential(nn.Conv2d(self.c_2 + 3, self.c_2, 3, 1, 1),
                                    Norm(self.c_2),
                                    nn.LeakyReLU(0.2)
                                    )

    def forward(self, x, y):
        b = x.shape[0]
        x_fixed = self.adaptive_pool1(x)  # [B, C1, H, W]-> [B, C1, 16, 16]
        y_fixed = self.adaptive_pool2(y)  # [B, C2, H, W]-> [B, C2, 16, 16]
        x_fixed = self.conv1(x_fixed)  # [B, C1, 16, 16]-> [B, C2, 16, 16]
        y_fixed = self.conv2(y_fixed)  # [B, C2, 16, 16]-> [B, C1, 16, 16]
        ## 特征展开
        x_unfold = x_fixed.view(b, self.c_2, self.fixed_dim ** 2)  # [B, C2, 16, 16]-> [B, C2, 256]
        y_unfold = y_fixed.view(b, self.c_1, self.fixed_dim ** 2)  # [B, C1, 16, 16]-> [B, C1, 256]
        # x_unfold = torch.clamp(x_unfold, -100, 100)
        # y_unfold = torch.clamp(y_unfold, -100, 100)
        ## 特征映射（计算卷积核）
        kernel_x3 = self.linear1_3(x_unfold).view(b, self.c_2, 3, 3).mean(dim=0,
                                                                          keepdim=True)  # [B, C2, 256]-> [1, C2, 3, 3]
        kernel_x5 = self.linear1_5(x_unfold).view(b, self.c_2, 5, 5).mean(dim=0,
                                                                          keepdim=True)  # [B, C2, 256]-> [1, C2, 5, 5]
        kernel_x7 = self.linear1_7(x_unfold).view(b, self.c_2, 7, 7).mean(dim=0, keepdim=True)
        kernel_y3 = self.linear2_3(y_unfold).view(b, self.c_1, 3, 3).mean(dim=0,
                                                                          keepdim=True)  # [B, C1, 256]-> [1, C1, 3, 3]
        kernel_y5 = self.linear2_5(y_unfold).view(b, self.c_1, 5, 5).mean(dim=0,
                                                                          keepdim=True)  # [B, C1, 256]-> [1, C1, 5, 5]
        kernel_y7 = self.linear2_7(y_unfold).view(b, self.c_1, 7, 7).mean(dim=0, keepdim=True)
        ## 卷积计算
        x_3 = F.conv2d(input=x, weight=kernel_y3, stride=1, padding=1)  # [B, C1, H, W]-> [B, 1, H, W]
        x_5 = F.conv2d(input=x, weight=kernel_y5, stride=1, padding=2)  # [B, C1, H, W]-> [B, 1, H, W]
        x_7 = F.conv2d(input=x, weight=kernel_y7, stride=1, padding=3)  # [B, C1, H, W]-> [B, 1, H, W]
        y_3 = F.conv2d(input=y, weight=kernel_x3, stride=1, padding=1)
        y_5 = F.conv2d(input=y, weight=kernel_x5, stride=1, padding=2)
        y_7 = F.conv2d(input=y, weight=kernel_x7, stride=1, padding=3)
        ## 特征层拼接
        x_out = torch.cat([x, x_3, x_5, x_7], dim=1)
        y_out = torch.cat([y, y_3, y_5, y_7], dim=1)

        x_out = self.layer1(x_out)
        y_out = self.layer2(y_out)
        return x_out, y_out


