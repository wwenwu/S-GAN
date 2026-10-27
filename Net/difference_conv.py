import torch
from torch import nn


class ConvHD(nn.Module):
    """
    Horizontal difference convolution
    """
    def __init__(self, c_in):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_in, 3, 1, 1, bias=True)
        self._init_hd_weight()
        # regisiter a hook
        self.conv.weight.register_hook(self._constrain_gradient)

    def _init_hd_weight(self):
        conv_weight = self.conv.weight.data
        hd_weight = torch.zeros_like(conv_weight)
        hd_weight[:, :, :, 0] = conv_weight[:, :, :, 0]
        hd_weight[:, :, :, 2] = -hd_weight[:, :, :, 0]
        self.conv.weight = nn.Parameter(hd_weight)

    def _constrain_gradient(self, grad):
        grad[:, :, :, 2] = -grad[:, :, :, 0].clone()
        grad[:, :, :, 1] = 0.0
        return grad

    def forward(self, x):
        return self.conv(x)


class ConvVD(nn.Module):
    """
    Vertical difference convolution
    """
    def __init__(self, c_in):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_in, 3, 1, 1, bias=False)
        self._init_hd_weight()
        # regisiter a hook
        self.conv.weight.register_hook(self._constrain_gradient)

    def _init_hd_weight(self):
        # self.conv.weights = nn.Parameter(torch.ones_like(self.conv.weights))
        conv_weight = self.conv.weight.data
        hd_weight = torch.zeros_like(conv_weight)
        hd_weight[:, :, 0, :] = conv_weight[:, :, 0, :]
        hd_weight[:, :, 2, :] = -hd_weight[:, :, 2, :]
        self.conv.weight = nn.Parameter(hd_weight)

    def _constrain_gradient(self, grad):
        grad[:, :, 2, :] = -grad[:, :, 0, :].clone()
        grad[:, :, 1, :] = 0.0
        return grad

    def forward(self, x):
        return self.conv(x)


class ConvCD(nn.Module):
    """
    Center difference convolution
    """
    def __init__(self, c_in):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_in, 3, 1, 1)
        self._init_cd_weight()
        self.conv.weight.register_hook(self._constrain_gradient)

    def _init_cd_weight(self):
        conv_weight = self.conv.weight.data
        cd_weight = conv_weight.clone()
        center = conv_weight[:, :, 1:2, 1:2]
        # 生成掩码：中心位置为0，周围位置为1（用于标记哪些元素需要调整）
        mask = torch.ones_like(conv_weight)
        mask[:, :, 1:2, 1:2] = 0
        # 周围元素 = 原周围元素 - 中心元素（通过掩码实现只修改周围）
        cd_weight = cd_weight - center * mask
        self.conv.weight = nn.Parameter(cd_weight)

    def _constrain_gradient(self, grad):
        """约束梯度：周围元素的梯度 = 原梯度 - 中心元素的梯度"""
        center_grad = grad[:, :, 1:2, 1:2]  # [c_out, c_in, 1, 1]
        # 生成掩码（中心为0，周围为1）
        mask = torch.ones_like(grad)
        mask[:, :, 1:2, 1:2] = 0
        # 周围元素的梯度调整：减去中心梯度（确保更新后仍满足"周围 = 原周围 - 中心"）
        grad = grad - center_grad * mask
        return grad

    def forward(self, x):
        return self.conv(x)