import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet
from ssd.src import SSD300, Backbone
from torchvision import models, transforms


class VggPerceptionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = models.vgg19(pretrained=True).features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        # 定义常用图层映射
        self.layers = ['conv3_1', 'conv4_1', 'conv5_2']
        self.weights = [0.3, 0.3, 0.4]
        self.layer_mapping = {
            '10': 'conv3_1',
            '19': 'conv4_1',
            '30': 'conv5_2'
        }

    def get_features(self, x, layers=None):
        if layers is None:
            layers = ['Conv5_2']
        x = (x + 1.0) / 2.0
        x = self.normalize(x)
        features = {}
        x0 = x
        for name, layer in self.vgg._modules.items():
            x0 = layer(x0)
            if self.layer_mapping.get(name) in layers:
                mean = x0.detach().mean(dim=[2, 3], keepdim=True)
                std = x0.detach().std(dim=[2, 3], keepdim=True) + 1e-6
                x0 = (x0 - mean) / std

                features[self.layer_mapping[name]] = x0
        return features

    def forward(self, pred, gt):
        """计算多图层特征的 MSE 损失之和"""
        pred_f = self.get_features(pred, self.layers)
        gt_f = self.get_features(gt, self.layers)
        loss = 0.0
        for layer, weight in zip(self.layers, self.weights):
            loss += weight * torch.mean((gt_f[layer] - pred_f[layer]) ** 2)
        return loss




class DetectionLoss(nn.Module):
    ## 通过检测置信度结果计算损失
    def __init__(self):
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to('cuda')
        self.std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to('cuda')
        ## 初始化损失函数
        self.loss = nn.MSELoss(reduction='mean')
        ## 搭建模型
        checkpoint = "./ssd/save_weights/ssd300-99.pth"
        backbone = Backbone()
        self.model = SSD300(backbone=backbone, num_classes=6)
        # 加载模型
        self.model.load_state_dict(torch.load(checkpoint, map_location='cuda')['model'])
        self.model.to('cuda')
        for p in self.model.parameters():
            p.requires_grad_(False)  # ✅ SSD 不被更新
        self.model.eval()

    def img_processing(self, img):
        """计算多图层特征的 MSE 损失之和"""
        ## 图像处理
        img = (img * 127.5) + 127.5
        img = torch.clamp(img, 0, 255)
        img = F.interpolate(img, size=(300, 300), mode='bilinear', align_corners=True)
        img = img / 255.0
        img = (img - self.mean) / self.std
        return img

    def has_target(self, imgs):
        """
        :param imgs: 批量图
        :return: 判断单张图像是否包含目标。
        """
        target_list = []
        for img in imgs:
            img = img.unsqueeze(0)
            pred = self.model(img)[0]
            _, _, scores = pred
            ## 筛选高置信度结果，判断是是否包含目标。
            gt_target = scores[scores >= 0.1]
            target_list.append(gt_target)
        return target_list

    def forward(self, enhanced):
        enhanced_recover = self.img_processing(enhanced)
        enhanced_tar_list = self.has_target(enhanced_recover)
        ## 解析结果
        loss = torch.tensor(0.0, device='cuda', requires_grad=True)
        i = 0   # 一个批量内包含目标的图像数量
        for tar_list in enhanced_tar_list:
            if tar_list.numel() > 0:
                i += 1
                loss = loss + self.loss(tar_list, torch.ones_like(tar_list))
        if i == 0:
            loss = loss + 0.05
        else:
            loss = loss / i
        return loss