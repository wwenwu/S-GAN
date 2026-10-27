import torch
import random
import numpy as np
from PIL import Image
from pathlib import Path
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from Net.utils import Norm
from Net.fusion import Fusion
from Net.attention import MultiScaleCrossConv
from Net.base_block import GenUp, GenDown, DisUp, ResBlock, DisPerception, GenPerception
from unet.SEM import UNetEncoder


class SganDataset(Dataset):
    def __init__(self, root_dir, resize=(256, 256), mode='val'):
        super().__init__()
        self.resize = resize
        self.img_dir = Path(root_dir) / "input"
        self.gt_dir = Path(root_dir) / "gt"
        self.img_list = list(self.img_dir.glob("*.*"))
        self.mode = mode

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = self.img_list[index]
        gt = self.gt_dir / img.name

        ## 归一化
        img = (img - 127.5) / 127.5
        gt = (gt - 127.5) / 127.5

        return img, gt


class SGAN(nn.Module):

    def __init__(self):
        super().__init__()
        ## 提取分割语义信息并处理
        self.unet = UNetEncoder()
        self.gen_sem_layer = nn.Sequential(nn.MaxPool2d(2, 2),
                                           nn.Conv2d(192, 128, 3, 1, 1),
                                           Norm(128),
                                           nn.LeakyReLU(0.2)
                                           )
        ## G和D的输入层，提取初始特征
        self.gen_input_layer = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1),
                                             Norm(16),
                                             nn.LeakyReLU(0.2)
                                             )

        ## ------------------------initialize Discriminator-------------------------------
        self.dis_up1 = DisUp(128)
        self.dis_block4 = DisPerception(64)
        self.dis_up2 = DisUp(64)
        self.dis_block5 = DisPerception(32)
        self.dis_up3 = DisUp(32)
        self.dis_block6 = DisPerception(16)
        # self.dis_out_layer = nn.Sequential(nn.Conv2d(16, 1, 3, 1, 1),
        #                                    nn.Sigmoid(),
        #                                    nn.MaxPool2d(2, 2)
        #                                    )
        self.dis_out_layer = nn.Conv2d(16, 1, 3, 2, 1)
        ## ------------------------initialize Generator-----------------------------------
        ## Encoder of Generator
        self.gen_block1 = GenPerception(16)
        self.gen_down1 = GenDown(16)
        self.gen_block2 = GenPerception(32)
        self.gen_down2 = GenDown(32)
        self.gen_block3 = GenPerception(64)
        self.gen_down3 = GenDown(64)

        # fusion modules -----------------------------------------------------------------
        self.gen_fusion0 = Fusion(128)  # 融合语义信息
        self.gen_fusion1 = Fusion(64)
        self.gen_fusion2 = Fusion(32)
        self.gen_fusion3 = Fusion(16)

        # Decoder of Generator
        self.gen_up1 = GenUp(128)
        self.gen_block4 = GenPerception(64)
        self.gen_layer1 = nn.Sequential(nn.Conv2d(128, 64, 1, 1),
                                        Norm(64),
                                        nn.LeakyReLU(0.2)
                                        )
        self.gen_up2 = GenUp(64)
        self.gen_block5 = GenPerception(32)
        self.gen_layer2 = nn.Sequential(nn.Conv2d(64, 32, 1, 1),
                                        Norm(32),
                                        nn.LeakyReLU(0.2)
                                        )
        self.gen_up3 = GenUp(32)
        self.gen_block6 = GenPerception(16)
        self.gen_layer3 = nn.Sequential(nn.Conv2d(32, 16, 1, 1),
                                        Norm(16),
                                        nn.LeakyReLU(0.2)
                                        )

        self.gen_out_layer1 = nn.Sequential(nn.ConvTranspose2d(16, 3, 4, 2, 1),
                                            Norm(3),
                                            nn.LeakyReLU(0.2)
                                            )
        self.gen_out_layer2 = nn.Sequential(nn.Conv2d(6, 3, 3, 1, 1),
                                            nn.Tanh()
                                            )

        ## ------------------------initialize BottleNeck-----------------------------------
        ## 同时负责融合语义信息
        self.gen_bottle_layer = GenPerception(128)
        self.dis_bottle_layer = ResBlock(128)

        ## -------------------------initialize skipping--------------------------------------
        self.gen_att = MultiScaleCrossConv(32, 64)
        self.gen_skip1 = nn.Conv2d(16, 16, 3, 1, 1)

    def forward(self, x):
        x0 = x
        s0 = self.unet(x).detach()
        sem = self.gen_sem_layer(s0)
        g0 = self.gen_input_layer(x)    # c=16

        ## ---------------------------Generator_Encoder-----------------------------------
        g0 = self.gen_block1(g0)
        g1 = self.gen_down1(g0)     # c=32
        g1 = self.gen_block2(g1)
        g2 = self.gen_down2(g1)
        g2 = self.gen_block3(g2)
        g3 = self.gen_down3(g2)

        ## ---------------------------------skip------------------------------------------
        g3 = self.gen_fusion0(g3, sem)
        g3 = self.gen_bottle_layer(g3)
        g_1, g_2 = self.gen_att(g1, g2)

        ## -------------------------Discriminator_Decoder---------------------------------
        d3 = self.dis_bottle_layer(g3.detach())
        d3 = self.dis_up1(d3)  # 该 g3 融合了语义信息，且经过了瓶颈层的处理。
        d3 = self.dis_block4(d3)
        d4 = self.dis_up2(d3)
        d4 = self.dis_block5(d4)
        d5 = self.dis_up3(d4)
        d5 = self.dis_block6(d5)
        d_out = self.dis_out_layer(d5)

        ## -------------------------Generator_Decoder---------------------------------
        ## 生成器：上采样->融合->处理
        g3 = self.gen_up1(g3)
        g3 = self.gen_fusion1(g3, d3.detach())
        g3 = torch.cat([g3, g_2], dim=1)
        g3 = self.gen_layer1(g3)
        g3 = self.gen_block4(g3)

        g4 = self.gen_up2(g3)
        g4 = self.gen_fusion2(g4, d4.detach())
        g4 = torch.cat([g4, g_1], dim=1)
        g4 = self.gen_layer2(g4)
        g4 = self.gen_block5(g4)

        g5 = self.gen_up3(g4)
        g5 = self.gen_fusion3(g5, d5.detach())
        g5 = torch.cat([g5, g0], dim=1)
        g5 = self.gen_layer3(g5)
        g5 = self.gen_block6(g5)

        ## output layers
        g_out = torch.cat([self.gen_out_layer1(g5), x0], dim=1)
        g_out = self.gen_out_layer2(g_out)
        return g_out, d_out

