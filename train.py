import os
import time
import torch
import argparse
import numpy as np
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from SGAN.sgan import SGAN
from Net.utils import save_ckpt, write_loss_2_txt
from SGAN.sgan import SganDataset
from Loss.loss import VggPerceptionLoss, DetectionLoss

torch.cuda.empty_cache()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = 'cuda'


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch implementation of SGAN")
    # parser.add_argument('-t', '--train_path', type=str, default="./Dataset/UIEB/train/")
    parser.add_argument('-t', '--train_path', type=str, default="E:\\WORK_DIR\\Experiment\\datasets\\TFUI\\train")
    parser.add_argument('-v', '--valid_path', type=str, default="./Dataset/UIEB/val/")
    parser.add_argument('-s', '--save_path', type=str, default='')
    parser.add_argument('-k', '--ckpt_save_path', type=str, default="")
    parser.add_argument('-lc', '--load_ckpt', type=bool, default=False)
    parser.add_argument('-l', '--ckpt_load_path', type=str,
                        default="")
    parser.add_argument('-r', '--resize', type=tuple, default=(256, 256))
    parser.add_argument('-g', '--gpu', type=int, default=0)
    # Training hyperparameters
    parser.add_argument('-gl', '--G_learning_rate', help='learning rate', type=float, default=0.0003)
    parser.add_argument('-dl', '--D_learning_rate', help='learning rate', type=float, default=0.00001)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', type=any, default=(0.5, 0.999))
    parser.add_argument('-b', '--batch_size', help='minibatch size', type=int, default=1)
    parser.add_argument('-e', '--epochs', help='number of epochs', type=int, default=500)
    parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('--plot_stats', help='plot stats after every epoch', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    ## ----------------------define Dataset and Dataloader--------------------------------
    # noinspection PyUnboundLocalVariable
    train_dataset = SganDataset(args.train_path, args.resize, 'train')
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0,
                              drop_last=True,
                              pin_memory=True,
                              persistent_workers=True)

    ## -------------------------------load model------------------------------------------
    model = SGAN().to(device)

    if args.load_ckpt:
        print('load ckpt from {}'.format(args.ckpt_load_path))
        model.load_state_dict(torch.load(args.ckpt_load_path))
    ## --------------------define optimizer and scheduler---------------------------------
    params_G = []
    params_D = []
    for name, param in model.named_parameters():
        if name.split('_')[0] == 'gen':
            params_G.append(param)
        else:
            params_D.append(param)

    optimizer_G = optim.AdamW(params_G, lr=args.G_learning_rate, betas=args.adam, weight_decay=5e-3)
    optimizer_D = optim.AdamW(params_D, lr=args.D_learning_rate, betas=args.adam, weight_decay=5e-3)

    # 采用余弦退火调度器-----------------------------------------------------------------------------
    T_MAX = 500  # 周期长度
    scheduler_G = CosineAnnealingLR(
        optimizer_G,
        T_max=T_MAX,
        eta_min=2e-6
    )
    scheduler_D = CosineAnnealingLR(
        optimizer_D,
        T_max=T_MAX,
        eta_min=1e-6
    )

    ## --------------------------define loss function-------------------------------------
    GAN_Loss = nn.MSELoss().to(device)
    Pixel_Loss = nn.SmoothL1Loss(reduction='mean').to(device)
    Perception_Loss = VggPerceptionLoss().to(device)
    SSIM_Loss = SSIM().to(device)
    Detection_Loss = DetectionLoss().to(device)

    ## 记录各个损失
    G_loss = []
    D_loss = []
    per_loss = []
    gan_loss = []
    sim_loss = []
    sem_loss = []
    pix_loss = []
    det_loss = []
    lr_list = []
    loss_dic = {}
    best = 20

    ## -------------------------------train loop------------------------------------------
    scaler = GradScaler()  # 初始化混合精度缩放器
    for epoch in range(args.epochs):
        start_time = time.time()
        G_epoch_loss = []
        D_epoch_loss = []
        epoch_per_loss = []
        epoch_gan_loss = []
        epoch_sim_loss = []
        epoch_pix_loss = []
        epoch_det_loss = []

        if epoch < 30:
            lambda_pixel = 50
            lambda_Perception = 50
            lambda_ssim = 50
            lambda_gan = 1
            for data in train_loader:
                raw, gt = data  # raw和gt经过同样的预处理，结果为：像素值[-1~1]的tensor张量
                raw = raw.to(device)
                gt = gt.to(device)
                ## ---------------------------train Discriminator-----------------------------
                optimizer_D.zero_grad()
                with autocast():
                    with torch.no_grad():
                        g_enhanced, _ = model(raw)  # 计算D损失用不上，所以不建计算图。
                    _, d_enhanced = model(g_enhanced.detach())
                    _, d_gt = model(gt)
                    D_real_loss = GAN_Loss(d_gt, torch.zeros_like(d_gt) + 0.9)
                    D_fake_loss = GAN_Loss(d_enhanced, torch.zeros_like(d_enhanced) + 0.1)
                    D_total_loss = (D_real_loss + D_fake_loss) / 2
                    D_epoch_loss.append(D_total_loss.item())
                scaler.scale(D_total_loss).backward()
                scaler.unscale_(optimizer_D)
                nn.utils.clip_grad_norm_(params_D, 1)
                scaler.step(optimizer_D)
                scaler.update()

                ## ---------------------------train Generator---------------------------------
                for _ in range(2):
                    optimizer_G.zero_grad()
                    with autocast():
                        enhanced, _ = model(raw)    # 输出[-1,1]的tensor(b, c, h, w)
                        _, d_enhanced_out = model(enhanced)
                        ## Calculate losses
                        G_gan_loss = GAN_Loss(d_enhanced_out, torch.zeros_like(d_enhanced_out) + 0.9)
                        epoch_gan_loss.append(G_gan_loss.item())
                        G_pixel_loss = Pixel_Loss(enhanced, gt)
                        epoch_pix_loss.append(G_pixel_loss.item())
                        G_perception_loss = Perception_Loss(enhanced, gt)
                        epoch_per_loss.append(G_perception_loss.item())
                        G_ssim_loss = 1 - SSIM_Loss(enhanced, gt)
                        epoch_sim_loss.append(G_ssim_loss.item())
                        G_det_loss = Detection_Loss(enhanced)
                        epoch_det_loss.append(G_det_loss.item())

                        G_total_loss = (lambda_gan * G_gan_loss +
                                        lambda_pixel * G_pixel_loss +
                                        lambda_Perception * G_perception_loss +
                                        lambda_ssim * G_ssim_loss
                                        )
                        G_epoch_loss.append(G_total_loss.item())
                    scaler.scale(G_total_loss).backward()
                    scaler.unscale_(optimizer_G)
                    nn.utils.clip_grad_norm_(params_G, 5)
                    scaler.step(optimizer_G)
                    scaler.update()

        else:
            for data in train_loader:
                lambda_pixel = 50
                lambda_Perception = 50
                lambda_ssim = 50
                lambda_det = 5
                lambda_gan = 1

                raw, gt = data  # raw和gt经过同样的预处理，结果为：像素值[-1~1]的tensor张量
                raw = raw.to(device)
                gt = gt.to(device)
                ## ---------------------------train Discriminator-----------------------------
                optimizer_D.zero_grad()
                with autocast():
                    with torch.no_grad():
                        g_enhanced, _ = model(raw)  # 计算D损失用不上，所以不建计算图。
                    _, d_enhanced = model(g_enhanced.detach())
                    _, d_gt = model(gt)
                    D_real_loss = GAN_Loss(d_gt, torch.zeros_like(d_gt) + 0.9)
                    D_fake_loss = GAN_Loss(d_enhanced, torch.zeros_like(d_enhanced) + 0.1)
                    D_total_loss = (D_real_loss + D_fake_loss) / 2
                    D_epoch_loss.append(D_total_loss.item())
                scaler.scale(D_total_loss).backward()
                scaler.unscale_(optimizer_D)
                nn.utils.clip_grad_norm_(params_D, 1)
                scaler.step(optimizer_D)
                scaler.update()
                ## ---------------------------train Generator---------------------------------
                for _ in range(2):
                    optimizer_G.zero_grad()
                    with autocast():
                        enhanced, _ = model(raw)  # 输出[-1,1]的tensor(b, c, h, w)
                        _, d_enhanced_out = model(enhanced)
                        ## Calculate losses
                        G_gan_loss = GAN_Loss(d_enhanced_out, torch.zeros_like(d_enhanced_out) + 0.9)
                        epoch_gan_loss.append(G_gan_loss.item())
                        G_pixel_loss = Pixel_Loss(enhanced, gt)
                        epoch_pix_loss.append(G_pixel_loss.item())
                        G_perception_loss = Perception_Loss(enhanced, gt)
                        epoch_per_loss.append(G_perception_loss.item())
                        G_ssim_loss = 1 - SSIM_Loss(enhanced, gt)
                        epoch_sim_loss.append(G_ssim_loss.item())
                        G_det_loss = Detection_Loss(enhanced)
                        epoch_det_loss.append(G_det_loss.item())

                        G_total_loss = (lambda_gan * G_gan_loss +
                                        lambda_pixel * G_pixel_loss +
                                        lambda_Perception * G_perception_loss +
                                        lambda_ssim * G_ssim_loss +
                                        lambda_det * G_det_loss
                                        )
                        G_epoch_loss.append(G_total_loss.item())
                    scaler.scale(G_total_loss).backward()
                    scaler.unscale_(optimizer_G)
                    nn.utils.clip_grad_norm_(params_G, 5)
                    scaler.step(optimizer_G)
                    scaler.update()
        end_time = time.time()
        epoch_time = end_time - start_time
        ## 保存损失
        g_loss = np.mean(G_epoch_loss)
        d_loss = np.mean(D_epoch_loss)
        G_loss.append(g_loss)
        D_loss.append(d_loss)
        gan_loss.append(np.mean(epoch_gan_loss))
        pix_loss.append(np.mean(epoch_pix_loss))
        per_loss.append(np.mean(epoch_per_loss))
        sim_loss.append(np.mean(epoch_sim_loss))
        det_loss.append(np.mean(epoch_det_loss))
        lr_list.append(scheduler_G.get_last_lr()[0])

        # save best model
        if best > g_loss.item():
            best = g_loss.item()
            save_ckpt(model, optim, args.ckpt_save_path)

        print("Epoch:{}, lr:{}, G_loss: {}, D_Loss:{}, time:{}".format(epoch, scheduler_G.get_last_lr()[0], g_loss,
                                                                       d_loss, epoch_time))
        scheduler_G.step()
        scheduler_D.step()

    loss_dic["gan"] = gan_loss
    loss_dic["pix"] = pix_loss
    loss_dic["per"] = per_loss
    loss_dic["sim"] = sim_loss
    loss_dic["det"] = sem_loss
    loss_dic["lr"] = lr_list
    loss_dic["G_loss"] = G_loss
    loss_dic["D_loss"] = D_loss
    # write_loss_2_txt(loss_dic, "/root/autodl-tmp/S-GAN/loss.txt")
    # curve_plot(G_loss, D_loss)

