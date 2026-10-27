import os
import torch
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from SGAN.sgan import SGAN
from Net.utils import pad_to_multiple_of_32, crop_padded_image

torch.cuda.empty_cache()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch implementation of S-GAN")
    # parser.add_argument('-t', '--test_path', type=str,
                        # default="") 
    parser.add_argument('-t', '--test_path', type=str,

    # ---------------------------------------保存路径------------------------------------------------------
    # parser.add_argument('-s', '--save_path', type=str,
    #                     default="")
    

    parser.add_argument('-l', '--ckpt_load_path', type=str,
                        default="")
    return parser.parse_args()


args = parse_args()

# 加载图像名
img_dir = Path(args.test_path)
res_dir = Path(args.save_path)
img_list = list(img_dir.glob("*"))

## -------------------------------load model------------------------------------------
model = SGAN().to(device).eval()
ckpt = torch.load(args.ckpt_load_path, map_location='cuda')
# model.load_state_dict(ckpt["model_state_dict"])
model.load_state_dict(ckpt['model_state_dict'])


## 输出参数量
# from torch.utils.flop_counter import FlopCounterMode
# dummy = torch.ones(1, 3, 256, 256).to("cuda")
# with torch.no_grad(), FlopCounterMode(display=False) as fc:
#     model(dummy)
# flops  = fc.get_total_flops()
# params = sum(p.numel() for p in model.parameters())

with torch.no_grad():
    for img_path in img_list:
        ## 图片处理 图像需要被 32 整除
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if img.size[0] != 256 or img.size[0]!=256:
            img = img.resize((256, 256), resample=Image.Resampling.BICUBIC)
        img = np.array(img, dtype=np.float32)   # [H, W, C]
        # img, pad_info = pad_to_multiple_of_32(img, 32)
        img = (img - 127.5) / 127.5
        img = torch.from_numpy(img).permute(2, 0, 1).float().to("cuda")    # [C, H, W]
        img = img.unsqueeze(0)    # [1, C, H, W]
        # process
        res, _ = model(img)    # [1, C, H, W]
        a = res.max()
        b = res.min()
        ## 恢复图像
        res = res.squeeze(0).permute(1, 2, 0).cpu()
        res = (res * 127.5) + 127.5

        res = torch.clamp(res, 0, 255)
        res = res.numpy().astype(np.uint8)
        # res = crop_padded_image(res, pad_info, (h, w))
        res = Image.fromarray(res)
        # res = image_restore(res, h, w)
        ## 保存图像
        # new_name = img_path.stem + "_sgan.jpg"
        res.save(res_dir / img_path.name)
        print(f"已成功处理图像：{img_path.name}")














