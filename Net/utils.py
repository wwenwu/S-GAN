import torch
import datetime
import numpy as np
from pathlib import Path
import cv2
from matplotlib import pyplot as plt


class Norm(torch.nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.norm = torch.nn.InstanceNorm2d(c_in, affine=True)

    def forward(self, x):
        return self.norm(x)


def save_ckpt(model, optim, save_path):
    # check dir
    path = Path(save_path)
    if not path.is_dir():
        AssertionError(f'{path} is not a valid path')
    now = datetime.datetime.now().strftime('%Y-%m-%d')
    # 按日期创建文件夹
    new_path = path / now
    if not new_path.is_dir():
        new_path.mkdir(parents=True, exist_ok=True)

    save_path = new_path / f"latest.pth"
    # print(f"保存模型至: {save_path}")
    torch.save({
        'model_state': model.state_dict(),
        'optim_state': optim.state_dict(),
    }, save_path)


def curve_plot(y1, y2):
    x = np.arange(len(y1))
    plt.figure(figsize=(10, 6))

    # 绘制曲线
    plt.plot(x, y1, marker='o', markersize=4, linewidth=2, label='G_Loss')
    plt.plot(x, y2, marker='s', markersize=4, linewidth=2, label='D_Loss')

    # 添加标题和坐标轴标签
    plt.title("train loss", fontsize=15)
    plt.xlabel("steps", fontsize=12)
    plt.ylabel("loss", fontsize=12)

    # 添加网格和图例
    plt.grid(alpha=0.3)  # 网格线透明度
    plt.legend(fontsize=12)

    # 调整布局
    plt.tight_layout()

    # # 保存图片（如果指定路径）
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     print(f"图片已保存至: {save_path}")
    plt.show()


def pad_to_multiple_of_32(image_np, mul=32):
    """
    通过镜像填充将numpy数组图像调整为32的倍数尺寸
    Args:
        image_np: numpy数组格式的图像，shape为 (height, width, channels) 或 (height, width)
    Returns:
        padded_np: 填充后的numpy数组图像
        original_shape: 原始图像的shape (h, w, c) 或 (h, w)
        pad_info: 填充信息字典，包含{top, bottom, left, right}，用于后续裁剪
        new_shape: 填充后的图像shape
    """
    if not isinstance(image_np, np.ndarray):
        raise TypeError("输入必须是numpy数组格式的图像")
    if len(image_np.shape) not in (2, 3):
        raise ValueError("图像数组维度必须是2D (灰度) 或 3D (彩色)")

    # 获取原始尺寸
    h, w = image_np.shape[:2]
    # 计算目标尺寸（mul 的倍数，向上取整）
    target_w = (min(1, (w % mul)) + (w // mul)) * mul
    target_h = (min(1, (h % mul)) + (h // mul)) * mul
    # 计算需要填充的像素数
    pad_w = target_w - w
    pad_h = target_h - h
    # 分配上下左右的填充量（尽量对称填充，保证镜像效果自然）
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    pad_info = {
        "top": pad_top,
        "bottom": pad_bottom,
        "left": pad_left,
        "right": pad_right
    }
    # 执行镜像填充（BORDER_REFLECT：镜像填充，无重复边界像素）
    # 例如 [1,2,3] → [2,1,1,2,3,3,2]
    padded_np = cv2.copyMakeBorder(image_np, top=pad_top, bottom=pad_bottom, left=pad_left,
                                   right=pad_right, borderType=cv2.BORDER_REFLECT)   # 核心：镜像填充模式
    return padded_np, pad_info


def crop_padded_image(image_np, pad_info, original_shape):
    """
    从填充后的图片中裁剪掉填充部分，还原到原始尺寸
    Args:
        image_np: 填充后的numpy数组图像
        pad_info: 填充信息字典（pad_to_multiple_of_32返回的pad_info）
        original_shape: 原始图像的shape
    Returns:
        cropped_np: 裁剪后的原始尺寸图像
    """
    # 输入验证
    if not isinstance(image_np, np.ndarray):
        raise TypeError("输入必须是numpy数组格式的图像")
    # 提取填充信息
    pad_top = pad_info["top"]
    pad_bottom = pad_info["bottom"]
    pad_left = pad_info["left"]
    pad_right = pad_info["right"]
    # 计算裁剪范围（核心：反向删除填充部分）
    # 起始行：pad_top → 结束行：总行数 - pad_bottom
    # 起始列：pad_left → 结束列：总列数 - pad_right
    h_start = pad_top
    h_end = image_np.shape[0] - pad_bottom
    w_start = pad_left
    w_end = image_np.shape[1] - pad_right
    # 执行裁剪
    cropped_np = image_np[h_start:h_end, w_start:w_end]
    # 验证裁剪结果是否匹配原始尺寸
    if cropped_np.shape[:2] != original_shape:
        raise ValueError(f"裁剪后尺寸不匹配！预期{original_shape}，实际{cropped_np.shape[:2]}")

    return cropped_np


def write_loss_2_txt(loss_dic, path):
    """
    :param loss_dic: 列表字典
    :param path: 写入路径
    :return: None
    """
    with open(path, 'w', encoding='utf-8') as f:
        for loss_name, loss_list in loss_dic.items():
            f.write(f"==={loss_name}===\n")  # 先写入标题
            for item in loss_list:           # 再写入列表
                f.write(f"{item}\n")

            f.write("\n")
