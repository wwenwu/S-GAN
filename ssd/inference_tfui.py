import os
import torch
import cv2
import numpy as np
from PIL import Image
import transforms
from tqdm import tqdm  # 进度条，可选（pip install tqdm）

# -------------------------- 核心配置（必须修改！） --------------------------
# 1. 数据源根目录（包含10个增强方法子文件夹）
SRC_ROOT = "E:\\WORK_DIR\\TFUI\\Enhancement\\"  # 替换为你的数据源根路径
# 2. 检测结果根目录（会自动创建10个ssd_xxx子文件夹）
DEST_ROOT = "E:\\WORK_DIR\\TFUI\\detection_result\\ssd_result"
# 3. 训练好的SSD权重路径
WEIGHTS_PATH = "./save_weights/ssd300-99.pth"
# 4. 类别配置（仅实际目标类别，和训练的--num_classes=5对齐，顺序一致）
CLASS_NAMES = ["1", "2", "3", "4", "5"]
# 5. 模型配置（训练时的num_classes=5+1=6）
NUM_CLASSES_TRAIN = len(CLASS_NAMES) + 1
# 6. 推理参数
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CONF_THRESHOLD = 0.5  # 置信度阈值


# ---------------------------------------------------------------------------

def create_model(num_classes=NUM_CLASSES_TRAIN):
    """复刻训练脚本的模型创建逻辑"""
    from src import SSD300, Backbone
    backbone = Backbone()
    model = SSD300(backbone=backbone, num_classes=num_classes)
    return model


def load_model(model, weights_path, device):
    """加载训练权重"""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"权重文件不存在：{weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()  # 评估模式
    print(f"✅ 成功加载权重并初始化模型")
    return model


def preprocess_image(image_path):
    """图片预处理（和训练验证集一致）"""
    raw_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = raw_img.size

    # 验证集预处理流程
    val_transform = transforms.Compose([
        transforms.Resize(),
        transforms.ToTensor(),
        transforms.Normalization()
    ])
    img_tensor, _ = val_transform(raw_img, target=None)
    img_tensor = torch.unsqueeze(img_tensor, dim=0).to(DEVICE)
    return img_tensor, orig_w, orig_h


def inference_single_image(model, image_path):
    """单张图片推理，返回处理后的检测结果（类别-1，仅类别+置信度）"""
    # 预处理
    img_tensor, _, _ = preprocess_image(image_path)

    # 推理
    with torch.no_grad():
        predictions = model(img_tensor)[0]  # (boxes, labels, scores)

    # 解析结果
    boxes, labels, scores = predictions
    boxes = boxes.cpu().numpy()
    labels = labels.cpu().numpy()
    scores = scores.cpu().numpy()

    # 过滤低置信度
    valid_idx = scores >= CONF_THRESHOLD
    labels = labels[valid_idx]
    scores = scores[valid_idx]

    # 类别从0开始（模型输出labels>=1，减1）
    labels_0based = labels - 1

    # 组合结果：[(类别ID, 置信度), ...]
    results = list(zip(labels_0based, scores))
    return results


def save_detection_result(txt_path, img_name, detection_results):
    """保存检测结果为指定txt格式"""
    with open(txt_path, "w", encoding="utf-8") as f:
        # 第1行：图片名
        f.write(f"{img_name}\n")
        # 第2行：detr（固定值）
        f.write("detr\n")
        # 后续每行：类别ID(0开始) 置信度（保留4位小数）
        for cls_id, conf in detection_results:
            f.write(f"{int(cls_id)} {conf:.4f}\n")


def main():
    # 1. 创建并加载模型
    model = create_model(num_classes=NUM_CLASSES_TRAIN)
    model = load_model(model, WEIGHTS_PATH, DEVICE)

    # 2. 获取所有增强方法子文件夹（数据源根目录下的10个文件夹）
    augment_folders = [f for f in os.listdir(SRC_ROOT)
                       if os.path.isdir(os.path.join(SRC_ROOT, f))]
    if len(augment_folders) == 0:
        raise FileNotFoundError(f"数据源根目录 {SRC_ROOT} 下未找到任何子文件夹")
    print(f"📁 找到 {len(augment_folders)} 个增强方法文件夹：{augment_folders}")

    # 3. 遍历每个增强文件夹，批量推理
    for aug_folder in augment_folders:
        # 3.1 构建输入/输出路径
        aug_src_path = os.path.join(SRC_ROOT, aug_folder)
        aug_dest_path = os.path.join(DEST_ROOT, f"ssd_{aug_folder}")  # 输出文件夹名：ssd_增强算法名
        os.makedirs(aug_dest_path, exist_ok=True)  # 自动创建输出文件夹

        # 3.2 获取当前增强文件夹下的所有图片（支持jpg/png/jpeg）
        img_suffixes = [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]
        img_files = [f for f in os.listdir(aug_src_path)
                     if os.path.splitext(f)[1] in img_suffixes]
        if len(img_files) == 0:
            print(f"⚠️  增强文件夹 {aug_folder} 下无图片，跳过")
            continue

        # 3.3 逐张推理并保存结果
        print(f"\n🚀 开始处理增强方法：{aug_folder}（共 {len(img_files)} 张图片）")
        for img_file in tqdm(img_files, desc=f"处理 {aug_folder}"):
            # 图片完整路径
            img_path = os.path.join(aug_src_path, img_file)
            # 推理
            det_results = inference_single_image(model, img_path)
            # 结果txt路径（和图片同名，后缀改为txt）
            txt_name = os.path.splitext(img_file)[0] + ".txt"
            txt_path = os.path.join(aug_dest_path, txt_name)
            # 保存结果
            save_detection_result(txt_path, img_file, det_results)

    print(f"\n🎉 所有推理完成！结果保存在：{DEST_ROOT}")


if __name__ == "__main__":
    main()