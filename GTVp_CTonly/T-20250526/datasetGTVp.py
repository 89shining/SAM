import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from torchvision import transforms
import torch
# from torch.utils.data import DataLoader
# import albumentations as A
# import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import SimpleITK as sitk
from pathlib import Path

class SAMDataset(Dataset):
    def __init__(self, csv_path, root_dir, nii_dir, target_size):
        self.df = pd.read_csv(csv_path, header=None,names=["image", "mask"])
        self.root_dir = root_dir
        self.nii_dir = nii_dir
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    # train box
    # 1024 图像随机外扩框：四个方向不等随机外扩0-1cm
    def get_box_train(self, mask_np, spacing_x, spacing_y, max_expand_cm=1.0):
        y_indices, x_indices = np.where(mask_np > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None
        x_min = np.min(x_indices)
        x_max = np.max(x_indices)
        y_min = np.min(y_indices)
        y_max = np.max(y_indices)

        img_width = mask_np.shape[1]  # W
        img_height = mask_np.shape[0]  # H

        # 四个方向各自随机外扩 [0, max_expand_cm] cm
        expand_left_cm = np.random.uniform(0, max_expand_cm)
        expand_right_cm = np.random.uniform(0, max_expand_cm)
        expand_top_cm = np.random.uniform(0, max_expand_cm)
        expand_bottom_cm = np.random.uniform(0, max_expand_cm)

        # 换算成像素数
        expand_left_px = round(expand_left_cm / spacing_x)
        expand_right_px = round(expand_right_cm / spacing_x)
        expand_top_px = round(expand_top_cm / spacing_y)
        expand_bottom_px = round(expand_bottom_cm / spacing_y)

        # 应用扩展并裁剪边界
        x_min = max(x_min - expand_left_px, 0)
        x_max = min(x_max + expand_right_px, img_width - 1)
        y_min = max(y_min - expand_top_px, 0)
        y_max = min(y_max + expand_bottom_px, img_height - 1)

        box_train = np.array([x_min, y_min, x_max, y_max])
        return box_train

    # validation box
    # 1024 图像固定四方向外扩5mm
    def get_box_val(self, mask_np, spacing_x, spacing_y, expand_cm=0.5):
        y_indices, x_indices = np.where(mask_np > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None
        x_min = np.min(x_indices)
        x_max = np.max(x_indices)
        y_min = np.min(y_indices)
        y_max = np.max(y_indices)

        img_width = mask_np.shape[1]  # W
        img_height = mask_np.shape[0]  # H

        # 换算成像素数
        expand_x_px = round(expand_cm / spacing_x)
        expand_y_px = round(expand_cm / spacing_y)

        # 应用扩展并裁剪边界
        x_min = max(x_min - expand_x_px, 0)
        x_max = min(x_max + expand_x_px, img_width - 1)
        y_min = max(y_min - expand_y_px, 0)
        y_max = min(y_max + expand_y_px, img_height - 1)

        box_val = np.array([x_min, y_min, x_max, y_max])
        return box_val

    def __getitem__(self, idx):
        # 按列分别获取每一行的image和mask
        image_path = os.path.join(self.root_dir, self.df.iloc[idx]['image'].lstrip("/\\"))
        mask_path = os.path.join(self.root_dir, self.df.iloc[idx]['mask'].lstrip("/\\"))
        # image_path = self.root_dir + self.df.iloc[idx]['image']
        # mask_path = self.root_dir + self.df.iloc[idx]['mask']

        # 读取图像:1024
        image = Image.open(image_path)
        image = image.convert("RGB")  # RGB格式
        # 查看图像
        # img_np = np.array(image)
        # print(img_np.shape)
        # print(f"Min: {img_np.min()}, Max: {img_np.max()}")
        transforms_image = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor()   # [H, W, 3] -> [3, H, W], float32，并将输入[0,255]归一化到[0,1]
        ])
        image = transforms_image(image)  # 0-1,[3,H,W] float32

        # 读取 Mask:1024
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(self.target_size, resample=Image.NEAREST)  # 大小
        mask_np = (np.array(mask) > 0).astype(np.uint8)  # 转换为二值 mask（0/1）
        mask = torch.tensor(mask_np, dtype=torch.float32)  # 转换为 PyTorch Tensor 类型
        mask = mask.unsqueeze(0)    # 0/1,[1,H,W] float32

        # 计算spacing_x, spacing_y
        image_rel_path = self.df.iloc[idx]['image'].lstrip("/\\")   # "images/p_0/image.tiff"
        patient_id = os.path.basename(os.path.dirname(image_rel_path))  # → "p_0"
        nii_path = os.path.join(self.nii_dir, patient_id, "image.nii.gz")
        if not os.path.exists(nii_path):
            raise FileNotFoundError(f"Missing NIfTI image: {nii_path}")
        img_nii = sitk.ReadImage(nii_path)
        # 计算resize比例, GetSize()[W,H,D]
        resize_factor_x = self.target_size[1] / img_nii.GetSize()[0]  # W 1024 / 512 = 2.0
        resize_factor_y = self.target_size[0] / img_nii.GetSize()[1]  # H 同上
        # GetSpacing[W, H, D]
        spacing_x_resized = img_nii.GetSpacing()[0] / resize_factor_x / 10.0  # mm → cm
        spacing_y_resized = img_nii.GetSpacing()[1] / resize_factor_y / 10.0  # mm → cm

        # 生成box提示
        box_train = self.get_box_train(mask_np, spacing_x_resized, spacing_y_resized)
        box_val = self.get_box_val(mask_np, spacing_x_resized, spacing_y_resized)

        return image, mask, box_train, box_val

def visualize_image_mask_box(image, mask, box, save_path=None, alpha=0.5):
    """
    Args:
        image: (3, H, W) Tensor
        mask: (1, H, W) Tensor
        box: (4,) Tensor
        save_path: 保存路径
        alpha: mask透明度
    """
    image_np = image.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    mask_np = mask.squeeze(0).cpu().numpy()  # (H, W)
    box_np = box.cpu().numpy()  # (4,)

    fig, ax = plt.subplots(1)
    ax.imshow(image_np)
    ax.imshow(mask_np, cmap='Greens', alpha=alpha)

    x_min, y_min, x_max, y_max = box_np
    width = x_max - x_min
    height = y_max - y_min
    rect = patches.Rectangle((x_min, y_min), width, height,
                             linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()


# if __name__ == '__main__':
#     dataset = SAMDataset(
#         csv_path="C:/Users/dell/Desktop/SAM/GTVp_CTonly/20250515/Dataset/train/train_tiff.csv",
#         root_dir="C:/Users/dell/Desktop/SAM/GTVp_CTonly/20250515/Dataset/train",    # train数据文件夹
#         nii_dir="C:/Users/dell/Desktop/SAM/GTVp_CTonly/20250515/datanii/traindatanii",   # trainnii数据文件夹
#         target_size=(1024, 1024)
#     )
#
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
#
#     for batch_idx, (image, mask, box_train, box_val) in enumerate(dataloader):
#         print("Image shape:", image.shape)
#         print("Mask shape:", mask.shape)
#         print("Box shape:", box_train.shape)

#         # 保存第一个样本的可视化
#         visualize_image_mask_box(
#             image[0], mask[0], box_train[0],
#             save_path=f"./vis_train_batch{batch_idx + 1}.png"
#         )
#
#         # 保存第一个样本的可视化
#         visualize_image_mask_box(
#             image[0], mask[0], box_val[0],
#             save_path=f"./vis_val_batch{batch_idx + 1}.png"
#         )
#         break  # 只保存第一个 batch


