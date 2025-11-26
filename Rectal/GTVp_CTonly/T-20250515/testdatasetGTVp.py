import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import matplotlib.patches as patches
import SimpleITK as sitk


class TestDataset(Dataset):
    def __init__(self, csv_path, root_dir, nii_dir, target_size):
        self.df = pd.read_csv(csv_path, header=None, names=["image", "mask"])
        self.root_dir = root_dir
        self.nii_dir = nii_dir
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    # 1024 图像随机外扩框：四个方向不等随机外扩
    def get_box(self, mask_np, spacing_x, spacing_y, max_expand_cm=1.0):
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
        expand_left_px = int(expand_left_cm / spacing_x)
        expand_right_px = int(expand_right_cm / spacing_x)
        expand_top_px = int(expand_top_cm / spacing_y)
        expand_bottom_px = int(expand_bottom_cm / spacing_y)

        # 应用扩展并裁剪边界
        x_min = max(x_min - expand_left_px, 0)
        x_max = min(x_max + expand_right_px, img_width - 1)
        y_min = max(y_min - expand_top_px, 0)
        y_max = min(y_max + expand_bottom_px, img_height - 1)

        box = np.array([x_min, y_min, x_max, y_max])
        return box

    def __getitem__(self, idx):
        # 按列分别获取每一行的image和mask
        image_path = os.path.join(self.root_dir, self.df.iloc[idx]['image'].lstrip("/\\"))
        mask_path = os.path.join(self.root_dir, self.df.iloc[idx]['mask'].lstrip("/\\"))
        # image_path = self.root_dir + self.df.iloc[idx]['image']
        # mask_path = self.root_dir + self.df.iloc[idx]['mask']

        # 读取图像
        image = Image.open(image_path)
        image = image.convert("RGB")  # RGB格式
        original_size = image.size[::-1]  # (H, W)
        # print(original_size)
        # # 查看图像
        # img_np = np.array(image)
        # print(f"Min: {img_np.min()}, Max: {img_np.max()}")
        transforms_image = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor()
        ])
        image = transforms_image(image)  # [0,1],[C,H,W] float32

        # 读取 Mask
        mask = (Image.open(mask_path).convert("L"))
        mask = mask.resize(self.target_size, resample=Image.NEAREST)  # 大小
        mask_np = (np.array(mask) > 0).astype(np.uint8)  # 转换为二值 mask（0/1）
        # print(mask_np.shape)
        mask = torch.tensor(mask_np, dtype=torch.long)  # 转换为 PyTorch Tensor，int64 类型
        mask = mask.unsqueeze(0)

        # 计算spacing_x, spacing_y
        image_rel_path = self.df.iloc[idx]['image'].lstrip("/\\")  # "images/p_0/image.tiff"
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
        box = self.get_box(mask_np, spacing_x_resized, spacing_y_resized)

        return image, mask, box, original_size, image_path

# if __name__ == '__main__':
#     dataset = TestDataset(
#         csv_path="C:/Users/dell/Desktop/task/train_tiff.csv",
#         root_dir="C:/Users/dell/Desktop/task",
#         target_size=(1024, 1024)
#     )
#
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
#
#     for batch_idx, (image, mask, box, original_size) in enumerate(dataloader):
# #         # print("Image shape:", image.shape)
#        print("Mask shape:", mask.shape)
# #         # print("Box shape:", box.shape)
# #         # print("Original size:", original_size)


