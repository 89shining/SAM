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

    # 1024 图像随机外扩框：四个方向固定外扩 pixel
    def get_box(self, mask_np, resize_factor, expand_pixel):
        y_indices, x_indices = np.where(mask_np > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None
        x_min = np.min(x_indices)
        x_max = np.max(x_indices)
        y_min = np.min(y_indices)
        y_max = np.max(y_indices)

        img_width = mask_np.shape[1]  # W
        img_height = mask_np.shape[0]  # H

        # resize后的外扩像素值
        expand_pixel_resized = round(expand_pixel * resize_factor)

        # 四个方向固定外扩
        x_min = max(x_min - expand_pixel_resized, 0)
        x_max = min(x_max + expand_pixel_resized, img_width - 1)
        y_min = max(y_min - expand_pixel_resized, 0)
        y_max = min(y_max + expand_pixel_resized, img_height - 1)

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

        # 计算缩放比例
        resize_factor = self.target_size[0] / original_size[0]

        # 生成box提示
        box = self.get_box(mask_np, resize_factor=resize_factor, expand_pixel=3.0)

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


