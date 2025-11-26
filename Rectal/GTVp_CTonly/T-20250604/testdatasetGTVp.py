import os

import cv2
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
    def get_box(self, resized_mask, resize_factor, expand_pixel):
        y_indices, x_indices = np.where(resized_mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None
        x_min = np.min(x_indices)
        x_max = np.max(x_indices)
        y_min = np.min(y_indices)
        y_max = np.max(y_indices)

        img_width = resized_mask.shape[1]  # W
        img_height = resized_mask.shape[0]  # H

        # resize后的外扩像素值
        expand_pixel_resized = round(expand_pixel * resize_factor)

        # 四个方向固定外扩
        x_min = max(x_min - expand_pixel_resized, 0)
        x_max = min(x_max + expand_pixel_resized, img_width - 1)
        y_min = max(y_min - expand_pixel_resized, 0)
        y_max = min(y_max + expand_pixel_resized, img_height - 1)

        box = np.array([x_min, y_min, x_max, y_max]).astype(np.float32)
        box = torch.tensor(box).unsqueeze(0)
        return box

    def __getitem__(self, idx):
        # # 按列分别获取每一行的image和mask
        # image_path = os.path.join(self.root_dir, self.df.iloc[idx]['image'].lstrip("/\\"))
        # mask_path = os.path.join(self.root_dir, self.df.iloc[idx]['mask'].lstrip("/\\"))
        # # image_path = self.root_dir + self.df.iloc[idx]['image']
        # # mask_path = self.root_dir + self.df.iloc[idx]['mask']

        # 获取 image 和 mask 相对路径，并清除开头斜杠
        image_rel = self.df.iloc[idx]['image'].lstrip("/\\")
        mask_rel = self.df.iloc[idx]['mask'].lstrip("/\\")
        # 拼接为完整路径
        image_path = os.path.normpath(os.path.join(self.root_dir, image_rel))
        mask_path = os.path.normpath(os.path.join(self.root_dir, mask_rel))

        image = Image.open(image_path)
        original_size = image.size[::-1]  # (H, W)
        # print(image.shape, image.dtype, image.mode)
        # 调整窗宽窗位， 0-255
        image = image.resize(self.target_size, resample=Image.BILINEAR)  # (1024,1024)
        image = np.array(image).astype(np.float32)  # float 32
        image = torch.from_numpy(image).permute(2, 0, 1)  # [H,W,3] -> [3,H,W]

        # 读取 Mask: nii uint8
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        mask = Image.fromarray(mask).convert("L")  # 灰度pil
        mask_np = (np.array(mask) > 0).astype(np.uint8)  # 转换为二值 mask（0/1）
        mask = torch.tensor(mask_np, dtype=torch.float32)  # float32
        mask = mask.unsqueeze(0)  # 0/1,[1,H,W] float32
        # 1024 mask
        resized_mask = cv2.resize(mask.squeeze(0).numpy(), self.target_size, interpolation=cv2.INTER_NEAREST)

        # 计算缩放比例
        resize_factor = self.target_size[0] / original_size[0]

        # 生成box提示
        box = self.get_box(resized_mask, resize_factor=resize_factor, expand_pixel=0.0)

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


