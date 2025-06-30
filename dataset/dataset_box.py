import os

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd

class SAMDataset(Dataset):
    def __init__(self, csv_path, root_dir, target_size = (1024, 1024)):
        # 读取 CSV
        # 第一列image,第二列mask
        self.data = pd.read_csv(csv_path, header=None, names=["image", "mask"])
        self.root_dir = root_dir
        self.target_size = target_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取文件绝对路径
        image_path = self.root_dir + self.data.iloc[idx]["image"]
        mask_path = self.root_dir + self.data.iloc[idx]["mask"]
        print(image_path)

        # 读取图像
        image= cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    # RGB格式
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        image = np.array(image, dtype=np.uint8)
        image = torch.tensor(image).permute(2, 0, 1).float()  # (H, W, C) -> (C, H, W)

        # 读取 Mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 确保是单通道
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)  # 转换为二值 mask（0/1）
        mask_tensor = torch.tensor(mask).long() # 转换为 PyTorch Tensor，int64 类型

        # # 生成 Point Prompt（前景 & 背景点）
        # foreground_points = np.column_stack(np.where(mask == 1))
        # background_points = np.column_stack(np.where(mask == 0))
        #
        # input_point = np.array([
        #     foreground_points[np.random.randint(len(foreground_points))],  # 选一个前景点
        #     background_points[np.random.randint(len(background_points))]   # 选一个背景点
        # ])
        # input_label = np.array([1, 0])  # 1=前景, 0=背景

        # 生成 Box Prompt（边界框）
        y_indices, x_indices = np.where(mask > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        input_box = np.array([x_min, y_min, x_max, y_max])

        return {
            "image": image,
            # "point_coords": torch.tensor(input_point).float(),
            # "point_labels": torch.tensor(input_label).long(),
            "box": torch.tensor(input_box).float(),
            "mask": mask_tensor
        }

# # 使用 Dataset
# csv_path = "C:/Users/dell/Desktop/test46/train_test_tiff.csv"  # 你的 CSV 文件路径
# root_dir = "C:/Users/dell/Desktop/test46"
# dataset = SAMDataset(csv_path, root_dir, target_size=(1024, 1024))
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
#
# # 测试加载
# sample = dataset[0]
# print("Image shape:", sample["image"].shape)
# print("Image dtype:", sample["image"].dtype)
# # print("Point Coords:", sample["point_coords"])
# # print("Point Labels:", sample["point_labels"])
# print("Box:", sample["box"])
# print("Box shape:", sample["box"].shape)
# print("Mask shape:", sample["mask"].shape)
# print("Mask dtype:", sample["mask"].dtype)
