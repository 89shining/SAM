import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import matplotlib.patches as patches



class TestDataset(Dataset):
    def __init__(self, csv_path, root_dir, target_size, num_pos, num_neg):
        self.df = pd.read_csv(csv_path, header=None, names=["image", "mask"])
        self.root_dir = root_dir
        self.target_size = target_size
        self.num_pos = num_pos
        self.num_neg = num_neg

    def __len__(self):
        return len(self.df)

    # 获取中心点
    def get_point_center(self, mask_np):
        y_indices, x_indices = np.where(mask_np > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None
        x_center = int(np.mean(x_indices))
        y_center = int(np.mean(y_indices))
        point_coords = np.array([[x_center, y_center]])
        point_labels = np.array([1])

        return point_coords, point_labels

    # 随机生成若干前景点
    def get_point_pos(self, mask_np, num_pos):
        y_indices, x_indices = np.where(mask_np > 0)
        point_coords = []
        point_labels = []

        for i in range(num_pos):
            idx = np.random.randint(len(x_indices))
            point_coords.append([x_indices[idx], y_indices[idx]])
            point_labels.append(1)

        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)

        return point_coords, point_labels

    # 随机生成若干前景点和背景点
    def get_point_random(self, mask_np, num_pos, num_neg):
        y_pos, x_pos = np.where(mask_np > 0)
        y_neg, x_neg = np.where(mask_np == 0)
        if len(x_neg) < num_neg or len(x_pos) < num_pos:
            return None
        point_coords = []
        point_labels = []

        for i in range(num_pos):
            idx = np.random.randint(len(x_pos))
            point_coords.append([x_pos[idx], y_pos[idx]])
            point_labels.append(1)
        for i in range(num_neg):
            idx = np.random.randint(len(x_neg))
            point_coords.append([x_neg[idx], y_neg[idx]])
            point_labels.append(0)

        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)

        return point_coords, point_labels

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

        # # 生成中心点提示
        # point_coords, point_labels = self.get_point_center(mask_np)
        # # 随机生成若干前景点
        # point_coords, point_labels = self.get_point_pos(mask_np, self.num_pos)
        # 随机生成若干前景点和背景点
        point_coords, point_labels = self.get_point_random(mask_np, self.num_pos, self.num_neg)

        return image, mask, point_coords, point_labels, original_size, image_path

# if __name__ == '__main__':
#     dataset = TestDataset(
#         csv_path="C:/Users/dell/Desktop/task/train_tiff.csv",
#         root_dir="C:/Users/dell/Desktop/task",
#         target_size=(1024, 1024)
#     )
#
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
#
#     for batch_idx, (image, mask, point_coords, point_labels, original_size) in enumerate(dataloader):
# #         # print("Image shape:", image.shape)
#        print("Mask shape:", mask.shape)
# #         # print("Original size:", original_size)


