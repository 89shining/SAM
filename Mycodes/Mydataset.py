import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import albumentations as A
import torch.nn.functional as F


class SAMDataset(Dataset):
    def __init__(self, csv_path, root_dir, target_size, num_pos, num_neg):
        self.df = pd.read_csv(csv_path, header=None,names=["image", "mask"])
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

    # 随机外扩
    def get_box(self, mask_np):
        y_indices, x_indices = np.where(mask_np > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None
        x_min = np.min(x_indices)
        x_max = np.max(x_indices)
        y_min = np.min(y_indices)
        y_max = np.max(y_indices)
        box = np.array([x_min, y_min, x_max, y_max])
        return box

    def mask_prompt(self, mask_np):
        # Step 1a: 腐蚀 + 膨胀，模糊边界
        kernel = np.ones((3, 3), np.uint8)
        processed = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)    # 开运算去噪
        processed = cv2.dilate(processed, kernel, iterations=1)  # 稍膨胀
        processed = gaussian_filter(processed.astype(float), sigma=1.0)  # 模糊

        # Step 1b: 转回二值（也可保留soft mask用于SAM）
        soft_mask = (processed > 0.3).astype(np.uint8)

        # 创建Elastic变换（模拟配准形变）
        transform = A.ElasticTransform(
            alpha=50,  # 形变强度（大一点更明显）
            sigma=5,  # 高斯滤波尺度
            alpha_affine=10,  # 仿射噪声程度
            p=1.0  # 始终执行
        )

        # 应用于掩码（或图像）
        aug = transform(image=soft_mask * 255)  # 注意输入范围应是0~255
        transformed_mask = (aug['image'] > 127).astype(np.uint8)

        # 转为 Tensor 并调整为 [1, 1, H, W] 形状
        mask_prompt_tensor = torch.tensor(transformed_mask)[None, None, ...].float()
        # print("Mask_prompt_tensor shape:", mask_prompt_tensor.shape)

        # SAM 使用的 mask提示
        mask_prompt = F.interpolate(mask_prompt_tensor, size=(256,256), mode='bilinear', align_corners=False)
        mask_prompt = mask_prompt.squeeze(1)
        # print("Mask_prompt shape:", mask_prompt.shape)

        return mask_prompt

    def __getitem__(self, idx):
        # 按列分别获取每一行的image和mask
        image_path = os.path.join(self.root_dir, self.df.iloc[idx]['image'].lstrip("/\\"))
        mask_path = os.path.join(self.root_dir, self.df.iloc[idx]['mask'].lstrip("/\\"))
        # image_path = self.root_dir + self.df.iloc[idx]['image']
        # mask_path = self.root_dir + self.df.iloc[idx]['mask']

        # 读取图像
        image = Image.open(image_path)
        image = image.convert("RGB")  # RGB格式
        transforms_image = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor()
        ])
        image = transforms_image(image)  # [3，1024，1024]，[0,1]，float32
        # # 把对应的image路径保存下来
        # image_id = self.df.iloc[idx]['image'].replace('/images/', '').replace('.tiff', '').replace('.tif', '')

        # 读取 Mask
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(self.target_size, resample=Image.NEAREST)  # 大小
        mask_np = (np.array(mask) > 0).astype(np.uint8)  # 转换为二值 mask（0/1）
        mask = torch.tensor(mask_np, dtype=torch.long)  # 转换为 PyTorch Tensor，int64 类型
        mask = mask.unsqueeze(0)   # [1,1024,1024]

        # # 生成中心点提示
        # point_coords, point_labels = self.get_point_center(mask_np)
        # # 随机生成若干前景点
        # point_coords, point_labels = self.get_point_pos(mask_np, self.num_pos)
        # 随机生成若干前景点和背景点
        # point_coords [N,2]
        # point_labels [N]
        point_coords, point_labels = self.get_point_random(mask_np, self.num_pos, self.num_neg)
        # print(point_coords.shape)
        # print(point_labels.shape)

        # 生成box提示  [4]
        box = self.get_box(mask_np)
        # print(box.shape)

        # 手动处理的mask提示  [1, 256, 256]
        mask_input = self.mask_prompt(mask_np)
        # print(mask_input.shape)

        return image, mask, point_coords, point_labels, box, mask_input


# if __name__ == '__main__':
#     dataset = SAMDataset(
#         csv_path="C:/Users/dell/Desktop/SAM/GTVp_CTonly/20250515/Dataset/train/train_tiff.csv",
#         root_dir="C:/Users/dell/Desktop/SAM/GTVp_CTonly/20250515/Dataset/train",
#         target_size=(1024, 1024),
#         num_pos=2,
#         num_neg=2
#     )
#
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
#
#     for image, mask, point_coords, point_labels, box, mask_input in dataloader:
        # print("Image shape:", image.shape)     # [B,3,1024,1024]
        # print("Mask shape:", mask.shape)     # [B,1,1024,1024]
        # print("Point Coords shape:", point_coords.shape)     # [B,N,2]
        # print("Point Labels shape:", point_labels.shape)     # [B,N]
        # print("Box shape:", box.shape)      # [B,4]
        # print("Mask_input shape:", mask_input.shape)   # [B,1,256,256]

