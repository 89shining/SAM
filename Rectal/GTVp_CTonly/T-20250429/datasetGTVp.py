import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
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

    # 1024 图像随机外扩框
    def get_box(self, mask_np, max_expand_ratio=0.2):
        y_indices, x_indices = np.where(mask_np > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None
        x_min = np.min(x_indices)
        x_max = np.max(x_indices)
        y_min = np.min(y_indices)
        y_max = np.max(y_indices)

        # 按比例外扩
        width = x_max - x_min
        height = y_max - y_min
        rand_expand_ratio = np.random.uniform(0, max_expand_ratio)

        img_width = mask_np.shape[1]  # W
        img_height = mask_np.shape[0]  # H

        # 随机扩展
        x_min = max(int(x_min - width * rand_expand_ratio), 0)
        y_min = max(int(y_min - height * rand_expand_ratio), 0)
        x_max = min(int(x_max + width * rand_expand_ratio), img_width - 1)
        y_max = min(int(y_max + height * rand_expand_ratio), img_height - 1)

        box = np.array([x_min, y_min, x_max, y_max])
        return box

    # def mask_prompt(self, mask_np):
    #     # Step 1a: 腐蚀 + 膨胀，模糊边界
    #     kernel = np.ones((3, 3), np.uint8)
    #     processed = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)    # 开运算去噪
    #     processed = cv2.dilate(processed, kernel, iterations=1)  # 稍膨胀
    #     processed = gaussian_filter(processed.astype(float), sigma=1.0)  # 模糊
    #
    #     # Step 1b: 转回二值（也可保留soft mask用于SAM）
    #     soft_mask = (processed > 0.3).astype(np.uint8)
    #
    #     # 创建Elastic变换（模拟配准形变）
    #     transform = A.ElasticTransform(
    #         alpha=50,  # 形变强度（大一点更明显）
    #         sigma=5,  # 高斯滤波尺度
    #         alpha_affine=10,  # 仿射噪声程度
    #         p=1.0  # 始终执行
    #     )
    #
    #     # 应用于掩码（或图像）
    #     aug = transform(image=soft_mask * 255)  # 注意输入范围应是0~255
    #     transformed_mask = (aug['image'] > 127).astype(np.uint8)
    #
    #     # 转为 Tensor 并调整为 [1, 1, H, W] 形状
    #     mask_prompt_tensor = torch.tensor(transformed_mask)[None, None, ...].float()
    #     # print("Mask_prompt_tensor shape:", mask_prompt_tensor.shape)
    #
    #     # SAM 使用的 mask提示
    #     mask_prompt = F.interpolate(mask_prompt_tensor, size=(256,256), mode='bilinear', align_corners=False)
    #     mask_prompt = mask_prompt.squeeze(1)
    #     # print("Mask_prompt shape:", mask_prompt.shape)
    #
    #     return mask_prompt

    def __getitem__(self, idx):
        # 按列分别获取每一行的image和mask
        image_path = os.path.join(self.root_dir, self.df.iloc[idx]['image'].lstrip("/\\"))
        mask_path = os.path.join(self.root_dir, self.df.iloc[idx]['mask'].lstrip("/\\"))
        # image_path = self.root_dir + self.df.iloc[idx]['image']
        # mask_path = self.root_dir + self.df.iloc[idx]['mask']

        # 读取图像:1024
        image = Image.open(image_path)
        image = image.convert("RGB")  # RGB格式
        # # 查看图像
        # img_np = np.array(image)
        # print(f"Min: {img_np.min()}, Max: {img_np.max()}")
        transforms_image = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor()
        ])
        image = transforms_image(image)  # [0,1],[C,H,W] float32

        # 读取 Mask:1024
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(self.target_size, resample=Image.NEAREST)  # 大小
        mask_np = (np.array(mask) > 0).astype(np.uint8)  # 转换为二值 mask（0/1）
        mask = torch.tensor(mask_np, dtype=torch.long)  # 转换为 PyTorch Tensor，int64 类型
        mask = mask.unsqueeze(0)

        # # 生成中心点提示
        # point_coords, point_labels = self.get_point_center(mask_np)
        # # 随机生成若干前景点
        # point_coords, point_labels = self.get_point_pos(mask_np, self.num_pos)
        # # 随机生成若干前景点和背景点
        # point_coords, point_labels = self.get_point_random(mask_np, self.num_pos, self.num_neg)

        # 生成box提示
        box = self.get_box(mask_np)

        # # 手动处理的mask提示
        # mask_input = self.mask_prompt(mask_np)

        return image, mask, box

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
#         csv_path="C:/Users/dell/Desktop/task/train_tiff.csv",
#         root_dir="C:/Users/dell/Desktop/task",
#         target_size=(1024, 1024),
#         num_neg=None,
#         num_pos=None
#     )
#
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
#
#     for batch_idx, (image, mask, box) in enumerate(dataloader):
#         # print("Image shape:", image.shape)
#         # print("Mask shape:", mask.shape)
#         # print("Box shape:", box.shape)
#
#         # 保存第一个样本的可视化
#         visualize_image_mask_box(
#             image[0], mask[0], box[0],
#             save_path=f"./vis_train_batch{batch_idx + 1}.png"
#         )
#         break  # 只保存第一个 batch


