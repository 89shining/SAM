import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
import torch
# from torch.utils.data import DataLoader
# import albumentations as A
# import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import SimpleITK as sitk


class SAMDataset(Dataset):
    def __init__(self, csv_path, root_dir, nii_dir, target_size):
        self.df = pd.read_csv(csv_path, header=None,names=["image", "mask", "prompt", "prompt_available"])
        self.root_dir = root_dir
        self.nii_dir = nii_dir
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    # train box
    # 1024 图像随机外扩框：四个方向不等随机外扩0-1cm
    def get_box_train(self, resized_mask, spacing_x, spacing_y, max_expand_cm=1.0):
        y_indices, x_indices = np.where(resized_mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None
        x_min = np.min(x_indices)
        x_max = np.max(x_indices)
        y_min = np.min(y_indices)
        y_max = np.max(y_indices)

        img_width = resized_mask.shape[1]  # W
        img_height = resized_mask.shape[0]  # H

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

        box = np.array([x_min, y_min, x_max, y_max]).astype(np.float32)
        box_train = torch.tensor(box).unsqueeze(0)
        return box_train

    # validation box
    # 1024 图像固定四方向外扩5mm
    def get_box_val(self, resized_mask, spacing_x, spacing_y, expand_cm=0.5):
        y_indices, x_indices = np.where(resized_mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None
        x_min = np.min(x_indices)
        x_max = np.max(x_indices)
        y_min = np.min(y_indices)
        y_max = np.max(y_indices)

        img_width = resized_mask.shape[1]  # W
        img_height = resized_mask.shape[0]  # H

        # 换算成像素数
        expand_x_px = round(expand_cm / spacing_x)
        expand_y_px = round(expand_cm / spacing_y)

        # 应用扩展并裁剪边界
        x_min = max(x_min - expand_x_px, 0)
        x_max = min(x_max + expand_x_px, img_width - 1)
        y_min = max(y_min - expand_y_px, 0)
        y_max = min(y_max + expand_y_px, img_height - 1)

        box = np.array([x_min, y_min, x_max, y_max]).astype(np.float32)
        box_val = torch.tensor(box).unsqueeze(0)
        return box_val

    def __getitem__(self, idx):
        # 按列分别获取每一行的image和mask
        image_path = os.path.join(self.root_dir, self.df.iloc[idx]['image'].lstrip("/\\"))
        mask_path = os.path.join(self.root_dir, self.df.iloc[idx]['mask'].lstrip("/\\"))
        prompt_path = os.path.join(self.root_dir, self.df.iloc[idx]['prompt'].lstrip("/\\"))
        prompt_available = self.df.iloc[idx]["prompt_available"]

        # 读取图像:nii float32
        """
        最终输入train的image：RGB，float32, [3，1024，1024], 0-255
        """
        image_rgb = Image.open(image_path)
        # print(image.shape, image.dtype, image.mode)
        # 调整窗宽窗位， 0-255
        image_resize = image_rgb.resize(self.target_size, resample=Image.BILINEAR)  # (1024,1024)
        image = np.array(image_resize).astype(np.float32)  # float 32
        image = torch.from_numpy(image).permute(2, 0, 1)  # [H,W,3] -> [3,H,W]
        # print(image.shape, image.dtype, image.min(), image.max())

        # 读取 Mask: nii uint8
        """
        最终输入train的mask：L, float32, [1,512,512]，0/1
        """
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        mask = Image.fromarray(mask).convert("L")  # 灰度pil
        mask_np = (np.array(mask) > 0).astype(np.uint8)  # 转换为二值 mask（0/1）
        mask = torch.tensor(mask_np, dtype=torch.float32)  # float32
        mask = mask.unsqueeze(0)    # 0/1,[1,H,W] float32
        # 1024 mask
        resized_mask = cv2.resize(mask.squeeze(0).numpy(), self.target_size, interpolation=cv2.INTER_NEAREST)

        # 计算spacing_x, spacing_y
        image_rel_path = self.df.iloc[idx]['mask'].lstrip("/\\")   # "images/p_0/image.nii"
        patient_id = os.path.basename(os.path.dirname(image_rel_path))  # → "p_0"
        nii_path = os.path.join(self.nii_dir, patient_id, "GTVp.nii.gz")
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
        box_train = self.get_box_train(resized_mask, spacing_x_resized, spacing_y_resized)
        box_val = self.get_box_val(resized_mask, spacing_x_resized, spacing_y_resized)

        # prompt mask 读取与判断
        prompt_tensor = None
        if prompt_available == 1:
            try:
                prompt_np = sitk.GetArrayFromImage(sitk.ReadImage(prompt_path)).astype(np.uint8)
                prompt_resized = cv2.resize(prompt_np, (256, 256), interpolation=cv2.INTER_NEAREST)
                prompt_tensor = torch.tensor(prompt_resized, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
            except Exception as e:
                prompt_tensor = None

        # SAM输入的提示方式：prompt mask优先，其次box
        if prompt_available == 1:
            return {
                'image': image,
                # "image_resized":image_resize,
                'GT': mask,
                'mask_prompt': prompt_tensor,  # float32, 0/1
                'use_mask_prompt': True,
                'train_box':  torch.zeros((1, 4), dtype=torch.float32),
                'val_box':  torch.zeros((1, 4), dtype=torch.float32),
                'image path': image_path
            }
        else:
            return {
                'image': image,
                # "image_resized": image_resize,
                'GT': mask,
                "mask_prompt": torch.zeros((1, 256, 256), dtype=torch.float32),
                "use_mask_prompt": False,
                'train_box': box_train,
                'val_box': box_val,
                'image path': image_path
            }

def visualize_image_mask_box(image_resized, mask, box=None, prompt=None, save_path=None, alpha_gt=0.3, alpha_prompt=0.3):
    """
    显示图像 + 掩膜 + 提示 + 框
    - image: [3, H, W]
    - mask: [1, h1, w1]，将被resize
    - prompt: [1, h2, w2]，将被resize
    - box: 坐标是基于 image 的大小构造的（无需缩放）
    """

    image_np = np.array(image_resized)
    H, W = image_np.shape[:2]

    fig, ax = plt.subplots(1)
    ax.imshow(image_np)

    # 处理 GT mask
    if mask is not None:
        mask_np = mask.squeeze(0).cpu().numpy()
        if mask_np.shape != (H, W):
            mask_np = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_NEAREST)
        ax.imshow(mask_np, cmap='Greens', alpha=alpha_gt)

    # 处理 prompt mask
    if prompt is not None:
        prompt_np = prompt.squeeze(0).cpu().numpy()
        if prompt_np.shape != (H, W):
            prompt_np = cv2.resize(prompt_np, (W, H), interpolation=cv2.INTER_NEAREST)
        ax.imshow(prompt_np, cmap='Reds', alpha=alpha_prompt)

    # 框
    if box is not None:
        box_np = box.squeeze(0).cpu().numpy() if box.ndim == 2 else box.cpu().numpy()
        x_min, y_min, x_max, y_max = box_np
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height,
                                 linewidth=1.5, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)

    ax.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()


# if __name__ == '__main__':
#     dataset = SAMDataset(
#         csv_path="C:/Users/WS/Desktop/MRI_mask/dataset/rgb_dataset.csv",
#         root_dir="C:/Users/WS/Desktop/MRI_mask/dataset",
#         nii_dir="C:/Users/WS/Desktop/MRI_mask/datanii",
#         target_size=(1024, 1024)
#     )
#
#     for idx in range(len(dataset)):
#         sample = dataset[idx]
#         image = sample['image_resized']
#         mask = sample['GT']
#
#         if 'mask_prompt' in sample:
#             prompt = sample['mask_prompt']
#             visualize_image_mask_box(
#                 image_resized=image,
#                 mask=mask,
#                 prompt=prompt,
#                 box=None,
#                 save_path=f"./vis_prompt_sample{idx + 1}.png"
#             )
#             print(f"Saved: vis_prompt_sample{idx + 1}.png")
#         else:
#             box_train = sample['train_box']
#             box_val = sample['val_box']
#             visualize_image_mask_box(
#                 image_resized=image,
#                 mask=mask,
#                 box=box_train,
#                 prompt=None,
#                 save_path=f"./vis_train_sample{idx + 1}.png"
#             )
#             visualize_image_mask_box(
#                 image_resized=image,
#                 mask=mask,
#                 box=box_val,
#                 prompt=None,
#                 save_path=f"./vis_val_sample{idx + 1}.png"
#             )
#             print(f"Saved: vis_train_sample{idx + 1}.png and vis_val_sample{idx + 1}.png")
#
#         if idx >= 4:
#             break  # 只可视化前 5 个样本


