"""
测试框四边随机不等外扩0-n cm
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import SimpleITK as sitk


class TestDataset(Dataset):
    def __init__(self, csv_path, root_dir, nii_dir, target_size, max_expand_cm):
        self.df = pd.read_csv(csv_path, header=None, names=["image", "mask"])
        self.root_dir = root_dir
        self.nii_dir = nii_dir
        self.target_size = target_size
        self.max_expand_cm = max_expand_cm  # 支持外扩像素传参

    def __len__(self):
        return len(self.df)

    # 四边随机不等外扩
    def get_box(self, resized_mask, spacing_x, spacing_y, max_expand_cm):
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

        # 真正应用到图像上的“外扩长度（mm）”= 像素 * 每像素物理尺寸（cm) * 10
        dL_mm = expand_left_px * spacing_x * 10.0
        dR_mm = expand_right_px * spacing_x * 10.0
        dT_mm = expand_top_px * spacing_y * 10.0
        dB_mm = expand_bottom_px * spacing_y * 10.0

        # 应用扩展并裁剪边界
        x_min = max(x_min - expand_left_px, 0)
        x_max = min(x_max + expand_right_px, img_width - 1)
        y_min = max(y_min - expand_top_px, 0)
        y_max = min(y_max + expand_bottom_px, img_height - 1)

        box = np.array([x_min, y_min, x_max, y_max]).astype(np.float32)
        box = torch.tensor(box).unsqueeze(0)
        # 新增：连同“实际外扩的四边（mm）”一起返回
        expand_mm = (float(dL_mm), float(dR_mm), float(dT_mm), float(dB_mm))
        return box, expand_mm

    def __getitem__(self, idx):
        image_rel = self.df.iloc[idx]['image'].lstrip("/\\")
        mask_rel = self.df.iloc[idx]['mask'].lstrip("/\\")
        image_path = os.path.normpath(os.path.join(self.root_dir, image_rel))
        mask_path = os.path.normpath(os.path.join(self.root_dir, mask_rel))
        # 1024 image
        image = Image.open(image_path)
        original_size = image.size[::-1]  # (H, W)
        image = image.resize(self.target_size, resample=Image.BILINEAR)
        image = np.array(image).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)

        # 原始尺寸512mask
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        mask = Image.fromarray(mask).convert("L")
        mask_np = (np.array(mask) > 0).astype(np.uint8)
        mask = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)
        # 1024 mask
        resized_mask = cv2.resize(mask.squeeze(0).numpy(), self.target_size, interpolation=cv2.INTER_NEAREST)

        # 计算spacing_x, spacing_y
        image_rel_path = self.df.iloc[idx]['mask'].lstrip("/\\")  # "image/p_0/34.nii"
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

        # box = self.get_box(resized_mask, spacing_x_resized, spacing_y_resized,  max_expand_cm=self.max_expand_cm)
        # return image, mask, box, original_size, image_path

        # 改为（新增返回 resized_mask_t, spacing_x_resized, spacing_y_resized；去掉 box）：
        resized_mask_t = torch.from_numpy(resized_mask.astype(np.uint8))
        return image, mask, original_size, image_path, resized_mask_t, float(spacing_x_resized), float(spacing_y_resized)