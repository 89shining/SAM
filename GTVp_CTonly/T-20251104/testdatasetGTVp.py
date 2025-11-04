"""
测试dataset
四边等距外扩cm
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
    def __init__(self, csv_path, root_dir, nii_dir, target_size, expand_cm):
        self.df = pd.read_csv(csv_path, header=None, names=["image", "mask"])
        self.root_dir = root_dir
        self.nii_dir = nii_dir
        self.target_size = target_size
        self.expand_cm = expand_cm

        # spacing缓存（避免重复读取）
        self.spacing_cache = {}
        self._cache_spacing()

    def _cache_spacing(self):
        """提前缓存每个患者的 spacing 信息（与SAMDataset一致）"""
        for patient_id in sorted(os.listdir(self.nii_dir)):
            nii_path = os.path.join(self.nii_dir, patient_id, "GTVp.nii.gz")
            if not os.path.exists(nii_path):
                continue
            img_nii = sitk.ReadImage(nii_path)
            size_x, size_y = img_nii.GetSize()[:2]
            spacing_x, spacing_y = img_nii.GetSpacing()[:2]
            # mm→cm，考虑resize比例
            resize_factor_x = self.target_size[1] / size_x
            resize_factor_y = self.target_size[0] / size_y
            spacing_x_resized = spacing_x / resize_factor_x / 10.0
            spacing_y_resized = spacing_y / resize_factor_y / 10.0
            self.spacing_cache[patient_id] = (spacing_x_resized, spacing_y_resized)

    def __len__(self):
        return len(self.df)

    def get_box(self, resized_mask, spacing_x, spacing_y, expand_cm):
        y_indices, x_indices = np.where(resized_mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None
        x_min = np.min(x_indices)
        x_max = np.max(x_indices)
        y_min = np.min(y_indices)
        y_max = np.max(y_indices)

        img_width = resized_mask.shape[1]
        img_height = resized_mask.shape[0]

        expand_pixel_x = round(expand_cm / spacing_x)
        expand_pixel_y = round(expand_cm / spacing_y)

        x_min = max(x_min - expand_pixel_x, 0)
        x_max = min(x_max + expand_pixel_x, img_width - 1)
        y_min = max(y_min - expand_pixel_y, 0)
        y_max = min(y_max + expand_pixel_y, img_height - 1)

        box = np.array([x_min, y_min, x_max, y_max]).astype(np.float32)
        box = torch.tensor(box).unsqueeze(0)
        return box

    def __getitem__(self, idx):
        image_rel = self.df.iloc[idx]['image'].lstrip("/\\")
        mask_rel = self.df.iloc[idx]['mask'].lstrip("/\\")
        image_path = os.path.normpath(os.path.join(self.root_dir, image_rel))
        mask_path = os.path.normpath(os.path.join(self.root_dir, mask_rel))

        image = Image.open(image_path)
        original_size = image.size[::-1]  # (H, W)
        image = image.resize(self.target_size, resample=Image.BILINEAR)
        image = np.array(image).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)

        # 原始尺寸 mask
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        mask = Image.fromarray(mask).convert("L")
        mask_np = (np.array(mask) > 0).astype(np.uint8)
        mask = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)
        # 1024 mask
        resized_mask = cv2.resize(mask.squeeze(0).numpy(), self.target_size, interpolation=cv2.INTER_NEAREST)

        # 计算spacing_x, spacing_y
        image_rel_path = self.df.iloc[idx]['mask'].lstrip("/\\")
        patient_id = os.path.basename(os.path.dirname(image_rel_path))  # → "p_0"
        spacing_x, spacing_y = self.spacing_cache[patient_id]

        box = self.get_box(resized_mask, spacing_x, spacing_y, expand_cm=self.expand_cm)
        has_gt = float(box is not None) # 有框 → 1.0，无框 → 0.0

        return {
            'image': image,
            'GT': mask,
            'box': box,
            'has_gt': has_gt,
            'original_size': original_size,
            'image_path': image_path
        }