import os
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import SimpleITK as sitk


class TestDataset(Dataset):
    def __init__(self, csv_path, root_dir, nii_dir, target_size, expand_pixel=0):
        self.df = pd.read_csv(csv_path, header=None, names=["image", "mask"])
        self.root_dir = root_dir
        self.nii_dir = nii_dir
        self.target_size = target_size
        self.expand_pixel = expand_pixel  # 支持外扩像素传参

    def __len__(self):
        return len(self.df)

    def get_box(self, resized_mask, resize_factor, expand_pixel):
        y_indices, x_indices = np.where(resized_mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None
        x_min = np.min(x_indices)
        x_max = np.max(x_indices)
        y_min = np.min(y_indices)
        y_max = np.max(y_indices)

        img_width = resized_mask.shape[1]
        img_height = resized_mask.shape[0]

        expand_pixel_resized = round(expand_pixel * resize_factor)

        x_min = max(x_min - expand_pixel_resized, 0)
        x_max = min(x_max + expand_pixel_resized, img_width - 1)
        y_min = max(y_min - expand_pixel_resized, 0)
        y_max = min(y_max + expand_pixel_resized, img_height - 1)

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

        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        mask = Image.fromarray(mask).convert("L")
        mask_np = (np.array(mask) > 0).astype(np.uint8)
        mask = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)

        resized_mask = cv2.resize(mask.squeeze(0).numpy(), self.target_size, interpolation=cv2.INTER_NEAREST)
        resize_factor = self.target_size[0] / original_size[0]

        box = self.get_box(resized_mask, resize_factor, expand_pixel=self.expand_pixel)

        return image, mask, box, original_size, image_path
