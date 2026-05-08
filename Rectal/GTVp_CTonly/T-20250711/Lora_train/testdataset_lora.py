import os
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import SimpleITK as sitk


class TestDatasetLoRA(Dataset):
    def __init__(self, csv_path, root_dir, nii_dir, target_size=(1024, 1024), expand_cm=0.0):
        self.df = pd.read_csv(csv_path, header=None, names=["image", "mask"])
        self.root_dir = root_dir
        self.nii_dir = nii_dir
        self.target_size = target_size
        self.expand_cm = expand_cm

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _compute_box(mask_2d, spacing_x_cm, spacing_y_cm, expand_cm):
        ys, xs = np.where(mask_2d > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None

        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)

        h, w = mask_2d.shape
        ex = expand_cm / spacing_x_cm
        ey = expand_cm / spacing_y_cm

        x_min = max(x_min - ex, 0)
        x_max = min(x_max + ex, w - 1)
        y_min = max(y_min - ey, 0)
        y_max = min(y_max + ey, h - 1)

        box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
        return torch.tensor(box).unsqueeze(0)

    def __getitem__(self, idx):
        image_rel = self.df.iloc[idx]["image"].lstrip("/\\")
        mask_rel = self.df.iloc[idx]["mask"].lstrip("/\\")
        image_path = os.path.normpath(os.path.join(self.root_dir, image_rel))
        mask_path = os.path.normpath(os.path.join(self.root_dir, mask_rel))

        image = Image.open(image_path)
        original_size = image.size[::-1]  # (H, W)
        image = image.resize(self.target_size, resample=Image.BILINEAR)
        image_np = np.array(image).astype(np.float32)
        image_t = torch.from_numpy(image_np).permute(2, 0, 1)

        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        mask_img = Image.fromarray(mask_arr).convert("L")
        mask_np = (np.array(mask_img) > 0).astype(np.uint8)
        mask_t = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)

        resized_mask = cv2.resize(mask_np, self.target_size, interpolation=cv2.INTER_NEAREST)

        patient_id = os.path.basename(os.path.dirname(mask_rel))
        gt_nii_path = os.path.join(self.nii_dir, patient_id, "GTVp.nii.gz")
        if not os.path.exists(gt_nii_path):
            raise FileNotFoundError(f"Missing NIfTI: {gt_nii_path}")

        gt_nii = sitk.ReadImage(gt_nii_path)
        resize_factor_x = self.target_size[1] / gt_nii.GetSize()[0]
        resize_factor_y = self.target_size[0] / gt_nii.GetSize()[1]

        spacing_x_cm = gt_nii.GetSpacing()[0] / resize_factor_x / 10.0
        spacing_y_cm = gt_nii.GetSpacing()[1] / resize_factor_y / 10.0

        box = self._compute_box(resized_mask, spacing_x_cm, spacing_y_cm, self.expand_cm)

        return image_t, mask_t, box, original_size, image_path
