"""
nnUNet的mask作为提示输入
"""

import os
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2


def window_level_transform(img, window_center=40, window_width=400):
    img = img.astype(np.float32)
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    img = np.clip(img, lower, upper)
    img = ((img - lower) / window_width) * 255.0
    return img.astype(np.uint8)


class SAMDatasetFromNiiGz(Dataset):
    """
    直接从 3D nii.gz 生成 slice 级样本（仅 GT 非空切片）。
    返回：
      image: float32 [3, 1024, 1024], 0-255
      GT:    float32 [1, H, W]        (保持原始尺寸，不 resize)
      mask_prompt: float32 [1, 256, 256] nnUNet mask prompt
    """

    def __init__(
        self,
        nii_root_dir,
        target_image_size=(1024, 1024),
        mask_prompt_size=(256, 256),
        window_center=40,
        window_width=400,
        image_name="image.nii.gz",
        gt_name="CTV.nii.gz",
        nnunet_name="nnunet_pred.nii.gz",
    ):
        self.nii_root_dir = nii_root_dir
        self.target_image_size = target_image_size
        self.mask_prompt_size = mask_prompt_size
        self.window_center = window_center
        self.window_width = window_width
        self.image_name = image_name
        self.gt_name = gt_name
        self.nnunet_name = nnunet_name

        # 建立 (patient_id, slice_idx) 的索引列表：只保留 GT 非空切片
        self.index = []
        self.patients = sorted(
            [d for d in os.listdir(nii_root_dir) if os.path.isdir(os.path.join(nii_root_dir, d))],
            key=lambda x: int(x.lstrip("p_")) if x.startswith("p_") and x.lstrip("p_").isdigit() else x
        )

        for pid in self.patients:
            pdir = os.path.join(self.nii_root_dir, pid)
            gt_path = os.path.join(pdir, self.gt_name)
            if not os.path.exists(gt_path):
                continue

            gt_vol = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))  # (Z, H, W)
            for z in range(gt_vol.shape[0]):
                if np.max(gt_vol[z]) > 0:
                    self.index.append((pid, z))

        if len(self.index) == 0:
            raise RuntimeError("No valid GT slices found. Check GT file name/path or masks are empty.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        pid, z = self.index[idx]
        pdir = os.path.join(self.nii_root_dir, pid)

        image_path = os.path.join(pdir, self.image_name)
        gt_path = os.path.join(pdir, self.gt_name)
        nnunet_path = os.path.join(pdir, self.nnunet_name)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Missing GT: {gt_path}")
        if not os.path.exists(nnunet_path):
            raise FileNotFoundError(f"Missing nnUNet pred: {nnunet_path}")

        # 读取 3D volume
        img_vol = sitk.GetArrayFromImage(sitk.ReadImage(image_path))  # (Z, H, W)
        gt_vol = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))      # (Z, H, W)
        nn_vol = sitk.GetArrayFromImage(sitk.ReadImage(nnunet_path))  # (Z, H, W)

        # 取 slice
        img_slice = img_vol[z]
        gt_slice = gt_vol[z]
        nn_slice = nn_vol[z]

        # image: window/level -> 0-255 -> RGB -> resize(1024) -> tensor [3,1024,1024]
        img_255 = window_level_transform(img_slice, self.window_center, self.window_width)
        img_rgb = Image.fromarray(img_255).convert("RGB")
        img_rgb = img_rgb.resize(self.target_image_size, resample=Image.BILINEAR)
        img_np = np.array(img_rgb).astype(np.float32)
        image = torch.from_numpy(img_np).permute(2, 0, 1)

        # GT: 保持原始尺寸，不 resize
        gt_bin = (gt_slice > 0).astype(np.uint8)
        GT = torch.tensor(gt_bin, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        # mask_prompt: nnUNet -> binary -> resize(256) -> tensor [1,256,256]
        nn_bin = (nn_slice > 0).astype(np.uint8)
        nn_bin = cv2.resize(nn_bin, self.mask_prompt_size, interpolation=cv2.INTER_NEAREST)
        mask_prompt = torch.tensor(nn_bin, dtype=torch.float32).unsqueeze(0)

        return {
            "image": image,
            "GT": GT,
            "mask_prompt": mask_prompt,
            # "patient_id": pid,
            # "slice_idx": z,
        }
