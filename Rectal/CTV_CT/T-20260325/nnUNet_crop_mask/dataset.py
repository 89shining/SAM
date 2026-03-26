"""
nnUNet mask as prompt input
Optimized: patient-level volume cache to reduce repeated I/O.
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
    Build slice-level samples from 3D nii.gz volumes (only GT-positive slices).

    Returns:
      image: float32 [3, 1024, 1024], 0-255
      GT: float32 [1, H, W] (original size)
      mask_prompt: float32 [1, 256, 256]
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

        self.index = []
        self.patients = sorted(
            [d for d in os.listdir(nii_root_dir) if os.path.isdir(os.path.join(nii_root_dir, d))],
            key=lambda x: int(x.lstrip("p_")) if x.startswith("p_") and x.lstrip("p_").isdigit() else x
        )

        # Keep path mapping once.
        self._paths = {}
        for pid in self.patients:
            pdir = os.path.join(self.nii_root_dir, pid)
            self._paths[pid] = {
                "image": os.path.join(pdir, self.image_name),
                "gt": os.path.join(pdir, self.gt_name),
                "nn": os.path.join(pdir, self.nnunet_name),
            }

        # Patient-level cache, loaded lazily in __getitem__.
        self._cache = {}

        for pid in self.patients:
            gt_path = self._paths[pid]["gt"]
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

    def _load_patient_volumes(self, pid):
        if pid in self._cache:
            return self._cache[pid]

        image_path = self._paths[pid]["image"]
        gt_path = self._paths[pid]["gt"]
        nnunet_path = self._paths[pid]["nn"]

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Missing GT: {gt_path}")
        if not os.path.exists(nnunet_path):
            raise FileNotFoundError(f"Missing nnUNet pred: {nnunet_path}")

        item = {
            "img_vol": sitk.GetArrayFromImage(sitk.ReadImage(image_path)),
            "gt_vol": sitk.GetArrayFromImage(sitk.ReadImage(gt_path)),
            "nn_vol": sitk.GetArrayFromImage(sitk.ReadImage(nnunet_path)),
        }
        self._cache[pid] = item
        return item

    def __getitem__(self, idx):
        pid, z = self.index[idx]
        vols = self._load_patient_volumes(pid)

        img_slice = vols["img_vol"][z]
        gt_slice = vols["gt_vol"][z]
        nn_slice = vols["nn_vol"][z]

        # image: window/level -> 0-255 -> RGB -> resize(1024)
        img_255 = window_level_transform(img_slice, self.window_center, self.window_width)
        img_rgb = Image.fromarray(img_255).convert("RGB")
        img_rgb = img_rgb.resize(self.target_image_size, resample=Image.BILINEAR)
        img_np = np.array(img_rgb, dtype=np.float32)
        image = torch.from_numpy(img_np).permute(2, 0, 1)

        # GT: keep original size
        gt_bin = (gt_slice > 0).astype(np.uint8)
        GT = torch.tensor(gt_bin, dtype=torch.float32).unsqueeze(0)

        # mask_prompt: nnUNet -> binary -> resize(256)
        nn_bin = (nn_slice > 0).astype(np.uint8)
        nn_bin = cv2.resize(nn_bin, self.mask_prompt_size, interpolation=cv2.INTER_NEAREST)
        mask_prompt = torch.tensor(nn_bin, dtype=torch.float32).unsqueeze(0)

        return {
            "image": image,
            "GT": GT,
            "mask_prompt": mask_prompt,
            "patient_id": pid,
            "slice_idx": z,
        }
