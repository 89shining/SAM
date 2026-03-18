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


class SAMTestDatasetFromNiiGz(Dataset):
    """
    STRICT test dataset:
    - Slice definition is IDENTICAL to training:
        only slices with GT > 0
    """

    def __init__(
        self,
        nii_root_dir,
        expand_cm,
        target_image_size=(1024, 1024),
        window_center=40,
        window_width=400,
        image_name="image.nii.gz",
        gt_name="CTV.nii.gz",
        nnunet_name="prompt.nii.gz",

    ):
        self.nii_root_dir = nii_root_dir
        self.expand_cm = expand_cm
        self.target_image_size = target_image_size
        self.window_center = window_center
        self.window_width = window_width
        self.image_name = image_name
        self.gt_name = gt_name
        self.nnunet_name = nnunet_name


        # === 与训练完全一致：只保留 GT 非空 slice ===
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

            gt_vol = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))  # (Z,H,W)

            for z in range(gt_vol.shape[0]):
                if np.max(gt_vol[z]) > 0:
                    self.index.append((pid, z))

        if len(self.index) == 0:
            raise RuntimeError("No GT-positive slices found.")

    def __len__(self):
        return len(self.index)

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

        expand_pixel_x = expand_cm / spacing_x
        expand_pixel_y = expand_cm / spacing_y

        x_min = max(x_min - expand_pixel_x, 0)
        x_max = min(x_max + expand_pixel_x, img_width - 1)
        y_min = max(y_min - expand_pixel_y, 0)
        y_max = min(y_max + expand_pixel_y, img_height - 1)

        box = np.array([x_min, y_min, x_max, y_max]).astype(np.float32)
        box = torch.tensor(box).unsqueeze(0)
        return box

    def __getitem__(self, idx):
        pid, z = self.index[idx]
        pdir = os.path.join(self.nii_root_dir, pid)

        image_path = os.path.join(pdir, self.image_name)
        gt_path = os.path.join(pdir, self.gt_name)
        nnunet_path = os.path.join(pdir, self.nnunet_name)

        img_vol = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
        gt_vol = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
        nn_vol = sitk.GetArrayFromImage(sitk.ReadImage(nnunet_path))

        # 计算原始图像 spacing_x, spacing_y
        img_nii = sitk.ReadImage(image_path)
        # 计算resize比例, GetSize()[W,H,D]
        resize_factor_x = self.target_image_size[1] / img_nii.GetSize()[0]  # W 1024 / 512 = 2.0
        resize_factor_y = self.target_image_size[0] / img_nii.GetSize()[1]  # H 同上
        # GetSpacing[W, H, D]
        spacing_x_resized = img_nii.GetSpacing()[0] / resize_factor_x / 10.0  # mm → cm
        spacing_y_resized = img_nii.GetSpacing()[1] / resize_factor_y / 10.0  # mm → cm

        img_slice = img_vol[z]
        gt_slice = gt_vol[z]
        nn_slice = nn_vol[z]

        H, W = img_slice.shape

        # ========== Image (same as training) ==========
        img_255 = window_level_transform(img_slice, self.window_center, self.window_width)
        img_rgb = Image.fromarray(img_255).convert("RGB")
        img_rgb = img_rgb.resize(self.target_image_size, resample=Image.BILINEAR)
        img_np = np.array(img_rgb).astype(np.float32)
        image = torch.from_numpy(img_np).permute(2, 0, 1)

        # ========== GT (same as training) ==========
        GT = torch.tensor((gt_slice > 0).astype(np.uint8),
                          dtype=torch.float32).unsqueeze(0)

        # ========== box prompt (same as training) ==========
        nn_bin = (nn_slice > 0).astype(np.uint8)
        nn_bin_1024 = cv2.resize(nn_bin, self.target_image_size,
                            interpolation=cv2.INTER_NEAREST)

        box = self.get_box(nn_bin_1024, spacing_x_resized, spacing_y_resized, expand_cm=self.expand_cm)

        return {
            "image": image,
            "GT": GT,
            "box": box,
            "original_size": (H, W),
            "patient_id": pid,
            "slice_idx": z,
        }
