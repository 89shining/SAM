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
    Test dataset for SAM with nnUNet-derived box prompt.

    If use_gt_positive_only=False, iterate all slices to support full-volume metrics.
    """

    def __init__(
        self,
        nii_root_dir,
        expand_cm=0.0,
        target_image_size=(1024, 1024),
        window_center=40,
        window_width=400,
        image_name="image.nii.gz",
        gt_name="CTV.nii.gz",
        nnunet_name="prompt.nii.gz",
        use_gt_positive_only=False,
    ):
        self.nii_root_dir = nii_root_dir
        self.expand_cm = float(expand_cm)
        self.target_image_size = target_image_size
        self.window_center = window_center
        self.window_width = window_width
        self.image_name = image_name
        self.gt_name = gt_name
        self.nnunet_name = nnunet_name
        self.use_gt_positive_only = use_gt_positive_only

        self.index = []
        self.patients = sorted(
            [d for d in os.listdir(nii_root_dir) if os.path.isdir(os.path.join(nii_root_dir, d))],
            key=lambda x: int(x.lstrip("p_")) if x.startswith("p_") and x.lstrip("p_").isdigit() else x
        )

        self._paths = {}
        self._cache = {}

        for pid in self.patients:
            pdir = os.path.join(self.nii_root_dir, pid)
            self._paths[pid] = {
                "image": os.path.join(pdir, self.image_name),
                "gt": os.path.join(pdir, self.gt_name),
                "nn": os.path.join(pdir, self.nnunet_name),
            }

            gt_path = self._paths[pid]["gt"]
            if not os.path.exists(gt_path):
                continue

            gt_vol = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
            if self.use_gt_positive_only:
                for z in range(gt_vol.shape[0]):
                    if np.max(gt_vol[z]) > 0:
                        self.index.append((pid, z))
            else:
                for z in range(gt_vol.shape[0]):
                    self.index.append((pid, z))

        if len(self.index) == 0:
            raise RuntimeError("No valid test slices found.")

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _fallback_box(resized_mask):
        h, w = resized_mask.shape[:2]
        return torch.tensor([0, 0, w - 1, h - 1], dtype=torch.float32).unsqueeze(0)

    def get_box(self, resized_mask, spacing_x, spacing_y, expand_cm=0.0):
        y_indices, x_indices = np.where(resized_mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return self._fallback_box(resized_mask)

        x_min = np.min(x_indices)
        x_max = np.max(x_indices)
        y_min = np.min(y_indices)
        y_max = np.max(y_indices)

        img_width = resized_mask.shape[1]
        img_height = resized_mask.shape[0]

        expand_x_px = round(expand_cm / spacing_x) if spacing_x > 0 else 0
        expand_y_px = round(expand_cm / spacing_y) if spacing_y > 0 else 0

        x_min = max(x_min - expand_x_px, 0)
        x_max = min(x_max + expand_x_px, img_width - 1)
        y_min = max(y_min - expand_y_px, 0)
        y_max = min(y_max + expand_y_px, img_height - 1)

        return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32).unsqueeze(0)

    def _load_patient(self, pid):
        if pid in self._cache:
            return self._cache[pid]

        image_path = self._paths[pid]["image"]
        gt_path = self._paths[pid]["gt"]
        nn_path = self._paths[pid]["nn"]

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Missing GT: {gt_path}")
        if not os.path.exists(nn_path):
            raise FileNotFoundError(f"Missing nnUNet prompt: {nn_path}")

        img_nii = sitk.ReadImage(image_path)
        img_vol = sitk.GetArrayFromImage(img_nii)
        gt_vol = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
        nn_vol = sitk.GetArrayFromImage(sitk.ReadImage(nn_path))

        resize_factor_x = self.target_image_size[1] / img_nii.GetSize()[0]
        resize_factor_y = self.target_image_size[0] / img_nii.GetSize()[1]
        spacing_x_resized = img_nii.GetSpacing()[0] / resize_factor_x / 10.0
        spacing_y_resized = img_nii.GetSpacing()[1] / resize_factor_y / 10.0

        item = {
            "img_vol": img_vol,
            "gt_vol": gt_vol,
            "nn_vol": nn_vol,
            "spacing_x_resized": spacing_x_resized,
            "spacing_y_resized": spacing_y_resized,
        }
        self._cache[pid] = item
        return item

    def __getitem__(self, idx):
        pid, z = self.index[idx]
        item = self._load_patient(pid)

        img_slice = item["img_vol"][z]
        gt_slice = item["gt_vol"][z]
        nn_slice = item["nn_vol"][z]

        h, w = img_slice.shape

        img_255 = window_level_transform(img_slice, self.window_center, self.window_width)
        img_rgb = Image.fromarray(img_255).convert("RGB")
        img_rgb = img_rgb.resize(self.target_image_size, resample=Image.BILINEAR)
        img_np = np.array(img_rgb, dtype=np.float32)
        image = torch.from_numpy(img_np).permute(2, 0, 1)

        GT = torch.tensor((gt_slice > 0).astype(np.uint8), dtype=torch.float32).unsqueeze(0)

        nn_bin = (nn_slice > 0).astype(np.uint8)
        nn_bin_1024 = cv2.resize(nn_bin, self.target_image_size, interpolation=cv2.INTER_NEAREST)
        box = self.get_box(
            nn_bin_1024,
            item["spacing_x_resized"],
            item["spacing_y_resized"],
            expand_cm=self.expand_cm,
        )

        return {
            "image": image,
            "GT": GT,
            "box": box,
            "original_size": (h, w),
            "patient_id": pid,
            "slice_idx": z,
        }
