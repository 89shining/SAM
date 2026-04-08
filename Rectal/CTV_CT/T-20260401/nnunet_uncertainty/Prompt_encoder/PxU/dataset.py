"""
nnUNet probability + uncertainty npz as prompt input:
P (soft probability) and U (uncertainty) -> weighted prompt P*U -> logit
"""

import math
import os
import re
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
      mask_prompt: float32 [1, 256, 256], from logit(P*U)
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
        nnunet_prob_npz_dir="",
        nnunet_uncertainty_npz_dir="",
        npz_pattern="CTV_{:03d}.npz",
        prompt_class_idx=1,
        prompt_eps=1e-4,
    ):
        self.nii_root_dir = nii_root_dir
        self.target_image_size = target_image_size
        self.mask_prompt_size = mask_prompt_size
        self.window_center = window_center
        self.window_width = window_width
        self.image_name = image_name
        self.gt_name = gt_name
        self.nnunet_prob_npz_dir = nnunet_prob_npz_dir
        self.nnunet_uncertainty_npz_dir = nnunet_uncertainty_npz_dir
        self.npz_pattern = npz_pattern
        self.prompt_class_idx = prompt_class_idx
        self.prompt_eps = prompt_eps

        self.index = []
        self.patients = sorted(
            [d for d in os.listdir(nii_root_dir) if os.path.isdir(os.path.join(nii_root_dir, d))],
            key=lambda x: int(x.lstrip("p_")) if x.startswith("p_") and x.lstrip("p_").isdigit() else x,
        )

        if not os.path.isdir(self.nnunet_prob_npz_dir):
            raise FileNotFoundError(f"nnUNet probability npz directory not found: {self.nnunet_prob_npz_dir}")
        if not os.path.isdir(self.nnunet_uncertainty_npz_dir):
            raise FileNotFoundError(f"nnUNet uncertainty npz directory not found: {self.nnunet_uncertainty_npz_dir}")

        self._paths = {}
        for pid in self.patients:
            pdir = os.path.join(self.nii_root_dir, pid)
            case_id = self._extract_case_id(pid)
            prob_npz_path = os.path.join(self.nnunet_prob_npz_dir, self.npz_pattern.format(case_id))
            uncertainty_npz_path = os.path.join(self.nnunet_uncertainty_npz_dir, self.npz_pattern.format(case_id))

            self._paths[pid] = {
                "image": os.path.join(pdir, self.image_name),
                "gt": os.path.join(pdir, self.gt_name),
                "prob_npz": prob_npz_path,
                "uncertainty_npz": uncertainty_npz_path,
            }

            if not os.path.exists(prob_npz_path):
                raise FileNotFoundError(f"Missing nnUNet probability npz for patient {pid}: {prob_npz_path}")
            if not os.path.exists(uncertainty_npz_path):
                raise FileNotFoundError(f"Missing nnUNet uncertainty npz for patient {pid}: {uncertainty_npz_path}")

        self._cache = {}

        for pid in self.patients:
            gt_path = self._paths[pid]["gt"]
            if not os.path.exists(gt_path):
                continue

            gt_vol = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
            for z in range(gt_vol.shape[0]):
                if np.max(gt_vol[z]) > 0:
                    self.index.append((pid, z))

        if len(self.index) == 0:
            raise RuntimeError("No valid GT slices found. Check GT file name/path or masks are empty.")

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _extract_case_id(pid):
        nums = re.findall(r"\d+", pid)
        if len(nums) == 0:
            raise ValueError(f"Cannot parse numeric case id from patient id: {pid}")
        return int(nums[-1])

    @staticmethod
    def _pick_npz_array(npz_obj):
        preferred_keys = ["softmax", "probabilities", "probability", "pred", "prediction", "logits", "entropy"]
        for k in preferred_keys:
            if k in npz_obj.files:
                return npz_obj[k], k
        if len(npz_obj.files) == 1:
            k = npz_obj.files[0]
            return npz_obj[k], k
        raise KeyError(
            f"Cannot determine array key from npz keys={npz_obj.files}. "
            "Expected one of softmax/probabilities/probability/pred/prediction/logits/entropy."
        )

    def _load_prob_volume(self, npz_path, expected_zhw):
        with np.load(npz_path) as npz_obj:
            arr, key = self._pick_npz_array(npz_obj)
            arr = np.asarray(arr)

        if arr.ndim == 4:
            c = arr.shape[0]
            cls = self.prompt_class_idx if c > 1 else 0
            if cls < 0 or cls >= c:
                raise IndexError(f"prompt_class_idx={cls} out of range for {npz_path}, channels={c}, key={key}")
            prob = arr[cls]
        elif arr.ndim == 3:
            prob = arr
        else:
            raise ValueError(f"Unsupported probability npz shape {arr.shape} in {npz_path}, key={key}")

        if prob.shape != expected_zhw:
            raise ValueError(
                f"Probability shape mismatch for {npz_path}: prob={prob.shape}, expected={expected_zhw}. "
                "Please ensure npz and nii are in the same crop/spacing/order."
            )

        prob = prob.astype(np.float32)
        if float(prob.min()) < 0.0 or float(prob.max()) > 1.0:
            prob = 1.0 / (1.0 + np.exp(-prob))
        return np.clip(prob, 0.0, 1.0)

    @staticmethod
    def _to_uncertainty_0_1(u):
        u = u.astype(np.float32)
        if float(u.min()) >= 0.0 and float(u.max()) <= 1.0:
            return u

        ln2 = math.log(2.0)
        if float(u.min()) >= 0.0 and float(u.max()) <= ln2 + 1e-3:
            return np.clip(u / ln2, 0.0, 1.0)

        u_min = float(u.min())
        u_max = float(u.max())
        if u_max > u_min:
            return (u - u_min) / (u_max - u_min)
        return np.zeros_like(u, dtype=np.float32)

    def _load_uncertainty_volume(self, npz_path, expected_zhw):
        with np.load(npz_path) as npz_obj:
            arr, key = self._pick_npz_array(npz_obj)
            arr = np.asarray(arr)

        if arr.ndim == 4:
            c = arr.shape[0]
            cls = self.prompt_class_idx if c > 1 else 0
            if cls < 0 or cls >= c:
                raise IndexError(f"prompt_class_idx={cls} out of range for {npz_path}, channels={c}, key={key}")
            uncertainty = arr[cls]
        elif arr.ndim == 3:
            uncertainty = arr
        else:
            raise ValueError(f"Unsupported uncertainty npz shape {arr.shape} in {npz_path}, key={key}")

        if uncertainty.shape != expected_zhw:
            raise ValueError(
                f"Uncertainty shape mismatch for {npz_path}: uncertainty={uncertainty.shape}, expected={expected_zhw}. "
                "Please ensure npz and nii are in the same crop/spacing/order."
            )

        return self._to_uncertainty_0_1(uncertainty)

    def _load_patient_volumes(self, pid):
        if pid in self._cache:
            return self._cache[pid]

        image_path = self._paths[pid]["image"]
        gt_path = self._paths[pid]["gt"]
        prob_npz_path = self._paths[pid]["prob_npz"]
        uncertainty_npz_path = self._paths[pid]["uncertainty_npz"]

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Missing GT: {gt_path}")

        img_vol = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
        gt_vol = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
        prob_vol = self._load_prob_volume(prob_npz_path, expected_zhw=gt_vol.shape)
        uncertainty_vol = self._load_uncertainty_volume(uncertainty_npz_path, expected_zhw=gt_vol.shape)

        item = {
            "img_vol": img_vol,
            "gt_vol": gt_vol,
            "prob_vol": prob_vol,
            "uncertainty_vol": uncertainty_vol,
        }
        self._cache[pid] = item
        return item

    def __getitem__(self, idx):
        pid, z = self.index[idx]
        vols = self._load_patient_volumes(pid)

        img_slice = vols["img_vol"][z]
        gt_slice = vols["gt_vol"][z]
        prob_slice = vols["prob_vol"][z]
        uncertainty_slice = vols["uncertainty_vol"][z]

        img_255 = window_level_transform(img_slice, self.window_center, self.window_width)
        img_rgb = Image.fromarray(img_255).convert("RGB")
        img_rgb = img_rgb.resize(self.target_image_size, resample=Image.BILINEAR)
        img_np = np.array(img_rgb, dtype=np.float32)
        image = torch.from_numpy(img_np).permute(2, 0, 1)

        gt_bin = (gt_slice > 0).astype(np.uint8)
        gt = torch.tensor(gt_bin, dtype=torch.float32).unsqueeze(0)

        pxu = np.clip(prob_slice.astype(np.float32), 0.0, 1.0) * np.clip(uncertainty_slice.astype(np.float32), 0.0, 1.0)
        p = np.clip(pxu, self.prompt_eps, 1.0 - self.prompt_eps)
        logit = np.log(p / (1.0 - p))
        logit = cv2.resize(logit, self.mask_prompt_size, interpolation=cv2.INTER_LINEAR)
        mask_prompt = torch.tensor(logit, dtype=torch.float32).unsqueeze(0)

        return {
            "image": image,
            "GT": gt,
            "mask_prompt": mask_prompt,
            "patient_id": pid,
            "slice_idx": z,
        }
