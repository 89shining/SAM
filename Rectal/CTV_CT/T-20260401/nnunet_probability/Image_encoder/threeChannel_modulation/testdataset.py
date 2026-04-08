import os
import re
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
import cv2


def window_level_transform(img, window_center=40, window_width=400):
    img = img.astype(np.float32)
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    img = np.clip(img, lower, upper)
    img = ((img - lower) / window_width) * 255.0
    return img.astype(np.uint8)


class SAMImageEncoderTestDatasetFromNiiGz(Dataset):
    """
    Test dataset for SAM image-encoder prior experiments.
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
        nnunet_prompt_npz_dir="",
        npz_pattern="CTV_{:03d}.npz",
        prompt_class_idx=1,
        prompt_eps=1e-4,
        mode="modulation",
        use_gt_positive_only=False,
    ):
        self.nii_root_dir = nii_root_dir
        self.target_image_size = target_image_size
        self.mask_prompt_size = mask_prompt_size
        self.window_center = window_center
        self.window_width = window_width
        self.image_name = image_name
        self.gt_name = gt_name
        self.nnunet_prompt_npz_dir = nnunet_prompt_npz_dir
        self.npz_pattern = npz_pattern
        self.prompt_class_idx = prompt_class_idx
        self.prompt_eps = prompt_eps
        self.mode = mode
        self.use_gt_positive_only = use_gt_positive_only

        if self.mode not in ["modulation", "injection"]:
            raise ValueError(f"Unsupported mode: {self.mode}")

        if not os.path.isdir(self.nnunet_prompt_npz_dir):
            raise FileNotFoundError(f"nnUNet prompt npz directory not found: {self.nnunet_prompt_npz_dir}")

        self.index = []
        self.patients = sorted(
            [d for d in os.listdir(nii_root_dir) if os.path.isdir(os.path.join(nii_root_dir, d))],
            key=lambda x: int(x.lstrip("p_")) if x.startswith("p_") and x.lstrip("p_").isdigit() else x
        )

        self._paths = {}
        self._cache = {}

        for pid in self.patients:
            pdir = os.path.join(self.nii_root_dir, pid)
            case_id = self._extract_case_id(pid)
            npz_path = os.path.join(self.nnunet_prompt_npz_dir, self.npz_pattern.format(case_id))

            self._paths[pid] = {
                "image": os.path.join(pdir, self.image_name),
                "gt": os.path.join(pdir, self.gt_name),
                "npz": npz_path,
            }

            if not os.path.exists(npz_path):
                raise FileNotFoundError(f"Missing nnUNet npz for patient {pid}: {npz_path}")

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
    def _extract_case_id(pid):
        nums = re.findall(r"\d+", pid)
        if len(nums) == 0:
            raise ValueError(f"Cannot parse numeric case id from patient id: {pid}")
        return int(nums[-1])

    @staticmethod
    def _pick_npz_array(npz_obj):
        preferred_keys = ["softmax", "probabilities", "probability", "pred", "prediction", "logits"]
        for k in preferred_keys:
            if k in npz_obj.files:
                return npz_obj[k], k
        if len(npz_obj.files) == 1:
            k = npz_obj.files[0]
            return npz_obj[k], k
        raise KeyError(
            f"Cannot determine array key from npz keys={npz_obj.files}. "
            "Expected one of softmax/probabilities/probability/pred/prediction/logits."
        )

    def _load_probability_volume(self, npz_path, expected_zhw):
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
            raise ValueError(f"Unsupported npz array shape {arr.shape} in {npz_path}, key={key}")

        if prob.shape != expected_zhw:
            raise ValueError(
                f"Prompt shape mismatch for {npz_path}: prompt={prob.shape}, expected={expected_zhw}. "
                "Please ensure npz and nii are in the same crop/spacing/order."
            )

        prob = prob.astype(np.float32)
        if float(prob.min()) < 0.0 or float(prob.max()) > 1.0:
            prob = 1.0 / (1.0 + np.exp(-prob))
        return np.clip(prob, 0.0, 1.0)

    def _load_patient_volumes(self, pid):
        if pid in self._cache:
            return self._cache[pid]

        image_path = self._paths[pid]["image"]
        gt_path = self._paths[pid]["gt"]
        npz_path = self._paths[pid]["npz"]

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Missing GT: {gt_path}")

        img_vol = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
        gt_vol = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
        prob_vol = self._load_probability_volume(npz_path, expected_zhw=gt_vol.shape)

        item = {
            "img_vol": img_vol,
            "gt_vol": gt_vol,
            "prob_vol": prob_vol,
        }
        self._cache[pid] = item
        return item

    def _build_encoder_input(self, img_255, prob_slice):
        ct_1024 = cv2.resize(
            img_255.astype(np.float32),
            self.target_image_size,
            interpolation=cv2.INTER_LINEAR,
        )
        p_1024 = cv2.resize(
            prob_slice.astype(np.float32),
            self.target_image_size,
            interpolation=cv2.INTER_LINEAR,
        )
        p_1024 = np.clip(p_1024, 0.0, 1.0)

        if self.mode == "modulation":
            mod = ct_1024 * p_1024
            ch1 = mod
            ch2 = mod
            ch3 = mod
        else:
            ch1 = ct_1024
            ch2 = p_1024 * 255.0
            ch3 = ct_1024

        image = np.stack([ch1, ch2, ch3], axis=0).astype(np.float32)
        return torch.from_numpy(image)

    def __getitem__(self, idx):
        pid, z = self.index[idx]
        vols = self._load_patient_volumes(pid)

        img_slice = vols["img_vol"][z]
        gt_slice = vols["gt_vol"][z]
        prob_slice = vols["prob_vol"][z]

        h, w = img_slice.shape

        img_255 = window_level_transform(img_slice, self.window_center, self.window_width)
        image = self._build_encoder_input(img_255, prob_slice)

        gt = torch.tensor((gt_slice > 0).astype(np.uint8), dtype=torch.float32).unsqueeze(0)
        p = np.clip(prob_slice.astype(np.float32), self.prompt_eps, 1.0 - self.prompt_eps)
        logit = np.log(p / (1.0 - p))
        logit = cv2.resize(logit, self.mask_prompt_size, interpolation=cv2.INTER_LINEAR)
        mask_prompt = torch.tensor(logit, dtype=torch.float32).unsqueeze(0)

        return {
            "image": image,
            "GT": gt,
            "mask_prompt": mask_prompt,
            "original_size": (h, w),
            "patient_id": pid,
            "slice_idx": z,
        }
