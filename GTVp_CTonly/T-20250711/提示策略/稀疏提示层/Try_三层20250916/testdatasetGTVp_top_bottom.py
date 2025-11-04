"""
只给上下界框——插值——外扩0.5cm
"""

import os
import cv2
import torch
import pandas as pd
import numpy as np
import SimpleITK as sitk
from PIL import Image
from torch.utils.data import Dataset
from scipy.interpolate import interp1d


class TestDataset(Dataset):
    def __init__(self, csv_path, root_dir, nii_dir, target_size, expand_cm):
        self.df = pd.read_csv(csv_path, header=None, names=["image", "mask"])
        self.root_dir = root_dir
        self.nii_dir = nii_dir
        self.target_size = target_size
        self.expand_cm = expand_cm

    def __len__(self):
        return len(self.df)

    # 框插值函数，返回每层的box
    # 原图尺寸先插值——外扩0.5cm
    def get_box_interp(self, mask_np, spacing, expand_cm):
        Z, H, W = mask_np.shape

        # 找出有mask的层
        valid_z = []
        for z in range(Z):
            if mask_np[z].sum() > 0:
                valid_z.append(z)

        if len(valid_z) < 2:
            raise ValueError("有效层不足2层，无法插值")

        # 提取 top / bottom / max-area 层
        top_z = valid_z[0]
        bottom_z = valid_z[-1]
        key_z_list = [top_z, bottom_z]

        # 提取框
        box_dict = {}
        for z in key_z_list:
            mask = mask_np[z]
            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            box_dict[z] = [x0, y0, x1, y1]

        # 插值函数
            # 插值函数（只用上下界）
            key_z = np.array(list(box_dict.keys()))
            box_array = np.array([box_dict[z] for z in key_z])
            interp_funcs = [
                interp1d(key_z, box_array[:, i], kind="linear", bounds_error=True, assume_sorted=False)
                for i in range(4)
            ]


        # 计算外扩像素换算
        expand_x = int(round((expand_cm * 10) / spacing[0]))  # mm → cm，x方向
        expand_y = int(round((expand_cm * 10) / spacing[1]))  # mm → cm，y方向

        all_box_dict = {}
        for z in valid_z:
            box = [float(f(z)) for f in interp_funcs]
            box = [int(round(b)) for b in box]

            # 外扩
            x0 = max(0, box[0] - expand_x)
            y0 = max(0, box[1] - expand_y)
            x1 = min(W, box[2] + expand_x)
            y1 = min(H, box[3] + expand_y)

            # 确保至少1像素宽高
            x1 = max(x0 + 1, x1)
            y1 = max(y0 + 1, y1)

            all_box_dict[z] = [x0, y0, x1, y1]

        return all_box_dict

    def __getitem__(self, idx):
        image_rel = self.df.iloc[idx]['image'].lstrip("/\\")
        mask_rel = self.df.iloc[idx]['mask'].lstrip("/\\")
        image_path = os.path.normpath(os.path.join(self.root_dir, image_rel))
        mask_path = os.path.normpath(os.path.join(self.root_dir, mask_rel))

        image = Image.open(image_path)
        original_size = image.size[::-1]  # (H, W)
        image = image.resize(self.target_size, resample=Image.BILINEAR)   # 1024
        image = np.array(image).astype(np.float32)     # numpy H W C
        image = torch.from_numpy(image).permute(2, 0, 1)   # C H W tensor

        # 读取原始mask并获取其3D结构   512
        mask_rel_path = self.df.iloc[idx]['mask'].lstrip("/\\")
        patient_id = os.path.basename(os.path.dirname(mask_rel_path))
        mask_nii_path = os.path.join(self.nii_dir, patient_id, "GTVp.nii.gz")
        if not os.path.exists(mask_nii_path):
            raise FileNotFoundError(f"Missing NIfTI image: {mask_nii_path}")
        mask_img = sitk.ReadImage(mask_nii_path)
        mask_np = sitk.GetArrayFromImage(mask_img)  # (Z, H, W)
        spacing = mask_img.GetSpacing()  # mm （x, y, z)

        # 当前切片索引（根据文件名）
        current_slice = int(os.path.splitext(os.path.basename(mask_rel))[0])

        # 原始mask
        mask_z = mask_np[current_slice]
        mask = torch.tensor((mask_z > 0).astype(np.float32)).unsqueeze(0)
        # 1024 resized_masK
        resized_mask = cv2.resize(mask_z, self.target_size, interpolation=cv2.INTER_NEAREST)


        # 获取该层插值box
        all_box_dict = self.get_box_interp(mask_np, spacing, self.expand_cm)   # 512 mask_np
        box_xyxy = all_box_dict[current_slice]  # [x0,y0,x1,y1]

        # 映射到 target_size
        orig_H, orig_W = mask_np.shape[1], mask_np.shape[2]
        scale_x = self.target_size[0] / orig_W
        scale_y = self.target_size[1] / orig_H

        x0, y0, x1, y1 = box_xyxy
        x0 = int(round(x0 * scale_x))
        x1 = int(round(x1 * scale_x))
        y0 = int(round(y0 * scale_y))
        y1 = int(round(y1 * scale_y))

        box_xyxy_resized = [x0, y0, x1, y1]
        # 1024
        box = torch.tensor(box_xyxy_resized, dtype=torch.float32).unsqueeze(0)

        return image, mask, box, original_size, image_path
