"""
上下界框 + 中间层（面积最大层/面积第二大层、面积第三大层）
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
        area_list = []
        valid_z = []
        for z in range(Z):
            area = mask_np[z].sum()
            if area > 0:
                valid_z.append(z)
                area_list.append(area)

        if len(valid_z) < 3:
            raise ValueError("有效层不足3层，无法插值")

        # 提取 top / bottom / max-area 层
        top_z = valid_z[0]
        bottom_z = valid_z[-1]

        # 构造中间层（排除 top 和 bottom）
        middle_z_list = []
        middle_area_list = []

        for z, a in zip(valid_z, area_list):
            if z != top_z and z != bottom_z:
                middle_z_list.append(z)
                middle_area_list.append(a)

        # 排序中间层面积
        sorted_indices = np.argsort(middle_area_list)[::-1]

        # 选面积层
        """
        [0] —— 最大层
        [1] —— 第二大层
        [2] —— 第三大层
        """
        mid_z = middle_z_list[sorted_indices[2]]

        key_z_list = [top_z, mid_z, bottom_z]

        # 提取三个框
        box_dict = {}
        for z in key_z_list:
            mask = mask_np[z]
            ys, xs = np.where(mask > 0)
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            box_dict[z] = [x0, y0, x1, y1]

        # 插值函数
        key_z = np.array(key_z_list)
        box_array = np.array([box_dict[z] for z in key_z_list])
        """
        f = interp1d(x, y, kind="linear", bounds_error=False, fill_value=np.nan, assume_sorted=False)

        参数说明：
            x: array_like
                自变量（已知点坐标，通常为一维数组），最好是递增排列。
            y: array_like
                因变量（已知点对应的函数值，可以是多维数组）。
            kind: str or int, optional
                插值方式：
                    "linear"    —— 线性插值（默认）
                    "nearest"   —— 最近邻插值
                    "zero"      —— 零阶保持（阶梯函数）
                    "slinear"   —— 一次样条插值（等价于线性）
                    "quadratic" —— 二次样条插值
                    "cubic"     —— 三次样条插值
                    int 值 n    —— n 次样条插值（如 n=1 等价 linear，n=3 等价 cubic）
            bounds_error: bool, optional
                是否在超出插值范围时报错：
                    True  —— 如果 x_new 超出 [x.min(), x.max()]，直接报错
                    False —— 不报错，返回 fill_value（默认）
            fill_value: float, array_like or str, optional
                超出 x 范围时返回的值：
                    np.nan        —— 默认，返回 NaN
                    "extrapolate" —— 允许外推（继续沿着边界趋势计算）
                    标量或数组    —— 固定值替代
            assume_sorted: bool, optional
                默认 False，会检查 x 是否递增；
                True 表示假设 x 已经排好序，可加快计算速度（不会重新检查）。
        """

        interp_funcs = [
            interp1d(key_z, box_array[:, i], kind="linear", bounds_error=True, assume_sorted=False)
            for i in range(4)
        ]

        # 有效层插值框
        all_box_dict = {}
        # 计算外扩像素换算
        expand_x = int(round((expand_cm * 10) / spacing[0]))  # mm → cm，x方向
        expand_y = int(round((expand_cm * 10) / spacing[1]))  # mm → cm，y方向

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
