"""
nnUNet mask改名
"""

import os
import shutil
import re

# ======== 配置路径 ========
SRC_DIR = r"/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_results/Dataset013_RectalCTV146pCrop/nnUNetTrainer__nnUNetPlans__3d_fullres/testResult_fold0"   # 放 CTV_000.nii.gz 的目录
DST_ROOT = r"/home/wusi/SAMdata/Rectal/20260316_CTV/Cropdatanii/test_nii"                # A 目录（包含 p_0, p_1, ...）

# ========================

pattern = re.compile(r"CTV_(\d+)\.nii\.gz$")

for fname in os.listdir(SRC_DIR):
    match = pattern.match(fname)
    if not match:
        continue

    idx = int(match.group(1))  # 000 -> 0
    src_path = os.path.join(SRC_DIR, fname)

    dst_dir = os.path.join(DST_ROOT, f"p_{idx}")
    if not os.path.isdir(dst_dir):
        print(f"[WARNING] 目标目录不存在: {dst_dir}")
        continue

    dst_path = os.path.join(dst_dir, "prompt.nii.gz")

    if os.path.exists(dst_path):
        print(f"[SKIP] 已存在: {dst_path}")
        continue

    shutil.copy2(src_path, dst_path)
    print(f"[OK] {fname} -> {dst_path}")
