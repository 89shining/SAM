"""
提取生成的预测.nii.gz按患者文件夹汇总
"""


import os
import shutil


"""
SAM训练提取
"""

# # 原始根目录
# root_dir = r"D:\SAM\GTVp_CTonly\20250701\TrainAll_pseudoRGB"
# save_dir = r"C:\Users\WS\Desktop/transfer"
# # 原始子文件夹
# source_folders = ["test_0_pixel", "test_3_pixel", "test_5_pixel", "test_7_pixel", "test_9_pixel"]
#
# # 遍历每个预测编号 pred_000.nii ~ pred_014.nii
# for i in range(15):  # 假设编号 0~14
#     patient_folder = f"p_{i}"
#     patient_path = os.path.join(save_dir, patient_folder)
#     os.makedirs(patient_path, exist_ok=True)
#
#     for folder in source_folders:
#         src_filename = f"pred_{i:03d}.nii.gz"
#         src_path = os.path.join(root_dir, folder, src_filename)
#         print(src_path)
#
#         # 从文件夹名提取数字，作为目标文件名
#         folder_idx = folder.split('_')[1]  # 如 "0" from "test_0_pixel"
#         dst_filename = f"TA_pRGB{folder_idx}.nii.gz"
#         dst_path = os.path.join(patient_path, dst_filename)
#         print(dst_path)
#
#         if os.path.exists(src_path):
#             shutil.copy(src_path, dst_path)
#         else:
#             print(f"⚠️ 未找到文件: {src_path}")

"""
nnunet提取
"""
import os
import shutil

# 源文件夹路径
src_dir = r"D:\SAM\GTVp_CTonly\20250701\nnUNet_RGB"

# 目标根目录
dst_root = r"C:\Users\WS\Desktop\transfer"

# 遍历编号
for i in range(15):  # 对应 GTVp_000.nii ~ GTVp_014.nii
    filename = f"RGB_{i:03d}.nii.gz"
    src_path = os.path.join(src_dir, filename)
    print(src_path)

    dst_folder = os.path.join(dst_root, f"p_{i}")
    os.makedirs(dst_folder, exist_ok=True)

    dst_path = os.path.join(dst_folder, "nnUNet_RGB.nii.gz")
    print(dst_path)

    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
    else:
        print(f"⚠️ 找不到源文件: {src_path}")
