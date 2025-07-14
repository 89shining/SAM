"""
生成预测_00N.nii.gz
"""

import csv
import imageio
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import nibabel as nib
from PIL import Image
import shutil
from testdatasetGTVp import TestDataset
from segment_anything import sam_model_registry

# ========== 模型配置 ==========
sam_checkpoint = "D:\\project\\segment-anything\\demo\\configs\\checkpoint\\sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
net = sam_model_registry[model_type](checkpoint=None)
net.to(device)
net.load_state_dict(torch.load(sam_checkpoint, map_location=device), strict=False)
net.load_state_dict(torch.load("D:\SAM\GTVp_CTonly/20250630/trainresults_kfold_freeze_encoder_decoder/fold_1\weights/best.pth", map_location=device))
net.eval()

# ========== 路径配置 ==========
csv_path = "D:\SAM\GTVp_CTonly/20250707\dataset/test/test_rgb_dataset.csv"
root_dir = "D:\SAM\GTVp_CTonly/20250707\dataset/test"
image_dir = "D:\SAM\GTVp_CTonly/20250707\dataset/test/rgb_images"
nii_dir = "D:\SAM\GTVp_CTonly/20250707/datanii/test_nii"
test_dir = "D:/SAM/GTVp_CTonly/20250707/Freeze_encoder_decoder"    # 测试结果保存地址
png_dir = os.path.join(test_dir, "tmp_png") # 临时保存 PNG
nii_out_dir = os.path.join(test_dir, "test_0_pixel")
os.makedirs(png_dir, exist_ok=True)
os.makedirs(nii_out_dir, exist_ok=True)

# ========== 数据加载 ==========
test_dataset = TestDataset(csv_path=csv_path, root_dir=root_dir, nii_dir=nii_dir, target_size=(1024, 1024))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ========== 预测并保存临时 PNG ==========
with torch.no_grad():
    for idx, (image, mask, box, original_size, image_path) in enumerate(test_loader):
        imgs = image.to(device).float()
        bbox = box.to(device).float()

        input_images = torch.stack([net.preprocess(im) for im in imgs], dim=0)
        image_embeddings = net.image_encoder(input_images)

        sparse_embeddings, dense_embeddings = net.prompt_encoder(
            points=None,
            boxes=bbox,
            masks=None
        )

        low_res_masks, _ = net.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=net.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        masks = net.postprocess_masks(
            low_res_masks,
            input_size=imgs.shape[-2:],
            original_size=original_size
        )

        prob_mask = torch.sigmoid(masks)
        resized_mask = (prob_mask > 0.5).float()

        # 保存为临时 PNG 文件
        save_mask = (resized_mask[0].squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
        rel_path = os.path.relpath(image_path[0], image_dir)
        patient_folder = Path(rel_path).parent.name
        image_stem = Path(rel_path).stem
        save_subdir = os.path.join(png_dir, patient_folder)
        os.makedirs(save_subdir, exist_ok=True)
        save_path = os.path.join(save_subdir, image_stem + ".png")
        imageio.imwrite(save_path, save_mask)


# ========== 函数：PNG → NII ==========
def pngs_to_nii(png_dir, reference_nii_path, output_nii_path, patient_id, all_mappings):
    # 读取参考NIfTI图像，提取空间信息
    ref_nii = nib.load(reference_nii_path)
    affine = ref_nii.affine
    header = ref_nii.header
    shape = ref_nii.shape  # (H, W, D)
    # print(shape)

    # 初始化全 0 体积，shape 为 (D, H, W)
    volume = np.zeros((shape[2], shape[0], shape[1]), dtype=np.uint8)

    # 存储索引和对应文件名
    slice_mapping = []

    for f in sorted(os.listdir(png_dir), key=lambda x: int(os.path.splitext(x)[0]) if x.endswith(".png") and os.path.splitext(x)[0].isdigit() else float('inf')):
        if not f.endswith(".png"):
            continue
        try:
            # 提取数字作为切片索引
            slice_idx = int(os.path.splitext(f)[0])
        except ValueError:
            print(f"跳过无法识别的文件名：{f}")
            continue

        img = Image.open(os.path.join(png_dir, f)).convert('L')
        arr = np.array(img)
        arr = np.rot90(arr, k=3)
        arr = np.fliplr(arr)

        if slice_idx >= volume.shape[0]:
            print(f"切片编号 {slice_idx} 超出体积深度 {volume.shape[0]}，跳过。")
            continue

        volume[slice_idx] = arr
        slice_mapping.append((patient_id, slice_idx, f))

        # 转换为 (H, W, D)
    volume = np.transpose(volume, (1, 2, 0))

    nii_img = nib.Nifti1Image(volume, affine=affine, header=header)
    nib.save(nii_img, output_nii_path)
    print(f"Saved NIfTI: {output_nii_path}")

    all_mappings.extend(slice_mapping)

# 示例调用
datanii_dir = nii_dir  # 原始测试数据nii目录
pred_dir = png_dir  # 预测mask结果png目录
vis_dir = nii_out_dir   # pred_nii拟储存目录

all_slice_mappings = []

for pa in os.listdir(datanii_dir):
    print(pa)
    nid = pa.split('_', -1)[-1]
    pa_path = os.path.join(datanii_dir, pa)
    image_nii_path = os.path.join(pa_path, "image.nii.gz")
    pre_png_dir = os.path.join(pred_dir, pa)
    # output_dir = os.path.join(vis_dir, pa)
    # os.makedirs(output_dir, exist_ok=True)
    # output_path = os.path.join(output_dir, "pred.nii.gz")
    output_path = os.path.join(vis_dir, f"pred_{int(nid):03d}.nii.gz")
    pngs_to_nii(
    png_dir=pre_png_dir,
    reference_nii_path=image_nii_path,
    output_nii_path=output_path,
    patient_id=pa,
    all_mappings=all_slice_mappings
    )

    # for filename in ["image.nii.gz", "GTVp.nii.gz"]:
    #     src_file = os.path.join(pa_path, filename)
    #     tgt_file = os.path.join(output_dir, filename)
    #
    #     if os.path.exists(src_file):
    #         shutil.copy(src_file, tgt_file)
    #         print(f"Copied {filename} to {output_dir}")
    #     else:
    #         print(f"源文件缺失: {src_file}")

# # 保存为总的 CSV
# output_csv = os.path.join(vis_dir, "all_slice_orders.csv")
# with open(output_csv, mode='w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(["patient_id", "slice_index", "file_name"])
#     writer.writerows(sorted(all_slice_mappings))
#
# print(f"所有患者堆叠顺序已保存至：{output_csv}")

# ========== 自动清理临时 PNG ==========
shutil.rmtree(png_dir)
print(f"🧹 已删除临时目录: {png_dir}")
