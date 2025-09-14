"""
上下界 + 面积最大层，插值——外扩0.5cm
"""

import os
import sys
sys.path.append("/home/wusi/segment-anything")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import re
import csv
import torch
import imageio
import nibabel as nib
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
from testdatasetGTVp_3 import TestDataset
import shutil

# ========= 配置路径（请根据实际路径修改） =========
fold_ckpts = [
    "/home/wusi/SAMdata/20250711/trainresult_Freeze_image_encoder/fold_1/weights/best.pth",                      # 每个fold的best权重路径
    "/home/wusi/SAMdata/20250711/trainresult_Freeze_image_encoder/fold_2/weights/best.pth",
    "/home/wusi/SAMdata/20250711/trainresult_Freeze_image_encoder/fold_3/weights/best.pth",
    "/home/wusi/SAMdata/20250711/trainresult_Freeze_image_encoder/fold_4/weights/best.pth",
    "/home/wusi/SAMdata/20250711/trainresult_Freeze_image_encoder/fold_5/weights/best.pth"
]
sam_checkpoint = "/home/wusi/segment-anything/demo/configs/checkpoint/sam_vit_b_01ec64.pth"  # 原始SAM模型权重路径（如sam_vit_b_01ec64.pth）
model_type = "vit_b"
csv_path = "/home/wusi/SAMdata/20250711/test/test_rgb.csv"   # 测试数据CSV文件路径
root_dir = "/home/wusi/SAMdata/20250711/test"                         # 测试集根目录
image_dir = "/home/wusi/SAMdata/20250711/test/rgb_images"             # 测试image
ii_dir = "/home/wusi/SAMdata/20250711/test_nii"                      # 对应的参考NIfTI图像路径（含image.nii.gz）
base_output_dir = "/home/wusi/SAMdata/20250711/testresults/Prompt_maxarea" # 预测输出结果根目录
expand_cm_list = [0.5]  # 外扩距离（单位：cm）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for expand_cm in expand_cm_list:
    print(f"\n=== 正在处理外扩距离: {expand_cm} cm ===")
    output_dir = os.path.join(base_output_dir, f"expand_{expand_cm:.1f}cm")
    os.makedirs(output_dir, exist_ok=True)
    tmp_png_dir = os.path.join(output_dir, "tmp_png")
    os.makedirs(tmp_png_dir, exist_ok=True)

    # ========= 数据加载 =========
    test_dataset = TestDataset(csv_path=csv_path, root_dir=root_dir, nii_dir=ii_dir, target_size=(1024, 1024), expand_cm=expand_cm)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # ========= 加载模型 =========
    nets = []
    for ckpt in fold_ckpts:
        model = sam_model_registry[model_type](checkpoint=None)
        model.to(device)
        model.load_state_dict(torch.load(sam_checkpoint, map_location=device), strict=False)
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        model.eval()
        nets.append(model)

    # ========= 融合预测 =========
    with torch.no_grad():
        for idx, (image, mask, box, original_size, image_path) in enumerate(test_loader):
            imgs = image.to(device).float()
            bbox = box.to(device).float()
            prob_list = []

            for net in nets:
                input_images = torch.stack([net.preprocess(im) for im in imgs], dim=0)
                image_embeddings = net.image_encoder(input_images)
                sparse_embeddings, dense_embeddings = net.prompt_encoder(points=None, boxes=bbox, masks=None)
                low_res_masks, _ = net.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )
                masks = net.postprocess_masks(low_res_masks, input_size=imgs.shape[-2:], original_size=original_size)
                prob_mask = torch.sigmoid(masks)
                prob_list.append(prob_mask)

            avg_prob = torch.mean(torch.stack(prob_list, dim=0), dim=0)
            final_mask = (avg_prob > 0.5).float()

            rel_path = os.path.relpath(image_path[0], image_dir)  # 确认image地址
            patient_folder = Path(rel_path).parent.name
            image_stem = Path(rel_path).stem
            save_subdir = os.path.join(tmp_png_dir, patient_folder)
            os.makedirs(save_subdir, exist_ok=True)
            save_path = os.path.join(save_subdir, image_stem + ".png")
            save_mask = (final_mask[0].squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
            imageio.imwrite(save_path, save_mask)

    # ========= PNG → NII 并按 nnU-Net 命名规范保存 =========
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

        for f in sorted(os.listdir(png_dir),
                        key=lambda x: int(os.path.splitext(x)[0]) if x.endswith(".png") and os.path.splitext(x)[
                            0].isdigit() else float('inf')):
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
    datanii_dir = ii_dir  # 原始测试数据nii目录
    pred_dir = tmp_png_dir  # 预测mask结果png目录
    vis_dir = base_output_dir  # pred_nii拟储存目录

    all_slice_mappings = []

    for pa in os.listdir(datanii_dir):
        match = re.search(r'\d+', pa)
        if not match:
            print(f"跳过无效文件夹：{pa}")
            continue
        idx = match.group(0).zfill(3)
        pa_path = os.path.join(datanii_dir, pa)
        image_nii_path = os.path.join(pa_path, "image.nii.gz")
        pre_png_dir = os.path.join(pred_dir, pa)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"GTVp_{idx}.nii.gz")
        pngs_to_nii(
            png_dir=pre_png_dir,
            reference_nii_path=image_nii_path,
            output_nii_path=output_path,
            patient_id=pa,
            all_mappings=all_slice_mappings
        )


    shutil.rmtree(tmp_png_dir)
    print(f"🧹 已删除临时目录: {tmp_png_dir}")
    print(f"外扩 {expand_cm}cm 融合预测完成，结果保存在: {output_dir}")
