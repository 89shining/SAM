"""
四边等距外扩固定值每层给框测试
支持 box=None（无提示层）
"""


import os
import sys
sys.path.append("/home/wusi/segment-anything")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
from testdatasetGTVp import TestDataset
import shutil

# ========= 模型权重路径 =========
ckpt_root = "/home/wusi/SAMdata/20250711/TrainResults/TrainAll"
fold_ckpts = [
    os.path.join(ckpt_root, f"fold_{i}/weights/best.pth") for i in range(1, 6)
]

sam_checkpoint = "/home/wusi/segment-anything/demo/configs/checkpoint/sam_vit_b_01ec64.pth"  # 原始SAM模型权重路径（如sam_vit_b_01ec64.pth）
model_type = "vit_b"
csv_path = "/home/wusi/SAMdata/20250711/test/test_rgb.csv"   # 测试数据CSV文件路径
root_dir = "/home/wusi/SAMdata/20250711/test"                         # 测试集根目录
image_dir = "/home/wusi/SAMdata/20250711/test/rgb_images"             # 测试image
nii_dir = "/home/wusi/SAMdata/20250711/test_nii"                      # 对应的参考NIfTI图像路径（含image.nii.gz）
base_output_dir = "/home/wusi/SAMdata/20250711/TestResults/cm/TrainAll" # 预测输出结果根目录
expand_cm_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]  # 不同外扩值

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for expand_cm in expand_cm_list:
    print(f"\n=== 正在处理外扩cm: {expand_cm} ===")
    output_dir = os.path.join(base_output_dir, f"expand_{expand_cm}cm")
    os.makedirs(output_dir, exist_ok=True)
    tmp_png_dir = os.path.join(output_dir, "tmp_png")
    os.makedirs(tmp_png_dir, exist_ok=True)

    # ========= 数据加载 =========
    test_dataset = TestDataset(csv_path=csv_path, root_dir=root_dir, nii_dir=nii_dir, target_size=(1024, 1024), expand_cm=expand_cm)
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
        for batch in test_loader:
            imgs = batch["image"].to(device).float()
            bbox = batch["box"][0]
            has_gt = batch["has_gt"]
            original_size = batch["original_size"][0]
            image_path = batch["image_path"][0]
            prob_list = []

            for net in nets:
                input_images = torch.stack([net.preprocess(im) for im in imgs], dim=0)
                image_embeddings = net.image_encoder(input_images)

                sparse_embeddings, dense_embeddings = net.prompt_encoder(
                    points=None,
                    boxes=(bbox.to(device) if bbox is not None else None),
                    masks=None
                )
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

            rel_path = os.path.relpath(image_path, image_dir)  # 确认image地址
            patient_folder = Path(rel_path).parent.name
            image_stem = Path(rel_path).stem
            save_subdir = os.path.join(tmp_png_dir, patient_folder)
            os.makedirs(save_subdir, exist_ok=True)
            save_path = os.path.join(save_subdir, image_stem + ".png")
            save_mask = (final_mask[0].squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
            imageio.imwrite(save_path, save_mask)


        # ========= PNG → NII =========
        def pngs_to_nii(png_dir, ref_nii_path, out_nii_path):
            ref = nib.load(ref_nii_path)
            affine, header = ref.affine, ref.header
            shape = ref.shape  # (H, W, D)
            vol = np.zeros((shape[2], shape[0], shape[1]), dtype=np.uint8)

            # 从文件名中提取 slice index
            pattern = re.compile(r"slice(\d+)", re.IGNORECASE)
            files = sorted(
                [f for f in os.listdir(png_dir) if f.endswith(".png")],
                key=lambda x: int(pattern.search(x).group(1)) if pattern.search(x) else 1e9,
            )

            for f in files:
                m = pattern.search(f)
                if not m:
                    continue
                idx = int(m.group(1))  # 层号
                if idx >= shape[2]:
                    continue
                img = np.array(Image.open(os.path.join(png_dir, f)).convert("L"))
                img = np.rot90(img, k=3)
                img = np.fliplr(img)
                vol[idx] = img

            vol = np.transpose(vol, (1, 2, 0))
            nii = nib.Nifti1Image(vol, affine=affine, header=header)
            nib.save(nii, out_nii_path)
            print(f"✅ Saved: {out_nii_path}")

        for pa in os.listdir(nii_dir):
            idx = re.search(r"\d+", pa)
            if not idx:
                continue
            idx = idx.group(0).zfill(3)
            pa_path = os.path.join(nii_dir, pa)
            ref_nii = os.path.join(pa_path, "image.nii.gz")
            pre_png_dir = os.path.join(tmp_png_dir, pa)
            if not os.path.exists(pre_png_dir):
                continue
            os.makedirs(output_dir, exist_ok=True)
            out_nii = os.path.join(output_dir, f"GTVp_{idx}.nii.gz")
            pngs_to_nii(pre_png_dir, ref_nii, out_nii)

        shutil.rmtree(tmp_png_dir, ignore_errors=True)
        print(f"🧹 已删除临时目录: {tmp_png_dir}")
        del nets
        torch.cuda.empty_cache()
        print(f"✅ 外扩 {expand_cm} cm 推理完成，结果保存在: {output_dir}")