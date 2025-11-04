"""
单折推理（指定best fold）
四边等距外扩固定值每层给框测试
支持 box=None（无提示层）
"""

import os
import sys
sys.path.append("/home/wusi/segment-anything")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
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

# ========= 配置路径 =========
ckpt_root = "/home/wusi/SAMdata/20250711/TrainResults/TrainAll"
best_fold = 3  # 手动指定最优fold（根据训练log）
ckpt_path = os.path.join(ckpt_root, f"fold_{best_fold}/weights/best.pth")

sam_checkpoint = "/home/wusi/segment-anything/demo/configs/checkpoint/sam_vit_b_01ec64.pth"  # 原始SAM模型权重路径
model_type = "vit_b"
csv_path = "/home/wusi/SAMdata/20250711/test/test_rgb.csv"   # 测试数据CSV文件路径
root_dir = "/home/wusi/SAMdata/20250711/test"                # 测试集根目录
image_dir = "/home/wusi/SAMdata/20250711/test/rgb_images"    # 测试image
nii_dir = "/home/wusi/SAMdata/20250711/test_nii"             # 对应的NIfTI图像路径（含image.nii.gz）
base_output_dir = "/home/wusi/SAMdata/20250711/TestResults/cm/TrainAll"  # 预测输出结果根目录
expand_cm_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]  # 不同外扩值

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= 加载模型 =========
print(f" Loading SAM model (fold_{best_fold})...")
model = sam_model_registry[model_type](checkpoint=None)
model.to(device)
model.load_state_dict(torch.load(sam_checkpoint, map_location=device), strict=False)
model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
model.eval()
print(f" Model loaded from: {ckpt_path}")

# ========= 多外扩推理 =========
for expand_cm in expand_cm_list:
    print(f"\n=== 正在处理外扩cm: {expand_cm} ===")
    output_dir = os.path.join(base_output_dir, f"expand_{expand_cm}cm")
    os.makedirs(output_dir, exist_ok=True)
    tmp_png_dir = os.path.join(output_dir, "tmp_png")
    os.makedirs(tmp_png_dir, exist_ok=True)

    # ========= 数据加载 =========
    test_dataset = TestDataset(csv_path=csv_path, root_dir=root_dir, nii_dir=nii_dir,
                               target_size=(1024, 1024), expand_cm=expand_cm)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # ========= 预测 =========
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device).float()
            bbox = batch["box"][0]
            has_gt = batch["has_gt"]
            original_size = batch["original_size"][0]
            image_path = batch["image_path"][0]

            # --- 推理 ---
            input_images = torch.stack([model.preprocess(im) for im in imgs], dim=0)
            image_embeddings = model.image_encoder(input_images)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None,
                boxes=(bbox.to(device) if bbox is not None else None),
                masks=None
            )
            low_res_masks, _ = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            masks = model.postprocess_masks(low_res_masks, input_size=imgs.shape[-2:], original_size=original_size)
            prob_mask = torch.sigmoid(masks)
            final_mask = (prob_mask > 0.5).float()

            # --- 保存 PNG ---
            rel_path = os.path.relpath(image_path, image_dir)
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
            idx = int(m.group(1))
            if idx >= shape[2]:
                continue
            img = np.array(Image.open(os.path.join(png_dir, f)).convert("L"))
            img = np.rot90(img, k=3)
            img = np.fliplr(img)
            vol[idx] = img

        vol = np.transpose(vol, (1, 2, 0))
        nii = nib.Nifti1Image(vol, affine=affine, header=header)
        nib.save(nii, out_nii_path)
        print(f" Saved: {out_nii_path}")

    # --- 按患者保存 NII ---
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

    # --- 清理缓存 ---
    shutil.rmtree(tmp_png_dir, ignore_errors=True)
    torch.cuda.empty_cache()
    print(f"✅ 外扩 {expand_cm} cm 推理完成，结果保存在: {output_dir}")
