"""
第一轮：box
后续多轮预测mask迭代
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
from testdataset import TestDataset
import shutil

# ===================== 配置 =====================
fold_ckpts = [
    "/home/wusi/SAMdata/Eso/20260104_CTV/box_predmask/TrainResult/fold_3/weights/best.pth"
]

sam_checkpoint = "/home/wusi/segment-anything/demo/configs/checkpoint/sam_vit_b_01ec64.pth"
model_type = "vit_b"

csv_path = "/home/wusi/SAMdata/Eso/20251217_CTV/dataset/test/test_rgb.csv"
root_dir = "/home/wusi/SAMdata/Eso/20251217_CTV/dataset/test"
image_dir = "/home/wusi/SAMdata/Eso/20251217_CTV/dataset/test/rgb_images"
nii_dir = "/home/wusi/SAMdata/Eso/20260104_CTV/nnUNet_mask/cropdatanii/test_nii"

base_output_dir = "/home/wusi/SAMdata/Eso/20260104_CTV/box_predmask/multi-iter/TestResult"

expand_cm_list = [0.5]
num_iters = 3     # ⭐ 多轮迭代次数（1=只box，2=一次refine，>=3 多轮）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =================================================

print("\n======================= SAM 测试配置 =======================")
print(f"迭代轮数 num_iters = {num_iters}")
print(f"输出目录: {base_output_dir}")
print("===========================================================\n")

for expand_cm in expand_cm_list:
    print(f"\n=== 外扩 {expand_cm} cm ===")

    output_dir = os.path.join(base_output_dir, f"expand_{expand_cm}cm_iter{num_iters}")
    os.makedirs(output_dir, exist_ok=True)

    tmp_png_dir = os.path.join(output_dir, "tmp_png")
    os.makedirs(tmp_png_dir, exist_ok=True)

    # ================= Dataset =================
    test_dataset = TestDataset(
        csv_path=csv_path,
        root_dir=root_dir,
        nii_dir=nii_dir,
        target_size=(1024, 1024),
        expand_cm=expand_cm
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # ================= Load models =================
    nets = []
    for ckpt in fold_ckpts:
        model = sam_model_registry[model_type](checkpoint=None)
        model.to(device)

        model.load_state_dict(torch.load(sam_checkpoint, map_location=device), strict=False)
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)

        model.eval()
        nets.append(model)

    # ================= 推理 =================
    with torch.no_grad():
        for idx, (image, mask, box, original_size, image_path) in enumerate(test_loader):

            imgs = image.to(device).float()
            bbox = box.to(device).float()

            prob_list = []

            for net in nets:
                # image encoder
                input_images = torch.stack([net.preprocess(im) for im in imgs], dim=0)
                image_embeddings = net.image_encoder(input_images)

                prev_low_res = None

                # ========= 多轮 mask refine =========
                for it in range(num_iters):
                    sparse_embeddings, dense_embeddings = net.prompt_encoder(
                        points=None,
                        boxes=bbox,
                        masks=prev_low_res   # 第0轮 None，仅 box
                    )

                    low_res_masks, _ = net.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False
                    )

                    prev_low_res = low_res_masks.detach()

                # postprocess
                masks = net.postprocess_masks(
                    prev_low_res,
                    input_size=imgs.shape[-2:],
                    original_size=original_size
                )

                prob_list.append(torch.sigmoid(masks))

            # ========= 融合 =========
            avg_prob = torch.mean(torch.stack(prob_list, dim=0), dim=0)
            final_mask = (avg_prob > 0.5).float()

            # ========= 保存 PNG =========
            rel_path = os.path.relpath(image_path[0], image_dir)
            patient_folder = Path(rel_path).parent.name
            image_stem = Path(rel_path).stem

            save_subdir = os.path.join(tmp_png_dir, patient_folder)
            os.makedirs(save_subdir, exist_ok=True)

            save_path = os.path.join(save_subdir, image_stem + ".png")
            save_mask = (final_mask[0].squeeze().cpu().numpy() > 0).astype(np.uint8) * 255
            imageio.imwrite(save_path, save_mask)

    # ================= PNG → NIfTI =================
    def pngs_to_nii(png_dir, reference_nii_path, output_nii_path):
        ref = nib.load(reference_nii_path)
        affine = ref.affine
        header = ref.header
        shape = ref.shape  # (H,W,D)

        volume = np.zeros((shape[2], shape[0], shape[1]), dtype=np.uint8)

        for f in sorted(os.listdir(png_dir), key=lambda x: int(os.path.splitext(x)[0])):
            slice_idx = int(os.path.splitext(f)[0])
            img = Image.open(os.path.join(png_dir, f)).convert("L")
            arr = np.array(img)

            arr = np.rot90(arr, k=3)
            arr = np.fliplr(arr)

            if slice_idx < volume.shape[0]:
                volume[slice_idx] = arr

        volume = np.transpose(volume, (1, 2, 0))
        nib.save(nib.Nifti1Image(volume, affine, header), output_nii_path)

    for pa in os.listdir(nii_dir):
        match = re.search(r'\d+', pa)
        if not match:
            continue

        idx = match.group(0).zfill(3)
        ref_nii = os.path.join(nii_dir, pa, "image.nii.gz")
        png_dir = os.path.join(tmp_png_dir, pa)

        out_nii = os.path.join(output_dir, f"CTV_{idx}.nii.gz")
        pngs_to_nii(png_dir, ref_nii, out_nii)
        print(f"Saved: {out_nii}")

    shutil.rmtree(tmp_png_dir)
    print(f"🧹 清理临时目录: {tmp_png_dir}")
