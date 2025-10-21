"""
测试不同提示层数性能 + 可视化提示框
"""

import os
import sys
sys.path.append("/home/wusi/segment-anything")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import re
import torch
import imageio
import nibabel as nib
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
from testdatasetGTVp_n import TestDataset
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import measure


def visualize_prediction_original_scale(image_1024, gt_mask, pred_mask, box_1024, original_size, save_path, spacing=None, expand_cm=None):
    # ---------- 0) mask -> 2D numpy ----------
    def to_2d_np(x):
        if torch.is_tensor(x):
            a = x.detach().cpu().numpy()
        else:
            a = np.asarray(x)
        a = np.squeeze(a)
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        return a

    # ---------- 1) original_size ----------
    if isinstance(original_size, (list, tuple)):
        H_orig = int(original_size[0]) if not torch.is_tensor(original_size[0]) else int(original_size[0].item())
        W_orig = int(original_size[1]) if not torch.is_tensor(original_size[1]) else int(original_size[1].item())
    elif torch.is_tensor(original_size):
        arr = original_size.detach().cpu().reshape(-1)
        H_orig, W_orig = int(arr[0].item()), int(arr[1].item())
    else:
        H_orig, W_orig = map(int, original_size)

    scale_x = float(W_orig) / 1024.0
    scale_y = float(H_orig) / 1024.0

    # ---------- 2) image ----------
    if torch.is_tensor(image_1024):
        img_t = image_1024.squeeze(0) if image_1024.dim() == 4 else image_1024
        img_np = img_t.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    else:
        img_np = np.array(image_1024)
        if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
            img_np = np.transpose(img_np, (1, 2, 0)).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

    image_resized = cv2.resize(img_np, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)

    # ---------- 3) 插值框 (黄) ----------
    if torch.is_tensor(box_1024):
        b = box_1024.detach().cpu().reshape(-1)
        x0f, y0f, x1f, y1f = [float(v.item()) for v in b]
    else:
        b = np.asarray(box_1024, dtype=float).reshape(-1)
        x0f, y0f, x1f, y1f = b.tolist()

    x0 = int(round(x0f * scale_x))
    y0 = int(round(y0f * scale_y))
    x1 = int(round(x1f * scale_x))
    y1 = int(round(y1f * scale_y))

    # ---------- 4) mask ----------
    gt_np   = to_2d_np(gt_mask)
    pred_np = to_2d_np(pred_mask)
    if pred_np.dtype != np.uint8:
        pred_np = (pred_np > 0.5).astype(np.uint8)

    # ---------- 5) 绘图 ----------
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(image_resized)

    # (a) GT 轮廓（绿，细）
    for c in measure.find_contours(gt_np, 0.5):
        ax.plot(c[:, 1], c[:, 0], linewidth=0.8, color='lime')

    # (b) Pred 轮廓（红，细）
    for c in measure.find_contours(pred_np, 0.5):
        ax.plot(c[:, 1], c[:, 0], linewidth=0.8, color='red')

    # (c) 插值框（黄）
    rect = patches.Rectangle((x0, y0), max(1, x1 - x0), max(1, y1 - y0),
                             linewidth=0.8, edgecolor='yellow', facecolor='none')
    ax.add_patch(rect)

    # (d) 基于 GT 的物理外扩 框（蓝）
    coords = np.argwhere(gt_np > 0)
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        if spacing is not None:
            sx, sy = spacing  # (mm/px)
            expand_x = int(round((expand_cm * 10) / sx))
            expand_y = int(round((expand_cm * 10)  / sy))
        else:
            raise ValueError("Spacing information is required for cm-to-pixel conversion, but got None.")

        x_min = max(0, x_min - expand_x)
        x_max = min(W_orig - 1, x_max + expand_x)
        y_min = max(0, y_min - expand_y)
        y_max = min(H_orig - 1, y_max + expand_y)

        rect_gt = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                    linewidth=0.8, edgecolor='blue', facecolor='none')
        ax.add_patch(rect_gt)

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()



# ========= 配置路径（请根据实际路径修改） =========
fold_ckpts = [
    "/home/wusi/SAMdata/20250711/TrainResults/trainresult_Freeze_image_encoder/fold_1/weights/best.pth",                      # 每个fold的best权重路径
    "/home/wusi/SAMdata/20250711/TrainResults/trainresult_Freeze_image_encoder/fold_2/weights/best.pth",
    "/home/wusi/SAMdata/20250711/TrainResults/trainresult_Freeze_image_encoder/fold_3/weights/best.pth",
    "/home/wusi/SAMdata/20250711/TrainResults/trainresult_Freeze_image_encoder/fold_4/weights/best.pth",
    "/home/wusi/SAMdata/20250711/TrainResults/trainresult_Freeze_image_encoder/fold_5/weights/best.pth"
]
sam_checkpoint = "/home/wusi/segment-anything/demo/configs/checkpoint/sam_vit_b_01ec64.pth"  # 原始SAM模型权重路径（如sam_vit_b_01ec64.pth）
model_type = "vit_b"
csv_path = "/home/wusi/SAMdata/20250711/test/test_rgb.csv"   # 测试数据CSV文件路径
root_dir = "/home/wusi/SAMdata/20250711/test"                         # 测试集根目录
image_dir = "/home/wusi/SAMdata/20250711/test/rgb_images"             # 测试image
ii_dir = "/home/wusi/SAMdata/20250711/test_nii"                      # 对应的参考NIfTI图像路径（含image.nii.gz）
base_output_dir = "/home/wusi/SAMdata/20250711/TestResults/Num_box_prompts" # 预测输出结果根目录
expand_cm_list = [0.5]  # 外扩距离（单位：cm）
num_prompts_list = [2, 3, 5]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for expand_cm in expand_cm_list:
    print(f"\n=== 正在处理外扩距离: {expand_cm} cm ===")
    for num_prompts in num_prompts_list:
        print(f"\n=== 提示层数: {num_prompts} 层 ===")
        output_dir = os.path.join(base_output_dir, f"expand_{expand_cm:.1f}cm",  f"prompt_{num_prompts}_vis")
        os.makedirs(output_dir, exist_ok=True)
        tmp_png_dir = os.path.join(output_dir, "tmp_png")
        os.makedirs(tmp_png_dir, exist_ok=True)

        # ========= 数据加载 =========
        test_dataset = TestDataset(csv_path=csv_path, root_dir=root_dir, nii_dir=ii_dir, target_size=(1024, 1024),
                                   expand_cm=expand_cm,
                                   num_prompts=num_prompts)
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

                # 取患者文件夹名 + 切片文件名
                nii_path = os.path.join(ii_dir, patient_folder, "image.nii.gz")
                nii = nib.load(nii_path)
                spacing = nii.header.get_zooms()[:2]  # (sx, sy)
                slice_name = Path(image_path[0]).stem

                # 为每个患者单独建子文件夹
                vis_save_dir = os.path.join(output_dir, "vis", patient_folder)
                os.makedirs(vis_save_dir, exist_ok=True)

                # 保存为 {患者ID}/{切片号}_vis.png
                vis_save_path = os.path.join(vis_save_dir, f"{slice_name}_vis.png")

                visualize_prediction_original_scale(
                    image_1024=image,
                    gt_mask=mask,
                    pred_mask=final_mask,
                    box_1024=box,
                    original_size=original_size,
                    save_path=vis_save_path,
                    spacing=spacing,
                    expand_cm=expand_cm
                )


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
        print(f"提示层数 {num_prompts} 实验完成，结果保存在: {output_dir}")
