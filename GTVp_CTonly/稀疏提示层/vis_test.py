"""
可视化稀疏插值提示框和预测结果
在 predict_interp.py 基础上修改：直接保存 NIfTI（命名 GTVp_001.nii.gz）
"""

import os
import sys
import re
sys.path.append("/home/wusi/segment-anything")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
from testdatasetGTVp_3 import TestDataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import measure


# ========= 可视化函数 =========
def visualize_prediction_original_scale(image_1024, gt_mask, pred_mask, box_1024,
                                        original_size, save_path, spacing=None, expand_cm=None):
    def to_2d_np(x):
        if torch.is_tensor(x):
            a = x.detach().cpu().numpy()
        else:
            a = np.asarray(x)
        a = np.squeeze(a)
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        return a

    # 原始大小
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

    # 图像
    if torch.is_tensor(image_1024):
        img_t = image_1024.squeeze(0) if image_1024.dim() == 4 else image_1024
        img_np = img_t.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    else:
        img_np = np.array(image_1024)
        if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
            img_np = np.transpose(img_np, (1, 2, 0)).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

    # mask
    gt_np = to_2d_np(gt_mask)
    pred_np = to_2d_np(pred_mask)
    if pred_np.dtype != np.uint8:
        pred_np = (pred_np > 0.5).astype(np.uint8)

    # 框
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

    # 绘图
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(img_np)

    # (a) GT 轮廓（绿）
    for c in measure.find_contours(gt_np, 0.5):
        ax.plot(c[:, 1], c[:, 0], linewidth=0.8, color='lime')

    # (b) Pred 轮廓（红）
    for c in measure.find_contours(pred_np, 0.5):
        ax.plot(c[:, 1], c[:, 0], linewidth=0.8, color='red')

    # (c) 插值框（黄）
    rect = patches.Rectangle((x0, y0), max(1, x1 - x0), max(1, y1 - y0),
                             linewidth=0.8, edgecolor='yellow', facecolor='none')
    ax.add_patch(rect)

    # (d) 基于 GT 的物理外扩 框（蓝）
    coords = np.argwhere(gt_np > 0)
    if coords.size > 0 and spacing is not None and expand_cm is not None:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        sx, sy = spacing  # (mm/px)
        expand_x = int(round((expand_cm * 10) / sx))
        expand_y = int(round((expand_cm * 10) / sy))
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


# ========= 配置 =========
fold_ckpts = [
    "/home/wusi/SAMdata/20250711/TrainResults/trainresult_Freeze_image_encoder/fold_1/weights/best.pth",
    "/home/wusi/SAMdata/20250711/TrainResults/trainresult_Freeze_image_encoder/fold_2/weights/best.pth",
    "/home/wusi/SAMdata/20250711/TrainResults/trainresult_Freeze_image_encoder/fold_3/weights/best.pth",
    "/home/wusi/SAMdata/20250711/TrainResults/trainresult_Freeze_image_encoder/fold_4/weights/best.pth",
    "/home/wusi/SAMdata/20250711/TrainResults/trainresult_Freeze_image_encoder/fold_5/weights/best.pth"
]
sam_checkpoint = "/home/wusi/segment-anything/demo/configs/checkpoint/sam_vit_b_01ec64.pth"
model_type = "vit_b"
csv_path = "/home/wusi/SAMdata/20250711/test/test_rgb.csv"
root_dir = "/home/wusi/SAMdata/20250711/test"
ii_dir = "/home/wusi/SAMdata/20250711/test_nii"
base_output_dir = "/home/wusi/SAMdata/20250711/TestResults/Prompt_maxarea"
expand_cm_list = [0.5]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========= 主流程 =========
for expand_cm in expand_cm_list:
    print(f"\n=== 正在处理外扩距离: {expand_cm} cm ===")
    output_dir = os.path.join(base_output_dir, f"expand_{expand_cm:.1f}cm")
    os.makedirs(output_dir, exist_ok=True)

    # 数据
    test_dataset = TestDataset(csv_path=csv_path, root_dir=root_dir,
                               nii_dir=ii_dir, target_size=(1024, 1024),
                               expand_cm=expand_cm)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 模型
    nets = []
    for ckpt in fold_ckpts:
        model = sam_model_registry[model_type](checkpoint=None)
        model.to(device)
        model.load_state_dict(torch.load(sam_checkpoint, map_location=device), strict=False)
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        model.eval()
        nets.append(model)

    # 预测并保存
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

            # 找到病人 ID
            rel_path = os.path.relpath(image_path[0], root_dir)
            patient_folder = Path(rel_path).parent.name

            # 提取数字编号并补零
            match = re.search(r'\d+', patient_folder)
            if not match:
                raise ValueError(f"患者文件夹名 {patient_folder} 中未找到数字编号")
            idx_str = match.group(0).zfill(3)

            # 保存 NIfTI（命名 GTVp_001.nii.gz）
            ref_nii_path = os.path.join(ii_dir, patient_folder, "image.nii.gz")
            ref_nii = nib.load(ref_nii_path)
            affine, header = ref_nii.affine, ref_nii.header
            pred_arr = final_mask.squeeze().cpu().numpy().astype(np.uint8)
            pred_nii = nib.Nifti1Image(pred_arr, affine=affine, header=header)
            out_path = os.path.join(output_dir, f"GTVp_{idx_str}.nii.gz")
            nib.save(pred_nii, out_path)
            print(f"Saved: {out_path}")

            # 可视化 PNG
            vis_save_dir = os.path.join(output_dir, "vis", patient_folder)
            os.makedirs(vis_save_dir, exist_ok=True)
            slice_name = Path(rel_path).stem
            vis_save_path = os.path.join(vis_save_dir, f"{slice_name}_vis.png")
            spacing = ref_nii.header.get_zooms()[:2]
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

    print(f"外扩 {expand_cm}cm 融合预测完成，结果保存在: {output_dir}")
