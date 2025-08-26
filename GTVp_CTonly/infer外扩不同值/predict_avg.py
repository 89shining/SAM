"""
实验B: 切片级随机外扩
统计3D Dice 和 3D HD95 (mm)，每个患者当次r的所有切片平均外扩量
结果保存到 Excel (overview + 每个患者sheet)
"""

import os
import sys
sys.path.append("/home/wusi/segment-anything")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
from testdatasetGTVp_2 import TestDataset
import pandas as pd
from collections import defaultdict
import re as _re
from tqdm import tqdm
from scipy import ndimage as ndi

# ============ 工具函数 ============
def dice3d_numpy(pred_vol, gt_vol):
    P = (pred_vol > 0).astype(np.uint8)
    G = (gt_vol > 0).astype(np.uint8)
    inter = (P & G).sum()
    denom = P.sum() + G.sum()
    if denom == 0: return 1.0
    return 2.0 * inter / denom

def hd95_numpy(gt_vol, pred_vol, spacing=(1.0,1.0,1.0)) -> float:
    G = gt_vol > 0
    P = pred_vol > 0
    if not G.any() and not P.any():
        return 0.0
    if not G.any() or not P.any():
        return float('nan')
    conn = ndi.generate_binary_structure(3, 1)
    G_er = ndi.binary_erosion(G, structure=conn, iterations=1, border_value=0)
    P_er = ndi.binary_erosion(P, structure=conn, iterations=1, border_value=0)
    Sg = G ^ G_er
    Sp = P ^ P_er
    dt_P = ndi.distance_transform_edt(~Sp, sampling=spacing)
    dt_G = ndi.distance_transform_edt(~Sg, sampling=spacing)
    d_g2p = dt_P[Sg]
    d_p2g = dt_G[Sp]
    if d_g2p.size == 0 and d_p2g.size == 0:
        return 0.0
    d = np.concatenate([d_g2p, d_p2g]) if d_p2g.size else d_g2p
    return float(np.percentile(d, 95))

def _sheet_name(s: str) -> str:
    s = _re.sub(r'[\[\]\:\*\?\/\\]', '_', str(s))
    return s[:31]

# ========= 配置路径 =========
fold_ckpts = [
    "/home/wusi/SAMdata/20250711/trainresult_Freeze_image_encoder/fold_1/weights/best.pth",
    "/home/wusi/SAMdata/20250711/trainresult_Freeze_image_encoder/fold_2/weights/best.pth",
    "/home/wusi/SAMdata/20250711/trainresult_Freeze_image_encoder/fold_3/weights/best.pth",
    "/home/wusi/SAMdata/20250711/trainresult_Freeze_image_encoder/fold_4/weights/best.pth",
    "/home/wusi/SAMdata/20250711/trainresult_Freeze_image_encoder/fold_5/weights/best.pth"
]
sam_checkpoint = "/home/wusi/segment-anything/demo/configs/checkpoint/sam_vit_b_01ec64.pth"
model_type = "vit_b"
csv_path = "/home/wusi/SAMdata/20250711/test/test_rgb.csv"
root_dir = "/home/wusi/SAMdata/20250711/test"
image_dir = "/home/wusi/SAMdata/20250711/test/rgb_images"
ii_dir = "/home/wusi/SAMdata/20250711/test_nii"
base_output_dir = "/home/wusi/SAMdata/20250711/testresults/max_expand_cm"
max_expand_cm_list = [0.5, 1, 1.5, 2.0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= 每例重复次数 =========
n_repeat = 20  # 可调小，正式跑再设大

for max_expand_cm in max_expand_cm_list:
    print(f"\n=== 正在处理外扩尺寸cm: {max_expand_cm} ===")
    # output_dir = os.path.join(base_output_dir, f"expand_{max_expand_cm}cm")
    output_dir = base_output_dir
    os.makedirs(output_dir, exist_ok=True)

    test_dataset = TestDataset(csv_path, root_dir, ii_dir, target_size=(1024, 1024), max_expand_cm=max_expand_cm)
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

    # ========= 缓存 image_embeddings =========
    embedding_cache = {}  # { (img_path, fold_idx): embedding }

    def get_embeddings(imgs, img_path):
        emb_list = []
        for i, net in enumerate(nets):
            key = (img_path, i)
            if key in embedding_cache:
                emb = embedding_cache[key]
            else:
                input_images = torch.stack([net.preprocess(im) for im in imgs], dim=0)
                emb = net.image_encoder(input_images)
                embedding_cache[key] = emb
            emb_list.append(emb)
        return emb_list

    # ========= 保存结果 =========
    patient_rows = defaultdict(list)

    for r in range(n_repeat):
        np.random.seed()
        torch.manual_seed(np.random.randint(0, 100000))

        patient_slices = defaultdict(dict)

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(test_loader, desc=f"外扩{max_expand_cm}cm 第{r+1}/{n_repeat}次", leave=False)):
                image, mask, original_size, image_path, resized_mask_t, spacing_x_cm, spacing_y_cm = batch
                imgs = image.to(device).float()
                img_path_str = image_path[0]
                rel_path = os.path.relpath(img_path_str, image_dir)
                patient_folder = Path(rel_path).parent.name
                slice_idx = int(Path(rel_path).stem)
                resized_mask_np = resized_mask_t.squeeze(0).numpy()
                sx_cm, sy_cm = float(spacing_x_cm), float(spacing_y_cm)

                res = test_dataset.get_box(resized_mask=resized_mask_np,
                                           spacing_x=sx_cm, spacing_y=sy_cm,
                                           max_expand_cm=max_expand_cm)
                if res is None:
                    continue
                bbox, (dL_mm, dR_mm, dT_mm, dB_mm) = res
                bbox = bbox.to(device).float()

                emb_list = get_embeddings(imgs, img_path_str)

                prob_list = []
                for net, image_embeddings in zip(nets, emb_list):
                    sparse_embeddings, dense_embeddings = net.prompt_encoder(points=None, boxes=bbox, masks=None)
                    low_res_masks, _ = net.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False
                    )
                    input_size = (int(imgs.shape[-2]), int(imgs.shape[-1]))
                    osz = (int(original_size[0]), int(original_size[1]))
                    masks = net.postprocess_masks(low_res_masks, input_size=input_size, original_size=osz)
                    prob_list.append(torch.sigmoid(masks))

                avg_prob = torch.mean(torch.stack(prob_list, dim=0), dim=0)
                final_mask = (avg_prob > 0.5).float().cpu().numpy()[0, 0]

                # 保存切片预测和外扩信息
                patient_slices[patient_folder][slice_idx] = (final_mask, (dL_mm, dR_mm, dT_mm, dB_mm))

        # ===== 每个病人算 Dice 和 HD95 =====
        for pa, slices in patient_slices.items():
            pa_path = os.path.join(ii_dir, pa)
            gt_img = nib.load(os.path.join(pa_path, "GTVp.nii.gz"))
            G = (gt_img.get_fdata() > 0).astype(np.uint8)

            pred_vol = np.zeros_like(G, dtype=np.uint8)
            expand_vals = {"dL": [], "dR": [], "dT": [], "dB": []}

            for slice_idx, (arr, expand_mm) in slices.items():
                if slice_idx < pred_vol.shape[2]:
                    arr = np.rot90(arr, k=3)
                    arr = np.fliplr(arr)
                    pred_vol[:, :, slice_idx] = arr.astype(np.uint8)
                    dL_mm, dR_mm, dT_mm, dB_mm = expand_mm
                    expand_vals["dL"].append(dL_mm)
                    expand_vals["dR"].append(dR_mm)
                    expand_vals["dT"].append(dT_mm)
                    expand_vals["dB"].append(dB_mm)

            spacing = gt_img.header.get_zooms()[:3]
            dice_val = dice3d_numpy(pred_vol, G)
            hd95_val = hd95_numpy(G, pred_vol, spacing=spacing)

            patient_rows[pa].append({
                "r": f"{r+1:03d}",
                "dice3d": dice_val,
                "hd95_mm": hd95_val,
                "dL_mean_mm": np.mean(expand_vals["dL"]),
                "dR_mean_mm": np.mean(expand_vals["dR"]),
                "dT_mean_mm": np.mean(expand_vals["dT"]),
                "dB_mean_mm": np.mean(expand_vals["dB"])
            })

    # ===== 写 Excel =====
    xlsx_path = os.path.join(output_dir, f"expand_{max_expand_cm}cm.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_all = []
        for pa, rows in patient_rows.items():
            if not rows: continue
            _df = pd.DataFrame(rows); _df["patient_id"] = pa
            df_all.append(_df)

        if df_all:
            df_all = pd.concat(df_all, ignore_index=True)
            overview = (df_all.groupby("r", as_index=False)
                        .agg(dice3d=("dice3d","mean"),
                             hd95_mm=("hd95_mm","mean"),
                             dL_mean_mm=("dL_mean_mm","mean"),
                             dR_mean_mm=("dR_mean_mm","mean"),
                             dT_mean_mm=("dT_mean_mm","mean"),
                             dB_mean_mm=("dB_mean_mm","mean"),
                             n_patients=("patient_id","nunique")))
            overview[["dice3d","hd95_mm","dL_mean_mm","dR_mean_mm","dT_mean_mm","dB_mean_mm"]] = \
                overview[["dice3d","hd95_mm","dL_mean_mm","dR_mean_mm","dT_mean_mm","dB_mean_mm"]].round(2)
            overview.to_excel(writer, sheet_name="OVERVIEW", index=False)

        for pa, rows in sorted(patient_rows.items(), key=lambda x: x[0]):
            df = pd.DataFrame(rows)
            mean_row = {"r":"MEAN",
                        "dice3d":df["dice3d"].mean(),
                        "hd95_mm":df["hd95_mm"].mean(),
                        "dL_mean_mm":df["dL_mean_mm"].mean(),
                        "dR_mean_mm":df["dR_mean_mm"].mean(),
                        "dT_mean_mm":df["dT_mean_mm"].mean(),
                        "dB_mean_mm":df["dB_mean_mm"].mean()}
            df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
            df[["dice3d","hd95_mm","dL_mean_mm","dR_mean_mm","dT_mean_mm","dB_mean_mm"]] = \
                df[["dice3d","hd95_mm","dL_mean_mm","dR_mean_mm","dT_mean_mm","dB_mean_mm"]].round(2)
            df.to_excel(writer, sheet_name=_sheet_name(pa), index=False)

    print(f"外扩 {max_expand_cm}cm 完成，结果保存在: {xlsx_path}")
