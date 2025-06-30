import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage import measure
from pathlib import Path
import SimpleITK as sitk

def visualize_overlay_mask(image_path, gt_mask_path, pred_mask_path, save_path=None, title_info=None):
    image_np = Image.open(image_path)

    # 使用 SimpleITK 读取 NIfTI 格式 Ground Truth
    gt_sitk = sitk.ReadImage(gt_mask_path)
    gt_np = sitk.GetArrayFromImage(gt_sitk)
    if gt_np.ndim == 3:
        gt_np = gt_np[0]  # 默认取第0张切片
    if gt_np.ndim != 2:
        raise ValueError(f"Unexpected GT mask shape: {gt_np.shape}")
    gt_mask_np = (gt_np > 0).astype(np.uint8)

    pred_mask = Image.open(pred_mask_path).convert("L")
    pred_mask_np = (np.array(pred_mask) > 0).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_np, interpolation='bilinear')

    if gt_mask_np.ndim == 2:
        contours_gt = measure.find_contours(gt_mask_np, 0.5)
        for contour in contours_gt:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=0.4, color='lime', label='GT')

    contours_pred = measure.find_contours(pred_mask_np, 0.5)
    for contour in contours_pred:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=0.4, color='red', label='Pred')

    if title_info:
        ax.set_title(title_info, fontsize=12)
    else:
        ax.set_title("Contour Overlay (Green: GT, Red: Pred)", fontsize=12)

    ax.axis('off')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    csv_path = "C:/Users/dell/Desktop/20250604/dataset/test/test_pseudo_rgb_dataset.csv"
    root_dir = "C:/Users/dell/Desktop/20250604/dataset/test"
    image_dir = "C:/Users/dell/Desktop/20250604/dataset/test/pseudo_rgb_images"
    pred_dir = "C:/Users/dell/Desktop/20250604/testresults/pseudorgb/pseudorgb_0_pixel/masks_pred"
    save_dir = "C:/Users/dell/Desktop/20250604/testresults/pseudorgb/pseudorgb_0_pixel/vis_compare"
    dice_csv_path = "C:/Users/dell/Desktop/20250604/testresults/pseudorgb/pseudorgb_0_pixel/dice.csv"

    os.makedirs(save_dir, exist_ok=True)

    df_paths = pd.read_csv(csv_path, header=None, names=["image", "mask"])
    df_dice = pd.read_csv(dice_csv_path)

    # 处理 df_dice 第一列去除扩展名并转换为相对路径格式
    df_dice["image_key"] = df_dice.iloc[:, 0].apply(lambda x: os.path.splitext(x.replace("\\", "/"))[0])
    dice_dict = dict(zip(df_dice["image_key"], df_dice.iloc[:, 1]))

    for idx, row in df_paths.iterrows():
        image_path = os.path.join(root_dir, row["image"].lstrip("/\\"))
        gt_mask_path = os.path.join(root_dir, row["mask"].lstrip("/\\"))

        rel_path = os.path.relpath(image_path, image_dir)
        image_name = os.path.splitext(rel_path.replace("\\", "/"))[0]  # 使用统一路径分隔符

        pred_mask_path = os.path.join(pred_dir, image_name + ".png")

        # 保存路径：按患者分类
        patient_folder = Path(image_name).parent.name
        image_stem = Path(image_name).stem
        save_subdir = os.path.join(save_dir, patient_folder)
        os.makedirs(save_subdir, exist_ok=True)
        save_path = os.path.join(save_subdir, image_stem + "_compare.png")

        dice_score = dice_dict.get(image_name, 'N/A')
        title_str = f"{image_name}  Dice {dice_score}"

        print(title_str)
        visualize_overlay_mask(image_path, gt_mask_path, pred_mask_path, save_path, title_info=title_str)
