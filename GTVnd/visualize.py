import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage import measure
from PIL import Image
import numpy as np

def load_and_window_tiff(image_path, window_center=40, window_width=350):
    img = Image.open(image_path)

    if img.mode in ['I;16', 'I']:  # 16-bit grayscale
        img = np.array(img).astype(np.float32)
        # 简单 windowing: normalize to 0~255
        lower = window_center - window_width // 2
        upper = window_center + window_width // 2
        img = np.clip((img - lower) / (upper - lower), 0, 1) * 255
        img = img.astype(np.uint8)
        img_rgb = np.stack([img]*3, axis=-1)  # Convert to RGB
        return img_rgb
    else:
        # Already 8-bit, standard RGB load
        return np.array(img.convert("RGB"))


def visualize_three_panel(image_path, gt_mask_path, pred_mask_path, save_path=None):
    img_np = load_and_window_tiff(image_path)
    gt_mask = Image.open(gt_mask_path).convert("L")
    pred_mask = Image.open(pred_mask_path).convert("L")

    # Convert to numpy
    gt_np = np.array(gt_mask)
    pred_np = np.array(pred_mask)

    # Create figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img_np)
    axs[0].set_title("Original Image")
    axs[1].imshow(gt_np, cmap='gray')
    axs[1].set_title("Ground Truth Mask")
    axs[2].imshow(pred_np, cmap='gray')
    axs[2].set_title("Predicted Mask")

    for ax in axs:
        ax.axis('off')

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_overlay_mask(image_path, gt_mask_path, pred_mask_path, save_path=None,title_info=None):
    # 加载图像和掩膜
    image_np = load_and_window_tiff(image_path)
    gt_mask = Image.open(gt_mask_path).convert("L")
    pred_mask = Image.open(pred_mask_path).convert("L")

    # 转为 NumPy
    gt_mask_np = (np.array(gt_mask) > 0).astype(np.uint8)
    pred_mask_np = (np.array(pred_mask) > 0).astype(np.uint8)

    # 创建绘图
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_np, interpolation='bilinear')

    # 绘制 Ground Truth 轮廓：绿色
    contours_gt = measure.find_contours(gt_mask_np, 0.5)
    for contour in contours_gt:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=0.4, color='lime', label='GT')

    # 绘制 Predicted 轮廓：红色
    contours_pred = measure.find_contours(pred_mask_np, 0.5)
    for contour in contours_pred:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=0.4, color='red', label='Pred')

    if title_info:
        ax.set_title(title_info, fontsize=12)
    else:
        ax.set_title("Contour Overlay (Green: GT, Red: Pred)", fontsize=12)

    ax.axis('off')

    # 避免重复图例项
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
    csv_path = "C:/Users/dell/Desktop/SAM/GTVnd/Dataset/test/test_tiff.csv"
    root_dir = "C:/Users/dell/Desktop/SAM/GTVnd/Dataset/test"
    image_dir = "C:/Users/dell/Desktop/SAM/GTVnd/Dataset/test/images"
    pred_dir = "C:/Users/dell/Desktop/SAM/GTVnd/20250522/testresults/pos1_neg1/masks_pred"  # 预测结果png文件夹
    save_dir = "C:/Users/dell/Desktop/SAM/GTVnd/20250522/testresults/pos1_neg1/vis_compare"    # 可视化对比保存文件夹
    dice_csv_path = "C:/Users/dell/Desktop/SAM/GTVnd/20250522/testresults/pos1_neg1/dice.csv"
    os.makedirs(save_dir, exist_ok=True)

    # 一次性读取两个 CSV
    df_paths = pd.read_csv(csv_path, header=None, names=["image", "mask"])
    df_dice = pd.read_csv(dice_csv_path)

    dice_dict = dict(zip(df_dice.iloc[:, 0], df_dice.iloc[:, 1]))

    for idx, row in df_paths.iterrows():
        image_path = os.path.join(root_dir, row["image"].lstrip("/\\"))
        gt_mask_path = os.path.join(root_dir, row["mask"].lstrip("/\\"))

        rel_path = os.path.relpath(image_path, image_dir)
        image_name = os.path.splitext(rel_path)[0]
        pred_mask_path = os.path.join(pred_dir, image_name + ".png")
        save_path = os.path.join(save_dir, image_name.replace("/", "_") + "_compare.png")

        # 获取对应的dice值
        dice_score = dice_dict.get(image_name, 'N/A')  # 若找不到则显示N/A
        # 通过 image_name 匹配 Dice 值
        title_str = f"{image_name}  Dice {dice_score}"

        print(title_str)

        visualize_overlay_mask(image_path, gt_mask_path, pred_mask_path, save_path, title_info=title_str)