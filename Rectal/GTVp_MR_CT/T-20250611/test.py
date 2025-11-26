import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import csv

from datasetGTVp import SAMDataset
from segment_anything import sam_model_registry

def sigmoid_mask_to_image(mask, threshold=0.5):
    mask = torch.sigmoid(mask)
    mask = (mask > threshold).float()
    mask_np = mask.squeeze().cpu().numpy().astype(np.uint8) * 255
    return Image.fromarray(mask_np)

def dice_coefficient(pred, target, epsilon=1e-6):
    pred = torch.sigmoid(pred) > 0.5
    pred = pred.float()
    target = target.float()
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.item()

if __name__ == '__main__':
    # === 路径配置 ===
    test_csv = "C:/Users/WS/Desktop/GTVp_MRI/dataset/test/test_rgb_dataset.csv"
    test_root = "C:/Users/WS/Desktop/GTVp_MRI/dataset/test"
    test_nii_dir = "C:/Users/WS/Desktop/GTVp_MRI/datanii/testdatanii"
    save_mask_root = "C:/Users/WS/Desktop/GTVp_MRI/testresults/masks_pred"
    dice_csv_path = "C:/Users/WS/Desktop/GTVp_MRI/testresults/dice.csv"
    os.makedirs(save_mask_root, exist_ok=True)

    sam_checkpoint = "D:\\learning\\segment-anything\\demo\\configs\\checkpoint\\sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    net = sam_model_registry[model_type](checkpoint=None)
    net.to(device=device)
    state_dict = torch.load(sam_checkpoint, map_location=device)
    net.load_state_dict(state_dict, strict=False)
    net.load_state_dict(torch.load("C:/Users/WS/Desktop/GTVp_MRI/trainresults_kfold_MRI/fold_4/weights/best.pth",
                                   map_location='cuda'))
    net.eval()

    # === 数据加载 ===
    test_dataset = SAMDataset(test_csv, test_root, test_nii_dir, target_size=(1024, 1024))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_list = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            image = batch["image"].to(device)  # [1, 3, H, W]
            input_image = torch.stack([net.preprocess(im) for im in image], dim=0)
            image_embedding = net.image_encoder(input_image)

            # 使用提示（box 或 mask）
            use_mask_prompt = batch['use_mask_prompt'][0]
            if use_mask_prompt:
                mask_prompt = batch['mask_prompt'].to(device)
                sparse_emb, dense_emb = net.prompt_encoder(points=None, boxes=None, masks=mask_prompt)
            else:
                val_box = batch['val_box'].to(device)
                sparse_emb, dense_emb = net.prompt_encoder(points=None, boxes=val_box, masks=None)

            low_res_masks, _ = net.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=net.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False
            )

            # 还原到 GT 尺寸
            mask_pred = F.interpolate(low_res_masks, size=batch['GT'].shape[-2:], mode='bilinear', align_corners=False)

            # === 保存预测图像 ===
            pred_img = sigmoid_mask_to_image(mask_pred)

            rel_image_path = batch['image path'][0]
            image_rel = os.path.relpath(rel_image_path, os.path.join(test_root, "rgb_images")).replace("\\", "/")
            image_key = os.path.splitext(image_rel)[0]

            patient_folder = os.path.dirname(image_key)
            image_name = os.path.basename(image_key)
            save_dir = os.path.join(save_mask_root, patient_folder)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, image_name + ".png")
            pred_img.save(save_path)

            # === Dice 计算并记录 ===
            true_mask = batch["GT"].to(device)
            dice = dice_coefficient(mask_pred, true_mask)
            dice_list.append((image_key, round(dice, 4)))

    # === 保存 Dice CSV ===
    dice_scores_only = [score for _, score in dice_list]
    average_dice = round(np.mean(dice_scores_only), 4)

    # 添加平均行写入文件
    with open(dice_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "dice_score"])
        writer.writerows(dice_list)
        writer.writerow(["Average", average_dice])

    print(f"Saved {len(dice_list)} Dice scores to: {dice_csv_path}")
    print(f"Average Dice: {average_dice}")
