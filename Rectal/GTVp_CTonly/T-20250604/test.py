import logging
import os
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from testdatasetGTVp import TestDataset
from segment_anything import sam_model_registry
from dice_loss import dice_loss
from pathlib import Path

sam_checkpoint = "D:\\project\\segment-anything\\demo\\configs\\checkpoint\\sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
net = sam_model_registry[model_type](checkpoint=None)
net.to(device=device)
state_dict = torch.load(sam_checkpoint, map_location=device)
net.load_state_dict(state_dict, strict=False)
net.load_state_dict(torch.load("C:/Users/dell/Desktop/20250604/trainresults_kfold_pseudoRGB/fold_4/weights/best.pth", map_location='cuda'))
net.eval()

# 路径配置
csv_path = "C:/Users/dell/Desktop/20250604/dataset/test/test_pseudo_rgb_dataset.csv"
root_dir = "C:/Users/dell/Desktop/20250604/dataset/test"
image_dir = "C:/Users/dell/Desktop/20250604/dataset/test/pseudo_rgb_images"
nii_dir = "C:/Users/dell/Desktop/20250604/datanii/test_nii"
test_dir = "C:/Users/dell/Desktop/20250604/testresults/pseudorgb/pseudorgb_0_pixel"
save_dir = os.path.join(test_dir, "masks_pred")
dice_csv_path = os.path.join(test_dir, "dice.csv")

os.makedirs(test_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

log_file = os.path.join(test_dir, "dice_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

with open(dice_csv_path, 'w') as f:
    f.write("image,dice\n")

# 加载数据
test_dataset = TestDataset(csv_path=csv_path, root_dir=root_dir, nii_dir=nii_dir, target_size=(1024, 1024))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

dice_scores = []

with torch.no_grad():
    for idx, (image, mask, box, original_size, image_path) in enumerate(test_loader):
        imgs = image.to(device).float()
        true_masks = mask.to(device).float()
        bbox = box.to(device).float()

        # 图像预处理
        input_images = torch.stack([net.preprocess(im) for im in imgs], dim=0)
        image_embeddings = net.image_encoder(input_images)

        sparse_embeddings, dense_embeddings = net.prompt_encoder(
            points=None,
            boxes=bbox,
            masks=None
        )

        low_res_masks, _ = net.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=net.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        masks = net.postprocess_masks(
            low_res_masks,
            input_size=imgs.shape[-2:],
            original_size=original_size
        )

        prob_mask = torch.sigmoid(masks)
        resized_mask = (prob_mask > 0.5).float()
        # resized_mask = F.interpolate(prob_mask, size=true_masks.shape[-2:], mode='bilinear', align_corners=False)

        # 保存 mask 图像
        save_mask = (resized_mask[0].squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
        rel_path = os.path.relpath(image_path[0], image_dir)
        patient_folder = Path(rel_path).parent.name
        image_stem = Path(rel_path).stem
        save_subdir = os.path.join(save_dir, patient_folder)
        os.makedirs(save_subdir, exist_ok=True)
        save_path = os.path.join(save_subdir, image_stem + ".png")
        imageio.imwrite(save_path, save_mask)

        # Dice
        dice = dice_loss(resized_mask, true_masks)
        logging.info(f"{patient_folder}/{image_stem}, Dice: {dice:.4f}")
        dice_scores.append(dice.item())

        with open(dice_csv_path, 'a') as f:
            f.write(f"{patient_folder}/{image_stem},{dice:.4f}\n")

mean_dice = sum(dice_scores) / len(dice_scores)
logging.info(f"Mean Dice on the Test Set: {mean_dice:.4f}")
