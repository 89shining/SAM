import logging
import os.path
import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from testdatasetGTVp import TestDataset
from segment_anything import sam_model_registry
from dice_loss import dice_loss


sam_checkpoint = "D:\\project\\segment-anything\\demo\\configs\\checkpoint\\sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
net = sam_model_registry[model_type](checkpoint=None)
net.to(device=device)
# 加载原始权重
state_dict = torch.load(sam_checkpoint, map_location=device)
net.load_state_dict(state_dict, strict=False)

# 加载预训练权重
net.load_state_dict(torch.load("C:/Users/dell/Desktop/SAM/GTVp_CTonly/20250515/trainresults_kfold_0522/fold_1/weights/best.pth", map_location='cuda'))
net.eval()
net.to(device)

csv_path = "C:/Users/dell/Desktop/SAM/GTVp_CTonly/20250515/Dataset/test/test_tiff.csv"   # test.csv
root_dir = "C:/Users/dell/Desktop/SAM/GTVp_CTonly/20250515/Dataset/test"   # testdata文件夹
image_dir = "C:/Users/dell/Desktop/SAM/GTVp_CTonly/20250515/Dataset/test/images"
nii_dir = "C:/Users/dell/Desktop/SAM/GTVp_CTonly/20250515/datanii/testdatanii"   # testnii数据文件夹
test_dir = "C:/Users/dell/Desktop/SAM/GTVp_CTonly/20250515/testresults/20250522"     # 测试结果保存主目录
save_dir = "C:/Users/dell/Desktop/SAM/GTVp_CTonly/20250515/testresults/20250522/masks_pred"     # 预测mask保存目录
dice_csv_path ="C:/Users/dell/Desktop/SAM/GTVp_CTonly/20250515/testresults/20250522/dice.csv"
os.makedirs(test_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.dirname(dice_csv_path), exist_ok=True)

# 初始化日志设置（写入到文件，也显示在终端）
log_file = os.path.join(test_dir, "dice_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# 初始化 Dice CSV 文件
with open(dice_csv_path, 'w') as f:
    f.write("image,dice\n")

# 加载测试数据
test_dataset = TestDataset(
    csv_path=csv_path,
    root_dir=root_dir,
    nii_dir=nii_dir,
    target_size=(1024, 1024)
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 评估dice值
dice_scores = []

with torch.no_grad():
    for idx, (image, mask, box, original_size, image_path) in enumerate(test_loader):
        imgs = image.to(device).float()
        true_masks = mask.to(device).float()
        bbox = box.to(device).float()
        # 手动传入
        input_images = torch.stack([net.preprocess(im) for im in imgs], dim=0)
        image_embeddings = net.image_encoder(input_images)

        logits_list = []
        for i in range(len(imgs)):
            sparse_embeddings, dense_embeddings = net.prompt_encoder(
                points=None,
                boxes=bbox[i].unsqueeze(0),
                masks=None
            )
            low_res_masks, _ = net.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=net.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            masks = net.postprocess_masks(
                low_res_masks,
                input_size=imgs[i].shape[-2:],
                original_size=original_size,
            )
            predmasks = masks > net.mask_threshold
            # 保存预测结果png
            save_mask = predmasks[i].squeeze().cpu().numpy().astype(np.uint8) * 255
            # print(image_path[i])
            rel_path = os.path.relpath(image_path[i], image_dir)
            image_name = os.path.splitext(rel_path)[0]  # 'p_0/32'
            # print(image_name)
            save_path = os.path.join(save_dir, image_name + ".png")
            # print(save_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.imwrite(save_path, save_mask)

            logits_list.append(low_res_masks)

        masks_pred = torch.stack([x.squeeze(0) for x in logits_list], dim=0)
        # print(masks_pred.shape)
        masks_pred = torch.sigmoid(masks_pred)
        masks_pred = masks_pred.to(device)


        if true_masks.dim() == 3:
            true_masks = true_masks.unsqueeze(1)

        true_masks = F.interpolate(true_masks, size=masks_pred.shape[-2:], mode='bilinear', align_corners=False)  # 与 pred 对齐 [1, 1, 256, 256]
        # print(true_masks.shape)

        dice = dice_loss(masks_pred, true_masks)
        # print(f"Dice：{dice:.4f}")
        logging.info(f"{image_name}, Dice: {dice:.4f}")
        dice_scores.append(dice.item())

        # 追加写入 CSV
        with open(dice_csv_path, 'a') as f:
            f.write(f"{image_name},{dice:.4f}\n")

mean_dice = sum(dice_scores) / len(dice_scores)
# print(f"Mean Dice on the Test Set：{mean_dice:.4f}")
logging.info(f"Mean Dice on the Test Set: {mean_dice:.4f}")



