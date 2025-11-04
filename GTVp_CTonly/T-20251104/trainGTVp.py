# 五折交叉验证 GTVp训练
import os
import sys
sys.path.append("/home/wusi/segment-anything")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import random
import logging
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from matplotlib.ticker import MaxNLocator
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dice_loss import BCEDiceLoss
from datasetGTVp import SAMDataset
from segment_anything import sam_model_registry

# 设置随机种子
manual_seed = int.from_bytes(os.urandom(4), 'little')
random.seed(manual_seed)
torch.manual_seed(manual_seed)

def custom_collate_fn(batch):
    """允许 batch 中包含 None（例如 box=None 的情况）"""
    result = {}
    keys = batch[0].keys()
    for key in keys:
        vals = [item[key] for item in batch]
        # 对 has_gt 手动 stack（list of float -> tensor）
        if key == 'has_gt':
            result[key] = torch.tensor(vals, dtype=torch.float32)
        elif any(v is None for v in vals):  # 例如 box
            result[key] = vals  # 保留 list
        elif isinstance(vals[0], torch.Tensor):
            result[key] = torch.stack(vals, dim=0)
        else:
            result[key] = vals
    return result


def dice_coefficient(pred, target, threshold=0.5, eps=1e-5):
    """
    pred: [B, 1, H, W]
    target: [B, 1, H, W]
    """
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    target = (target > 0.5).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean().item()

def train_one_fold(fold, train_idx, val_idx, all_image_paths, dataset, net, device,
                   epochs, batch_size, lr, save_dir):
    fold_dir = os.path.join(save_dir, f"fold_{fold + 1}")
    os.makedirs(fold_dir, exist_ok=True)
    os.makedirs(os.path.join(fold_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(fold_dir, 'runs'), exist_ok=True)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    writer = SummaryWriter(os.path.join(fold_dir, 'runs'))

    # 日志信息
    logging.info(f'''Starting training:
            Fold:            {fold + 1}
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {len(train_idx)}
            Validation size: {len(val_idx)}
            Device:          {device.type}
        ''')

    # 损失函数
    criterion = BCEDiceLoss()

    # 学习率
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=7,
        min_lr=1e-6
    )

    scaler = torch.amp.GradScaler('cuda', enabled=True)  # 自动混合精度

    best_val_loss = float('inf')
    no_improve_epochs = 0
    early_stop = 15

    train_losses, val_losses = [], []
    best_val_dice = 0.0

    for epoch in range(epochs):
        net.train()
        running_loss, valid_count = 0.0, 0
        for batch in tqdm(train_loader, desc=f"[Train {fold+1}] Epoch {epoch+1}/{epochs}", leave=False):
            imgs = batch['image'].to(device)
            true_masks = batch['GT'].to(device)
            bbox = batch['train_box']
            has_gt = batch['has_gt'].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=True):
                input_images = torch.stack([net.preprocess(im) for im in imgs], dim=0)
                image_embeddings = net.image_encoder(input_images)

                logits_list = []
                for i in range(len(imgs)):
                    sparse_embeddings, dense_embeddings = net.prompt_encoder(
                        points=None,
                        boxes=(bbox[i].unsqueeze(0).to(device) if bbox[i] is not None else None),
                        masks=None)
                    low_res_masks, _ = net.mask_decoder(
                        image_embeddings=image_embeddings[i].unsqueeze(0),
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False)
                    logits_list.append(low_res_masks)

                masks_pred = torch.stack([x.squeeze(0) for x in logits_list], dim=0)
                if true_masks.dim() == 3:
                    true_masks = true_masks.unsqueeze(1)
                true_masks = F.interpolate(true_masks, size=masks_pred.shape[-2:], mode='bilinear', align_corners=False)

                loss_all = criterion(masks_pred, true_masks)

                # 加入空层加权机制
                loss_weight = torch.where(has_gt > 0, 1.0, 0.3).to(device)
                loss = (loss_all * loss_weight.mean())

            scaler.scale(loss).backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            valid_count += 1

        train_loss = running_loss / valid_count
        train_losses.append(train_loss)
        writer.add_scalar('Loss/Train', train_loss, epoch + 1)

        # Validation
        net.eval()
        val_running_loss, val_count = 0.0, 0
        val_dice_list = []  # 每个 batch 的 Dice 暂存
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Val {fold+1}] Epoch {epoch+1}", leave=False):
                imgs = batch['image'].to(device)
                true_masks = batch['GT'].to(device)
                bbox = batch['val_box']
                has_gt = batch['has_gt'].to(device)

                with torch.amp.autocast('cuda', enabled=True):
                    input_images = torch.stack([net.preprocess(im) for im in imgs], dim=0)
                    image_embeddings = net.image_encoder(input_images)
                    logits_list = []
                    for i in range(len(imgs)):
                        sparse_embeddings, dense_embeddings = net.prompt_encoder(
                            points=None,
                            boxes=(bbox[i].unsqueeze(0).to(device) if bbox[i] is not None else None),
                            masks=None)
                        low_res_masks, _ = net.mask_decoder(
                            image_embeddings=image_embeddings[i].unsqueeze(0),
                            image_pe=net.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False)
                        logits_list.append(low_res_masks)
                    masks_pred = torch.stack([x.squeeze(0) for x in logits_list], dim=0)
                    if true_masks.dim() == 3:
                        true_masks = true_masks.unsqueeze(1)
                    true_masks = F.interpolate(true_masks, size=masks_pred.shape[-2:], mode='bilinear',
                                               align_corners=False)

                    val_loss_all = criterion(masks_pred, true_masks)
                    loss_weight = torch.where(has_gt > 0, 1.0, 0.3).to(device)
                    val_loss = val_loss_all * loss_weight.mean()

                val_running_loss += val_loss.item()
                val_count += 1

                # ---- 计算 Dice（保持与 val_loss 同级缩进）----
                val_dice = dice_coefficient(masks_pred, true_masks)
                val_dice_list.append(val_dice)

        # epoch 汇总
        val_mean_loss = val_running_loss / val_count
        val_mean_dice = np.mean(val_dice_list)
        val_losses.append(val_mean_loss)

        writer.add_scalar('Loss/Val', val_mean_loss, epoch + 1)
        writer.add_scalar('Metrics/ValDice', val_mean_dice, epoch + 1)

        scheduler.step(val_mean_loss)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR', current_lr, epoch + 1)

        logging.info(
            f"Epoch {epoch + 1}/{epochs} | Train={train_loss:.4f} | Val={val_mean_loss:.4f} | Dice={val_mean_dice:.4f} | LR={current_lr:.6e}")


        # ---------- 保存最优 (基于Dice) ----------
        if val_mean_dice > best_val_dice:
            best_val_dice = val_mean_dice
            torch.save(net.state_dict(), os.path.join(fold_dir, 'weights', 'best.pth'))
            logging.info(f" New best model (Dice={best_val_dice:.4f}, Loss={val_mean_loss:.4f}) saved.")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stop:
                logging.info(f" Early stopping at epoch {epoch + 1} (no Dice improvement)")
                break

    # 保存loss曲线
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val')
    plt.legend();
    plt.xlabel('Epoch');
    plt.ylabel('Loss')
    plt.savefig(os.path.join(fold_dir, 'loss_curve.jpg'))
    plt.close()
    writer.close()

if __name__ == '__main__':
    root_dir = '/home/wusi/SAMdata/20251104_GTVp/dataset/train'  # traindataset的目录
    csv_path = '/home/wusi/SAMdata/20251104_GTVp/dataset/train/train_rgb.csv'
    nii_dir = "/home/wusi/SAMdata/20250711_GTVp/datanii/train_nii"  # trainnii数据文件夹
    save_dir = '/home/wusi/SAMdata/20251104_GTVp/TrainResult/TrainAll'  # 训练结果保存文件夹
    os.makedirs(save_dir, exist_ok=True)


    dataset = SAMDataset(csv_path=csv_path, root_dir=root_dir, nii_dir = nii_dir, target_size=(1024, 1024))
    all_image_paths = pd.read_csv(csv_path, header=None, names=["image", "mask"])["image"].tolist()

    sam_checkpoint = "/home/wusi/segment-anything/demo/configs/checkpoint/sam_vit_b_01ec64.pth"
    model_type = "vit_b"

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        # 只训练第4折和5折
        # if fold not in [2]:
            # continue

        # Logging setup
        log_path = os.path.join(save_dir, f'fold_{fold + 1}/train_fold{fold + 1}.log')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # 需要先移除已存在的 handler（否则重复 logging 会出错）
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path, mode='w'),
                logging.StreamHandler(sys.stdout)]
        )
        logging.info(f"[Fold {fold + 1}] Logging initialized.")
        logging.info(f'Using device {device}')

        # 每次重新初始化网络
        net = sam_model_registry[model_type](checkpoint=None)
        state_dict = torch.load(sam_checkpoint, map_location=device)
        net.load_state_dict(state_dict, strict=False)
        logging.info(f"[Info] Loaded SAM checkpoint from {sam_checkpoint} with strict=False.")
        net.to(device)

        # # 冻结图像编码器
        # for param in net.image_encoder.parameters():
        #     param.requires_grad = False
        
        # 冻结解码器
        # for param in net.mask_decoder.parameters():
            # param.requires_grad = False

        trainable_params = [name for name, param in net.named_parameters() if param.requires_grad]
        logging.info(f"Trainable parameters ({len(trainable_params)}):")
        # print("Trainable parameters:")
        for name in trainable_params:
            logging.info(f"  {name}")
            # print(name)

        train_one_fold(fold, train_idx, val_idx, all_image_paths, dataset, net, device,
                       epochs=100, batch_size=4, lr=0.001, save_dir=save_dir)
        logging.info(f"Training Fold{fold + 1} completed.")

        torch.cuda.empty_cache()

    print("Five-fold cross-validation completed.")
