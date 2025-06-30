# 五折交叉验证 GTVp训练
import os
import sys
sys.path.append("/home/intern/wusi/segment-anything")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
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
from pseudo_datasetGTVp import SAMDataset
from segment_anything import sam_model_registry

# 设置随机种子
manual_seed = int.from_bytes(os.urandom(4), 'little')
random.seed(manual_seed)
torch.manual_seed(manual_seed)


def train_one_fold(fold, train_idx, val_idx, all_image_paths, dataset, net, device,
                   epochs, batch_size, lr, save_dir):
    fold_dir = os.path.join(save_dir, f"fold_{fold + 1}")
    os.makedirs(fold_dir, exist_ok=True)
    os.makedirs(os.path.join(fold_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(fold_dir, 'runs'), exist_ok=True)

    # 日志记录
    logging.info(f'Auto-generated seed: {manual_seed}')

    train_ids = [all_image_paths[i].replace('/images/', '').replace('.tiff', '').replace('.tif', '') for i in train_idx]
    val_ids = [all_image_paths[i].replace('/images/', '').replace('.tiff', '').replace('.tif', '') for i in val_idx]

    with open(os.path.join(fold_dir, 'train_ids.txt'), 'w') as f:
        f.writelines(f"{id}\n" for id in train_ids)
    with open(os.path.join(fold_dir, 'val_ids.txt'), 'w') as f:
        f.writelines(f"{id}\n" for id in val_ids)

    # 同时保存到日志文件
    logging.info(f"Train IDs ({len(train_ids)} samples): {train_ids}")
    logging.info(f"Val IDs ({len(val_ids)} samples): {val_ids}")

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

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
    scalelr = lr
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=scalelr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=7,
        min_lr=1e-6
    )

    bestloss = float('inf')
    no_improve_epochs = 0
    early_stop_patience = 15
    trainLoss, valLoss = [], []

    for epoch in range(epochs):
        net.train()
        train_epoch_loss = 0
        LOSS = []
        train_n_loss = 0
        with tqdm(total=len(train_loader), desc=f'[Train Fold {fold + 1}]', unit='batch', disable=True) as pbar:
            # 传入一个batch
            for batch_idx, batch in enumerate(train_loader):
                imgs = batch['image'].to(device)
                true_masks = batch['GT'].to(device)
                bbox = batch['train_box'].to(device)

                input_images = torch.stack([net.preprocess(im) for im in imgs], dim=0)
                image_embeddings = net.image_encoder(input_images)

                logits_list = []
                for i in range(len(imgs)):
                    sparse_embeddings, dense_embeddings = net.prompt_encoder(
                        points=None,
                        boxes=bbox[i].unsqueeze(0),
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
                train_loss = criterion(masks_pred, true_masks)
                # 返回当前batch的loss
                train_loss_batch = float(train_loss.item())
                # 当前epoch总loss
                train_epoch_loss += train_loss_batch
                train_n_loss += 1
                # 当前batch的loss反向传播
                optimizer.zero_grad()
                train_loss.backward()
                # 梯度裁剪：将梯度值限制在某个指定的范围内，防止梯度值过大导致训练不稳定
                # clip_value：梯度剪裁阈值，使梯度最大绝对值不超过0.1
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                # 优化器参数更新
                optimizer.step()
                # torch.cuda.empty_cache()

                # 更新进度条右侧的附加信息:当前epoch的平均loss dice bce
                pbar.set_postfix({'TrainLoss': f"{train_epoch_loss / train_n_loss:.4f}"})
                # 更新进度条（进度条前进1步）
                pbar.update(1)

        train_meanLoss = train_epoch_loss / train_n_loss  # 当前epoch每个batch的平均损失
        LOSS.append(train_meanLoss)  # LOSS列表保存平均损失
        trainLoss.append(LOSS[-1])
        writer.add_scalar('Loss/train_epoch_avg', train_meanLoss, epoch + 1)
        # torch.cuda.empty_cache()

        # Validation
        net.eval()
        val_epoch_loss = 0
        LOSS = []
        val_n_loss = 0
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f'[Train Fold {fold + 1}]', unit='batch', disable=True) as pbar:
                # 传入一个batch
                for batch_idx, batch in enumerate(val_loader):
                    imgs = batch['image'].to(device)
                    true_masks = batch['GT'].to(device)
                    bbox = batch['val_box'].to(device)

                    input_images = torch.stack([net.preprocess(im) for im in imgs], dim=0)
                    image_embeddings = net.image_encoder(input_images)
                    logits_list = []
                    for i in range(len(imgs)):
                        sparse_embeddings, dense_embeddings = net.prompt_encoder(
                            points=None,
                            boxes=bbox[i].unsqueeze(0),
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
                    val_loss = criterion(masks_pred, true_masks)
                    # 返回当前batch的loss
                    val_loss_batch = float(val_loss.item())
                    # 当前epoch总loss
                    val_epoch_loss += val_loss_batch
                    # print("epoch_loss")
                    # print(epoch_loss)
                    val_n_loss += 1

                    # 更新进度条右侧的附加信息:当前epoch的平均loss
                    pbar.set_postfix({'ValLoss': f"{val_epoch_loss / val_n_loss:.4f}"})
                    # 更新进度条（进度条前进1步）
                    pbar.update(1)
                    # torch.cuda.empty_cache()

        val_meanLoss = val_epoch_loss / val_n_loss  # 当前epoch每个batch的平均损失
        LOSS.append(val_meanLoss)  # LOSS列表保存平均损失
        valLoss.append(LOSS[-1])
        writer.add_scalar('Loss/Val_epoch_avg', val_meanLoss, epoch + 1)
        # torch.cuda.empty_cache()

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR', current_lr, epoch + 1)
        logging.info(
            f'Epoch {epoch + 1}: Train Loss={trainLoss[-1]:.4f}, Val Loss={valLoss[-1]:.4f}, lr={current_lr:.8f}')

        scheduler.step(val_meanLoss)

        if bestloss > val_meanLoss:
            bestloss = val_meanLoss
            no_improve_epochs = 0
            torch.save(net.state_dict(),
                       os.path.join(fold_dir, 'weights') + f'/best.pth')  # 将最优模型权重保存为best.pth
            logging.info(f'Best model updated with loss={bestloss:.4f}')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stop_patience:
                logging.info(f"Early stopping triggered at epoch {epoch + 1}, Best Val Loss: {bestloss:.4f}")
                break

        # 记录每个fold的最终结果
        with open(os.path.join(save_dir, 'summary.txt'), 'a') as f:
            f.write(f"Fold {fold + 1}: Best Val Loss = {bestloss:.4f}\n")

    # 绘制损失图
    plt.figure()
    plt.plot(range(1, len(trainLoss) + 1), trainLoss, label='Train Loss')
    plt.plot(range(1, len(valLoss) + 1), valLoss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold + 1} Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(fold_dir, 'loss_curve.jpg'))
    plt.close()
    writer.close()

if __name__ == '__main__':
    root_dir = '/home/intern/wusi/Dataset/GTVp0604/dataset/train'  # traindataset的目录
    csv_path = '/home/intern/wusi/Dataset/GTVp0604/dataset/train/pseudo_rgb_dataset.csv'
    nii_dir = "/home/intern/wusi/Dataset/GTVp0604/traindatanii"  # trainnii数据文件夹
    save_dir = '/home/intern/wusi/Dataset/GTVp0604/trainresults_kfold_pseudoRGB'  # 训练结果保存文件夹
    os.makedirs(save_dir, exist_ok=True)


    dataset = SAMDataset(csv_path=csv_path, root_dir=root_dir, nii_dir = nii_dir, target_size=(1024, 1024))
    all_image_paths = pd.read_csv(csv_path, header=None, names=["image", "mask"])["image"].tolist()

    sam_checkpoint = "/home/intern/wusi/segment-anything/demo/configs/checkpoint/sam_vit_b_01ec64.pth"
    model_type = "vit_b"

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        # 只训练第1折
        if fold not in [0]:
            continue

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
        #
        # # 冻结解码器
        # for param in net.mask_decoder.parameters():
        #     param.requires_grad = False

        trainable_params = [name for name, param in net.named_parameters() if param.requires_grad]
        logging.info(f"Trainable parameters ({len(trainable_params)}):")
        # print("Trainable parameters:")
        for name in trainable_params:
            logging.info(f"  {name}")
            # print(name)

        train_one_fold(fold, train_idx, val_idx, all_image_paths, dataset, net, device,
                       epochs=150, batch_size=2, lr=0.001, save_dir=save_dir)
        logging.info(f"Training Fold{fold + 1} completed.")

        torch.cuda.empty_cache()

    print("Five-fold cross-validation completed.")
