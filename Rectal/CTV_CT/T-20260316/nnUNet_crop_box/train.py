# Five-fold cross-validation training for CTV
import os
import sys

sys.path.append("/home/wusi/segment-anything")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import random
import logging
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dice_loss import BCEDiceLoss
from dataset import SAMDatasetFromNiiGz
from segment_anything import sam_model_registry

# Random seed
manual_seed = int.from_bytes(os.urandom(4), 'little')
random.seed(manual_seed)
torch.manual_seed(manual_seed)


def train_one_fold(fold, train_idx, val_idx, dataset, net, device,
                   epochs, batch_size, lr, save_dir):
    fold_dir = os.path.join(save_dir, f"fold_{fold + 1}")
    weights_dir = os.path.join(fold_dir, 'weights')
    runs_dir = os.path.join(fold_dir, 'runs')
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    last_ckpt_path = os.path.join(weights_dir, 'last.pth')
    best_ckpt_path = os.path.join(weights_dir, 'best.pth')
    done_flag_path = os.path.join(fold_dir, 'training_done.flag')

    if os.path.exists(done_flag_path):
        logging.info(f"[Fold {fold + 1}] Found done flag, skip this fold.")
        return

    logging.info(f'Auto-generated seed: {manual_seed}')

    train_ids = [f"{dataset.index[i][0]}_z{dataset.index[i][1]}" for i in train_idx]
    val_ids = [f"{dataset.index[i][0]}_z{dataset.index[i][1]}" for i in val_idx]

    with open(os.path.join(fold_dir, 'train_ids.txt'), 'w') as f:
        f.writelines(f"{id_}\n" for id_ in train_ids)
    with open(os.path.join(fold_dir, 'val_ids.txt'), 'w') as f:
        f.writelines(f"{id_}\n" for id_ in val_ids)

    logging.info(f"Train IDs ({len(train_ids)} samples): {train_ids}")
    logging.info(f"Val IDs ({len(val_ids)} samples): {val_ids}")

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

    writer = SummaryWriter(runs_dir)

    logging.info(f'''Starting training:
            Fold:            {fold + 1}
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {len(train_idx)}
            Validation size: {len(val_idx)}
            Device:          {device.type}
        ''')

    criterion = BCEDiceLoss()
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

    bestloss = float('inf')
    no_improve_epochs = 0
    early_stop_patience = 15
    trainLoss, valLoss = [], []
    start_epoch = 0

    if os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            net.load_state_dict(ckpt['model_state_dict'], strict=False)
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])

            bestloss = ckpt.get('best_loss', float('inf'))
            no_improve_epochs = ckpt.get('no_improve_epochs', 0)
            trainLoss = ckpt.get('train_loss_history', [])
            valLoss = ckpt.get('val_loss_history', [])
            start_epoch = int(ckpt.get('epoch', -1)) + 1
            logging.info(
                f"[Fold {fold + 1}] Resume from epoch {start_epoch}/{epochs}, "
                f"best val loss={bestloss:.4f}"
            )
        else:
            logging.warning(f"[Fold {fold + 1}] last.pth format incompatible, start from epoch 0.")

    if start_epoch >= epochs:
        logging.info(f"[Fold {fold + 1}] Already reached target epochs ({epochs}), mark done.")
        with open(done_flag_path, 'w') as f:
            f.write(f"fold={fold + 1}, status=done, best_val_loss={bestloss:.6f}\n")
        writer.close()
        return

    last_finished_epoch = start_epoch - 1

    for epoch in range(start_epoch, epochs):
        net.train()
        train_epoch_loss = 0.0
        train_n_loss = 0

        with tqdm(total=len(train_loader), desc=f'[Train Fold {fold + 1}]', unit='batch', disable=True) as pbar:
            for batch in train_loader:
                imgs = batch['image'].to(device)
                true_masks = batch['GT'].to(device)
                bbox = batch['box'].to(device)

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
                        multimask_output=False
                    )
                    logits_list.append(low_res_masks)

                masks_pred = torch.stack([x.squeeze(0) for x in logits_list], dim=0)
                if true_masks.dim() == 3:
                    true_masks = true_masks.unsqueeze(1)
                true_masks = F.interpolate(
                    true_masks, size=masks_pred.shape[-2:], mode='bilinear', align_corners=False
                )

                train_loss = criterion(masks_pred, true_masks)
                train_epoch_loss += float(train_loss.item())
                train_n_loss += 1

                optimizer.zero_grad()
                train_loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.set_postfix({'TrainLoss': f"{train_epoch_loss / max(train_n_loss, 1):.4f}"})
                pbar.update(1)

        train_meanLoss = train_epoch_loss / max(train_n_loss, 1)
        trainLoss.append(train_meanLoss)
        writer.add_scalar('Loss/train_epoch_avg', train_meanLoss, epoch + 1)

        net.eval()
        val_epoch_loss = 0.0
        val_n_loss = 0
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f'[Val Fold {fold + 1}]', unit='batch', disable=True) as pbar:
                for batch in val_loader:
                    imgs = batch['image'].to(device)
                    true_masks = batch['GT'].to(device)
                    bbox = batch['box'].to(device)

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
                            multimask_output=False
                        )
                        logits_list.append(low_res_masks)

                    masks_pred = torch.stack([x.squeeze(0) for x in logits_list], dim=0)
                    if true_masks.dim() == 3:
                        true_masks = true_masks.unsqueeze(1)
                    true_masks = F.interpolate(
                        true_masks, size=masks_pred.shape[-2:], mode='bilinear', align_corners=False
                    )

                    val_loss = criterion(masks_pred, true_masks)
                    val_epoch_loss += float(val_loss.item())
                    val_n_loss += 1

                    pbar.set_postfix({'ValLoss': f"{val_epoch_loss / max(val_n_loss, 1):.4f}"})
                    pbar.update(1)

        val_meanLoss = val_epoch_loss / max(val_n_loss, 1)
        valLoss.append(val_meanLoss)
        writer.add_scalar('Loss/Val_epoch_avg', val_meanLoss, epoch + 1)

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR', current_lr, epoch + 1)
        logging.info(
            f'Epoch {epoch + 1}: Train Loss={trainLoss[-1]:.4f}, '
            f'Val Loss={valLoss[-1]:.4f}, lr={current_lr:.8f}'
        )

        scheduler.step(val_meanLoss)

        if bestloss > val_meanLoss:
            bestloss = val_meanLoss
            no_improve_epochs = 0
            torch.save(net.state_dict(), best_ckpt_path)
            logging.info(f'Best model updated with loss={bestloss:.4f}')
        else:
            no_improve_epochs += 1

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': bestloss,
                'no_improve_epochs': no_improve_epochs,
                'train_loss_history': trainLoss,
                'val_loss_history': valLoss,
            },
            last_ckpt_path
        )
        last_finished_epoch = epoch

        if no_improve_epochs >= early_stop_patience:
            logging.info(f"Early stopping triggered at epoch {epoch + 1}, Best Val Loss: {bestloss:.4f}")
            break

    with open(os.path.join(save_dir, 'summary.txt'), 'a') as f:
        f.write(f"Fold {fold + 1}: Best Val Loss = {bestloss:.4f}\n")

    with open(done_flag_path, 'w') as f:
        f.write(
            f"fold={fold + 1}, status=done, last_epoch={last_finished_epoch + 1}, "
            f"best_val_loss={bestloss:.6f}\n"
        )

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
    nii_root_dir = "/home/wusi/segment-anything/SAMdata/Rectal/20260325_CTV/Cropdatanii/train_nii"
    save_dir = '/home/wusi/segment-anything/SAMdata/Rectal/20260325_CTV/nnUNet_crop_box/TrainResult'
    os.makedirs(save_dir, exist_ok=True)

    dataset = SAMDatasetFromNiiGz(
        nii_root_dir=nii_root_dir,
        target_image_size=(1024, 1024),
        image_name="image.nii.gz",
        gt_name="CTV.nii.gz",
        nnunet_name="prompt.nii.gz"
    )

    sam_checkpoint = "/home/wusi/segment-anything/demo/configs/checkpoint/sam_vit_b_01ec64.pth"
    model_type = "vit_b"

    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        log_path = os.path.join(save_dir, f'fold_{fold + 1}/train_fold{fold + 1}.log')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path, mode='a'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info(f"[Fold {fold + 1}] Logging initialized.")
        logging.info(f'Using device {device}')

        net = sam_model_registry[model_type](checkpoint=None)
        state_dict = torch.load(sam_checkpoint, map_location=device)
        net.load_state_dict(state_dict, strict=False)
        logging.info(f"[Info] Loaded SAM checkpoint from {sam_checkpoint} with strict=False.")
        net.to(device)

        for param in net.image_encoder.parameters():
            param.requires_grad = False

        trainable_params = [name for name, param in net.named_parameters() if param.requires_grad]
        logging.info(f"Trainable parameters ({len(trainable_params)}):")
        for name in trainable_params:
            logging.info(f"  {name}")

        train_one_fold(
            fold, train_idx, val_idx, dataset, net, device,
            epochs=200, batch_size=8, lr=0.001, save_dir=save_dir
        )
        logging.info(f"Training Fold{fold + 1} completed.")

        torch.cuda.empty_cache()

    print("Five-fold cross-validation completed.")
