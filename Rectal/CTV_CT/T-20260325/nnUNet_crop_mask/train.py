# Five-fold cross-validation training for CTV (nnUNet mask prompt)
import json
import os
import sys

sys.path.append("/home/wusi/segment-anything")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import random
import logging
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dice_loss import BCEDiceLoss
from dataset import SAMDatasetFromNiiGz
from segment_anything import sam_model_registry

manual_seed = int.from_bytes(os.urandom(4), 'little')
random.seed(manual_seed)
torch.manual_seed(manual_seed)


def _build_loader(dataset, indices, batch_size, shuffle, device):
    cpu_count = os.cpu_count() or 0
    num_workers = min(8, cpu_count) if cpu_count > 0 else 0
    pin_memory = device.type == 'cuda'

    kwargs = {
        "dataset": Subset(dataset, indices),
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return DataLoader(**kwargs)


def _forward_masks_batched(net, imgs, mask_inputs, freeze_image_encoder=True):
    """
    SAM mask_decoder in this codebase assumes image_embeddings shape starts with 1.
    So we keep image encoder batched, but run prompt+decoder per sample.
    """
    input_images = torch.stack([net.preprocess(im) for im in imgs], dim=0)

    if freeze_image_encoder:
        with torch.no_grad():
            image_embeddings = net.image_encoder(input_images)
    else:
        image_embeddings = net.image_encoder(input_images)

    low_res_list = []
    for i in range(imgs.shape[0]):
        sparse_embeddings, dense_embeddings = net.prompt_encoder(
            points=None,
            boxes=None,
            masks=mask_inputs[i].unsqueeze(0),
        )

        low_res_masks, _ = net.mask_decoder(
            image_embeddings=image_embeddings[i].unsqueeze(0),
            image_pe=net.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        low_res_list.append(low_res_masks)

    return torch.cat(low_res_list, dim=0)


def _dice_from_binary(pred_bin, target_bin, eps=1e-6):
    inter = float((pred_bin * target_bin).sum().item())
    denom = float(pred_bin.sum().item() + target_bin.sum().item())
    return (2.0 * inter + eps) / (denom + eps), inter, float(pred_bin.sum().item()), float(target_bin.sum().item())


def train_one_fold(fold, train_idx, val_idx, dataset, net, device,
                   epochs, batch_size, lr, save_dir):
    fold_dir = os.path.join(save_dir, f"fold_{fold + 1}")
    weights_dir = os.path.join(fold_dir, 'weights')
    os.makedirs(fold_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(os.path.join(fold_dir, 'runs'), exist_ok=True)

    last_ckpt_path = os.path.join(weights_dir, 'last.pth')
    best_ckpt_path = os.path.join(weights_dir, 'best_by_dice.pth')
    done_flag_path = os.path.join(fold_dir, 'training_done.flag')
    metrics_path = os.path.join(fold_dir, 'best_metrics.json')

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

    train_patients = sorted({dataset.index[i][0] for i in train_idx})
    val_patients = sorted({dataset.index[i][0] for i in val_idx})
    logging.info(f"Train patients ({len(train_patients)}): {train_patients}")
    logging.info(f"Val patients ({len(val_patients)}): {val_patients}")

    train_loader = _build_loader(dataset, train_idx, batch_size=batch_size, shuffle=True, device=device)
    val_loader = _build_loader(dataset, val_idx, batch_size=batch_size, shuffle=False, device=device)

    writer = SummaryWriter(os.path.join(fold_dir, 'runs'))

    logging.info(f'''Starting training:
            Fold:                 {fold + 1}
            Epochs:               {epochs}
            Batch size:           {batch_size}
            Learning rate:        {lr}
            Training slices:      {len(train_idx)}
            Validation slices:    {len(val_idx)}
            Training patients:    {len(train_patients)}
            Validation patients:  {len(val_patients)}
            Device:               {device.type}
        ''')

    criterion = BCEDiceLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=lr,
        weight_decay=1e-4
    )

    # Dice-driven scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=7,
        min_lr=1e-6
    )

    best_dice = -1.0
    best_epoch = -1
    best_loss_at_best_dice = float('inf')
    no_improve_epochs = 0
    early_stop_patience = 15

    trainLoss, valLoss = [], []
    valSliceDiceHist, valPatientDiceHist = [], []
    start_epoch = 0

    if os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            net.load_state_dict(ckpt['model_state_dict'], strict=False)
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])

            best_dice = float(ckpt.get('best_dice', -1.0))
            best_epoch = int(ckpt.get('best_epoch', -1))
            best_loss_at_best_dice = float(ckpt.get('best_loss_at_best_dice', float('inf')))
            no_improve_epochs = int(ckpt.get('no_improve_epochs', 0))
            trainLoss = ckpt.get('train_loss_history', [])
            valLoss = ckpt.get('val_loss_history', [])
            valSliceDiceHist = ckpt.get('val_slice_dice_history', [])
            valPatientDiceHist = ckpt.get('val_patient_dice_history', [])
            start_epoch = int(ckpt.get('epoch', -1)) + 1
            logging.info(
                f"[Fold {fold + 1}] Resume from epoch {start_epoch}/{epochs}, "
                f"best patient-dice={best_dice:.6f}"
            )
        else:
            logging.warning(f"[Fold {fold + 1}] last.pth format incompatible, start from epoch 0.")

    if start_epoch >= epochs:
        logging.info(f"[Fold {fold + 1}] Already reached target epochs ({epochs}), mark done.")
        with open(done_flag_path, 'w') as f:
            f.write(f"fold={fold + 1}, status=done, best_patient_dice={best_dice:.6f}\n")
        writer.close()
        return

    last_finished_epoch = start_epoch - 1

    for epoch in range(start_epoch, epochs):
        net.train()
        train_epoch_loss = 0.0
        train_n_loss = 0

        with tqdm(total=len(train_loader), desc=f'[Train Fold {fold + 1}]', unit='batch', disable=True) as pbar:
            for batch in train_loader:
                imgs = batch['image'].to(device, non_blocking=True)
                true_masks = batch['GT'].to(device, non_blocking=True)
                mask_inputs = batch['mask_prompt'].to(device, non_blocking=True)

                masks_pred = _forward_masks_batched(net, imgs, mask_inputs, freeze_image_encoder=True)

                if true_masks.dim() == 3:
                    true_masks = true_masks.unsqueeze(1)
                true_masks = F.interpolate(
                    true_masks,
                    size=masks_pred.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
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

        train_mean_loss = train_epoch_loss / max(train_n_loss, 1)
        trainLoss.append(train_mean_loss)
        writer.add_scalar('Loss/train_epoch_avg', train_mean_loss, epoch + 1)

        net.eval()
        val_epoch_loss = 0.0
        val_n_loss = 0
        val_slice_dices = []
        patient_stats = {}

        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f'[Val Fold {fold + 1}]', unit='batch', disable=True) as pbar:
                for batch in val_loader:
                    imgs = batch['image'].to(device, non_blocking=True)
                    true_masks = batch['GT'].to(device, non_blocking=True)
                    mask_inputs = batch['mask_prompt'].to(device, non_blocking=True)
                    patient_ids = batch['patient_id']

                    masks_pred = _forward_masks_batched(net, imgs, mask_inputs, freeze_image_encoder=True)

                    if true_masks.dim() == 3:
                        true_masks = true_masks.unsqueeze(1)
                    true_masks = F.interpolate(
                        true_masks,
                        size=masks_pred.shape[-2:],
                        mode='bilinear',
                        align_corners=False,
                    )

                    val_loss = criterion(masks_pred, true_masks)
                    val_epoch_loss += float(val_loss.item())
                    val_n_loss += 1

                    pred_bin = (torch.sigmoid(masks_pred) > 0.5).float()
                    gt_bin = (true_masks > 0.5).float()

                    bsz = pred_bin.shape[0]
                    for i in range(bsz):
                        pid = patient_ids[i]
                        dice_i, inter_i, pred_i, gt_i = _dice_from_binary(pred_bin[i, 0], gt_bin[i, 0])
                        val_slice_dices.append(dice_i)

                        if pid not in patient_stats:
                            patient_stats[pid] = {"inter": 0.0, "pred": 0.0, "gt": 0.0}
                        patient_stats[pid]["inter"] += inter_i
                        patient_stats[pid]["pred"] += pred_i
                        patient_stats[pid]["gt"] += gt_i

                    pbar.set_postfix({'ValLoss': f"{val_epoch_loss / max(val_n_loss, 1):.4f}"})
                    pbar.update(1)

        val_mean_loss = val_epoch_loss / max(val_n_loss, 1)
        valLoss.append(val_mean_loss)
        writer.add_scalar('Loss/Val_epoch_avg', val_mean_loss, epoch + 1)

        val_slice_dice = float(np.mean(val_slice_dices)) if len(val_slice_dices) > 0 else 0.0
        valSliceDiceHist.append(val_slice_dice)
        writer.add_scalar('Dice/Val_slice_avg', val_slice_dice, epoch + 1)

        patient_dices = []
        eps = 1e-6
        for pid, st in patient_stats.items():
            patient_dice = (2.0 * st["inter"] + eps) / (st["pred"] + st["gt"] + eps)
            patient_dices.append(patient_dice)
        val_patient_dice = float(np.mean(patient_dices)) if len(patient_dices) > 0 else 0.0
        valPatientDiceHist.append(val_patient_dice)
        writer.add_scalar('Dice/Val_patient_avg', val_patient_dice, epoch + 1)

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR', current_lr, epoch + 1)

        logging.info(
            f"Epoch {epoch + 1}: Train Loss={train_mean_loss:.4f}, "
            f"Val Loss={val_mean_loss:.4f}, Val Slice Dice={val_slice_dice:.4f}, "
            f"Val Patient Dice={val_patient_dice:.4f}, lr={current_lr:.8f}"
        )

        scheduler.step(val_patient_dice)

        if val_patient_dice > best_dice:
            best_dice = val_patient_dice
            best_epoch = epoch + 1
            best_loss_at_best_dice = val_mean_loss
            no_improve_epochs = 0
            torch.save(net.state_dict(), best_ckpt_path)
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        "fold": fold + 1,
                        "best_epoch": best_epoch,
                        "best_val_patient_dice": best_dice,
                        "val_loss_at_best_dice": best_loss_at_best_dice,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            logging.info(f"Best model updated by patient-dice={best_dice:.6f} at epoch {best_epoch}")
        else:
            no_improve_epochs += 1

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'best_epoch': best_epoch,
                'best_loss_at_best_dice': best_loss_at_best_dice,
                'no_improve_epochs': no_improve_epochs,
                'train_loss_history': trainLoss,
                'val_loss_history': valLoss,
                'val_slice_dice_history': valSliceDiceHist,
                'val_patient_dice_history': valPatientDiceHist,
            },
            last_ckpt_path
        )
        last_finished_epoch = epoch

        if no_improve_epochs >= early_stop_patience:
            logging.info(
                f"Early stopping by patient-dice at epoch {epoch + 1}. "
                f"Best epoch={best_epoch}, best patient-dice={best_dice:.6f}"
            )
            break

    with open(os.path.join(save_dir, 'summary.txt'), 'a') as f:
        f.write(
            f"Fold {fold + 1}: Best Val Patient Dice = {best_dice:.6f} "
            f"(epoch {best_epoch}, val_loss={best_loss_at_best_dice:.6f})\n"
        )

    with open(done_flag_path, 'w') as f:
        f.write(
            f"fold={fold + 1}, status=done, last_epoch={last_finished_epoch + 1}, "
            f"best_epoch={best_epoch}, best_patient_dice={best_dice:.6f}\n"
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

    if len(valPatientDiceHist) > 0:
        plt.figure()
        plt.plot(range(1, len(valPatientDiceHist) + 1), valPatientDiceHist, label='Val Patient Dice')
        plt.plot(range(1, len(valSliceDiceHist) + 1), valSliceDiceHist, label='Val Slice Dice')
        plt.xlabel('Epoch')
        plt.ylabel('Dice')
        plt.title(f'Fold {fold + 1} Dice Curve')
        plt.legend()
        plt.savefig(os.path.join(fold_dir, 'dice_curve.jpg'))
        plt.close()

    writer.close()


if __name__ == '__main__':
    nii_root_dir = "/home/wusi/segment-anything/SAMdata/Rectal/20260325_CTV/Cropdatanii/train_nii"
    save_dir = '/home/wusi/segment-anything/SAMdata/Rectal/20260325_CTV/nnUNet_crop_mask/TrainResult'
    os.makedirs(save_dir, exist_ok=True)

    dataset = SAMDatasetFromNiiGz(
        nii_root_dir=nii_root_dir,
        target_image_size=(1024, 1024),
        mask_prompt_size=(256, 256),
        image_name="image.nii.gz",
        gt_name="CTV.nii.gz",
        nnunet_name="prompt.nii.gz"
    )

    sam_checkpoint = "/home/wusi/segment-anything/demo/configs/checkpoint/sam_vit_b_01ec64.pth"
    model_type = "vit_b"

    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Patient-level split by group
    groups = [dataset.index[i][0] for i in range(len(dataset))]
    gkf = GroupKFold(n_splits=5)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(np.zeros(len(dataset)), groups=groups)):
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
        logging.info(f"Training Fold {fold + 1} completed.")

        torch.cuda.empty_cache()

    print("Five-fold cross-validation completed.")

