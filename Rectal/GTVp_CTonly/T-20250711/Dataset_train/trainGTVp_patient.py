import os
import sys
import random
import logging
import argparse

import torch
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter
from tqdm import tqdm

sys.path.append('/home/wusi/segment-anything')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dice_loss import BCEDiceLoss
from datasetGTVp import SAMDataset
from segment_anything import sam_model_registry


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train GTVp with different patient counts (patient-level random subset + 5-fold CV).'
    )
    parser.add_argument('--root_dir', type=str, default='/home/wusi/segment-anything/SAMdata/Rectal/20250711_GTVp/dataset/train')
    parser.add_argument('--csv_path', type=str, default='/home/wusi/segment-anything/SAMdata/Rectal/20250711_GTVp/dataset/train/train_rgb.csv')
    parser.add_argument('--nii_dir', type=str, default='/home/wusi/segment-anything/SAMdata/Rectal/20250711_GTVp/datanii/train_nii')
    parser.add_argument('--save_dir', type=str, default='/home/wusi/segment-anything/SAMdata/Rectal/20250711_GTVp/TrainResults/DaatasetSize_fre_img')

    parser.add_argument('--sam_checkpoint', type=str,
                        default='/home/wusi/segment-anything/demo/configs/checkpoint/sam_vit_b_01ec64.pth')
    parser.add_argument('--model_type', type=str, default='vit_b')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--target_h', type=int, default=1024)
    parser.add_argument('--target_w', type=int, default=1024)

    parser.add_argument('--patient_counts', type=int, nargs='+', default=[10, 20, 30, 40, 50, 60])
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--subset_seed', type=int, default=20250711,
                        help='Random seed for patient subset sampling.')
    parser.add_argument('--fold_seed', type=int, default=42,
                        help='Random seed for KFold split among selected patients.')
    parser.add_argument('--cuda_visible_devices', type=str, default='3')

    return parser.parse_args()


def extract_patient_id_from_mask(mask_rel_path: str) -> str:
    # e.g. masks/p_0/34.nii -> p_0
    return os.path.basename(os.path.dirname(mask_rel_path.lstrip('/\\')))


def build_patient_to_indices(csv_path):
    df = pd.read_csv(csv_path, header=None, names=['image', 'mask'])
    patient_to_indices = {}
    for idx, row in df.iterrows():
        patient_id = extract_patient_id_from_mask(str(row['mask']))
        patient_to_indices.setdefault(patient_id, []).append(idx)
    return df, patient_to_indices


def train_one_fold(fold, train_idx, val_idx, all_image_paths, dataset, net, device,
                   epochs, batch_size, lr, save_dir, manual_seed):
    fold_dir = os.path.join(save_dir, f'fold_{fold + 1}')
    os.makedirs(fold_dir, exist_ok=True)
    os.makedirs(os.path.join(fold_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(fold_dir, 'runs'), exist_ok=True)

    logging.info(f'Auto-generated seed: {manual_seed}')

    train_ids = [all_image_paths[i].replace('/images/', '').replace('.tiff', '').replace('.tif', '') for i in train_idx]
    val_ids = [all_image_paths[i].replace('/images/', '').replace('.tiff', '').replace('.tif', '') for i in val_idx]

    with open(os.path.join(fold_dir, 'train_ids.txt'), 'w') as f:
        f.writelines(f'{id_}\n' for id_ in train_ids)
    with open(os.path.join(fold_dir, 'val_ids.txt'), 'w') as f:
        f.writelines(f'{id_}\n' for id_ in val_ids)

    logging.info(f'Train IDs ({len(train_ids)} samples): {train_ids}')
    logging.info(f'Val IDs ({len(val_ids)} samples): {val_ids}')

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

    writer = SummaryWriter(os.path.join(fold_dir, 'runs'))

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

    for epoch in range(epochs):
        net.train()
        train_epoch_loss = 0
        LOSS = []
        train_n_loss = 0
        with tqdm(total=len(train_loader), desc=f'[Train Fold {fold + 1}]', unit='batch', disable=True) as pbar:
            for batch in train_loader:
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

                train_loss_batch = float(train_loss.item())
                train_epoch_loss += train_loss_batch
                train_n_loss += 1

                optimizer.zero_grad()
                train_loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.set_postfix({'TrainLoss': f'{train_epoch_loss / train_n_loss:.4f}'})
                pbar.update(1)

        train_meanLoss = train_epoch_loss / train_n_loss
        LOSS.append(train_meanLoss)
        trainLoss.append(LOSS[-1])
        writer.add_scalar('Loss/train_epoch_avg', train_meanLoss, epoch + 1)

        net.eval()
        val_epoch_loss = 0
        LOSS = []
        val_n_loss = 0
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f'[Val Fold {fold + 1}]', unit='batch', disable=True) as pbar:
                for batch in val_loader:
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
                    true_masks = F.interpolate(true_masks, size=masks_pred.shape[-2:], mode='bilinear', align_corners=False)
                    val_loss = criterion(masks_pred, true_masks)

                    val_loss_batch = float(val_loss.item())
                    val_epoch_loss += val_loss_batch
                    val_n_loss += 1

                    pbar.set_postfix({'ValLoss': f'{val_epoch_loss / val_n_loss:.4f}'})
                    pbar.update(1)

        val_meanLoss = val_epoch_loss / val_n_loss
        LOSS.append(val_meanLoss)
        valLoss.append(LOSS[-1])
        writer.add_scalar('Loss/Val_epoch_avg', val_meanLoss, epoch + 1)

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR', current_lr, epoch + 1)
        logging.info(
            f'Epoch {epoch + 1}: Train Loss={trainLoss[-1]:.4f}, Val Loss={valLoss[-1]:.4f}, lr={current_lr:.8f}')

        scheduler.step(val_meanLoss)

        if bestloss > val_meanLoss:
            bestloss = val_meanLoss
            no_improve_epochs = 0
            torch.save(net.state_dict(), os.path.join(fold_dir, 'weights', 'best.pth'))
            logging.info(f'Best model updated with loss={bestloss:.4f}')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stop_patience:
                logging.info(f'Early stopping triggered at epoch {epoch + 1}, Best Val Loss: {bestloss:.4f}')
                break

        with open(os.path.join(save_dir, 'summary.txt'), 'a') as f:
            f.write(f'Fold {fold + 1}: Best Val Loss = {bestloss:.4f}\n')

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


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Keep training randomness behavior consistent with original script.
    manual_seed = int.from_bytes(os.urandom(4), 'little')
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = SAMDataset(
        csv_path=args.csv_path,
        root_dir=args.root_dir,
        nii_dir=args.nii_dir,
        target_size=(args.target_h, args.target_w)
    )

    df, patient_to_indices = build_patient_to_indices(args.csv_path)
    all_image_paths = df['image'].tolist()
    all_patients = sorted(patient_to_indices.keys())

    if len(all_patients) < max(args.patient_counts):
        raise ValueError(
            f'Not enough patients. Found {len(all_patients)}, but max requested is {max(args.patient_counts)}.'
        )
    if min(args.patient_counts) < args.n_splits:
        raise ValueError(
            f'Each patient count must be >= n_splits ({args.n_splits}), got {args.patient_counts}.'
        )

    for patient_count in args.patient_counts:
        exp_dir = os.path.join(args.save_dir, f'sample_{patient_count}')
        os.makedirs(exp_dir, exist_ok=True)

        # Make each patient-count subset independently random (no sequential RNG coupling).
        count_subset_seed = args.subset_seed + patient_count * 100003
        subset_rng = random.Random(count_subset_seed)
        selected_patients = subset_rng.sample(all_patients, patient_count)
        selected_patients = sorted(selected_patients)

        with open(os.path.join(exp_dir, 'selected_patients.txt'), 'w') as f:
            f.writelines(f'{pid}\n' for pid in selected_patients)
        pd.DataFrame({
            'order': list(range(1, len(selected_patients) + 1)),
            'patient_id': selected_patients,
            'subset_seed_for_this_count': [count_subset_seed] * len(selected_patients)
        }).to_csv(os.path.join(exp_dir, 'selected_patients.csv'), index=False)

        logging.getLogger().handlers.clear()
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(exp_dir, 'experiment.log'), mode='w'),
                logging.StreamHandler(sys.stdout)
            ]
        )

        logging.info(f'Patient-count experiment: {patient_count}')
        logging.info(f'Auto-generated seed: {manual_seed}')
        logging.info(f'Base subset seed: {args.subset_seed}')
        logging.info(f'Subset seed for sample_{patient_count}: {count_subset_seed}')
        logging.info(f'Fold seed: {args.fold_seed}')
        logging.info(f'Selected patients ({len(selected_patients)}): {selected_patients}')
        logging.info(f'Using device {device}')

        patient_kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.fold_seed)
        fold_split_rows = []

        for fold, (train_p_idx, val_p_idx) in enumerate(patient_kf.split(selected_patients)):
            train_patients = [selected_patients[i] for i in train_p_idx]
            val_patients = [selected_patients[i] for i in val_p_idx]
            for pid in train_patients:
                fold_split_rows.append({'fold': fold + 1, 'subset': 'train', 'patient_id': pid})
            for pid in val_patients:
                fold_split_rows.append({'fold': fold + 1, 'subset': 'val', 'patient_id': pid})

            train_idx = []
            for pid in train_patients:
                train_idx.extend(patient_to_indices[pid])

            val_idx = []
            for pid in val_patients:
                val_idx.extend(patient_to_indices[pid])

            fold_dir = os.path.join(exp_dir, f'fold_{fold + 1}')
            os.makedirs(fold_dir, exist_ok=True)

            with open(os.path.join(fold_dir, 'train_patients.txt'), 'w') as f:
                f.writelines(f'{pid}\n' for pid in train_patients)
            with open(os.path.join(fold_dir, 'val_patients.txt'), 'w') as f:
                f.writelines(f'{pid}\n' for pid in val_patients)

            log_path = os.path.join(fold_dir, f'train_fold{fold + 1}.log')
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_path, mode='w'),
                    logging.StreamHandler(sys.stdout)
                ]
            )

            logging.info(f'[Sample {patient_count}][Fold {fold + 1}] Logging initialized.')
            logging.info(f'[Sample {patient_count}][Fold {fold + 1}] Train patients ({len(train_patients)}): {train_patients}')
            logging.info(f'[Sample {patient_count}][Fold {fold + 1}] Val patients ({len(val_patients)}): {val_patients}')
            logging.info(f'[Sample {patient_count}][Fold {fold + 1}] Train slices: {len(train_idx)}')
            logging.info(f'[Sample {patient_count}][Fold {fold + 1}] Val slices: {len(val_idx)}')
            logging.info(f'Using device {device}')

            net = sam_model_registry[args.model_type](checkpoint=None)
            state_dict = torch.load(args.sam_checkpoint, map_location=device)
            net.load_state_dict(state_dict, strict=False)
            logging.info(f'[Info] Loaded SAM checkpoint from {args.sam_checkpoint} with strict=False.')

            # Freeze image encoder parameters.
            for param in net.image_encoder.parameters():
                param.requires_grad = False

            net.to(device)

            trainable_params = [name for name, param in net.named_parameters() if param.requires_grad]
            logging.info(f'Trainable parameters ({len(trainable_params)}):')
            for name in trainable_params:
                logging.info(f'  {name}')

            train_one_fold(
                fold=fold,
                train_idx=train_idx,
                val_idx=val_idx,
                all_image_paths=all_image_paths,
                dataset=dataset,
                net=net,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                save_dir=exp_dir,
                manual_seed=manual_seed
            )

            logging.info(f'[Sample {patient_count}] Training Fold {fold + 1} completed.')
            torch.cuda.empty_cache()

        pd.DataFrame(fold_split_rows).to_csv(
            os.path.join(exp_dir, 'fold_patient_split.csv'),
            index=False
        )

        # Restore experiment-level log for post-fold summary.
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(exp_dir, 'experiment.log'), mode='a'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info(f'[Sample {patient_count}] 5-fold completed.')

    print('All patient-count experiments completed.')


if __name__ == '__main__':
    main()
