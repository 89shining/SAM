# 五折交叉验证 GTVp 训练（Image Encoder LoRA）
import os
import sys
import math
import torch
import random
import logging
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter
from tqdm import tqdm

sys.path.append("/home/wusi/segment-anything")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dice_loss import BCEDiceLoss
from datasetGTVp import SAMDataset
from segment_anything import sam_model_registry


# 设置随机种子（保持原脚本行为：每次运行随机生成）
manual_seed = int.from_bytes(os.urandom(4), 'little')
random.seed(manual_seed)
torch.manual_seed(manual_seed)


class LoRAQKV(nn.Module):
    """
    LoRA wrapper for merged qkv projection.
    Apply LoRA only to q and v branches (Hu et al.-style low-rank update):
      y = W0 x + (alpha/r) * B(A(dropout(x)))
    where LoRA is injected into q and v slices of qkv output.
    """

    def __init__(self, base_linear: nn.Linear, dim: int, r: int = 4, alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("LoRALinear expects nn.Linear as base layer.")
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0.")

        self.base_linear = base_linear
        self.dim = dim
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(p=dropout)

        in_features = base_linear.in_features
        # q branch
        self.lora_q_A = nn.Linear(in_features, r, bias=False)
        self.lora_q_B = nn.Linear(r, dim, bias=False)
        # v branch
        self.lora_v_A = nn.Linear(in_features, r, bias=False)
        self.lora_v_B = nn.Linear(r, dim, bias=False)

        # Requirement: freeze pretrained W0
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        # Requirement: A Kaiming uniform, B zeros
        nn.init.kaiming_uniform_(self.lora_q_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_q_B.weight)
        nn.init.kaiming_uniform_(self.lora_v_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_v_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_linear(x)  # [..., 3*dim]
        dropped = self.lora_dropout(x)
        delta_q = self.lora_q_B(self.lora_q_A(dropped)) * self.scaling  # [..., dim]
        delta_v = self.lora_v_B(self.lora_v_A(dropped)) * self.scaling  # [..., dim]

        out = base.clone()
        out[..., :self.dim] = out[..., :self.dim] + delta_q
        out[..., -self.dim:] = out[..., -self.dim:] + delta_v
        return out


def inject_lora_to_sam_image_encoder(sam_model: nn.Module, r: int = 4, alpha: int = 16, dropout: float = 0.1) -> nn.Module:
    """
    Inject LoRA into SAM image encoder attention qkv projection.
    For merged qkv implementation, LoRA is applied only to q and v outputs.
    """
    for blk in sam_model.image_encoder.blocks:
        dim = blk.attn.qkv.in_features
        blk.attn.qkv = LoRAQKV(blk.attn.qkv, dim=dim, r=r, alpha=alpha, dropout=dropout)
    return sam_model


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def configure_trainable_parameters(sam_model: nn.Module) -> None:
    """
    Requirement:
    - only LoRA trainable inside image_encoder
    - prompt_encoder trainable
    - mask_decoder trainable
    """
    # Freeze all first
    for p in sam_model.parameters():
        p.requires_grad = False

    # Unfreeze only LoRA parameters in image encoder
    for name, p in sam_model.image_encoder.named_parameters():
        if "lora_" in name:
            p.requires_grad = True

    # Unfreeze prompt & mask decoders
    _set_requires_grad(sam_model.prompt_encoder, True)
    _set_requires_grad(sam_model.mask_decoder, True)


def count_params(module: nn.Module, only_trainable: bool = False, include_name_filter=None, exclude_name_filter=None) -> int:
    total = 0
    for name, p in module.named_parameters():
        if only_trainable and not p.requires_grad:
            continue
        if include_name_filter is not None and not include_name_filter(name):
            continue
        if exclude_name_filter is not None and not exclude_name_filter(name):
            continue
        total += p.numel()
    return total


def log_trainable_breakdown(net: nn.Module) -> None:
    image_encoder_original_trainable = count_params(
        net.image_encoder,
        only_trainable=True,
        exclude_name_filter=lambda n: ("lora_" not in n),
    )
    image_encoder_lora_trainable = count_params(
        net.image_encoder,
        only_trainable=True,
        include_name_filter=lambda n: ("lora_" in n),
    )
    prompt_encoder_trainable = sum(p.numel() for p in net.prompt_encoder.parameters() if p.requires_grad)
    mask_decoder_trainable = sum(p.numel() for p in net.mask_decoder.parameters() if p.requires_grad)
    total_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)

    logging.info(f"image_encoder original trainable params: {image_encoder_original_trainable}")
    logging.info(f"image_encoder LoRA trainable params: {image_encoder_lora_trainable}")
    logging.info(f"prompt_encoder trainable params: {prompt_encoder_trainable}")
    logging.info(f"mask_decoder trainable params: {mask_decoder_trainable}")
    logging.info(f"total trainable parameters: {total_trainable}")


def train_one_fold(fold, train_idx, val_idx, all_image_paths, dataset, net, device,
                   epochs, batch_size, lr, save_dir):
    fold_dir = os.path.join(save_dir, f"fold_{fold + 1}")
    os.makedirs(fold_dir, exist_ok=True)
    os.makedirs(os.path.join(fold_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(fold_dir, 'runs'), exist_ok=True)

    logging.info(f'Auto-generated seed: {manual_seed}')

    train_ids = [all_image_paths[i].replace('/images/', '').replace('.tiff', '').replace('.tif', '') for i in train_idx]
    val_ids = [all_image_paths[i].replace('/images/', '').replace('.tiff', '').replace('.tif', '') for i in val_idx]

    with open(os.path.join(fold_dir, 'train_ids.txt'), 'w') as f:
        f.writelines(f"{id}\n" for id in train_ids)
    with open(os.path.join(fold_dir, 'val_ids.txt'), 'w') as f:
        f.writelines(f"{id}\n" for id in val_ids)

    logging.info(f"Train IDs ({len(train_ids)} samples): {train_ids}")
    logging.info(f"Val IDs ({len(val_ids)} samples): {val_ids}")

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

    # Requirement: optimizer only on requires_grad=True params
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
            for _, batch in enumerate(train_loader):
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

                pbar.set_postfix({'TrainLoss': f"{train_epoch_loss / train_n_loss:.4f}"})
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
                for _, batch in enumerate(val_loader):
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

                    val_epoch_loss += float(val_loss.item())
                    val_n_loss += 1

                    pbar.set_postfix({'ValLoss': f"{val_epoch_loss / val_n_loss:.4f}"})
                    pbar.update(1)

        val_meanLoss = val_epoch_loss / val_n_loss
        LOSS.append(val_meanLoss)
        valLoss.append(LOSS[-1])
        writer.add_scalar('Loss/Val_epoch_avg', val_meanLoss, epoch + 1)

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR', current_lr, epoch + 1)
        logging.info(f'Epoch {epoch + 1}: Train Loss={trainLoss[-1]:.4f}, Val Loss={valLoss[-1]:.4f}, lr={current_lr:.8f}')

        scheduler.step(val_meanLoss)

        if bestloss > val_meanLoss:
            bestloss = val_meanLoss
            no_improve_epochs = 0
            # Requirement: save full model state_dict
            torch.save(net.state_dict(), os.path.join(fold_dir, 'weights', 'best.pth'))
            logging.info(f'Best model updated with loss={bestloss:.4f}')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stop_patience:
                logging.info(f"Early stopping triggered at epoch {epoch + 1}, Best Val Loss: {bestloss:.4f}")
                break

        with open(os.path.join(save_dir, 'summary.txt'), 'a') as f:
            f.write(f"Fold {fold + 1}: Best Val Loss = {bestloss:.4f}\n")

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
    root_dir = '/home/wusi/segment-anything/SAMdata/Rectal/20250711_GTVp/dataset/train'
    csv_path = '/home/wusi/segment-anything/SAMdata/Rectal/20250711_GTVp/dataset/train/train_rgb.csv'
    nii_dir = '/home/wusi/segment-anything/SAMdata/Rectal/20250711_GTVp/datanii/train_nii'
    save_dir = '/home/wusi/segment-anything/SAMdata/Rectal/20250711_GTVp/TrainResults/trainresult_TrainAll_lora'
    os.makedirs(save_dir, exist_ok=True)

    dataset = SAMDataset(csv_path=csv_path, root_dir=root_dir, nii_dir=nii_dir, target_size=(1024, 1024))
    all_image_paths = pd.read_csv(csv_path, header=None, names=['image', 'mask'])['image'].tolist()

    sam_checkpoint = '/home/wusi/segment-anything/demo/configs/checkpoint/sam_vit_b_01ec64.pth'
    model_type = 'vit_b'

    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
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
                logging.FileHandler(log_path, mode='w'),
                logging.StreamHandler(sys.stdout)]
        )
        logging.info(f"[Fold {fold + 1}] Logging initialized.")
        logging.info(f'Using device {device}')

        net = sam_model_registry[model_type](checkpoint=None)
        state_dict = torch.load(sam_checkpoint, map_location=device)
        net.load_state_dict(state_dict, strict=False)
        logging.info(f"[Info] Loaded SAM checkpoint from {sam_checkpoint} with strict=False.")

        # LoRA config (defaults requested by user)
        lora_rank = 4
        lora_alpha = 16
        lora_dropout = 0.1
        logging.info(f"LoRA config: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")

        net = inject_lora_to_sam_image_encoder(net, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        configure_trainable_parameters(net)
        log_trainable_breakdown(net)

        net.to(device)

        trainable_params = [name for name, param in net.named_parameters() if param.requires_grad]
        logging.info(f"Trainable parameters ({len(trainable_params)}):")
        for name in trainable_params:
            logging.info(f"  {name}")

        train_one_fold(
            fold, train_idx, val_idx, all_image_paths, dataset, net, device,
            epochs=100, batch_size=2, lr=0.001, save_dir=save_dir
        )
        logging.info(f"Training Fold{fold + 1} completed.")

        torch.cuda.empty_cache()

    print("Five-fold cross-validation completed.")
