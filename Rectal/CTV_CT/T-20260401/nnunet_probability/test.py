import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from segment_anything import sam_model_registry
from testdataset import SAMTestDatasetFromNiiGz


def parse_args():
    parser = argparse.ArgumentParser(description="Test SAM with nnUNet probability prompt.")
    parser.add_argument("--datanii-dir", required=True, help="Path to test_nii root directory.")
    parser.add_argument("--nnunet-prompt-npz-dir", required=True, help="Directory containing CTV_XXX.npz files.")
    parser.add_argument("--output-dir", required=True, help="Directory to save predicted nii.gz and metrics.json.")
    parser.add_argument("--train-result-dir", required=True, help="Directory containing fold_*/best_metrics.json and weights.")
    parser.add_argument("--sam-ckpt", required=True, help="Path to base SAM checkpoint.")
    parser.add_argument("--model-type", default="vit_b", help="SAM model type key in sam_model_registry.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--use-gt-positive-only", action="store_true", help="Only evaluate GT-positive slices.")
    return parser.parse_args()


def dice_score(pred, gt, eps=1e-6):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = float((pred * gt).sum())
    denom = float(pred.sum() + gt.sum())
    return (2.0 * inter + eps) / (denom + eps)


def build_loader(dataset, batch_size):
    cpu_count = os.cpu_count() or 0
    num_workers = min(8, cpu_count) if cpu_count > 0 else 0

    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def resolve_best_checkpoint(train_result_dir):
    candidates = []
    if not os.path.isdir(train_result_dir):
        raise FileNotFoundError(f"TrainResult directory not found: {train_result_dir}")

    for name in os.listdir(train_result_dir):
        fold_dir = os.path.join(train_result_dir, name)
        if not os.path.isdir(fold_dir) or not name.startswith("fold_"):
            continue

        metrics_path = os.path.join(fold_dir, "best_metrics.json")
        ckpt_path = os.path.join(fold_dir, "weights", "best_by_dice.pth")
        if not os.path.exists(metrics_path) or not os.path.exists(ckpt_path):
            continue

        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            dice = float(m.get("best_val_patient_dice", -1.0))
            epoch = int(m.get("best_epoch", -1))
            candidates.append(
                {
                    "fold_name": name,
                    "dice": dice,
                    "epoch": epoch,
                    "ckpt": ckpt_path,
                    "metrics": metrics_path,
                }
            )
        except Exception:
            continue

    if len(candidates) == 0:
        raise RuntimeError(
            "No valid fold metrics/checkpoint found. Expected fold_*/best_metrics.json and fold_*/weights/best_by_dice.pth"
        )

    candidates.sort(key=lambda x: (x["dice"], x["epoch"]), reverse=True)
    return candidates[0], candidates


def _forward_masks_batched(net, imgs, mask_inputs):
    input_images = torch.stack([net.preprocess(im) for im in imgs], dim=0)
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


def _get_hw_from_batch(batch, i):
    ori = batch["original_size"]
    if isinstance(ori, (tuple, list)) and len(ori) == 2:
        h, w = ori[0], ori[1]
        if isinstance(h, torch.Tensor):
            h = int(h[i].item())
        else:
            h = int(h[i])
        if isinstance(w, torch.Tensor):
            w = int(w[i].item())
        else:
            w = int(w[i])
        return h, w

    item = ori[i]
    if isinstance(item, torch.Tensor):
        return int(item[0].item()), int(item[1].item())
    return int(item[0]), int(item[1])


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disable_cudnn = os.environ.get("SAM_DISABLE_CUDNN", "1") == "1"
    if disable_cudnn:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = False
    os.makedirs(args.output_dir, exist_ok=True)

    best_item, all_items = resolve_best_checkpoint(args.train_result_dir)
    finetuned_ckpt = best_item["ckpt"]
    print(f"[AutoSelect] Use best fold: {best_item['fold_name']}, patient_dice={best_item['dice']:.6f}, ckpt={finetuned_ckpt}")

    test_dataset = SAMTestDatasetFromNiiGz(
        nii_root_dir=args.datanii_dir,
        target_image_size=(1024, 1024),
        mask_prompt_size=(256, 256),
        image_name="image.nii.gz",
        gt_name="CTV.nii.gz",
        nnunet_prompt_npz_dir=args.nnunet_prompt_npz_dir,
        npz_pattern="CTV_{:03d}.npz",
        prompt_class_idx=1,
        prompt_eps=1e-4,
        use_gt_positive_only=args.use_gt_positive_only,
    )
    test_loader = build_loader(test_dataset, batch_size=args.batch_size)

    net = sam_model_registry[args.model_type](checkpoint=None)
    net.load_state_dict(torch.load(args.sam_ckpt, map_location=device), strict=False)
    net.load_state_dict(torch.load(finetuned_ckpt, map_location=device), strict=False)
    net.to(device)
    net.eval()

    pred_volumes = defaultdict(dict)
    gt_slices = defaultdict(dict)
    slice_dice_list = []

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device, non_blocking=True).float()
            mask_prompt = batch["mask_prompt"].to(device, non_blocking=True)
            true_masks = batch["GT"].to(device, non_blocking=True)
            patient_ids = batch["patient_id"]
            slice_idxs = batch["slice_idx"]

            low_res_masks = _forward_masks_batched(net, imgs, mask_prompt)

            bsz = imgs.shape[0]
            for i in range(bsz):
                h, w = _get_hw_from_batch(batch, i)
                masks = net.postprocess_masks(
                    low_res_masks[i].unsqueeze(0),
                    input_size=imgs.shape[-2:],
                    original_size=(h, w),
                )

                pid = patient_ids[i]
                z = int(slice_idxs[i])
                pred_np = (torch.sigmoid(masks) > 0.5).float()[0, 0].cpu().numpy().astype(np.uint8)
                gt_np = (true_masks[i, 0] > 0.5).float().cpu().numpy().astype(np.uint8)

                pred_volumes[pid][z] = pred_np
                gt_slices[pid][z] = gt_np
                slice_dice_list.append(dice_score(pred_np, gt_np))

    patient_dice = {}
    for pa in os.listdir(args.datanii_dir):
        pdir = os.path.join(args.datanii_dir, pa)
        if not os.path.isdir(pdir):
            continue

        ref_img = sitk.ReadImage(os.path.join(pdir, "image.nii.gz"))
        ref_arr = sitk.GetArrayFromImage(ref_img)
        pred_arr = np.zeros_like(ref_arr, dtype=np.uint8)
        gt_arr = np.zeros_like(ref_arr, dtype=np.uint8)

        if pa in pred_volumes:
            for z, m in pred_volumes[pa].items():
                pred_arr[z] = m
        if pa in gt_slices:
            for z, m in gt_slices[pa].items():
                gt_arr[z] = m

        patient_dice[pa] = dice_score(pred_arr, gt_arr)
        pred_img = sitk.GetImageFromArray(pred_arr)
        pred_img.CopyInformation(ref_img)

        match = re.search(r"\d+", pa)
        if match:
            idx = match.group(0).zfill(3)
            save_name = f"CTV_{idx}.nii.gz"
        else:
            save_name = f"{pa}_pred.nii.gz"
        sitk.WriteImage(pred_img, os.path.join(args.output_dir, save_name))

    metrics = {
        "num_slices": len(slice_dice_list),
        "slice_dice_mean": float(np.mean(slice_dice_list)) if len(slice_dice_list) > 0 else 0.0,
        "slice_dice_std": float(np.std(slice_dice_list)) if len(slice_dice_list) > 0 else 0.0,
        "num_patients": len(patient_dice),
        "patient_dice_mean": float(np.mean(list(patient_dice.values()))) if len(patient_dice) > 0 else 0.0,
        "patient_dice_std": float(np.std(list(patient_dice.values()))) if len(patient_dice) > 0 else 0.0,
        "patient_dice": patient_dice,
        "selected_checkpoint": finetuned_ckpt,
        "selected_fold": best_item["fold_name"],
        "selected_fold_best_val_patient_dice": best_item["dice"],
        "all_fold_candidates": all_items,
        "use_gt_positive_only": bool(args.use_gt_positive_only),
    }

    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("DONE.")
    print(
        f"selected_fold={metrics['selected_fold']}, "
        f"slice_dice_mean={metrics['slice_dice_mean']:.6f}, "
        f"patient_dice_mean={metrics['patient_dice_mean']:.6f}, "
        f"num_patients={metrics['num_patients']}"
    )


if __name__ == "__main__":
    main()
