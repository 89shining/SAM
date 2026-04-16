"""
评估nnunet预测结果GT和pred的非空切片2d dsc和2d hd95的值
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute slice-wise Dice/HD95 between nnUNet predictions and GT."
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path(
            "/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_results/"
            "Dataset015_RectalCTV60pAll/nnUNetTrainer__nnUNetPlans__3d_fullres/"
            "testResult_fold2"
        ),
        help="Directory of prediction files.",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path(
            "/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/"
            "Dataset015_RectalCTV60pAll/labelsTs"
        ),
        help="Directory of GT files.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path(
            "/home/wusi/segment-anything/SAMdata/Rectal/20260325_CTV/"
            "nnUNet_all_bad/slice_metrics_total.csv"
        ),
        help="Output CSV path for the total slice-wise table.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--surface-batch-size",
        type=int,
        default=2048,
        help="Chunk size when computing pairwise distances for HD95.",
    )
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_nii(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # [Z, H, W]
    spacing = img.GetSpacing()  # (x, y, z)
    return arr, spacing


def list_nii_gz_files(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.name.endswith(".nii.gz")])


def get_nonzero_bounds(mask_3d: np.ndarray):
    z_nonzero = np.where(mask_3d.reshape(mask_3d.shape[0], -1).sum(axis=1) > 0)[0]
    if len(z_nonzero) == 0:
        return None, None
    return int(z_nonzero.min()), int(z_nonzero.max())


def dice_2d_torch(gt_slice: torch.Tensor, pred_slice: torch.Tensor) -> float:
    inter = torch.logical_and(gt_slice, pred_slice).sum(dtype=torch.float32)
    denom = gt_slice.sum(dtype=torch.float32) + pred_slice.sum(dtype=torch.float32)
    if denom.item() == 0:
        return float("nan")
    return float((2.0 * inter / denom).item())


def _surface_points_2d(binary_mask: torch.Tensor) -> torch.Tensor:
    m = binary_mask.to(torch.uint8)
    up = torch.zeros_like(m)
    down = torch.zeros_like(m)
    left = torch.zeros_like(m)
    right = torch.zeros_like(m)
    up[1:, :] = m[:-1, :]
    down[:-1, :] = m[1:, :]
    left[:, 1:] = m[:, :-1]
    right[:, :-1] = m[:, 1:]
    eroded = m & up & down & left & right
    surface = m & (~eroded)
    points = torch.nonzero(surface, as_tuple=False).to(torch.float32)  # [N,2] as (y, x)
    return points


def _directed_min_distances(
    src: torch.Tensor,
    dst: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    mins = []
    for i in range(0, src.shape[0], chunk_size):
        part = src[i : i + chunk_size]
        dist = torch.cdist(part, dst, p=2.0)
        mins.append(dist.min(dim=1).values)
    return torch.cat(mins, dim=0)


def hd95_2d_torch(
    gt_slice: torch.Tensor,
    pred_slice: torch.Tensor,
    spacing_yx,
    chunk_size: int = 2048,
) -> float:
    gt_surface = _surface_points_2d(gt_slice)
    pred_surface = _surface_points_2d(pred_slice)
    if gt_surface.numel() == 0 or pred_surface.numel() == 0:
        return float("nan")

    scale = torch.tensor(
        [spacing_yx[0], spacing_yx[1]],
        device=gt_slice.device,
        dtype=torch.float32,
    )
    gt_surface = gt_surface * scale
    pred_surface = pred_surface * scale

    d_gt_to_pred = _directed_min_distances(gt_surface, pred_surface, chunk_size=chunk_size)
    d_pred_to_gt = _directed_min_distances(pred_surface, gt_surface, chunk_size=chunk_size)
    all_dist = torch.cat([d_gt_to_pred, d_pred_to_gt], dim=0)
    return float(torch.quantile(all_dist, 0.95).item())


def main():
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not args.gt_dir.exists():
        raise FileNotFoundError(f"GT directory not found: {args.gt_dir}")
    if not args.pred_dir.exists():
        raise FileNotFoundError(f"Pred directory not found: {args.pred_dir}")

    device = resolve_device(args.device)
    print(f"[Info] Using device: {device}")

    gt_files = list_nii_gz_files(args.gt_dir)
    pred_files = list_nii_gz_files(args.pred_dir)
    gt_map = {p.name: p for p in gt_files}
    pred_map = {p.name: p for p in pred_files}

    gt_names = set(gt_map.keys())
    pred_names = set(pred_map.keys())

    only_in_gt = sorted(gt_names - pred_names)
    only_in_pred = sorted(pred_names - gt_names)
    common_names = sorted(gt_names & pred_names)

    print(f"[Info] GT nii.gz count: {len(gt_files)}")
    print(f"[Info] Pred nii.gz count: {len(pred_files)} (non-nii.gz are ignored)")
    print(f"[Info] Matched pairs: {len(common_names)}")
    if only_in_gt:
        print(f"[Warn] Missing in Pred (first 10): {only_in_gt[:10]}")
    if only_in_pred:
        print(f"[Warn] Missing in GT (first 10): {only_in_pred[:10]}")

    rows = []

    for idx, case_name in enumerate(common_names, 1):
        gt_path = gt_map[case_name]
        pred_path = pred_map[case_name]

        gt_np, gt_spacing = read_nii(gt_path)
        pred_np, _ = read_nii(pred_path)

        if gt_np.shape != pred_np.shape:
            print(f"[Skip] shape mismatch: {case_name}, GT={gt_np.shape}, Pred={pred_np.shape}")
            continue

        gt_np = (gt_np > 0).astype(np.uint8)
        pred_np = (pred_np > 0).astype(np.uint8)

        gt_lower_z, gt_upper_z = get_nonzero_bounds(gt_np)
        voxelspacing_2d = (float(gt_spacing[1]), float(gt_spacing[0]))  # (y, x)

        gt_t = torch.from_numpy(gt_np).to(device=device, dtype=torch.bool)
        pred_t = torch.from_numpy(pred_np).to(device=device, dtype=torch.bool)

        for z in range(gt_t.shape[0]):
            gt_slice = gt_t[z]
            pred_slice = pred_t[z]

            gt_nonempty = bool(gt_slice.any().item())
            pred_nonempty = bool(pred_slice.any().item())

            if (not gt_nonempty) and (not pred_nonempty):
                continue

            dice_val = dice_2d_torch(gt_slice, pred_slice)

            if gt_nonempty and pred_nonempty:
                try:
                    hd95_val = hd95_2d_torch(
                        gt_slice,
                        pred_slice,
                        spacing_yx=voxelspacing_2d,
                        chunk_size=args.surface_batch_size,
                    )
                except RuntimeError as e:
                    print(f"[Warn] HD95 failed for {case_name} z={z}: {e}")
                    hd95_val = float("nan")
            else:
                hd95_val = float("nan")

            rows.append(
                {
                    "case": case_name,
                    "z": int(z),
                    "gt_lower_z": gt_lower_z,
                    "gt_upper_z": gt_upper_z,
                    "z_relative_to_gt_lower": (int(z) - gt_lower_z) if gt_lower_z is not None else None,
                    "z_relative_to_gt_upper": (int(z) - gt_upper_z) if gt_upper_z is not None else None,
                    "gt_nonempty": int(gt_nonempty),
                    "pred_nonempty": int(pred_nonempty),
                    "dice_2d": float(dice_val),
                    "hd95_2d_mm": float(hd95_val),
                }
            )

        print(f"[Progress] {idx}/{len(common_names)} done: {case_name}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print(f"[Done] Saved to: {args.out_csv}")
    print(f"[Done] Total valid slices saved: {len(df)}")


if __name__ == "__main__":
    main()
