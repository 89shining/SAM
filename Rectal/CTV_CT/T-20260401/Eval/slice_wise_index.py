"""
评估SAM预测结果 GT和pred的非空切片2d dsc和2d hd95的值
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch


ID_PATTERN = re.compile(r"(\d+)")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute slice-wise Dice/HD95 for crop-mask dataset layout. "
            "Pred files are matched to GT folders by numeric id."
        )
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("/home/wusi/segment-anything/SAMdata/Rectal/20260401_CTV/nnunet_probability/Prompt_encoder/TestResult"),
        help="Prediction directory (contains .nii.gz files and maybe other files).",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("/home/wusi/segment-anything/SAMdata/Rectal/20260325_CTV/Cropdatanii/test_nii"),
        help="GT root directory (contains folders like p_22, each with CTV.nii.gz).",
    )
    parser.add_argument(
        "--gt-name",
        type=str,
        default="CTV.nii.gz",
        help="GT filename inside each case folder.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path(
            "/home/wusi/segment-anything/SAMdata/Rectal/20260401_CTV/"
            "nnunet_probability/Prompt_encoder/slice_metrics_total.csv"
        ),
        help="Output CSV path for total slice-wise metrics.",
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
        help="Chunk size for pairwise distance computation in HD95.",
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
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()  # (x, y, z)
    return arr, spacing


def extract_numeric_id(text: str):
    m = ID_PATTERN.search(text)
    if m is None:
        return None
    return int(m.group(1))


def collect_pred_map(pred_dir: Path):
    pred_files = sorted([p for p in pred_dir.iterdir() if p.is_file() and p.name.endswith(".nii.gz")])
    pred_map = {}
    for p in pred_files:
        pid = extract_numeric_id(p.stem)
        if pid is None:
            print(f"[Warn] Skip pred without numeric id: {p.name}")
            continue
        if pid in pred_map:
            raise RuntimeError(f"Duplicate pred id={pid}: {pred_map[pid].name} and {p.name}")
        pred_map[pid] = p
    return pred_map, pred_files


def collect_gt_map(gt_dir: Path, gt_name: str):
    gt_case_dirs = sorted([d for d in gt_dir.iterdir() if d.is_dir()])
    gt_map = {}
    for case_dir in gt_case_dirs:
        gid = extract_numeric_id(case_dir.name)
        if gid is None:
            print(f"[Warn] Skip GT folder without numeric id: {case_dir.name}")
            continue
        gt_path = case_dir / gt_name
        if not gt_path.exists():
            print(f"[Warn] GT file missing in {case_dir.name}: {gt_name}")
            continue
        if gid in gt_map:
            raise RuntimeError(
                f"Duplicate GT id={gid}: {gt_map[gid]['case_dir'].name} and {case_dir.name}"
            )
        gt_map[gid] = {"case_dir": case_dir, "gt_path": gt_path}
    return gt_map, gt_case_dirs


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
    return torch.nonzero(surface, as_tuple=False).to(torch.float32)  # (y, x)


def _directed_min_distances(src: torch.Tensor, dst: torch.Tensor, chunk_size: int):
    mins = []
    for i in range(0, src.shape[0], chunk_size):
        part = src[i : i + chunk_size]
        dist = torch.cdist(part, dst, p=2.0)
        mins.append(dist.min(dim=1).values)
    return torch.cat(mins, dim=0)


def hd95_2d_torch(gt_slice: torch.Tensor, pred_slice: torch.Tensor, spacing_yx, chunk_size: int):
    gt_surface = _surface_points_2d(gt_slice)
    pred_surface = _surface_points_2d(pred_slice)
    if gt_surface.numel() == 0 or pred_surface.numel() == 0:
        return float("nan")

    scale = torch.tensor([spacing_yx[0], spacing_yx[1]], device=gt_slice.device, dtype=torch.float32)
    gt_surface = gt_surface * scale
    pred_surface = pred_surface * scale

    d_gt_to_pred = _directed_min_distances(gt_surface, pred_surface, chunk_size)
    d_pred_to_gt = _directed_min_distances(pred_surface, gt_surface, chunk_size)
    all_dist = torch.cat([d_gt_to_pred, d_pred_to_gt], dim=0)
    return float(torch.quantile(all_dist, 0.95).item())


def main():
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not args.pred_dir.exists():
        raise FileNotFoundError(f"Pred directory not found: {args.pred_dir}")
    if not args.gt_dir.exists():
        raise FileNotFoundError(f"GT directory not found: {args.gt_dir}")

    device = resolve_device(args.device)
    print(f"[Info] Using device: {device}")

    pred_map, pred_files = collect_pred_map(args.pred_dir)
    gt_map, gt_case_dirs = collect_gt_map(args.gt_dir, args.gt_name)

    pred_ids = set(pred_map.keys())
    gt_ids = set(gt_map.keys())
    common_ids = sorted(pred_ids & gt_ids)

    only_in_pred = sorted(pred_ids - gt_ids)
    only_in_gt = sorted(gt_ids - pred_ids)

    print(f"[Info] Pred nii.gz count: {len(pred_files)} (non-nii.gz are ignored)")
    print(f"[Info] GT case dir count: {len(gt_case_dirs)}")
    print(f"[Info] Matched by numeric id: {len(common_ids)}")

    if only_in_pred:
        print(f"[Warn] Pred ids without GT (first 20): {only_in_pred[:20]}")
    if only_in_gt:
        print(f"[Warn] GT ids without Pred (first 20): {only_in_gt[:20]}")

    rows = []

    for idx, cid in enumerate(common_ids, 1):
        pred_path = pred_map[cid]
        gt_info = gt_map[cid]
        gt_case_dir = gt_info["case_dir"]
        gt_path = gt_info["gt_path"]

        gt_np, gt_spacing = read_nii(gt_path)
        pred_np, _ = read_nii(pred_path)

        if gt_np.shape != pred_np.shape:
            print(
                f"[Skip] shape mismatch id={cid}: GT={gt_np.shape}, Pred={pred_np.shape}; "
                f"GT={gt_path.name}, Pred={pred_path.name}"
            )
            continue

        gt_np = (gt_np > 0).astype(np.uint8)
        pred_np = (pred_np > 0).astype(np.uint8)

        gt_lower_z, gt_upper_z = get_nonzero_bounds(gt_np)
        voxelspacing_2d = (float(gt_spacing[1]), float(gt_spacing[0]))

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
                    print(f"[Warn] HD95 failed id={cid} z={z}: {e}")
                    hd95_val = float("nan")
            else:
                hd95_val = float("nan")

            rows.append(
                {
                    "case_id": int(cid),
                    "pred_file": pred_path.name,
                    "gt_case_dir": gt_case_dir.name,
                    "gt_file": gt_path.name,
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

        print(f"[Progress] {idx}/{len(common_ids)} done: id={cid}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print(f"[Done] Saved to: {args.out_csv}")
    print(f"[Done] Total valid slices saved: {len(df)}")


if __name__ == "__main__":
    main()
