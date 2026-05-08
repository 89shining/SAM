"""
Evaluate prompt variability with random middle-slice prompts and random asymmetric box expansion.

For each patient and each expansion upper limit, this script runs 10 independent trials:
1. Use three manual prompt slices: superior GTV slice, inferior GTV slice, and one random middle slice.
2. On the three prompt slices only, derive the GT bounding box and independently expand
   left/right/top/bottom by a random value sampled from [0, upper_limit_cm].
3. Generate all other in-GTV slice boxes by slice-wise linear interpolation between prompt boxes.
4. Predict only in the GTV z-extent with a single fine-tuned SAM ViT-B checkpoint.
5. Save one full 3D NIfTI prediction for every patient/trial and write prompt/metric Excel sheets.
"""

import argparse
import hashlib
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append("/home/wusi/segment-anything")

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from medpy import metric
from PIL import Image
from scipy import ndimage as ndi
from scipy.interpolate import interp1d
from segment_anything import sam_model_registry
from tqdm import tqdm


DEFAULT_CSV_PATH = "/home/wusi/segment-anything/SAMdata/Rectal/20251224_GTVp/dataset/test/test_rgb.csv"
DEFAULT_ROOT_DIR = "/home/wusi/segment-anything/SAMdata/Rectal/20251224_GTVp/dataset/test"
DEFAULT_IMAGE_DIR = "/home/wusi/segment-anything/SAMdata/Rectal/20251224_GTVp/dataset/test/rgb_images"
DEFAULT_NII_DIR = "/home/wusi/segment-anything/SAMdata/Rectal/20251224_GTVp/datanii/test_nii"
DEFAULT_OUTPUT_ROOT = "/home/wusi/segment-anything/SAMdata/Rectal/20251224_GTVp/TestResults/Prompt_variability/Random_box"
DEFAULT_SAM_CHECKPOINT = "/home/wusi/segment-anything/demo/configs/checkpoint/sam_vit_b_01ec64.pth"
DEFAULT_CKPT_PATH = "/home/wusi/segment-anything/SAMdata/Rectal/20250711_GTVp/TrainResults/trainresult_Freeze_image_encoder//fold_4/weights/best.pth"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Random middle-slice + random asymmetric box prompt variability evaluation."
    )
    parser.add_argument(
        "--ckpt_path",
        default=DEFAULT_CKPT_PATH,
        help="Fine-tuned single-fold SAM checkpoint. Defaults to the fold_4 checkpoint used by 提示位置/test_pos.py.",
    )
    parser.add_argument("--sam_checkpoint", default=DEFAULT_SAM_CHECKPOINT, help="Official SAM ViT-B checkpoint.")
    parser.add_argument("--model_type", default="vit_b", choices=["vit_b"], help="SAM model type.")
    parser.add_argument("--csv_path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--root_dir", default=DEFAULT_ROOT_DIR)
    parser.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--nii_dir", default=DEFAULT_NII_DIR)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--upper_limits_cm", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20251224, help="Base random seed.")
    parser.add_argument("--target_size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def stable_int(text):
    digest = hashlib.md5(str(text).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def fmt_cm(value):
    return f"{float(value):.1f}"


def patient_numeric_id(patient_id):
    match = re.search(r"\d+", str(patient_id))
    if match:
        return match.group(0).zfill(3)
    return str(patient_id)


def load_single_sam_model(ckpt_path, sam_checkpoint, model_type, device):
    model = sam_model_registry[model_type](checkpoint=None)
    model.to(device)
    model.load_state_dict(torch.load(sam_checkpoint, map_location=device), strict=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    model.eval()
    return model


def read_test_csv(csv_path, root_dir, image_dir):
    df = pd.read_csv(csv_path, header=None, names=["image", "mask"])
    rows_by_patient = defaultdict(dict)
    for _, row in df.iterrows():
        image_rel = str(row["image"]).lstrip("/\\")
        image_path = os.path.normpath(os.path.join(root_dir, image_rel))
        rel_to_images = os.path.relpath(image_path, image_dir)
        patient_id = Path(rel_to_images).parent.name
        slice_idx = int(Path(rel_to_images).stem)
        rows_by_patient[patient_id][slice_idx] = image_path
    return rows_by_patient


def get_positive_slices(gt_volume_zyx):
    return [int(z) for z in np.where(gt_volume_zyx.reshape(gt_volume_zyx.shape[0], -1).sum(axis=1) > 0)[0]]


def get_gt_box_from_mask(mask_2d):
    ys, xs = np.where(mask_2d > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def random_expand_box(box, spacing_x_cm, spacing_y_cm, upper_limit_cm, rng, image_width, image_height):
    x0, y0, x1, y1 = box
    expand_left_cm = float(rng.uniform(0.0, upper_limit_cm))
    expand_right_cm = float(rng.uniform(0.0, upper_limit_cm))
    expand_top_cm = float(rng.uniform(0.0, upper_limit_cm))
    expand_bottom_cm = float(rng.uniform(0.0, upper_limit_cm))

    expand_left_px = int(round(expand_left_cm / spacing_x_cm))
    expand_right_px = int(round(expand_right_cm / spacing_x_cm))
    expand_top_px = int(round(expand_top_cm / spacing_y_cm))
    expand_bottom_px = int(round(expand_bottom_cm / spacing_y_cm))

    expanded = [
        max(0, x0 - expand_left_px),
        max(0, y0 - expand_top_px),
        min(image_width - 1, x1 + expand_right_px),
        min(image_height - 1, y1 + expand_bottom_px),
    ]
    expanded[2] = max(expanded[2], expanded[0] + 1)
    expanded[3] = max(expanded[3], expanded[1] + 1)

    expand_info = {
        "expand_left_cm": round(expand_left_cm, 2),
        "expand_right_cm": round(expand_right_cm, 2),
        "expand_top_cm": round(expand_top_cm, 2),
        "expand_bottom_cm": round(expand_bottom_cm, 2),
    }
    return expanded, expand_info


def interpolate_boxes(prompt_boxes_by_z, z):
    key_z = np.array(sorted(prompt_boxes_by_z.keys()), dtype=np.float32)
    box_array = np.array([prompt_boxes_by_z[int(k)] for k in key_z], dtype=np.float32)
    funcs = [
        interp1d(key_z, box_array[:, i], kind="linear", bounds_error=True, assume_sorted=True)
        for i in range(4)
    ]
    return [int(round(float(f(z)))) for f in funcs]


def build_trial_boxes_and_prompt_records(
    patient_id,
    gt_volume_zyx,
    valid_z,
    spacing,
    target_size,
    upper_limit_cm,
    trial,
    middle_z,
    rng_box,
):
    if len(valid_z) < 3:
        raise ValueError(f"{patient_id}: fewer than 3 positive GTV slices, cannot use three-prompt strategy.")

    # Follow the dataset convention: the smaller slice index is the inferior boundary.
    inferior_z = int(valid_z[0])
    superior_z = int(valid_z[-1])
    prompt_z_by_type = {
        "superior": superior_z,
        "middle": middle_z,
        "inferior": inferior_z,
    }

    orig_h, orig_w = gt_volume_zyx.shape[1], gt_volume_zyx.shape[2]
    target_h, target_w = target_size
    resize_factor_x = target_w / orig_w
    resize_factor_y = target_h / orig_h
    # Same physical conversion as 提示位置/testdatasetGTVp_pos.py, expressed in resized coordinates:
    # original code expands by (cm * 10 / spacing_mm) on the native mask and then scales to 1024;
    # here spacing_mm is divided by the resize factor first, so cm / spacing_cm directly gives 1024 pixels.
    spacing_x_cm = spacing[0] / resize_factor_x / 10.0
    spacing_y_cm = spacing[1] / resize_factor_y / 10.0

    prompt_boxes_by_z = {}
    prompt_records = []
    for prompt_type, z in prompt_z_by_type.items():
        resized_mask = cv2.resize(
            (gt_volume_zyx[z] > 0).astype(np.uint8),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )
        gt_box = get_gt_box_from_mask(resized_mask)
        if gt_box is None:
            raise ValueError(f"{patient_id}: prompt slice {z} has empty GT mask.")

        expanded_box, expand_info = random_expand_box(
            gt_box,
            spacing_x_cm=spacing_x_cm,
            spacing_y_cm=spacing_y_cm,
            upper_limit_cm=upper_limit_cm,
            rng=rng_box,
            image_width=target_w,
            image_height=target_h,
        )
        prompt_boxes_by_z[z] = expanded_box

        prompt_records.append(
            {
                "patient_id": patient_id,
                "upper_limit_cm": round(float(upper_limit_cm), 2),
                "trial": trial,
                "prompt_type": prompt_type,
                "superior_slice_index": superior_z,
                "inferior_slice_index": inferior_z,
                "middle_slice_index": middle_z,
                "box_x0": expanded_box[0],
                "box_y0": expanded_box[1],
                "box_x1": expanded_box[2],
                "box_y1": expanded_box[3],
                **expand_info,
            }
        )

    all_boxes_by_z = {}
    for z in valid_z:
        box = interpolate_boxes(prompt_boxes_by_z, z)
        x0, y0, x1, y1 = box
        x0 = max(0, min(target_w - 1, x0))
        y0 = max(0, min(target_h - 1, y0))
        x1 = max(x0 + 1, min(target_w - 1, x1))
        y1 = max(y0 + 1, min(target_h - 1, y1))
        all_boxes_by_z[int(z)] = [x0, y0, x1, y1]

    return all_boxes_by_z, prompt_records


def load_image_tensor(image_path, target_size):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size[::-1]  # (H, W)
    image = image.resize((target_size[1], target_size[0]), resample=Image.BILINEAR)
    image_np = np.array(image).astype(np.float32)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    return image_tensor, original_size


def predict_slice_with_box(model, image_tensor, box, original_size, device):
    imgs = image_tensor.unsqueeze(0).to(device).float()
    bbox = torch.tensor(box, dtype=torch.float32, device=device).unsqueeze(0)
    input_images = torch.stack([model.preprocess(im) for im in imgs], dim=0)
    image_embeddings = model.image_encoder(input_images)
    sparse_embeddings, dense_embeddings = model.prompt_encoder(points=None, boxes=bbox, masks=None)
    low_res_masks, _ = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    masks = model.postprocess_masks(
        low_res_masks,
        input_size=(int(imgs.shape[-2]), int(imgs.shape[-1])),
        original_size=(int(original_size[0]), int(original_size[1])),
    )
    return (torch.sigmoid(masks)[0, 0] > 0.5).cpu().numpy().astype(np.uint8)


def save_pred_nifti(pred_vol, reference_nii_path, save_path):
    """Save a (H, W, D) uint8 prediction with the reference NIfTI spatial metadata."""
    ref_nii = nib.load(reference_nii_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pred_vol = (pred_vol > 0).astype(np.uint8)
    if pred_vol.shape != ref_nii.shape:
        raise ValueError(f"Prediction shape {pred_vol.shape} does not match reference shape {ref_nii.shape}.")
    nii_img = nib.Nifti1Image(pred_vol, affine=ref_nii.affine, header=ref_nii.header)
    nii_img.set_data_dtype(np.uint8)
    nib.save(nii_img, save_path)


def init_prediction_volume_dhw(reference_nii_path):
    """Match the reference code: accumulate slices as (D, H, W), then transpose to (H, W, D)."""
    ref_nii = nib.load(reference_nii_path)
    shape_hwd = ref_nii.shape
    return np.zeros((shape_hwd[2], shape_hwd[0], shape_hwd[1]), dtype=np.uint8)


def insert_slice_like_reference(volume_dhw, slice_idx, pred_slice_hw):
    """Apply the same orientation correction as 提示位置/test_pos.py before NIfTI assembly."""
    arr = np.rot90(pred_slice_hw, k=3)
    arr = np.fliplr(arr)
    if slice_idx >= volume_dhw.shape[0]:
        raise IndexError(f"Slice index {slice_idx} exceeds volume depth {volume_dhw.shape[0]}.")
    volume_dhw[slice_idx] = (arr > 0).astype(np.uint8)


def finalize_prediction_volume_hwd(volume_dhw):
    return np.transpose(volume_dhw, (1, 2, 0)).astype(np.uint8)


def dice_binary(pred, gt):
    pred = pred > 0
    gt = gt > 0
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def iou_binary(pred, gt):
    pred = pred > 0
    gt = gt > 0
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def hd95_binary(gt, pred, spacing):
    gt = gt > 0
    pred = pred > 0
    if not gt.any() and not pred.any():
        return 0.0
    if not gt.any() or not pred.any():
        return np.nan

    conn = ndi.generate_binary_structure(gt.ndim, 1)
    gt_surface = gt ^ ndi.binary_erosion(gt, structure=conn, iterations=1, border_value=0)
    pred_surface = pred ^ ndi.binary_erosion(pred, structure=conn, iterations=1, border_value=0)
    if not gt_surface.any() or not pred_surface.any():
        return 0.0

    dt_pred = ndi.distance_transform_edt(~pred_surface, sampling=spacing)
    dt_gt = ndi.distance_transform_edt(~gt_surface, sampling=spacing)
    distances = np.concatenate([dt_pred[gt_surface], dt_gt[pred_surface]])
    return float(np.percentile(distances, 95)) if distances.size else 0.0


def asd_mm_evaluate_sum_style(pred_vol, gt_vol, spacing_xyz):
    pred_arr = (pred_vol > 0).astype(np.uint8)
    gt_arr = (gt_vol > 0).astype(np.uint8)
    try:
        asd_voxel = metric.binary.asd(pred_arr, gt_arr)
        return round(float(asd_voxel) * float(np.mean(spacing_xyz)), 2)
    except Exception:
        return 0.0


def choose_middle_slice(valid_z, patient_id, trial, base_seed):
    if len(valid_z) < 3:
        raise ValueError(f"{patient_id}: fewer than 3 positive GTV slices, cannot choose middle slice.")
    middle_seed = base_seed + trial * 10_000 + stable_int(patient_id)
    rng_middle = np.random.default_rng(middle_seed)
    return int(rng_middle.choice(valid_z[1:-1]))


def compute_all_metrics(pred_vol, gt_vol, spacing):
    pred = pred_vol > 0
    gt = gt_vol > 0

    dsc_2d_values = []
    hd95_2d_values = []
    spacing_2d = spacing[:2]
    for z in range(gt.shape[2]):
        gt_slice = gt[:, :, z]
        pred_slice = pred[:, :, z]
        if gt_slice.any() or pred_slice.any():
            dsc_2d_values.append(dice_binary(pred_slice, gt_slice))
        if gt_slice.any() and pred_slice.any():
            hd = hd95_binary(gt_slice, pred_slice, spacing_2d)
            if not np.isnan(hd):
                hd95_2d_values.append(hd)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    sensitivity = np.nan if (tp + fn) == 0 else float(tp / (tp + fn) * 100.0)
    precision = np.nan if (tp + fp) == 0 else float(tp / (tp + fp) * 100.0)

    return {
        "DSC_2D": float(np.nanmean(dsc_2d_values)) if dsc_2d_values else np.nan,
        "DSC_3D": dice_binary(pred, gt),
        "HD95_2D_mm": float(np.nanmean(hd95_2d_values)) if hd95_2d_values else np.nan,
        "HD95_3D_mm": hd95_binary(gt, pred, spacing),
        "IoU": iou_binary(pred, gt),
        "ASD_mm": asd_mm_evaluate_sum_style(pred_vol, gt_vol, spacing),
        "Sensitivity_percent": sensitivity,
        "Precision_percent": precision,
    }


def mean_std_string(values):
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return "nan±nan"
    return f"{np.mean(arr):.2f}±{np.std(arr, ddof=0):.2f}"


def build_patient_summary(trial_metric_records):
    metric_cols = [
        "DSC_2D",
        "DSC_3D",
        "HD95_2D_mm",
        "HD95_3D_mm",
        "IoU",
        "ASD_mm",
        "Sensitivity_percent",
        "Precision_percent",
    ]
    df = pd.DataFrame(trial_metric_records)
    summary_records = []
    if df.empty:
        return summary_records

    for (patient_id, upper_limit_cm), group in df.groupby(["patient_id", "upper_limit_cm"], sort=True):
        row = {"patient_id": patient_id, "upper_limit_cm": upper_limit_cm}
        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[col] = round(float(values.mean()), 2) if values.notna().any() else np.nan
        summary_records.append(row)

    summary_df = pd.DataFrame(summary_records)
    for label, func in [("Mean", "mean"), ("STD", "std")]:
        row = {"patient_id": label, "upper_limit_cm": ""}
        for col in metric_cols:
            values = pd.to_numeric(summary_df[col], errors="coerce")
            if values.notna().any():
                if func == "mean":
                    row[col] = round(float(values.mean()), 2)
                else:
                    row[col] = round(float(values.std(ddof=0)), 2)
            else:
                row[col] = np.nan
        summary_records.append(row)
    return summary_records


def write_excel(prompt_records, trial_metric_records, patient_summary_records, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    metric_cols = [
        "DSC_2D",
        "DSC_3D",
        "HD95_2D_mm",
        "HD95_3D_mm",
        "IoU",
        "ASD_mm",
        "Sensitivity_percent",
        "Precision_percent",
    ]

    prompt_df = pd.DataFrame(prompt_records)
    metrics_df = pd.DataFrame(trial_metric_records)
    summary_df = pd.DataFrame(patient_summary_records)

    for col in ["upper_limit_cm", "expand_left_cm", "expand_right_cm", "expand_top_cm", "expand_bottom_cm"]:
        if col in prompt_df:
            prompt_df[col] = pd.to_numeric(prompt_df[col], errors="coerce").round(2)
    if not metrics_df.empty:
        for col in metric_cols + ["upper_limit_cm"]:
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce").round(2)

    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        prompt_df.to_excel(writer, sheet_name="Prompt_Info", index=False)
        metrics_df.to_excel(writer, sheet_name="Sum_Eval_Metrics", index=False)
        summary_df.to_excel(writer, sheet_name="Mean_patient", index=False)


def print_cohort_mean_std(upper_limit_cm, trial_metric_records):
    metric_cols = [
        "DSC_2D",
        "DSC_3D",
        "HD95_2D_mm",
        "HD95_3D_mm",
        "IoU",
        "ASD_mm",
        "Sensitivity_percent",
        "Precision_percent",
    ]
    df = pd.DataFrame(trial_metric_records)
    print(f"\nCohort-level mean±std for upper {fmt_cm(upper_limit_cm)} cm")
    if df.empty:
        print("  No metric records.")
        return
    for col in metric_cols:
        print(f"  {col}: {mean_std_string(df[col])}")


def main():
    args = parse_args()
    device = torch.device(args.device)
    target_size = tuple(args.target_size)
    pred_nii_root = os.path.join(args.output_root, "Pred_nii")
    excel_root = os.path.join(args.output_root, "Excel")

    print("\n================ Prompt Variability Evaluation ================")
    print(f"Fine-tuned checkpoint: {args.ckpt_path}")
    print(f"Official SAM checkpoint: {args.sam_checkpoint}")
    print(f"CSV: {args.csv_path}")
    print(f"NIfTI dir: {args.nii_dir}")
    print(f"Output root: {args.output_root}")
    print(f"Upper limits: {args.upper_limits_cm}")
    print(f"Trials per patient: {args.n_trials}")
    print(f"Seed: {args.seed}")
    print("==============================================================\n")

    rows_by_patient = read_test_csv(args.csv_path, args.root_dir, args.image_dir)
    model = load_single_sam_model(args.ckpt_path, args.sam_checkpoint, args.model_type, device)

    for upper_idx, upper_limit_cm in enumerate(args.upper_limits_cm):
        print(f"\n=== Processing expansion upper limit: {upper_limit_cm} cm ===")
        prompt_records = []
        trial_metric_records = []

        for patient_id in tqdm(sorted(rows_by_patient.keys()), desc=f"upper {fmt_cm(upper_limit_cm)}cm"):
            patient_dir = os.path.join(args.nii_dir, patient_id)
            gt_nii_path = os.path.join(patient_dir, "GTVp.nii.gz")
            image_nii_path = os.path.join(patient_dir, "image.nii.gz")
            if not os.path.exists(gt_nii_path):
                print(f"Skip {patient_id}: missing {gt_nii_path}")
                continue
            if not os.path.exists(image_nii_path):
                print(f"Skip {patient_id}: missing {image_nii_path}")
                continue

            gt_sitk = sitk.ReadImage(gt_nii_path)
            gt_volume_zyx = (sitk.GetArrayFromImage(gt_sitk) > 0).astype(np.uint8)
            valid_z = get_positive_slices(gt_volume_zyx)
            if len(valid_z) < 3:
                print(f"Skip {patient_id}: fewer than 3 positive GTV slices.")
                continue

            gt_nib = nib.load(gt_nii_path)
            gt_vol = (gt_nib.get_fdata() > 0).astype(np.uint8)
            spacing = gt_nib.header.get_zooms()[:3]
            patient_images = rows_by_patient[patient_id]

            for trial in range(1, args.n_trials + 1):
                middle_z = choose_middle_slice(valid_z, patient_id, trial, args.seed)
                box_seed = args.seed + upper_idx * 1_000_000 + trial * 10_000 + stable_int(patient_id)
                rng_box = np.random.default_rng(box_seed)

                try:
                    boxes_by_z, records = build_trial_boxes_and_prompt_records(
                        patient_id=patient_id,
                        gt_volume_zyx=gt_volume_zyx,
                        valid_z=valid_z,
                        spacing=gt_sitk.GetSpacing(),
                        target_size=target_size,
                        upper_limit_cm=upper_limit_cm,
                        trial=trial,
                        middle_z=middle_z,
                        rng_box=rng_box,
                    )
                except Exception as exc:
                    print(f"Skip {patient_id} trial {trial}: {exc}")
                    continue

                prompt_records.extend(records)
                pred_volume_dhw = init_prediction_volume_dhw(image_nii_path)

                with torch.no_grad():
                    for z in valid_z:
                        if z not in patient_images:
                            print(f"Missing image for {patient_id} slice {z}, keep empty prediction.")
                            continue
                        image_tensor, original_size = load_image_tensor(patient_images[z], target_size)
                        pred_slice = predict_slice_with_box(
                            model=model,
                            image_tensor=image_tensor,
                            box=boxes_by_z[z],
                            original_size=original_size,
                            device=device,
                        )
                        try:
                            insert_slice_like_reference(pred_volume_dhw, z, pred_slice)
                        except IndexError as exc:
                            print(f"{patient_id}: {exc}; keep empty prediction for this slice.")

                pred_vol = finalize_prediction_volume_hwd(pred_volume_dhw)

                upper_dir = os.path.join(pred_nii_root, f"upper_{fmt_cm(upper_limit_cm)}cm", patient_id)
                nii_name = f"GTVp_{patient_numeric_id(patient_id)}_{trial}.nii.gz"
                save_pred_nifti(pred_vol, image_nii_path, os.path.join(upper_dir, nii_name))

                metrics = compute_all_metrics(pred_vol, gt_vol, spacing)
                trial_metric_records.append(
                    {
                        "patient_id": patient_id,
                        "upper_limit_cm": round(float(upper_limit_cm), 2),
                        "trial": trial,
                        **metrics,
                    }
                )

        patient_summary_records = build_patient_summary(trial_metric_records)
        excel_path = os.path.join(excel_root, f"prompt_variability_{fmt_cm(upper_limit_cm)}cm.xlsx")
        write_excel(prompt_records, trial_metric_records, patient_summary_records, excel_path)
        print(f"Saved Excel: {excel_path}")
        print_cohort_mean_std(upper_limit_cm, trial_metric_records)


if __name__ == "__main__":
    main()
