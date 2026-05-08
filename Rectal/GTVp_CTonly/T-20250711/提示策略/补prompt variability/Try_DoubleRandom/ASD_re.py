import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from medpy import metric


"""
Recompute ASD values in prompt variability Excel sheets using the same method as evaluate_sum.py:
    asd_mm = round(metric.binary.asd(pred_array, gt_array) * np.mean(gt_spacing), 2)

Expected prediction layout (from prompt_variability_random.py):
    {pred_root}/upper_{upper_limit_cm:.1f}cm/{patient_id}/GTVp_{patient_num}_{trial}.nii.gz
Expected GT layout:
    {nii_dir}/{patient_id}/GTVp.nii.gz
"""


def patient_numeric_id(patient_id: str) -> str:
    m = re.search(r"\d+", str(patient_id))
    return m.group(0).zfill(3) if m else str(patient_id)


def fmt_upper(value: float) -> str:
    return f"{float(value):.1f}"


def compute_asd_mm(gt_path: Path, pred_path: Path) -> float:
    gt_img = sitk.ReadImage(str(gt_path))
    pred_img = sitk.ReadImage(str(pred_path))

    gt_array = (sitk.GetArrayFromImage(gt_img) > 0).astype(np.uint8)
    pred_array = (sitk.GetArrayFromImage(pred_img) > 0).astype(np.uint8)
    spacing = gt_img.GetSpacing()

    try:
        asd_voxel = metric.binary.asd(pred_array, gt_array)
        return round(float(asd_voxel) * float(np.mean(spacing)), 2)
    except Exception:
        return 0.0


def update_mean_patient_sheet(mean_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    if mean_df.empty:
        return mean_df

    case_mask = ~mean_df["patient_id"].astype(str).isin(["Mean", "STD"])
    case_rows = mean_df.loc[case_mask].copy()

    if "ASD_mm" not in case_rows.columns:
        return mean_df

    grouped = (
        metrics_df.groupby(["patient_id", "upper_limit_cm"], dropna=False)["ASD_mm"]
        .mean()
        .round(2)
        .reset_index()
    )

    merged = case_rows.drop(columns=["ASD_mm"], errors="ignore").merge(
        grouped,
        on=["patient_id", "upper_limit_cm"],
        how="left",
    )

    out = mean_df.copy()
    out.loc[case_mask, "ASD_mm"] = merged["ASD_mm"].to_numpy()

    all_case_asd = pd.to_numeric(out.loc[case_mask, "ASD_mm"], errors="coerce")
    mean_val = round(float(all_case_asd.mean()), 2) if all_case_asd.notna().any() else np.nan
    std_val = round(float(all_case_asd.std(ddof=0)), 2) if all_case_asd.notna().any() else np.nan

    mean_row_mask = out["patient_id"].astype(str) == "Mean"
    std_row_mask = out["patient_id"].astype(str) == "STD"
    if mean_row_mask.any():
        out.loc[mean_row_mask, "ASD_mm"] = mean_val
    if std_row_mask.any():
        out.loc[std_row_mask, "ASD_mm"] = std_val

    return out


def process_one_excel(excel_path: Path, pred_root: Path, nii_dir: Path, overwrite: bool = False) -> Path:
    try:
        xls = pd.ExcelFile(excel_path, engine="openpyxl")
    except Exception as e:
        raise ValueError(f"Failed to open Excel file: {excel_path}. It may be corrupted or not a real .xlsx. {e}") from e

    if "Sum_Eval_Metrics" not in xls.sheet_names:
        raise ValueError(f"{excel_path} missing sheet: Sum_Eval_Metrics")

    metrics_df = pd.read_excel(excel_path, sheet_name="Sum_Eval_Metrics", engine="openpyxl")
    if "ASD_mm" not in metrics_df.columns:
        raise ValueError(f"{excel_path} missing column ASD_mm in Sum_Eval_Metrics")

    required_cols = {"patient_id", "upper_limit_cm", "trial"}
    missing = required_cols - set(metrics_df.columns)
    if missing:
        raise ValueError(f"{excel_path} missing required columns: {sorted(missing)}")

    new_asd = []
    not_found = 0

    for _, row in metrics_df.iterrows():
        patient_id = str(row["patient_id"])
        upper = float(row["upper_limit_cm"])
        trial = int(row["trial"])

        pred_path = (
            pred_root
            / f"upper_{fmt_upper(upper)}cm"
            / patient_id
            / f"GTVp_{patient_numeric_id(patient_id)}_{trial}.nii.gz"
        )
        gt_path = nii_dir / patient_id / "GTVp.nii.gz"

        if (not pred_path.exists()) or (not gt_path.exists()):
            new_asd.append(np.nan)
            not_found += 1
            continue

        new_asd.append(compute_asd_mm(gt_path, pred_path))

    metrics_df["ASD_mm"] = pd.to_numeric(pd.Series(new_asd), errors="coerce").round(2)

    mean_df = None
    if "Mean_patient" in xls.sheet_names:
        mean_df = pd.read_excel(excel_path, sheet_name="Mean_patient", engine="openpyxl")
        if {"patient_id", "upper_limit_cm"}.issubset(set(mean_df.columns)):
            mean_df = update_mean_patient_sheet(mean_df, metrics_df)

    passthrough_sheets = {}
    for sheet in xls.sheet_names:
        if sheet not in {"Sum_Eval_Metrics", "Mean_patient"}:
            passthrough_sheets[sheet] = pd.read_excel(excel_path, sheet_name=sheet, engine="openpyxl")

    output_path = excel_path if overwrite else excel_path.with_name(excel_path.stem + "_ASD_fixed.xlsx")
    write_path = output_path
    if overwrite:
        write_path = excel_path.with_name(excel_path.stem + ".__tmp_asd_write__.xlsx")

    with pd.ExcelWriter(write_path, engine="openpyxl") as writer:
        for sheet in xls.sheet_names:
            if sheet == "Sum_Eval_Metrics":
                metrics_df.to_excel(writer, sheet_name=sheet, index=False)
            elif sheet == "Mean_patient" and mean_df is not None:
                mean_df.to_excel(writer, sheet_name=sheet, index=False)
            elif sheet == "Mean_patient" and mean_df is None:
                pd.read_excel(excel_path, sheet_name=sheet, engine="openpyxl").to_excel(
                    writer, sheet_name=sheet, index=False
                )
            else:
                passthrough_sheets[sheet].to_excel(writer, sheet_name=sheet, index=False)

    if overwrite:
        write_path.replace(excel_path)

    total = len(metrics_df)
    fixed = int(pd.to_numeric(metrics_df["ASD_mm"], errors="coerce").notna().sum())
    print(f"[OK] {excel_path}")
    print(f"     rows={total}, recalculated={fixed}, missing_pred_or_gt={not_found}")
    print(f"     output={output_path}")
    return output_path


def discover_excels(excel_root: Path):
    return sorted(excel_root.glob("*.xlsx"))


def main():
    parser = argparse.ArgumentParser(description="Recalculate ASD in prompt variability Excel sheets.")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/home/wusi/segment-anything/SAMdata/Rectal/20251224_GTVp/TestResults/Prompt_variability"),
        help="Same as prompt_variability_random.py --output_root.",
    )
    parser.add_argument(
        "--pred_root",
        type=Path,
        default=None,
        help="Prediction root. Default: {output_root}/Pred_nii",
    )
    parser.add_argument(
        "--excel_root",
        type=Path,
        default=None,
        help="Excel root. Default: {output_root}/Excel",
    )
    parser.add_argument(
        "--nii_dir",
        type=Path,
        default=Path("/home/wusi/segment-anything/SAMdata/Rectal/20251224_GTVp/datanii/test_nii"),
        help="Same as prompt_variability_random.py --nii_dir (GT root).",
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=None,
        help="Optional single excel file path to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the original xlsx instead of writing *_ASD_fixed.xlsx",
    )
    args = parser.parse_args()

    pred_root = args.pred_root if args.pred_root is not None else args.output_root / "Pred_nii"
    excel_root = args.excel_root if args.excel_root is not None else args.output_root / "Excel"

    if args.excel is not None:
        excel_list = [args.excel]
    else:
        excel_list = discover_excels(excel_root)

    if not excel_list:
        raise FileNotFoundError(f"No xlsx found. excel_root={excel_root}")

    print("=== ASD Recalculation (evaluate_sum.py style) ===")
    print(f"pred_root:  {pred_root}")
    print(f"excel_root: {excel_root}")
    print(f"nii_dir:    {args.nii_dir}")
    print(f"overwrite:  {args.overwrite}")
    print(f"files:      {len(excel_list)}")

    for excel_path in excel_list:
        try:
            process_one_excel(excel_path, pred_root=pred_root, nii_dir=args.nii_dir, overwrite=args.overwrite)
        except Exception as e:
            print(f"[FAILED] {excel_path}")
            print(f"         {e}")


if __name__ == "__main__":
    main()
