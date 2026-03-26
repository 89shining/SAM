"""
python check_empty_prompt_slices.py \
  --nii_root_dir /home/wusi/segment-anything/SAMdata/Rectal/20260325_CTV/Cropdatanii/train_nii \
  --gt_name CTV.nii.gz \
  --prompt_name prompt.nii.gz

"""

import os
import argparse
import numpy as np
import SimpleITK as sitk


def list_patients(root_dir):
    patients = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    return sorted(
        patients,
        key=lambda x: int(x.lstrip("p_")) if x.startswith("p_") and x.lstrip("p_").isdigit() else x,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nii_root_dir", required=True, help="Root dir containing p_xxx folders")
    parser.add_argument("--gt_name", default="CTV.nii.gz")
    parser.add_argument("--prompt_name", default="prompt.nii.gz")
    args = parser.parse_args()

    total_gt_pos_slices = 0
    empty_prompt_on_gt_pos = []
    missing_prompt_patients = []

    for pid in list_patients(args.nii_root_dir):
        pdir = os.path.join(args.nii_root_dir, pid)
        gt_path = os.path.join(pdir, args.gt_name)
        prompt_path = os.path.join(pdir, args.prompt_name)

        if not os.path.exists(gt_path):
            continue

        if not os.path.exists(prompt_path):
            missing_prompt_patients.append(pid)
            continue

        gt_vol = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))  # (Z,H,W)
        prompt_vol = sitk.GetArrayFromImage(sitk.ReadImage(prompt_path))  # (Z,H,W)

        if gt_vol.shape != prompt_vol.shape:
            print(f"[SHAPE_MISMATCH] {pid}: GT {gt_vol.shape}, PROMPT {prompt_vol.shape}")
            continue

        for z in range(gt_vol.shape[0]):
            gt_pos = np.max(gt_vol[z]) > 0
            if not gt_pos:
                continue
            total_gt_pos_slices += 1

            prompt_pos = np.max(prompt_vol[z]) > 0
            if not prompt_pos:
                empty_prompt_on_gt_pos.append((pid, z))

    print("\n===== CHECK SUMMARY =====")
    print(f"GT-positive slices: {total_gt_pos_slices}")
    print(f"Empty prompt on GT-positive slices: {len(empty_prompt_on_gt_pos)}")
    print(f"Patients missing prompt file: {len(missing_prompt_patients)}")

    if missing_prompt_patients:
        print("\n[Missing prompt files]")
        for pid in missing_prompt_patients:
            print(f"  {pid}")

    if empty_prompt_on_gt_pos:
        print("\n[Empty prompt slice list]")
        for pid, z in empty_prompt_on_gt_pos:
            print(f"  {pid}, z={z}")


if __name__ == "__main__":
    main()
