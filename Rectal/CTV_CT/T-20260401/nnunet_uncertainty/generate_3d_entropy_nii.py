import os
import argparse
import numpy as np
import SimpleITK as sitk


DEFAULT_NPZ_DIR = (
    "/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_results/"
    "Dataset014_RectalCTV60pCrop/nnUNetTrainer__nnUNetPlans__3d_fullres/"
    "testResult_5folds_uncertainty/3D_entropy"
)

# 这里放与你的 npz 一一对应的参考 nii
DEFAULT_REF_DIR = (
    "/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/"
    "Dataset014_RectalCTV60pCrop/labelsTs"
)

DEFAULT_SAVE_DIR = (
    "/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_results/"
    "Dataset014_RectalCTV60pCrop/nnUNetTrainer__nnUNetPlans__3d_fullres/"
    "testResult_5folds_uncertainty/3D_entropy_nii"
)


def find_reference_nii(ref_dir: str, base_name: str) -> str:
    """
    根据 base_name 在 ref_dir 中寻找对应的 nii/nii.gz 文件
    """
    candidates = [
        os.path.join(ref_dir, base_name + ".nii.gz"),
        os.path.join(ref_dir, base_name + ".nii"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Reference nii not found for base_name={base_name} in {ref_dir}")


def npz_to_nii_one(npz_path: str, ref_nii_path: str, save_path: str, key: str = "entropy"):
    """
    将一个 npz 文件中的 3D 数组转为 nii.gz，并复制参考图像的空间信息
    """
    with np.load(npz_path, allow_pickle=False) as data:
        if key not in data.files:
            raise KeyError(f"Key '{key}' not found in {npz_path}. Available keys: {data.files}")
        arr = data[key]

    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape={arr.shape} from {npz_path}")

    ref_img = sitk.ReadImage(ref_nii_path)
    ref_arr = sitk.GetArrayFromImage(ref_img)

    if tuple(arr.shape) != tuple(ref_arr.shape):
        raise ValueError(
            f"Shape mismatch for {os.path.basename(npz_path)}: "
            f"npz shape={arr.shape}, ref nii shape={ref_arr.shape}"
        )

    out_img = sitk.GetImageFromArray(arr)
    out_img.CopyInformation(ref_img)

    sitk.WriteImage(out_img, save_path)
    print(f"[Done] {os.path.basename(npz_path)} -> {save_path}")


def batch_npz_to_nii(npz_dir: str, ref_dir: str, save_dir: str, key: str = "entropy"):
    os.makedirs(save_dir, exist_ok=True)

    npz_files = [f for f in os.listdir(npz_dir) if f.endswith(".npz")]
    npz_files.sort()

    if len(npz_files) == 0:
        raise RuntimeError(f"No npz files found in {npz_dir}")

    failed = []

    for i, fname in enumerate(npz_files, start=1):
        npz_path = os.path.join(npz_dir, fname)

        # 去掉 .npz 后缀作为病例名
        base_name = os.path.splitext(fname)[0]

        try:
            ref_nii_path = find_reference_nii(ref_dir, base_name)
            save_path = os.path.join(save_dir, base_name + ".nii.gz")
            npz_to_nii_one(npz_path, ref_nii_path, save_path, key=key)
            print(f"[{i:03d}/{len(npz_files):03d}] OK: {fname}")
        except Exception as e:
            failed.append((fname, str(e)))
            print(f"[ERROR] {fname}: {e}")

    print("\n[Summary]")
    print(f"Success: {len(npz_files) - len(failed)} / {len(npz_files)}")
    print(f"Failed : {len(failed)} / {len(npz_files)}")

    if failed:
        print("[Failed cases]")
        for fname, err in failed:
            print(f"  - {fname}: {err}")


def main():
    parser = argparse.ArgumentParser(description="Convert entropy npz files to nii.gz using reference nii metadata.")
    parser.add_argument("--npz_dir", type=str, default=DEFAULT_NPZ_DIR, help="Directory of entropy npz files.")
    parser.add_argument("--ref_dir", type=str, default=DEFAULT_REF_DIR, help="Directory of reference nii files.")
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR, help="Directory to save nii.gz files.")
    parser.add_argument("--key", type=str, default="entropy", help="Key in npz to convert.")
    args = parser.parse_args()

    batch_npz_to_nii(
        npz_dir=args.npz_dir,
        ref_dir=args.ref_dir,
        save_dir=args.save_dir,
        key=args.key,
    )


if __name__ == "__main__":
    main()