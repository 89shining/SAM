import argparse
import os
from typing import Tuple, List

import numpy as np


DEFAULT_PROB_DIR = (
    "/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_results/"
    "Dataset014_RectalCTV60pCrop/nnUNetTrainer__nnUNetPlans__3d_fullres/"
    "testResult_5folds_probability"
)
DEFAULT_SAVE_DIR = (
    "/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_results/"
    "Dataset014_RectalCTV60pCrop/nnUNetTrainer__nnUNetPlans__3d_fullres/"
    "testResult_5folds_uncertainty/3D_entropy"
)


def compute_entropy_map(
    prob_array: np.ndarray,
    normalize: bool = True,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Compute voxel-wise binary entropy from probability map.

    Args:
        prob_array: array in [0, 1], shape [Z,H,W]
        normalize: whether to divide by log(2), so output is in [0, 1]
        eps: numerical stability

    Returns:
        entropy map, float32, shape [Z,H,W]
    """
    prob_array = np.clip(prob_array.astype(np.float32), 0.0, 1.0)

    entropy = -(
        prob_array * np.log(prob_array + eps) +
        (1.0 - prob_array) * np.log(1.0 - prob_array + eps)
    )

    if normalize:
        entropy = entropy / np.log(2.0)

    return entropy.astype(np.float32)


def logits_to_prob(x: np.ndarray) -> np.ndarray:
    """
    Convert logits to probability using sigmoid.
    """
    x = x.astype(np.float32)
    x = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x))


def to_probability_map(raw: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Convert input array to probability map if needed.

    Returns:
        prob_array: float32 array in [0, 1]
        mode: description string
    """
    arr = np.asarray(raw, dtype=np.float32)

    if not np.isfinite(arr).all():
        raise ValueError("Input map contains NaN or Inf.")

    vmin = float(arr.min())
    vmax = float(arr.max())

    # Already probability
    if 0.0 <= vmin and vmax <= 1.0:
        return np.clip(arr, 0.0, 1.0), "prob_0_1"

    # Otherwise treat as logits
    return logits_to_prob(arr), "logits_sigmoid"


def list_supported_files(prob_dir: str) -> List[str]:
    """
    List all .npz files in input directory.
    """
    files = [f for f in os.listdir(prob_dir) if f.endswith(".npz")]
    files.sort()
    return files


def pick_npz_array(npz_obj) -> Tuple[np.ndarray, str]:
    """
    Pick the most likely prediction array from npz.
    """
    preferred_keys = [
        "softmax",
        "probabilities",
        "probability",
        "pred",
        "prediction",
        "logits"
    ]

    for k in preferred_keys:
        if k in npz_obj.files:
            return np.asarray(npz_obj[k]), k

    if len(npz_obj.files) == 1:
        k = npz_obj.files[0]
        return np.asarray(npz_obj[k]), k

    raise KeyError(
        f"Cannot determine array key from npz keys={npz_obj.files}. "
        f"Expected one of {preferred_keys}."
    )


def to_single_channel_volume(arr: np.ndarray, class_idx: int) -> Tuple[np.ndarray, str]:
    """
    Convert array to single-channel 3D volume.

    Supported:
        [Z,H,W]
        [C,Z,H,W]
        [Z,H,W,C]

    Returns:
        volume: [Z,H,W]
        format_desc: description of detected layout
    """
    arr = np.asarray(arr)

    if arr.ndim == 3:
        return arr.astype(np.float32), "3d"

    if arr.ndim != 4:
        raise ValueError(
            f"Unsupported array shape: {arr.shape}. "
            f"Expected [Z,H,W], [C,Z,H,W], or [Z,H,W,C]."
        )

    # Heuristic 1: channel-first
    if arr.shape[0] <= 10:
        c = arr.shape[0]
        use_cls = class_idx if c > 1 else 0
        if use_cls < 0 or use_cls >= c:
            raise IndexError(
                f"class_idx={use_cls} out of range for channel-first array with channels={c}"
            )
        return arr[use_cls].astype(np.float32), "4d_channel_first"

    # Heuristic 2: channel-last
    if arr.shape[-1] <= 10:
        c = arr.shape[-1]
        use_cls = class_idx if c > 1 else 0
        if use_cls < 0 or use_cls >= c:
            raise IndexError(
                f"class_idx={use_cls} out of range for channel-last array with channels={c}"
            )
        return arr[..., use_cls].astype(np.float32), "4d_channel_last"

    raise ValueError(
        f"Cannot determine channel dimension from shape {arr.shape}. "
        "Expected one channel dimension <= 10."
    )


def save_entropy_npz(
    save_path: str,
    entropy_array: np.ndarray,
    source_key: str,
    input_mode: str,
    array_format: str,
    class_idx: int,
    normalize: bool
) -> None:
    """
    Save entropy result to compressed npz.
    """
    np.savez_compressed(
        save_path,
        entropy=entropy_array.astype(np.float32),
        source_key=np.array(source_key),
        input_mode=np.array(input_mode),
        array_format=np.array(array_format),
        class_idx=np.array(class_idx, dtype=np.int32),
        normalized=np.array(int(normalize), dtype=np.int32),
    )


def batch_generate_3d_entropy(
    prob_dir: str,
    save_dir: str,
    normalize: bool = True,
    class_idx: int = 1
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    files = list_supported_files(prob_dir)
    if len(files) == 0:
        raise RuntimeError(f"No .npz files found in: {prob_dir}")

    print(f"[Info] Found {len(files)} files")
    print(f"[Info] Input : {prob_dir}")
    print(f"[Info] Output: {save_dir}")
    print(f"[Info] Normalize: {normalize}")
    print(f"[Info] Class idx : {class_idx}")

    failed_cases = []

    for i, fname in enumerate(files, start=1):
        prob_path = os.path.join(prob_dir, fname)
        save_path = os.path.join(save_dir, fname)

        try:
            with np.load(prob_path, allow_pickle=False) as npz_obj:
                raw_arr, src_key = pick_npz_array(npz_obj)

            raw_array, array_format = to_single_channel_volume(raw_arr, class_idx=class_idx)
            prob_array, mode = to_probability_map(raw_array)
            entropy_array = compute_entropy_map(prob_array, normalize=normalize)

            save_entropy_npz(
                save_path=save_path,
                entropy_array=entropy_array,
                source_key=src_key,
                input_mode=mode,
                array_format=array_format,
                class_idx=class_idx,
                normalize=normalize,
            )

            print(
                f"[{i:03d}/{len(files):03d}] Done: {fname} | "
                f"src_key={src_key} | format={array_format} | mode={mode} | "
                f"raw_range=({float(raw_array.min()):.4f}, {float(raw_array.max()):.4f}) | "
                f"entropy_range=({float(entropy_array.min()):.4f}, {float(entropy_array.max()):.4f})"
            )

        except Exception as e:
            failed_cases.append((fname, str(e)))
            print(f"[ERROR] {fname}: {e}")

    print("\n[Summary]")
    print(f"Success: {len(files) - len(failed_cases)} / {len(files)}")
    print(f"Failed : {len(failed_cases)} / {len(files)}")

    if failed_cases:
        print("[Failed cases]")
        for fname, err in failed_cases:
            print(f"  - {fname}: {err}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D entropy uncertainty maps from nnUNet probability/logit npz files."
    )
    parser.add_argument(
        "--prob_dir",
        type=str,
        default=DEFAULT_PROB_DIR,
        help="Directory of input probability/logit npz files."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=DEFAULT_SAVE_DIR,
        help="Directory to save entropy npz files."
    )
    parser.add_argument(
        "--class_idx",
        type=int,
        default=1,
        help="Foreground class index when input is multi-channel."
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable entropy normalization by log(2)."
    )

    args = parser.parse_args()

    normalize = not args.no_normalize

    batch_generate_3d_entropy(
        prob_dir=args.prob_dir,
        save_dir=args.save_dir,
        normalize=normalize,
        class_idx=args.class_idx,
    )


if __name__ == "__main__":
    main()