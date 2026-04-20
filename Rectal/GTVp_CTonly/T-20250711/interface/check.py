import os
import numpy as np
import SimpleITK as sitk


def read_img(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # (Z,H,W)
    return img, arr


def get_nonzero_slices(arr):
    nz = [z for z in range(arr.shape[0]) if np.any(arr[z] > 0)]
    if len(nz) == 0:
        return None
    return {
        "first_slice": nz[0],
        "last_slice": nz[-1],
        "num_nonzero_slices": len(nz),
    }


def summarize(path, is_mask=False):
    img, arr = read_img(path)
    info = {
        "path": path,
        "shape_zyx": arr.shape,
        "size_xyz": img.GetSize(),
        "spacing_xyz": img.GetSpacing(),
        "origin_xyz": img.GetOrigin(),
        "direction": img.GetDirection(),
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }

    if is_mask:
        uniq = np.unique(arr)
        info["unique_values_preview"] = uniq[:20]
        info["num_unique_values"] = len(uniq)
        info["nonzero_slice_info"] = get_nonzero_slices(arr)

    return info


def compare_image_and_mask(image_path, mask_path):
    image_info = summarize(image_path, is_mask=False)
    mask_info = summarize(mask_path, is_mask=True)

    print("\n[Input image]")
    for k, v in image_info.items():
        print(f"{k}: {v}")

    print("\n[Output mask]")
    for k, v in mask_info.items():
        print(f"{k}: {v}")

    same_shape = image_info["shape_zyx"] == mask_info["shape_zyx"]
    same_size = image_info["size_xyz"] == mask_info["size_xyz"]
    same_spacing = np.allclose(image_info["spacing_xyz"], mask_info["spacing_xyz"], atol=1e-6)
    same_origin = np.allclose(image_info["origin_xyz"], mask_info["origin_xyz"], atol=1e-6)
    same_direction = np.allclose(image_info["direction"], mask_info["direction"], atol=1e-6)

    print("\n[Check]")
    print(f"same_shape_zyx: {same_shape}")
    print(f"same_size_xyz: {same_size}")
    print(f"same_spacing: {same_spacing}")
    print(f"same_origin: {same_origin}")
    print(f"same_direction: {same_direction}")

    if same_shape and same_size and same_spacing and same_origin and same_direction:
        print("=> 输入图像和输出mask完全对齐")
    elif same_shape:
        print("=> shape一致，但几何信息不完全一致")
    else:
        print("=> 连shape都不一致，需要先处理后再用于界面对齐")

    uniq = mask_info["unique_values_preview"]
    if len(uniq) <= 2 and set(np.array(uniq).astype(float).tolist()).issubset({0.0, 1.0}):
        print("=> mask看起来是二值的(0/1)")
    elif len(uniq) <= 2 and set(np.array(uniq).astype(float).tolist()).issubset({0.0, 255.0}):
        print("=> mask看起来是二值的(0/255)")
    else:
        print("=> mask不一定是标准二值，建议再确认")


if __name__ == "__main__":
    image_path = r"C:\Users\dell\Desktop\image.nii.gz"
    mask_path = r"C:\Users\dell\Desktop\GTVp_000.nii.gz"
    compare_image_and_mask(image_path, mask_path)