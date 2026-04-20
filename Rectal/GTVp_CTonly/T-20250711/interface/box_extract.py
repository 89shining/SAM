import os
import numpy as np
import SimpleITK as sitk


def extract_3_boxes(mask_nii_path, expand_cm=0.5):
    # ---------- 1. 读取 mask ----------
    mask_img = sitk.ReadImage(mask_nii_path)
    mask_np = sitk.GetArrayFromImage(mask_img)  # (Z,H,W)
    spacing = mask_img.GetSpacing()  # (sx, sy, sz) in mm

    Z, H, W = mask_np.shape

    # ---------- 2. 找有效层 ----------
    valid_z = []
    area_list = []

    for z in range(Z):
        area = np.sum(mask_np[z] > 0)
        if area > 0:
            valid_z.append(z)
            area_list.append(area)

    if len(valid_z) < 3:
        raise ValueError("有效层不足3层，无法生成3个提示框")

    # ---------- 3. 计算 top / mid / bottom ----------
    top_z = valid_z[0]
    bottom_z = valid_z[-1]

    mid_idx = len(valid_z) // 2
    mid_z = valid_z[mid_idx]

    key_z_list = [top_z, mid_z, bottom_z]

    # ---------- 4. cm → pixel ----------
    sx, sy = spacing[0], spacing[1]
    expand_x = int(round((expand_cm * 10) / sx))  # cm→mm→pixel
    expand_y = int(round((expand_cm * 10) / sy))

    # ---------- 5. 提取框 ----------
    boxes = []

    for z in key_z_list:
        mask = mask_np[z] > 0
        ys, xs = np.where(mask)

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        # 外扩
        x0 = max(0, x0 - expand_x)
        y0 = max(0, y0 - expand_y)
        x1 = min(W - 1, x1 + expand_x)
        y1 = min(H - 1, y1 + expand_y)

        boxes.append({
            "slice_index": int(z),
            "x0": int(x0),
            "y0": int(y0),
            "x1": int(x1),
            "y1": int(y1)
        })

    return {
        "mask_path": mask_nii_path,
        "expand_cm": expand_cm,
        "boxes": boxes
    }


if __name__ == "__main__":
    mask_path = r"C:\Users\dell\Desktop\GTVp.nii.gz"

    result = extract_3_boxes(mask_path, expand_cm=0.5)

    print("\n=== 提取结果 ===")
    for b in result["boxes"]:
        print(b)

    # 可选：保存为 json 给前端
    # save_json = mask_path.replace(".nii.gz", "_boxes.json")
    # with open(save_json, "w") as f:
    #     json.dump(result, f, indent=4)

    # print(f"\n已保存到: {save_json}")