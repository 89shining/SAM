# -*- coding: utf-8 -*-
"""
Clinical SAM - 本地 CPU 测试版
=============================

用途：
1. 从指定路径读取 image.nii.gz（原始 3D CT）
2. 从指定路径读取 boxes.json（医生手动画的 3 个 box）
3. 对 3 个 box 之间逐层进行线性插值
4. 使用微调后的 SAM ViT-B 逐层预测
5. 在本代码所在目录输出 GTVp.nii.gz（二值 0/1 mask）

当前版本仅用于本地测试：
- 输入 image.nii.gz / boxes.json 可以放在任意位置
- 只需要在下面“路径配置”里填写它们的完整路径
- 输出暂时固定到本代码所在目录
- 默认 CPU 推理，不要求 GPU

boxes.json 格式：
{
    "boxes": [
        {"slice_index": 32, "x0": 240, "y0": 264, "x1": 269, "y1": 292},
        {"slice_index": 37, "x0": 232, "y0": 258, "x1": 274, "y1": 299},
        {"slice_index": 42, "x0": 241, "y0": 269, "x1": 274, "y1": 303}
    ]
}

注意：
- slice_index 为 NIfTI z 方向层号，从 0 开始
- x/y 坐标为“原始 axial CT 图像”的像素坐标
- (x0, y0) = 左上角
- (x1, y1) = 右下角
- 不进行 0.5 cm 外扩
"""

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
from segment_anything import sam_model_registry


# ============================================================
# 0. 路径配置 —— 你本地测试时主要改这里
# ============================================================

# 代码所在目录
BASE_DIR = Path(__file__).resolve().parent

# -------- 输入文件：可以放在任意位置 --------
# 请改成你自己电脑上的实际完整路径
IMAGE_PATH = Path(r"D:\ClinicalSAM\image.nii.gz")
BOXES_PATH = Path(r"D:\ClinicalSAM\boxes.json")

# -------- 模型权重：建议和本代码放在同一个文件夹 --------
SAM_CHECKPOINT = BASE_DIR / "sam_vit_b_01ec64.pth"
FINETUNED_CHECKPOINT = BASE_DIR / "ImageEncoder_finetune_fold4_best.pth"

# -------- 当前本地测试输出 --------
# 暂时直接输出到代码所在目录
OUTPUT_PATH = BASE_DIR / "GTVp.nii.gz"

# -------- 模型配置 --------
MODEL_TYPE = "vit_b"

# 本地电脑没有 GPU：固定使用 CPU
DEVICE = "cpu"

# 二值化阈值
THRESHOLD = 0.5

# SAM 输入尺寸
TARGET_SIZE = 1024

# 必须与训练/原测试数据预处理保持一致
WINDOW_WIDTH = 350.0
WINDOW_LEVEL = 40.0


# ============================================================
# 1. 读取并检查 boxes.json
# ============================================================

def load_boxes(json_path: Path, volume_shape):
    """
    读取 3 个医生手动画的 box，并按照 slice_index 从小到大排序。

    volume_shape:
        (Z, H, W)

    返回：
        [
            {
                "slice_index": int,
                "x0": float,
                "y0": float,
                "x1": float,
                "y1": float
            },
            ...
        ]
    """
    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if "boxes" not in data:
        raise ValueError('boxes.json 中缺少 "boxes" 字段。')

    if not isinstance(data["boxes"], list):
        raise ValueError('"boxes" 必须是一个列表。')

    boxes = data["boxes"]

    if len(boxes) != 3:
        raise ValueError(
            f"当前程序要求恰好 3 个提示框，但实际读取到 {len(boxes)} 个。"
        )

    Z, H, W = volume_shape
    required_keys = ["slice_index", "x0", "y0", "x1", "y1"]

    cleaned = []

    for i, box in enumerate(boxes, start=1):

        for key in required_keys:
            if key not in box:
                raise ValueError(
                    f"第 {i} 个 box 缺少字段：{key}"
                )

        z = int(box["slice_index"])
        x0 = float(box["x0"])
        y0 = float(box["y0"])
        x1 = float(box["x1"])
        y1 = float(box["y1"])

        # 检查 z
        if not (0 <= z < Z):
            raise ValueError(
                f"第 {i} 个 box 的 slice_index={z} 超出 CT 范围 0~{Z - 1}。"
            )

        # 检查 x
        if not (0 <= x0 < x1 < W):
            raise ValueError(
                f"第 {i} 个 box 的 x 坐标不合法："
                f"x0={x0}, x1={x1}, CT width={W}。"
            )

        # 检查 y
        if not (0 <= y0 < y1 < H):
            raise ValueError(
                f"第 {i} 个 box 的 y 坐标不合法："
                f"y0={y0}, y1={y1}, CT height={H}。"
            )

        cleaned.append({
            "slice_index": z,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
        })

    # 按 z 从小到大排序
    cleaned.sort(key=lambda x: x["slice_index"])

    z_list = [b["slice_index"] for b in cleaned]

    if len(set(z_list)) != 3:
        raise ValueError("3 个提示框必须位于 3 个不同的切片。")

    return cleaned


# ============================================================
# 2. 三个 box 之间进行分段线性插值
# ============================================================

def interpolate_two_boxes(box_a, box_b, z):
    """
    在两个医生提示框之间，对：
        x0, y0, x1, y1
    分别沿 z 方向做线性插值。
    """

    za = box_a["slice_index"]
    zb = box_b["slice_index"]

    if zb == za:
        raise ValueError("两个用于插值的 box 不能位于同一层。")

    t = (z - za) / float(zb - za)

    a = np.array(
        [box_a["x0"], box_a["y0"], box_a["x1"], box_a["y1"]],
        dtype=np.float32
    )

    b = np.array(
        [box_b["x0"], box_b["y0"], box_b["x1"], box_b["y1"]],
        dtype=np.float32
    )

    return a + t * (b - a)


def get_box_for_slice(boxes, z):
    """
    三个框：
        box1 -> box2：线性插值
        box2 -> box3：线性插值

    第 1、2、3 个提示层本身会精确使用医生给出的 box。
    """

    box1, box2, box3 = boxes

    z1 = box1["slice_index"]
    z2 = box2["slice_index"]
    z3 = box3["slice_index"]

    if z1 <= z <= z2:
        return interpolate_two_boxes(box1, box2, z)

    if z2 < z <= z3:
        return interpolate_two_boxes(box2, box3, z)

    raise ValueError(
        f"slice {z} 不在三个提示框定义的范围 {z1}~{z3} 内。"
    )


# ============================================================
# 3. CT 预处理
# ============================================================

def ct_slice_to_rgb(ct_slice):
    """
    与原 GTVp SAM 流程保持一致：

    1. WW = 350 HU
    2. WL = 40 HU
    3. 截断到 [-135, 215] HU
    4. 线性映射到 [0, 255]
    5. uint8
    6. 将同一张 axial CT 复制成 3 个通道
    """

    low = WINDOW_LEVEL - WINDOW_WIDTH / 2.0
    high = WINDOW_LEVEL + WINDOW_WIDTH / 2.0

    image = ct_slice.astype(np.float32)

    image = np.clip(
        image,
        low,
        high
    )

    image = (
        (image - low)
        / (high - low)
        * 255.0
    )

    image = np.clip(
        image,
        0,
        255
    ).astype(np.uint8)

    # 灰度复制为 3 通道 RGB
    image_rgb = np.stack(
        [image, image, image],
        axis=-1
    )

    return image_rgb


def prepare_image(ct_slice, device):
    """
    原始 axial CT:
        [H, W]

    -> RGB:
        [H, W, 3]

    -> resize:
        [1024, 1024, 3]

    -> torch:
        [1, 3, 1024, 1024]
    """

    rgb = ct_slice_to_rgb(ct_slice)

    pil_image = Image.fromarray(rgb)

    # 兼容不同 Pillow 版本
    if hasattr(Image, "Resampling"):
        resample_mode = Image.Resampling.BILINEAR
    else:
        resample_mode = Image.BILINEAR

    pil_image = pil_image.resize(
        (TARGET_SIZE, TARGET_SIZE),
        resample=resample_mode
    )

    image_np = np.asarray(
        pil_image,
        dtype=np.uint8
    )

    image_tensor = (
        torch.from_numpy(image_np.copy())
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        .to(device)
    )

    return image_tensor


# ============================================================
# 4. 将原始 CT 像素坐标的 box 映射到 1024×1024
# ============================================================

def scale_box_to_1024(
    box,
    original_h,
    original_w,
    device
):
    """
    输入：
        box = [x0, y0, x1, y1]
        坐标属于原始 CT

    输出：
        SAM 1024×1024 空间中的 box
        shape = [1, 1, 4]
    """

    scale_x = TARGET_SIZE / float(original_w)
    scale_y = TARGET_SIZE / float(original_h)

    x0, y0, x1, y1 = box

    box_1024 = [
        x0 * scale_x,
        y0 * scale_y,
        x1 * scale_x,
        y1 * scale_y,
    ]

    box_tensor = torch.tensor(
        box_1024,
        dtype=torch.float32,
        device=device
    ).view(1, 1, 4)

    return box_tensor


# ============================================================
# 5. 加载 SAM + 微调权重
# ============================================================

def load_model(device):

    print()
    print("==========================================")
    print("开始加载模型")
    print("==========================================")

    print(f"SAM基础权重：{SAM_CHECKPOINT}")
    print(f"微调权重：{FINETUNED_CHECKPOINT}")
    print(f"设备：{device}")

    # 创建 SAM ViT-B
    model = sam_model_registry[MODEL_TYPE](
        checkpoint=None
    )

    # ---------- 先加载官方 SAM 基础权重 ----------
    print("\n[1/2] 加载 SAM 官方基础权重...")

    base_state = torch.load(
        str(SAM_CHECKPOINT),
        map_location="cpu"
    )

    model.load_state_dict(
        base_state,
        strict=False
    )

    # ---------- 再加载你的微调权重 ----------
    print("[2/2] 加载 GTVp 微调权重...")

    fine_state = torch.load(
        str(FINETUNED_CHECKPOINT),
        map_location="cpu"
    )

    # 兼容常见 checkpoint 包装格式
    if isinstance(fine_state, dict) and "model_state_dict" in fine_state:
        fine_state = fine_state["model_state_dict"]

    elif isinstance(fine_state, dict) and "state_dict" in fine_state:
        fine_state = fine_state["state_dict"]

    # 兼容 DataParallel 保存的 module.xxx
    if isinstance(fine_state, dict):
        fine_state = {
            (
                key[7:]
                if key.startswith("module.")
                else key
            ): value
            for key, value in fine_state.items()
        }

    model.load_state_dict(
        fine_state,
        strict=False
    )

    model.to(device)
    model.eval()

    print("模型加载完成。")

    return model


# ============================================================
# 6. 单层 SAM 推理
# ============================================================

@torch.no_grad()
def predict_one_slice(
    model,
    ct_slice,
    box,
    device
):
    """
    单层推理：

    CT slice
        +
    当前层 box
        ↓
    SAM
        ↓
    二值 mask
    """

    H, W = ct_slice.shape

    # CT -> 1024 RGB tensor
    image = prepare_image(
        ct_slice,
        device
    )

    # 原图 box -> 1024 box
    bbox = scale_box_to_1024(
        box,
        H,
        W,
        device
    )

    # 与原测试代码保持相同 SAM 推理顺序
    input_images = torch.stack(
        [
            model.preprocess(im)
            for im in image
        ],
        dim=0
    )

    # image encoder
    image_embeddings = model.image_encoder(
        input_images
    )

    # prompt encoder
    sparse_embeddings, dense_embeddings = (
        model.prompt_encoder(
            points=None,
            boxes=bbox,
            masks=None
        )
    )

    # mask decoder
    low_res_masks, _ = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False
    )

    # 恢复到原始 CT 尺寸
    masks = model.postprocess_masks(
        low_res_masks,
        input_size=image.shape[-2:],
        original_size=(H, W)
    )

    probability = torch.sigmoid(
        masks
    )

    prediction = (
        probability[0, 0] > THRESHOLD
    ).cpu().numpy().astype(np.uint8)

    return prediction


# ============================================================
# 7. 主程序
# ============================================================

def main():

    print()
    print("==========================================")
    print("Clinical SAM 本地 CPU 测试")
    print("==========================================")

    # --------------------------------------------------------
    # A. 检查文件
    # --------------------------------------------------------

    required_files = {
        "image.nii.gz": IMAGE_PATH,
        "boxes.json": BOXES_PATH,
        "SAM基础权重": SAM_CHECKPOINT,
        "微调权重": FINETUNED_CHECKPOINT,
    }

    for name, path in required_files.items():

        print(f"{name}: {path}")

        if not path.exists():
            raise FileNotFoundError(
                f"\n找不到 {name}：\n{path}\n"
            )

    # --------------------------------------------------------
    # B. CPU
    # --------------------------------------------------------

    device = torch.device(DEVICE)

    if device.type == "cpu":
        # 防止 Windows CPU 线程开得过多
        cpu_threads = min(
            8,
            max(
                1,
                torch.get_num_threads()
            )
        )

        torch.set_num_threads(
            cpu_threads
        )

        print(
            f"\n当前使用 CPU 推理，PyTorch线程数 = {cpu_threads}"
        )

    # --------------------------------------------------------
    # C. 读取 3D CT
    # --------------------------------------------------------

    print("\n读取 CT...")

    ct_image = sitk.ReadImage(
        str(IMAGE_PATH)
    )

    # SimpleITK:
    # NIfTI -> numpy 后为 [Z, Y, X]
    ct_array = sitk.GetArrayFromImage(
        ct_image
    )

    if ct_array.ndim != 3:
        raise ValueError(
            f"输入必须是 3D CT，但读取到 shape={ct_array.shape}"
        )

    Z, H, W = ct_array.shape

    print(
        f"CT shape [Z,Y,X] = [{Z}, {H}, {W}]"
    )

    print(
        f"Spacing [x,y,z] = {ct_image.GetSpacing()}"
    )

    # --------------------------------------------------------
    # D. 读取三个医生 box
    # --------------------------------------------------------

    boxes = load_boxes(
        BOXES_PATH,
        ct_array.shape
    )

    box1, box2, box3 = boxes

    z1 = box1["slice_index"]
    z2 = box2["slice_index"]
    z3 = box3["slice_index"]

    print()
    print("医生提示层：")
    print(f"  Box 1: slice {z1}")
    print(f"  Box 2: slice {z2}")
    print(f"  Box 3: slice {z3}")

    print(
        f"\n模型仅预测 slice {z1} ~ {z3}"
    )

    print(
        "三个提示框之间使用分段线性插值。"
    )

    print(
        "本版本不进行 0.5 cm 外扩。"
    )

    # --------------------------------------------------------
    # E. 加载模型
    # --------------------------------------------------------

    model = load_model(
        device
    )

    # --------------------------------------------------------
    # F. 创建空的 3D mask
    # --------------------------------------------------------

    prediction_volume = np.zeros(
        (Z, H, W),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # G. 从第一个提示层预测到第三个提示层
    # --------------------------------------------------------

    total_slices = z3 - z1 + 1

    print()
    print("==========================================")
    print(
        f"开始预测，共 {total_slices} 层"
    )
    print("==========================================")

    for count, z in enumerate(
        range(z1, z3 + 1),
        start=1
    ):

        current_box = get_box_for_slice(
            boxes,
            z
        )

        box_text = [
            round(
                float(value),
                2
            )
            for value in current_box
        ]

        print(
            f"[{count}/{total_slices}] "
            f"slice={z}, "
            f"box={box_text}"
        )

        prediction = predict_one_slice(
            model=model,
            ct_slice=ct_array[z],
            box=current_box,
            device=device
        )

        prediction_volume[z] = prediction

    # --------------------------------------------------------
    # H. 保存 GTVp.nii.gz
    # --------------------------------------------------------

    print()
    print("正在保存结果...")

    output_image = sitk.GetImageFromArray(
        prediction_volume
    )

    # 输出与输入 CT 的物理空间信息完全一致
    output_image.CopyInformation(
        ct_image
    )

    sitk.WriteImage(
        output_image,
        str(OUTPUT_PATH),
        useCompression=True
    )

    # --------------------------------------------------------
    # I. 简单检查输出
    # --------------------------------------------------------

    unique_values = np.unique(
        prediction_volume
    )

    foreground_voxels = int(
        prediction_volume.sum()
    )

    print()
    print("==========================================")
    print("预测完成")
    print("==========================================")
    print(
        f"输出文件：{OUTPUT_PATH}"
    )
    print(
        f"输出 shape：{prediction_volume.shape}"
    )
    print(
        f"输出值：{unique_values.tolist()}"
    )
    print(
        f"前景体素数：{foreground_voxels}"
    )
    print("==========================================")


if __name__ == "__main__":
    main()
