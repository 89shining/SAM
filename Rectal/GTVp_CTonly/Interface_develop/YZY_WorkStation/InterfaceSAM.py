# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
from segment_anything import sam_model_registry


def get_program_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


BASE_DIR = get_program_dir()
CHECKPOINT_DIR = BASE_DIR / "checkpoint"
SAM_CHECKPOINT = CHECKPOINT_DIR / "sam_vit_b_01ec64.pth"
FINETUNED_CHECKPOINT = CHECKPOINT_DIR / "ImageEncoder_finetune_fold4_best.pth"

MODEL_TYPE = "vit_b"
DEVICE = "cpu"
THRESHOLD = 0.5
TARGET_SIZE = 1024
WINDOW_WIDTH = 350.0
WINDOW_LEVEL = 40.0


def parse_args():
    parser = argparse.ArgumentParser(description="Clinical SAM GTVp segmentation")
    parser.add_argument(
        "-nii", "--niiPath",
        type=str,
        required=True,
        help="待分割原始CT NIfTI文件的完整路径",
    )
    parser.add_argument(
        "-json", "--jsonPath",
        type=str,
        required=True,
        help="3个提示框boxes.json文件的完整路径",
    )
    parser.add_argument(
        "-out", "--outputPath",
        type=str,
        required=True,
        help="输出预测mask NIfTI文件的完整保存路径",
    )
    return parser.parse_args()


def load_boxes(json_path: Path, volume_shape):
    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if "boxes" not in data or not isinstance(data["boxes"], list):
        raise ValueError('boxes.json中必须包含列表字段"boxes"。')

    boxes = data["boxes"]
    if len(boxes) != 3:
        raise ValueError(f"当前程序要求恰好3个提示框，实际读取到{len(boxes)}个。")

    Z, H, W = volume_shape
    required_keys = ["slice_index", "x0", "y0", "x1", "y1"]
    cleaned = []

    for i, box in enumerate(boxes, start=1):
        for key in required_keys:
            if key not in box:
                raise ValueError(f"第{i}个box缺少字段：{key}")

        z = int(box["slice_index"])
        x0 = float(box["x0"])
        y0 = float(box["y0"])
        x1 = float(box["x1"])
        y1 = float(box["y1"])

        if not (0 <= z < Z):
            raise ValueError(f"第{i}个box的slice_index={z}超出CT范围0~{Z - 1}。")
        if not (0 <= x0 < x1 < W):
            raise ValueError(f"第{i}个box的x坐标不合法：x0={x0}, x1={x1}, width={W}。")
        if not (0 <= y0 < y1 < H):
            raise ValueError(f"第{i}个box的y坐标不合法：y0={y0}, y1={y1}, height={H}。")

        cleaned.append({
            "slice_index": z,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
        })

    cleaned.sort(key=lambda x: x["slice_index"])

    if len({b["slice_index"] for b in cleaned}) != 3:
        raise ValueError("3个提示框必须位于3个不同的切片。")

    return cleaned


def interpolate_two_boxes(box_a, box_b, z):
    za = box_a["slice_index"]
    zb = box_b["slice_index"]
    if zb == za:
        raise ValueError("两个用于插值的box不能位于同一层。")

    t = (z - za) / float(zb - za)
    a = np.array([box_a["x0"], box_a["y0"], box_a["x1"], box_a["y1"]], dtype=np.float32)
    b = np.array([box_b["x0"], box_b["y0"], box_b["x1"], box_b["y1"]], dtype=np.float32)
    return a + t * (b - a)


def get_box_for_slice(boxes, z):
    box1, box2, box3 = boxes
    z1 = box1["slice_index"]
    z2 = box2["slice_index"]
    z3 = box3["slice_index"]

    if z1 <= z <= z2:
        return interpolate_two_boxes(box1, box2, z)
    if z2 < z <= z3:
        return interpolate_two_boxes(box2, box3, z)

    raise ValueError(f"slice {z}不在提示框范围{z1}~{z3}内。")


def ct_slice_to_rgb(ct_slice):
    low = WINDOW_LEVEL - WINDOW_WIDTH / 2.0
    high = WINDOW_LEVEL + WINDOW_WIDTH / 2.0

    image = ct_slice.astype(np.float32)
    image = np.clip(image, low, high)
    image = (image - low) / (high - low) * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    return np.stack([image, image, image], axis=-1)


def prepare_image(ct_slice, device):
    rgb = ct_slice_to_rgb(ct_slice)
    pil_image = Image.fromarray(rgb)
    resample_mode = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    pil_image = pil_image.resize((TARGET_SIZE, TARGET_SIZE), resample=resample_mode)
    image_np = np.asarray(pil_image, dtype=np.uint8)

    return (
        torch.from_numpy(image_np.copy())
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        .to(device)
    )


def scale_box_to_1024(box, original_h, original_w, device):
    scale_x = TARGET_SIZE / float(original_w)
    scale_y = TARGET_SIZE / float(original_h)
    x0, y0, x1, y1 = box
    box_1024 = [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y]

    return torch.tensor(box_1024, dtype=torch.float32, device=device).view(1, 1, 4)


def load_model(device):
    print("\n==========================================")
    print("开始加载模型")
    print("==========================================")
    print(f"SAM基础权重：{SAM_CHECKPOINT}")
    print(f"微调权重：{FINETUNED_CHECKPOINT}")
    print(f"设备：{device}")

    model = sam_model_registry[MODEL_TYPE](checkpoint=None)

    print("\n[1/2] 加载 SAM 官方基础权重...")
    base_state = torch.load(str(SAM_CHECKPOINT), map_location="cpu")
    model.load_state_dict(base_state, strict=False)

    print("[2/2] 加载 GTVp 微调权重...")
    fine_state = torch.load(str(FINETUNED_CHECKPOINT), map_location="cpu")
    if isinstance(fine_state, dict) and "model_state_dict" in fine_state:
        fine_state = fine_state["model_state_dict"]
    elif isinstance(fine_state, dict) and "state_dict" in fine_state:
        fine_state = fine_state["state_dict"]

    if isinstance(fine_state, dict):
        fine_state = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in fine_state.items()
        }

    model.load_state_dict(fine_state, strict=False)
    model.to(device)
    model.eval()

    print("模型加载完成。")
    return model


@torch.no_grad()
def predict_one_slice(model, ct_slice, box, device):
    H, W = ct_slice.shape
    image = prepare_image(ct_slice, device)
    bbox = scale_box_to_1024(box, H, W, device)

    input_images = torch.stack([model.preprocess(im) for im in image], dim=0)
    image_embeddings = model.image_encoder(input_images)

    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=None,
        boxes=bbox,
        masks=None,
    )

    low_res_masks, _ = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    masks = model.postprocess_masks(
        low_res_masks,
        input_size=image.shape[-2:],
        original_size=(H, W),
    )

    probability = torch.sigmoid(masks)
    return (probability[0, 0] > THRESHOLD).cpu().numpy().astype(np.uint8)


def main():
    args = parse_args()

    image_path = Path(args.niiPath).expanduser()
    boxes_path = Path(args.jsonPath).expanduser()
    output_path = Path(args.outputPath).expanduser()
    print("\n==========================================")
    print("InterfaceSAM 输入输出")
    print("==========================================")
    print(f"输入 CT：{image_path}")
    print(f"输入 boxes：{boxes_path}")
    print(f"输出路径：{output_path}")

    required_files = {
        "NIfTI": image_path,
        "boxes.json": boxes_path,
        "SAM checkpoint": SAM_CHECKPOINT,
        "finetuned checkpoint": FINETUNED_CHECKPOINT,
    }
    for name, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}不存在：{path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE)
    if device.type == "cpu":
        torch.set_num_threads(min(8, max(1, torch.get_num_threads())))

    ct_image = sitk.ReadImage(str(image_path))
    ct_array = sitk.GetArrayFromImage(ct_image)
    if ct_array.ndim != 3:
        raise ValueError(f"输入必须是3D CT，当前shape={ct_array.shape}")

    Z, H, W = ct_array.shape
    boxes = load_boxes(boxes_path, ct_array.shape)
    z1 = boxes[0]["slice_index"]
    z2 = boxes[1]["slice_index"]
    z3 = boxes[2]["slice_index"]

    print("\n医生提示层：")
    print(f"  Box 1: slice {z1}")
    print(f"  Box 2: slice {z2}")
    print(f"  Box 3: slice {z3}")

    model = load_model(device)

    prediction_volume = np.zeros((Z, H, W), dtype=np.uint8)
    total_slices = z3 - z1 + 1

    print("\n==========================================")
    print(f"开始预测，共 {total_slices} 层")
    print("==========================================")

    for count, z in enumerate(range(z1, z3 + 1), start=1):
        current_box = get_box_for_slice(boxes, z)
        box_text = [round(float(v), 2) for v in current_box]
        print(f"[{count}/{total_slices}] slice={z}, box={box_text}")

        prediction_volume[z] = predict_one_slice(
            model=model,
            ct_slice=ct_array[z],
            box=current_box,
            device=device,
        )

    print("\n正在保存结果...")

    output_image = sitk.GetImageFromArray(prediction_volume)
    output_image.CopyInformation(ct_image)
    sitk.WriteImage(output_image, str(output_path), useCompression=True)

    print("\n==========================================")
    print("预测完成")
    print("==========================================")
    print(f"输出文件：{output_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
