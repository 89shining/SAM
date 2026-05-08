import argparse
import math
import os
import re
import shutil
import sys
from pathlib import Path

import imageio
import nibabel as nib
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..")))

from segment_anything import sam_model_registry
from testdataset_lora import TestDatasetLoRA


class LoRAQKV(nn.Module):
    def __init__(self, base_linear: nn.Linear, dim: int, r: int = 4, alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("LoRAQKV expects nn.Linear base layer")
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")

        self.base_linear = base_linear
        self.dim = dim
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(p=dropout)

        in_features = base_linear.in_features
        self.lora_q_A = nn.Linear(in_features, r, bias=False)
        self.lora_q_B = nn.Linear(r, dim, bias=False)
        self.lora_v_A = nn.Linear(in_features, r, bias=False)
        self.lora_v_B = nn.Linear(r, dim, bias=False)

        nn.init.kaiming_uniform_(self.lora_q_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_q_B.weight)
        nn.init.kaiming_uniform_(self.lora_v_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_v_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_linear(x)
        dropped = self.lora_dropout(x)
        delta_q = self.lora_q_B(self.lora_q_A(dropped)) * self.scaling
        delta_v = self.lora_v_B(self.lora_v_A(dropped)) * self.scaling

        out = base.clone()
        out[..., :self.dim] += delta_q
        out[..., -self.dim:] += delta_v
        return out


def inject_lora_to_sam_image_encoder(sam_model: nn.Module, r: int = 4, alpha: int = 16, dropout: float = 0.1):
    for blk in sam_model.image_encoder.blocks:
        dim = blk.attn.qkv.in_features
        blk.attn.qkv = LoRAQKV(blk.attn.qkv, dim=dim, r=r, alpha=alpha, dropout=dropout)
    return sam_model


def parse_expand_list(text: str):
    if not text:
        return [0.0]
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def build_model(model_type, sam_checkpoint, lora_checkpoint, device, lora_rank, lora_alpha, lora_dropout):
    # 1) Build base SAM and load foundation checkpoint.
    net = sam_model_registry[model_type](checkpoint=None)
    base_missing = net.load_state_dict(torch.load(sam_checkpoint, map_location=device), strict=False)

    # 2) Inject LoRA adapters into image encoder q/v branches (k unchanged).
    net = inject_lora_to_sam_image_encoder(net, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout)

    # 3) Load training checkpoint that contains LoRA parameters.
    #    Must be done after LoRA injection; strict=False avoids benign key mismatch.
    lora_missing = net.load_state_dict(torch.load(lora_checkpoint, map_location=device), strict=False)

    print(f"[Load] base SAM ckpt: {sam_checkpoint}")
    print(
        f"[Load] base strict=False -> missing={len(base_missing.missing_keys)}, "
        f"unexpected={len(base_missing.unexpected_keys)}"
    )
    print(f"[Load] LoRA train ckpt: {lora_checkpoint}")
    print(
        f"[Load] lora strict=False -> missing={len(lora_missing.missing_keys)}, "
        f"unexpected={len(lora_missing.unexpected_keys)}"
    )

    lora_total = 0
    lora_nonzero = 0
    for name, p in net.named_parameters():
        if "lora_" in name:
            lora_total += 1
            is_nonzero = bool(torch.any(p.detach() != 0).item())
            if is_nonzero:
                lora_nonzero += 1
            print(f"[LoRA] {name:80s} shape={tuple(p.shape)} nonzero={is_nonzero}")

    print(f"[LoRA] tensors: {lora_nonzero}/{lora_total} non-zero")
    if lora_total == 0:
        raise RuntimeError("No LoRA parameters found in model. LoRA injection may have failed.")
    if lora_nonzero == 0:
        print("[Warn] All LoRA tensors are zero. Check whether lora_checkpoint is the trained checkpoint.")

    # 4) Inference mode on GPU/selected device.
    net.to(device)
    net.eval()
    return net


def pngs_to_nii(png_dir, reference_nii_path, output_nii_path):
    ref_nii = nib.load(reference_nii_path)
    affine = ref_nii.affine
    header = ref_nii.header
    shape = ref_nii.shape  # (H, W, D)

    volume = np.zeros((shape[2], shape[0], shape[1]), dtype=np.uint8)

    png_files = sorted(
        [f for f in os.listdir(png_dir) if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 10**9,
    )

    for f in png_files:
        stem = os.path.splitext(f)[0]
        if not stem.isdigit():
            continue
        slice_idx = int(stem)
        if slice_idx >= volume.shape[0]:
            continue

        arr = np.array(Image.open(os.path.join(png_dir, f)).convert("L"))
        arr = np.rot90(arr, k=3)
        arr = np.fliplr(arr)
        volume[slice_idx] = arr

    volume = np.transpose(volume, (1, 2, 0))
    nib.save(nib.Nifti1Image(volume, affine=affine, header=header), output_nii_path)


def main():
    parser = argparse.ArgumentParser(description="LoRA SAM test script based on cm_test logic")
    parser.add_argument(
        "--fold_ckpts",
        nargs="+",
        default=[
            # "/home/wusi/segment-anything/SAMdata/Rectal/20250711_GTVp/TrainResults/trainresult_TrainAll_lora/fold_1/weights/best.pth",
            # "/home/wusi/segment-anything/SAMdata/Rectal/20250711_GTVp/TrainResults/trainresult_TrainAll_lora/fold_2/weights/best.pth",
            # "/home/wusi/segment-anything/SAMdata/Rectal/20250711_GTVp/TrainResults/trainresult_TrainAll_lora/fold_3/weights/best.pth",
            "/home/wusi/segment-anything/SAMdata/Rectal/20250711_GTVp/TrainResults/trainresult_TrainAll_lora/fold_4/weights/best.pth",
            # "/home/wusi/segment-anything/SAMdata/Rectal/20250711_GTVp/TrainResults/trainresult_TrainAll_lora/fold_5/weights/best.pth",
        ],
        help="List of fold best.pth paths",
    )
    parser.add_argument(
        "--sam_checkpoint",
        default="/home/wusi/segment-anything/demo/configs/checkpoint/sam_vit_b_01ec64.pth",
        help="Base SAM checkpoint path",
    )
    parser.add_argument("--model_type", default="vit_b")
    parser.add_argument("--csv_path", default="/home/wusi/segment-anything/SAMdata/Rectal/20251224_GTVp/dataset/test/test_rgb.csv")
    parser.add_argument("--root_dir", default="/home/wusi/segment-anything/SAMdata/Rectal/20251224_GTVp/dataset/test")
    parser.add_argument("--image_dir", default="/home/wusi/segment-anything/SAMdata/Rectal/20251224_GTVp/dataset/test/rgb_images")
    parser.add_argument("--nii_dir", default="/home/wusi/segment-anything/SAMdata/Rectal/20251224_GTVp/datanii/test_nii")
    parser.add_argument("--base_output_dir", default="/home/wusi/segment-anything/SAMdata/Rectal/20251224_GTVp/TestResults/cm/TrainAll_lora")
    parser.add_argument("--expand_cm_list", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.5")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--cuda_visible_devices", default=None)
    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expand_values = parse_expand_list(args.expand_cm_list)

    print("\n================ LoRA SAM Test Config ================")
    print(f"device: {device}")
    print(f"sam_checkpoint: {args.sam_checkpoint}")
    print("fold checkpoints:")
    for ck in args.fold_ckpts:
        print(f"  - {ck}")
    print(f"expand list: {expand_values}")
    print(f"output root: {args.base_output_dir}")
    print("=====================================================\n")

    nets = [
        build_model(
            args.model_type,
            args.sam_checkpoint,
            ckpt,
            device,
            args.lora_rank,
            args.lora_alpha,
            args.lora_dropout,
        )
        for ckpt in args.fold_ckpts
    ]

    for expand_cm in expand_values:
        print(f"=== running expand_cm={expand_cm} ===")
        output_dir = os.path.join(args.base_output_dir, f"expand_{expand_cm}cm")
        os.makedirs(output_dir, exist_ok=True)
        tmp_png_dir = os.path.join(output_dir, "tmp_png")
        os.makedirs(tmp_png_dir, exist_ok=True)

        dataset = TestDatasetLoRA(
            csv_path=args.csv_path,
            root_dir=args.root_dir,
            nii_dir=args.nii_dir,
            target_size=(1024, 1024),
            expand_cm=expand_cm,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        with torch.no_grad():
            for image, _, box, original_size, image_path in loader:
                imgs = image.to(device).float()
                bbox = box.to(device).float()

                prob_list = []
                for net in nets:
                    input_images = torch.stack([net.preprocess(im) for im in imgs], dim=0)
                    image_embeddings = net.image_encoder(input_images)
                    sparse_embeddings, dense_embeddings = net.prompt_encoder(points=None, boxes=bbox, masks=None)
                    low_res_masks, _ = net.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    masks = net.postprocess_masks(
                        low_res_masks,
                        input_size=imgs.shape[-2:],
                        original_size=original_size,
                    )
                    prob_list.append(torch.sigmoid(masks))

                avg_prob = torch.mean(torch.stack(prob_list, dim=0), dim=0)
                final_mask = (avg_prob > args.threshold).float()

                rel_path = os.path.relpath(image_path[0], args.image_dir)
                patient_folder = Path(rel_path).parent.name
                image_stem = Path(rel_path).stem
                save_subdir = os.path.join(tmp_png_dir, patient_folder)
                os.makedirs(save_subdir, exist_ok=True)
                save_path = os.path.join(save_subdir, image_stem + ".png")

                save_mask = (final_mask[0].squeeze().cpu().numpy() > 0).astype(np.uint8) * 255
                imageio.imwrite(save_path, save_mask)

        for patient in os.listdir(args.nii_dir):
            m = re.search(r"\d+", patient)
            if not m:
                continue

            idx = m.group(0).zfill(3)
            ref_nii = os.path.join(args.nii_dir, patient, "image.nii.gz")
            patient_png_dir = os.path.join(tmp_png_dir, patient)
            if not os.path.isdir(patient_png_dir):
                continue

            out_nii = os.path.join(output_dir, f"GTVp_{idx}.nii.gz")
            pngs_to_nii(patient_png_dir, ref_nii, out_nii)

        shutil.rmtree(tmp_png_dir)
        print(f"done expand_cm={expand_cm}, saved to {output_dir}")


if __name__ == "__main__":
    main()
