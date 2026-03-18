import os
import re
import torch
import numpy as np
import SimpleITK as sitk
from collections import defaultdict
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
from testdataset import SAMTestDatasetFromNiiGz   # 你的 dataset

# ================= 配置 =================
datanii_dir = "/home/wusi/segment-anything/SAMdata/Rectal/20260316_CTV/Cropdatanii/test_nii"
output_dir = "/home/wusi/segment-anything/SAMdata/Rectal/20260316_CTV/nnUNet_crop_box/TestResult"

sam_ckpt = "/home/wusi/segment-anything/demo/configs/checkpoint/sam_vit_b_01ec64.pth"
finetuned_ckpt = "/home/wusi/segment-anything/SAMdata/Rectal/20260316_CTV/nnUNet_crop_box/TrainResult/fold_2/weights/best.pth"
model_type = "vit_b"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(output_dir, exist_ok=True)

# ================= Dataset =================
test_dataset = SAMTestDatasetFromNiiGz(
    nii_root_dir=datanii_dir,
    expand_cm=0,
    target_image_size=(1024, 1024),
    image_name="image.nii.gz",
    gt_name="CTV.nii.gz",
    nnunet_name="prompt.nii.gz",
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ================= Model =================
net = sam_model_registry[model_type](checkpoint=None)
net.load_state_dict(torch.load(sam_ckpt, map_location=device), strict=False)
net.load_state_dict(torch.load(finetuned_ckpt, map_location=device), strict=False)
net.to(device)
net.eval()

# ================= Inference =================
pred_volumes = defaultdict(dict)  # pred_volumes[pa][z]

with torch.no_grad():
    for batch in test_loader:
        imgs = batch["image"].to(device).float()
        bbox = batch["box"].to(device)
        original_size = batch["original_size"]
        pa = batch["patient_id"][0]
        z = int(batch["slice_idx"][0])

        input_images = torch.stack([net.preprocess(im) for im in imgs], dim=0)
        image_embeddings = net.image_encoder(input_images)

        sparse_embeddings, dense_embeddings = net.prompt_encoder(
            points=None, boxes=bbox, masks=None
        )

        low_res_masks, _ = net.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=net.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        masks = net.postprocess_masks(
            low_res_masks,
            input_size=imgs.shape[-2:],
            original_size=original_size
        )

        final_mask = (torch.sigmoid(masks) > 0.5).float()
        pred_volumes[pa][z] = final_mask[0, 0].cpu().numpy().astype(np.uint8)

# ================= Save NIfTI（命名规则 100% 原样） =================
for pa in os.listdir(datanii_dir):
    match = re.search(r'\d+', pa)
    if not match:
        continue

    idx = match.group(0).zfill(3)
    ref_img = sitk.ReadImage(os.path.join(datanii_dir, pa, "image.nii.gz"))
    ref_arr = sitk.GetArrayFromImage(ref_img)  # (Z,H,W)

    pred_arr = np.zeros_like(ref_arr, dtype=np.uint8)
    if pa in pred_volumes:
        for z, mask in pred_volumes[pa].items():
            pred_arr[z] = mask

    pred_img = sitk.GetImageFromArray(pred_arr)
    pred_img.CopyInformation(ref_img)

    sitk.WriteImage(pred_img, os.path.join(output_dir, f"CTV_{idx}.nii.gz"))

print("DONE.")
