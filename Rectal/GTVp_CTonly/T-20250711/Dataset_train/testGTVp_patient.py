import os
import re
import sys
import csv
import shutil
from pathlib import Path

import cv2
import torch
import imageio
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image
from torch.utils.data import Dataset, DataLoader

sys.path.append('/home/wusi/segment-anything')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from segment_anything import sam_model_registry

# =========================
# Explicit configuration
# =========================
TRAIN_RESULTS_DIR = '/home/wusi/segment-anything/SAMdata/Rectal/20250711_GTVp/TrainResults/DaatasetSize_fre_img'
SAMPLE_SIZES = [10, 20, 30, 40, 50, 60]
FOLDS = [1, 2, 3, 4, 5]

SAM_CHECKPOINT = '/home/wusi/segment-anything/demo/configs/checkpoint/sam_vit_b_01ec64.pth'
MODEL_TYPE = 'vit_b'

TEST_CSV_PATH = '/home/wusi/segment-anything/SAMdata/Rectal/20251224_GTVp/dataset/test/test_rgb.csv'
TEST_ROOT_DIR = '/home/wusi/segment-anything/SAMdata/Rectal/20251224_GTVp/dataset/test'
TEST_NII_DIR = '/home/wusi/segment-anything/SAMdata/Rectal/20251224_GTVp/datanii/test_nii'
TEST_IMAGE_DIR = '/home/wusi/segment-anything/SAMdata/Rectal/20251224_GTVp/dataset/test/rgb_images'

BASE_OUTPUT_DIR = '/home/wusi/segment-anything/SAMdata/Rectal/20251224_GTVp/TestResults/cm/DatasetSize_fre_img'
EXPAND_CM_LIST = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]

TARGET_H = 1024
TARGET_W = 1024


class TestDataset(Dataset):
    """Same testing dataset logic as cm_test/testdataset_cm.py."""

    def __init__(self, csv_path, root_dir, nii_dir, target_size, expand_cm):
        self.df = pd.read_csv(csv_path, header=None, names=['image', 'mask'])
        self.root_dir = root_dir
        self.nii_dir = nii_dir
        self.target_size = target_size
        self.expand_cm = expand_cm

    def __len__(self):
        return len(self.df)

    def get_box(self, resized_mask, spacing_x, spacing_y, expand_cm):
        y_indices, x_indices = np.where(resized_mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None

        x_min = np.min(x_indices)
        x_max = np.max(x_indices)
        y_min = np.min(y_indices)
        y_max = np.max(y_indices)

        img_width = resized_mask.shape[1]
        img_height = resized_mask.shape[0]

        expand_pixel_x = expand_cm / spacing_x
        expand_pixel_y = expand_cm / spacing_y

        x_min = max(x_min - expand_pixel_x, 0)
        x_max = min(x_max + expand_pixel_x, img_width - 1)
        y_min = max(y_min - expand_pixel_y, 0)
        y_max = min(y_max + expand_pixel_y, img_height - 1)

        box = np.array([x_min, y_min, x_max, y_max]).astype(np.float32)
        return torch.tensor(box).unsqueeze(0)

    def __getitem__(self, idx):
        image_rel = self.df.iloc[idx]['image'].lstrip('/\\')
        mask_rel = self.df.iloc[idx]['mask'].lstrip('/\\')

        image_path = os.path.normpath(os.path.join(self.root_dir, image_rel))
        mask_path = os.path.normpath(os.path.join(self.root_dir, mask_rel))

        image = Image.open(image_path)
        original_size = image.size[::-1]  # (H, W)
        image = image.resize(self.target_size, resample=Image.BILINEAR)
        image = np.array(image).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)

        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        mask = Image.fromarray(mask).convert('L')
        mask_np = (np.array(mask) > 0).astype(np.uint8)
        mask = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)
        resized_mask = cv2.resize(mask.squeeze(0).numpy(), self.target_size, interpolation=cv2.INTER_NEAREST)

        patient_id = os.path.basename(os.path.dirname(mask_rel))
        nii_path = os.path.join(self.nii_dir, patient_id, 'GTVp.nii.gz')
        if not os.path.exists(nii_path):
            raise FileNotFoundError(f'Missing NIfTI image: {nii_path}')
        img_nii = sitk.ReadImage(nii_path)

        resize_factor_x = self.target_size[1] / img_nii.GetSize()[0]
        resize_factor_y = self.target_size[0] / img_nii.GetSize()[1]
        spacing_x_resized = img_nii.GetSpacing()[0] / resize_factor_x / 10.0
        spacing_y_resized = img_nii.GetSpacing()[1] / resize_factor_y / 10.0

        box = self.get_box(resized_mask, spacing_x_resized, spacing_y_resized, expand_cm=self.expand_cm)

        return image, mask, box, original_size, image_path


def build_fold_ckpts(train_results_dir, sample_size, folds):
    ckpts = []
    for fold in folds:
        ckpt = os.path.join(
            train_results_dir,
            f'sample_{sample_size}',
            f'fold_{fold}',
            'weights',
            'best.pth'
        )
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f'Missing checkpoint: {ckpt}')
        ckpts.append(ckpt)
    return ckpts


def pngs_to_nii(png_dir, reference_nii_path, output_nii_path, patient_id, all_mappings):
    ref_nii = nib.load(reference_nii_path)
    affine = ref_nii.affine
    header = ref_nii.header
    shape = ref_nii.shape  # (H, W, D)

    volume = np.zeros((shape[2], shape[0], shape[1]), dtype=np.uint8)
    slice_mapping = []

    for f in sorted(
        os.listdir(png_dir),
        key=lambda x: int(os.path.splitext(x)[0]) if x.endswith('.png') and os.path.splitext(x)[0].isdigit() else float('inf')
    ):
        if not f.endswith('.png'):
            continue

        try:
            slice_idx = int(os.path.splitext(f)[0])
        except ValueError:
            print(f'Skip unrecognized file: {f}')
            continue

        img = Image.open(os.path.join(png_dir, f)).convert('L')
        arr = np.array(img)
        arr = np.rot90(arr, k=3)
        arr = np.fliplr(arr)

        if slice_idx >= volume.shape[0]:
            print(f'Slice index {slice_idx} exceeds volume depth {volume.shape[0]}, skip.')
            continue

        volume[slice_idx] = arr
        slice_mapping.append((patient_id, slice_idx, f))

    volume = np.transpose(volume, (1, 2, 0))
    nii_img = nib.Nifti1Image(volume, affine=affine, header=header)
    nib.save(nii_img, output_nii_path)

    all_mappings.extend(slice_mapping)


def run_single_setting(sample_size, expand_cm, device):
    fold_ckpts = build_fold_ckpts(TRAIN_RESULTS_DIR, sample_size, FOLDS)

    output_dir = os.path.join(BASE_OUTPUT_DIR, f'sample_{sample_size}', f'expand_{expand_cm}cm')
    os.makedirs(output_dir, exist_ok=True)
    tmp_png_dir = os.path.join(output_dir, 'tmp_png')
    os.makedirs(tmp_png_dir, exist_ok=True)

    print(f'\n=== Sample {sample_size} | Expand {expand_cm}cm ===')
    print('Using checkpoints:')
    for ckpt in fold_ckpts:
        print(f'  - {ckpt}')

    test_dataset = TestDataset(
        csv_path=TEST_CSV_PATH,
        root_dir=TEST_ROOT_DIR,
        nii_dir=TEST_NII_DIR,
        target_size=(TARGET_H, TARGET_W),
        expand_cm=expand_cm
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    nets = []
    for ckpt in fold_ckpts:
        model = sam_model_registry[MODEL_TYPE](checkpoint=None)
        model.to(device)
        model.load_state_dict(torch.load(SAM_CHECKPOINT, map_location=device), strict=False)
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        model.eval()
        nets.append(model)

    with torch.no_grad():
        for image, _, box, original_size, image_path in test_loader:
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
                    multimask_output=False
                )
                masks = net.postprocess_masks(low_res_masks, input_size=imgs.shape[-2:], original_size=original_size)
                prob_mask = torch.sigmoid(masks)
                prob_list.append(prob_mask)

            avg_prob = torch.mean(torch.stack(prob_list, dim=0), dim=0)
            final_mask = (avg_prob > 0.5).float()

            rel_path = os.path.relpath(image_path[0], TEST_IMAGE_DIR)
            patient_folder = Path(rel_path).parent.name
            image_stem = Path(rel_path).stem
            save_subdir = os.path.join(tmp_png_dir, patient_folder)
            os.makedirs(save_subdir, exist_ok=True)

            save_path = os.path.join(save_subdir, image_stem + '.png')
            save_mask = (final_mask[0].squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
            imageio.imwrite(save_path, save_mask)

    datanii_dir = TEST_NII_DIR
    all_slice_mappings = []

    for pa in os.listdir(datanii_dir):
        match = re.search(r'\d+', pa)
        if not match:
            print(f'Skip invalid patient folder: {pa}')
            continue

        idx = match.group(0).zfill(3)
        pa_path = os.path.join(datanii_dir, pa)
        image_nii_path = os.path.join(pa_path, 'image.nii.gz')
        pre_png_dir = os.path.join(tmp_png_dir, pa)

        if not os.path.exists(pre_png_dir):
            print(f'Skip {pa}: no predicted PNG folder found.')
            continue

        output_path = os.path.join(output_dir, f'CTV_{idx}.nii.gz')
        pngs_to_nii(
            png_dir=pre_png_dir,
            reference_nii_path=image_nii_path,
            output_nii_path=output_path,
            patient_id=pa,
            all_mappings=all_slice_mappings
        )

    mapping_csv = os.path.join(output_dir, 'slice_mapping.csv')
    with open(mapping_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['patient_id', 'slice_idx', 'png_name'])
        writer.writerows(all_slice_mappings)

    shutil.rmtree(tmp_png_dir)
    print(f'Done: Sample {sample_size}, expand {expand_cm}cm -> {output_dir}')


def main():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('\n======================= Test Config =======================')
    print(f'Train results dir: {TRAIN_RESULTS_DIR}')
    print(f'Output root: {BASE_OUTPUT_DIR}')
    print(f'Sample sizes: {SAMPLE_SIZES}')
    print(f'Folds: {FOLDS}')
    print(f'Expand cm list: {EXPAND_CM_LIST}')
    print('===========================================================\n')

    for sample_size in SAMPLE_SIZES:
        for expand_cm in EXPAND_CM_LIST:
            run_single_setting(sample_size, expand_cm, device)

    print('\nAll testing jobs completed.')


if __name__ == '__main__':
    main()
