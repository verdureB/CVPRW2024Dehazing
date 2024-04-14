import segmentation_models_pytorch as smp

# autocast
from torch.cuda.amp import autocast

# 使用示例
import os
from tqdm import tqdm

DEVICE = 0
IMAGESIZE = 2000
OVERLAP = 500
OUTDIR = "testing/finetune"
CKPTPATH = "checkpoints/convnext_new_valid_resample_400_3_8e-5_1e-8_swinv2_lpips_smoothl1/epoch=0-valid_psnr=22.0754.ckpt"


from models.fusenet import convnext_plus_head

model = convnext_plus_head("convnext")


import torch

ckpt = torch.load(CKPTPATH, map_location="cpu")["state_dict"]
for k in list(ckpt.keys()):
    if "model." in k:
        ckpt[k.replace("model.", "")] = ckpt.pop(k)
    if "DNet" in k:
        ckpt.pop(k)
    if "gradloss" in k:
        ckpt.pop(k)
    if "lpips" in k:
        ckpt.pop(k)

model.load_state_dict(ckpt)
model.eval()
import cv2
import numpy as np
import torch


def split_image_into_patches_with_overlap(image, patch_size=IMAGESIZE, overlap=OVERLAP):
    patches = []
    coords = []
    h, w, _ = image.shape
    stride = patch_size - overlap
    # 确保即使在图像边缘也能正确处理
    for x in range(0, h, stride):
        for y in range(0, w, stride):
            x_end = min(x + patch_size, h)
            y_end = min(y + patch_size, w)
            patch = image[x:x_end, y:y_end]
            # 对于边缘小于patch_size的patch，使用cv2.copyMakeBorder进行填充
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                patch = cv2.copyMakeBorder(
                    patch,
                    0,
                    patch_size - patch.shape[0],
                    0,
                    patch_size - patch.shape[1],
                    cv2.BORDER_REFLECT,
                )
            patches.append(patch)
            coords.append((x, y))
    return patches, coords


def preprocess_image(image):
    image = image.astype(np.float32)
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()
    image = image / 127.5 - 1
    return image


def preprocess_batch_image(image):
    image = image.astype(np.float32)
    image = torch.tensor(image).permute(0, 3, 1, 2).float()
    image = image / 127.5 - 1
    return image


def reconstruct_image_with_overlap(
    patches, coords, image_shape, patch_size=IMAGESIZE, overlap=OVERLAP
):
    stride = patch_size - overlap
    vote_map = np.zeros(image_shape[:2], dtype=np.int32)
    reconstructed = np.zeros(image_shape, dtype=np.float32)
    for patch, (x, y) in zip(patches, coords):
        x_end = min(x + patch_size, image_shape[0])
        y_end = min(y + patch_size, image_shape[1])
        reconstructed[x:x_end, y:y_end] += patch[: x_end - x, : y_end - y]
        vote_map[x:x_end, y:y_end] += 1
    vote_map[vote_map == 0] = 1  # 避免除以零
    reconstructed /= vote_map[:, :, np.newaxis]
    return reconstructed


def predict_and_reconstruct_with_overlap(
    image_path, model, patch_size=IMAGESIZE, overlap=OVERLAP
):
    image = cv2.imread(image_path)
    image = np.uint16(image)
    original_shape = image.shape
    patches, coords = split_image_into_patches_with_overlap(image, patch_size, overlap)
    predicted_patches = []
    with torch.no_grad():
        for patch in tqdm(patches):
            patch = preprocess_image(patch)
            with torch.no_grad():
                patch = patch.cuda(DEVICE)
                pred1, pred2 = model(patch)
                output = patch * pred1 + patch + pred2
                output = output[0].permute(1, 2, 0).detach().cpu().numpy()
                output = (output + 1) * 127.5
                output = np.clip(output, 0, 255).astype(np.uint16)
                # 适应原始patch大小，特别是对于边缘部分
                predicted_patch = output[
                    : original_shape[0] - coords[0][0],
                    : original_shape[1] - coords[0][1],
                ]
                predicted_patches.append(predicted_patch)

    reconstructed_image = reconstruct_image_with_overlap(
        predicted_patches, coords, original_shape
    )
    return reconstructed_image


def predict_and_reconstruct_with_overlap_v2(
    image_path, model, patch_size=IMAGESIZE, overlap=OVERLAP
):
    image = cv2.imread(image_path)
    image = np.uint16(image)
    original_shape = image.shape
    patches, coords = split_image_into_patches_with_overlap(image, patch_size, overlap)
    patches = np.array(patches)
    patches = preprocess_batch_image(patches)
    predicted_patches = np.zeros(
        (len(patches), patch_size, patch_size, 3), dtype=np.uint16
    )
    STEP = 3  # 47 94
    with torch.no_grad():
        for i in tqdm(range(len(patches) // STEP)):
            with torch.no_grad():
                patch = patches[i * STEP : (i + 1) * STEP].cuda(DEVICE)
                # with autocast():
                #     pred1, pred2 = model(patch)
                # output = patch * pred1 + patch + pred2
                output = model(patch)
                output = output.permute(0, 2, 3, 1).detach().cpu().numpy()
                output = (output + 1) * 127.5
                output = np.clip(output, 0, 255).astype(np.uint16)
                # 适应原始patch大小，特别是对于边缘部分
                predicted_patch = output[
                    :,
                    :,
                    : original_shape[0] - coords[0][0],
                    : original_shape[1] - coords[0][1],
                ]
                predicted_patches[i * STEP : (i + 1) * STEP] = predicted_patch

    reconstructed_image = reconstruct_image_with_overlap(
        predicted_patches, coords, original_shape
    )
    return reconstructed_image


# 假设 model 已经加载
model = model.cuda(DEVICE)


valid_list = sorted(os.listdir("data/final"))
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)
for _, valid in enumerate(valid_list):
    output_image = predict_and_reconstruct_with_overlap_v2(
        f"data/final/{valid}", model, patch_size=IMAGESIZE, overlap=OVERLAP
    )
    alpha = cv2.imread(f"data/final/{valid}", cv2.IMREAD_UNCHANGED)[..., -1]
    print(valid)
    cv2.imwrite(
        OUTDIR + f"/{valid}",
        np.concatenate((output_image[..., :3], alpha[..., None]), axis=-1),
    )
    # break
