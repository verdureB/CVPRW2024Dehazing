import cv2
import numpy as np
import torch
import os
import glob
import pandas as pd
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from models.fusenet import convnext_plus_head

DOWNSIZE = 2
DEVICE = 2
IMAGESIZE = 2048
OVERLAP = 1024
DATAMODE = "val"
EXPNAME = "v1"

CKPTPATH = glob.glob(f"./checkpoints/{EXPNAME}/*.ckpt")[1]
OUTDIR = f"/home/ubuntu/Competition/LowLevel/dehaze_data_{DOWNSIZE}/{DATAMODE}_pred"
TESTPATH = f"/home/ubuntu/Competition/LowLevel/dehaze_data_{DOWNSIZE}/{DATAMODE}/input"
GTPATH = f"/home/ubuntu/Competition/LowLevel/dehaze_data_{DOWNSIZE}/{DATAMODE}/gt"

model = convnext_plus_head("convnext")

print(CKPTPATH)
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


def preprocess_batch_image(image):
    image = image.astype(np.float32)
    image = torch.tensor(image).permute(0, 3, 1, 2).float()
    image = image / 127.5 - 1
    return image


def reconstruct_image_with_overlap(
    patches, coords, image_shape, patch_size=IMAGESIZE, overlap=OVERLAP
):
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
    STEP = 1  # 47 94
    with torch.no_grad():
        for i in tqdm(range(len(patches) // STEP)):
            with torch.no_grad():
                patch = patches[i * STEP : (i + 1) * STEP].cuda(DEVICE)
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

valid_list = sorted(os.listdir(TESTPATH))
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

psnr_list = []
ssim_list = []
for _, valid in enumerate(valid_list):
    input_image_path = f"{TESTPATH}/{valid}"
    gt_image_path = f"{GTPATH}/{valid}"

    # 预测图像
    output_image = predict_and_reconstruct_with_overlap_v2(
        input_image_path, model, patch_size=IMAGESIZE, overlap=OVERLAP
    )

    # 读取输入和GT图像
    input_image = cv2.imread(input_image_path).astype(np.uint16)
    if os.path.exists(gt_image_path):
        gt_image = cv2.imread(gt_image_path).astype(np.uint16)
    else:
        gt_image = np.zeros_like(input_image)

    psnr_value = psnr(output_image, gt_image, data_range=255)
    ssim_value = ssim(output_image, gt_image, data_range=255, channel_axis=2)
    psnr_list.append(psnr_value)
    ssim_list.append(ssim_value)
    # 缩小图像
    input_image_resized = cv2.resize(input_image, (0, 0), fx=0.5, fy=0.5)
    output_image_resized = cv2.resize(output_image, (0, 0), fx=0.5, fy=0.5)
    gt_image_resized = cv2.resize(gt_image, (0, 0), fx=0.5, fy=0.5)

    # 拼接图像
    concatenated_image = np.concatenate(
        (input_image_resized, output_image_resized, gt_image_resized), axis=1
    )

    # 保存拼接后的图像
    cv2.imwrite(OUTDIR + f"/{valid}", concatenated_image)
df = pd.DataFrame(
    {
        "image": valid_list,
        "psnr": psnr_list,
        "ssim": ssim_list,
    }
)
df.to_csv(f"{OUTDIR}/metrics.csv", mode="w", header=False, index=False)
print(df)
print(df.describe())
