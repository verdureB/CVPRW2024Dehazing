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

# --- Configuration ---
DOWNSIZE = 1
DEVICE = 1
IMAGESIZE = 2048  # Adjusted for memory constraints
OVERLAP = 512  # Maintain a good overlap
DATAMODE = "test"
EXPNAME = "v3->cautiou+dpath0.2+dropout0.2+extra_data+cc"
TESTPATH = f"/home/ubuntu/Competition/LowLevel/dehaze_data_{DOWNSIZE}/{DATAMODE}/input"
GTPATH = f"/home/ubuntu/Competition/LowLevel/dehaze_data_{DOWNSIZE}/{DATAMODE}/gt"
CKPTPATHS = [
    glob.glob(f"./checkpoints/{EXPNAME}/*.ckpt")[1],
]
CONFIGS = [
    {"tta": True, "ckpt_index": 0, "name": "tta"},
]
BASE_OUTDIR = (
    f"/home/ubuntu/Competition/LowLevel/dehaze_data_{DOWNSIZE}/{DATAMODE}_pred"
)


# --- Helper Functions ---
def split_image_into_patches_with_overlap(image, patch_size=IMAGESIZE, overlap=OVERLAP):
    patches = []
    coords = []
    h, w, _ = image.shape
    stride = patch_size - overlap

    for x in range(0, h, stride):
        for y in range(0, w, stride):
            x_end = min(x + patch_size, h)
            y_end = min(y + patch_size, w)

            # --- Zero-padding on all sides ---
            top_pad = 0
            bottom_pad = 0
            left_pad = 0
            right_pad = 0

            if x_end - x < patch_size:
                bottom_pad = patch_size - (x_end - x)
            if y_end - y < patch_size:
                right_pad = patch_size - (y_end - y)

            patch = image[x:x_end, y:y_end]
            patch = cv2.copyMakeBorder(
                patch,
                top_pad,
                bottom_pad,
                left_pad,
                right_pad,
                cv2.BORDER_REFLECT,
                # value=0,
            )

            patches.append(patch)
            coords.append((x, y))

    return patches, coords


def preprocess_batch_image(image):
    image = image.astype(np.float32)
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    image = image / 127.5 - 1
    return image


def reconstruct_image_with_overlap(
    patches, coords, image_shape, patch_size=IMAGESIZE, overlap=OVERLAP
):
    reconstructed = torch.zeros(image_shape, dtype=torch.float32)
    vote_map = torch.zeros(image_shape[:2], dtype=torch.int32)

    for patch, (x, y) in zip(patches, coords):
        x_end = min(x + patch_size, image_shape[0])
        y_end = min(y + patch_size, image_shape[1])

        patch = patch[: x_end - x, : y_end - y]

        reconstructed[x:x_end, y:y_end, :] += patch
        vote_map[x:x_end, y:y_end] += 1

    vote_map[vote_map == 0] = 1
    reconstructed /= vote_map.unsqueeze(-1)

    return reconstructed.cpu().numpy()


def predict_and_reconstruct_with_overlap_v2(
    image_path, model, enable_tta, patch_size=IMAGESIZE, overlap=OVERLAP
):
    image = cv2.imread(image_path)
    image = np.uint16(image)
    original_shape = image.shape
    patches, coords = split_image_into_patches_with_overlap(image, patch_size, overlap)

    predicted_patches = []

    with torch.no_grad():
        for patch in tqdm(patches):
            patch_tensor = preprocess_batch_image(patch).cuda(DEVICE)

            if enable_tta:
                out1 = model(patch_tensor)
                out2 = torch.flip(model(torch.flip(patch_tensor, [3])), [3])
                output = (out1 + out2) / 2
            else:
                output = model(patch_tensor)
            output = output.squeeze(0).permute(1, 2, 0).detach().cpu()
            output = (output + 1) * 127.5
            output = torch.clamp(output, 0, 255).type(torch.uint16)
            predicted_patches.append(output)

    reconstructed_image = reconstruct_image_with_overlap(
        predicted_patches, coords, original_shape
    )
    return reconstructed_image


# --- Main Loop ---
valid_list = sorted(os.listdir(TESTPATH))

for config in CONFIGS:
    ENABLE_TTA = config["tta"]
    OUTDIR = f"{BASE_OUTDIR}_{config['name']}"
    CKPTPATH = CKPTPATHS[config["ckpt_index"]]

    model = convnext_plus_head()
    print(f"Loading checkpoint: {CKPTPATH}")
    ckpt = torch.load(CKPTPATH, map_location="cpu", weights_only=False)["state_dict"]
    for k in list(ckpt.keys()):
        if "lpips" in k:
            ckpt.pop(k)
        elif "DNet" in k:
            ckpt.pop(k)
        elif "gradloss" in k:
            ckpt.pop(k)
        elif "model." in k:
            ckpt[k.replace("model.", "")] = ckpt.pop(k)
        else:
            ckpt.pop(k)
    model.load_state_dict(ckpt)
    model.eval()
    model = model.cuda(DEVICE)

    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    psnr_list = []
    ssim_list = []
    for _, valid in enumerate(valid_list):
        input_image_path = f"{TESTPATH}/{valid}"
        gt_image_path = f"{GTPATH}/{valid}"

        output_image = predict_and_reconstruct_with_overlap_v2(
            input_image_path, model, ENABLE_TTA, patch_size=IMAGESIZE, overlap=OVERLAP
        )

        input_image = cv2.imread(input_image_path).astype(np.uint16)
        if os.path.exists(gt_image_path):
            gt_image = cv2.imread(gt_image_path).astype(np.uint16)
        else:
            gt_image = np.zeros_like(input_image)

        psnr_value = psnr(output_image, gt_image, data_range=255)
        ssim_value = ssim(output_image, gt_image, data_range=255, channel_axis=2)
        psnr_list.append(psnr_value)
        ssim_list.append(ssim_value)

        input_image_resized = cv2.resize(input_image, (0, 0), fx=0.5, fy=0.5)
        output_image_resized = cv2.resize(output_image, (0, 0), fx=0.5, fy=0.5)
        gt_image_resized = cv2.resize(gt_image, (0, 0), fx=0.5, fy=0.5)

        concatenated_image = np.concatenate(
            (input_image_resized, output_image_resized, gt_image_resized), axis=1
        )
        cv2.imwrite(OUTDIR + f"/{valid}", concatenated_image)

    df = pd.DataFrame(
        {
            "image": valid_list,
            "psnr": psnr_list,
            "ssim": ssim_list,
        }
    )
    df.to_csv(f"{OUTDIR}/metrics.csv", index=False)

    print(df)
    print(df.describe())
