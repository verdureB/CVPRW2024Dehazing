import os
import glob
import torch
import cv2
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from option import get_option
from sklearn.model_selection import train_test_split

opt = get_option()

train_transform = A.Compose(
    [
        A.RandomCrop(opt.image_size, opt.image_size),
        A.RandomGridShuffle((2, 2)),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(transpose_mask=True),
    ]
)

valid_transform = A.Compose(
    [
        # A.PadIfNeeded(
        #     opt.valid_image_size, opt.valid_image_size, border_mode=cv2.BORDER_REFLECT
        # ),
        # A.CenterCrop(opt.valid_image_size, opt.valid_image_size),
        ToTensorV2(transpose_mask=True),
    ]
)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, phase, opt, transform=None):
        self.phase = phase
        self.opt = opt
        self.dataset_root = opt.dataset_root
        self.transform = transform
        self.crops_per_image = opt.crops_per_image if phase == "train" else 1
        self.dataset_root = os.path.join(self.dataset_root, "train")
        self.image_list = os.listdir(os.path.join(self.dataset_root, "gt"))

        # Split data into train and validation sets
        train_images, val_images = train_test_split(
            self.image_list, test_size=opt.valid_image_rate, random_state=413
        )

        if self.phase == "train":
            self.image_list = train_images
        elif self.phase == "valid":
            self.image_list = val_images
            print(val_images)

        self.load_images_in_parallel()

        if self.phase == "train" and opt.extra_data:
            self.load_extra_data()

    def load_image(self, path):
        image = cv2.imread(path)
        if image.shape[0] == 4000:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif image.shape[0] != 6000:
            image = cv2.resize(image, (4000, 6000))
        return image

    def load_extra_data(self):
        extra_input = glob.glob(
            "/home/ubuntu/Competition/LowLevel/dehaze_data_2/train/input/*.png"
        )
        extra_label = glob.glob(
            "/home/ubuntu/Competition/LowLevel/dehaze_data_2/train/gt/*.png"
        )

        def load_pair(paths):
            input_path, label_path = paths
            return self.load_image(input_path), self.load_image(label_path)

        with ThreadPoolExecutor(max_workers=24) as executor:
            extra_images = list(
                tqdm(
                    executor.map(load_pair, zip(extra_input, extra_label)),
                    total=len(extra_input),
                    desc="Loading extra data",
                )
            )

        for input_img, label_img in extra_images:
            self.input_list.append(input_img)
            self.target_list.append(label_img)

    def load_images_in_parallel(self):
        def load_images(image_paths, root_dir):
            with ThreadPoolExecutor(max_workers=24) as executor:
                images = list(
                    tqdm(
                        executor.map(
                            self.load_image,
                            [os.path.join(root_dir, img) for img in image_paths],
                        ),
                        total=len(image_paths),
                        desc=f"Loading {self.phase} images",
                    )
                )
            return images

        self.input_list = load_images(
            self.image_list, os.path.join(self.dataset_root, "input")
        )
        self.target_list = load_images(
            self.image_list, os.path.join(self.dataset_root, "gt")
        )

    def __getitem__(self, index):
        real_index = index // self.crops_per_image

        low_image = self.input_list[real_index]
        high_image = self.target_list[real_index]

        if self.transform:
            transformed = self.transform(image=low_image, mask=high_image)
            low_image = transformed["image"]
            high_image = transformed["mask"]

        low_image = low_image / 127.5 - 1.0
        high_image = high_image / 127.5 - 1.0

        if random.random() < self.opt.ori_image_rate and self.phase == "train":
            return high_image.float(), high_image.float()
        else:
            return low_image.float(), high_image.float()

    def __len__(self):
        return len(self.input_list) * self.crops_per_image


def get_dataloader(opt):
    train_dataset = Dataset(phase="train", opt=opt, transform=train_transform)
    valid_dataset = Dataset(phase="valid", opt=opt, transform=valid_transform)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    return train_dataloader, valid_dataloader


if __name__ == "__main__":
    train_dataloader, valid_dataloader = get_dataloader(opt)
    for i, (low, high) in enumerate(train_dataloader):
        pass
        print(low.shape, high.shape)
        if i == 0:
            break
    for i, (low, high) in enumerate(valid_dataloader):
        pass
        print(low.shape, high.shape)
        if i == 0:
            break
