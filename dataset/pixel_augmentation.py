"""Implements the cutmix image augmentations."""

import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
import random
from PIL import Image
from torchvision.utils import save_image
from glob import glob
from tqdm import tqdm
import argparse
import numpy as np
import os
import torchvision.transforms.functional as Fun


random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)


def check_overlap(
    rect1_x1, rect1_y1, rect1_x2, rect1_y2, rect2_x1, rect2_y1, rect2_x2, rect2_y2
):
    """Checks if two rectangles overlap."""
    if rect1_x1 > rect2_x2 or rect2_x1 > rect1_x2:
        return False
    if rect1_y1 > rect2_y2 or rect2_y1 > rect1_y2:
        return False
    return True


def prepare_images(pil_base_image, pil_cutmix_image,base_label,cutmix_label, num_classes,device,base_image_size=256, n=5):
    """Prepares the cutmix images.

    :param pil_base_image: PIL image of the base image.
    :param pil_cutmix_image: PIL image of the cutmix image.
    :param base_image_size: Size of the base image. Default: 256.
    :param n: Setting for the cutmix image. Default: 5.
    """
    pil_base_image = Fun.to_pil_image(pil_base_image)
    pil_cutmix_image = Fun.to_pil_image(pil_cutmix_image)
    ## Augmentations ##
    ## 1. For the base image ##
    base_transform = transforms.Compose(
        [
            transforms.Resize(base_image_size),
            transforms.CenterCrop(base_image_size),
            transforms.ToTensor(),
        ]
    )

    ## 2. For the cutmix image ##

    if n == 5:
        n = random.randint(1, 4)
    
    base_label = F.one_hot(base_label,num_classes=num_classes ).to(device)
    cutmix_label = F.one_hot(cutmix_label,num_classes=num_classes).to(device)

    ## Scale = same as base --> Setting 1 ##
    if n == 1:
        final_label = 0.5*base_label+0.5*cutmix_label

        cutmix_transform_list = [
            transforms.Resize(base_image_size),
            transforms.CenterCrop(base_image_size),
            transforms.ToTensor(),
        ]

        m = random.randint(1, 2)
        if m == 1:
            mask = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
            x_start = base_image_size // 2
            y_start = 0
            x_end = base_image_size
            y_end = base_image_size

        else:
            mask = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
            x_start = 0
            y_start = base_image_size // 2
            x_end = base_image_size
            y_end = base_image_size

        ## Upscaling the mask ##
        mask = F.upsample(
            mask.unsqueeze(0).unsqueeze(0),
            size=(base_image_size, base_image_size),
            mode="nearest",
        )

    ## Scale = half and placed at corners --> Setting 2 ##
    elif n == 2:
        final_label = 0.75*base_label+0.25*cutmix_label
        cutmix_image_size = base_image_size // 2
        cutmix_transform_list = [
            transforms.Resize(cutmix_image_size),
            transforms.CenterCrop(cutmix_image_size),
            transforms.ToTensor(),
        ]

        m = random.randint(1, 4)
        if m == 1:
            mask = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
            x_start = 0
            y_start = 0
            x_end = cutmix_image_size
            y_end = cutmix_image_size
            cutmix_transform_list.append(
                transforms.Pad((0, 0, cutmix_image_size, cutmix_image_size), fill=0)
            )

        elif m == 2:
            mask = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
            x_start = base_image_size - cutmix_image_size
            y_start = 0
            x_end = base_image_size
            y_end = cutmix_image_size
            cutmix_transform_list.append(
                transforms.Pad((cutmix_image_size, 0, 0, cutmix_image_size), fill=0)
            )

        elif m == 3:
            mask = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
            x_start = 0
            y_start = base_image_size - cutmix_image_size
            x_end = cutmix_image_size
            y_end = base_image_size
            cutmix_transform_list.append(
                transforms.Pad((0, cutmix_image_size, cutmix_image_size, 0), fill=0)
            )

        else:
            mask = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
            x_start = base_image_size - cutmix_image_size
            y_start = base_image_size - cutmix_image_size
            x_end = base_image_size
            y_end = base_image_size
            cutmix_transform_list.append(
                transforms.Pad((cutmix_image_size, cutmix_image_size, 0, 0), fill=0)
            )

        ## Upscaling the mask ##
        mask = F.upsample(
            mask.unsqueeze(0).unsqueeze(0),
            size=(base_image_size, base_image_size),
            mode="nearest",
        )

    ## Scale = 1/3 and placed along the borders --> Setting 3 ##
    elif n == 3:
        final_label = (8/9) * base_label+ (1/9) * cutmix_label

        # cutmix_image_size = base_image_size // 4
        cutmix_image_size = base_image_size // 3
        cutmix_transform_list = [
            transforms.Resize(cutmix_image_size),
            transforms.CenterCrop(cutmix_image_size),
            transforms.ToTensor(),
        ]

        mask = torch.zeros(base_image_size, base_image_size)

        m = random.randint(1, 4)

        if m == 1:
            x_start = 0
            y_start = random.randint(0, base_image_size - cutmix_image_size)
            cutmix_transform_list.append(
                transforms.Pad(
                    (
                        0,
                        y_start,
                        base_image_size - cutmix_image_size,
                        base_image_size - (y_start + cutmix_image_size),
                    ),
                    fill=0,
                )
            )

        elif m == 2:
            x_start = random.randint(0, base_image_size - cutmix_image_size)
            y_start = 0
            cutmix_transform_list.append(
                transforms.Pad(
                    (
                        x_start,
                        0,
                        base_image_size - (x_start + cutmix_image_size),
                        base_image_size - cutmix_image_size,
                    ),
                    fill=0,
                )
            )

        elif m == 3:
            x_start = base_image_size - cutmix_image_size
            y_start = random.randint(0, base_image_size - cutmix_image_size)
            cutmix_transform_list.append(
                transforms.Pad(
                    (
                        x_start,
                        y_start,
                        0,
                        base_image_size - (y_start + cutmix_image_size),
                    ),
                    fill=0,
                )
            )

        else:
            x_start = random.randint(0, base_image_size - cutmix_image_size)
            y_start = base_image_size - cutmix_image_size
            cutmix_transform_list.append(
                transforms.Pad(
                    (
                        x_start,
                        y_start,
                        base_image_size - (x_start + cutmix_image_size),
                        0,
                    ),
                    fill=0,
                )
            )

        mask[
            y_start : y_start + cutmix_image_size, x_start : x_start + cutmix_image_size
        ] = 1.0
        x_end = x_start + cutmix_image_size
        y_end = y_start + cutmix_image_size

        mask = mask.unsqueeze(0).unsqueeze(0)

    ## Scale quarter but not in the border and not in the center --> Setting 4 ##
    else:
        final_label = (15/16) * base_label+ (1/16) * cutmix_label
        center = base_image_size // 2
        red_zone_x_start = center - int(0.1 * base_image_size)
        red_zone_x_end = center + int(0.1 * base_image_size)
        red_zone_y_start = center - int(0.1 * base_image_size)
        red_zone_y_end = center + int(0.1 * base_image_size)

        cutmix_image_size = base_image_size // 4

        val = True
        while val:
            x_start = random.randint(
                int(0.05 * base_image_size),
                base_image_size - cutmix_image_size - int(0.05 * base_image_size),
            )
            y_start = random.randint(
                int(0.05 * base_image_size),
                base_image_size - cutmix_image_size - int(0.05 * base_image_size),
            )
            x_end = x_start + cutmix_image_size
            y_end = y_start + cutmix_image_size

            if check_overlap(
                red_zone_x_start,
                red_zone_y_start,
                red_zone_x_end,
                red_zone_y_end,
                x_start,
                y_start,
                x_end,
                y_end,
            ):
                continue

            else:
                val = False

        # print(red_zone_x_start, red_zone_x_end, red_zone_y_start, red_zone_y_end)
        # print(x_start, x_end, y_start, y_end)

        cutmix_transform_list = [
            transforms.Resize(cutmix_image_size),
            transforms.CenterCrop(cutmix_image_size),
            transforms.ToTensor(),
            transforms.Pad(
                (
                    x_start,
                    y_start,
                    base_image_size - x_end,
                    base_image_size - y_end,
                ),
                fill=0,
            ),
        ]

        mask = torch.zeros(base_image_size, base_image_size)
        mask[
            y_start : y_start + cutmix_image_size, x_start : x_start + cutmix_image_size
        ] = 1.0

        mask = mask.unsqueeze(0).unsqueeze(0)

    ## Creating the final cutmix transform ##
    cutmix_transform = transforms.Compose(cutmix_transform_list)
    ## Apply the transformations ##
    base_image = base_transform(pil_base_image)
    cutmix_image = cutmix_transform(pil_cutmix_image)

    # print(f"Scale : {n}")
    final_img = base_image * (1 - mask.to(base_image.device)) + cutmix_image * mask.to(
        base_image.device
    )
    return torch.squeeze(final_img), final_label

def create_all_images(
    x, y,base_image_size, setting,num_classes,device,dataset_times=1
):
    """Creates all the cutmix images.

    :param img_dir: Directory of the images.
    :param base_image_size: Size of the base image.
    :param setting: Setting for the cutmix image.
    :param dataset_times: How many times of imagenet is the cutmix dataset?
    :param out_dir: Directory to save the images.
    :param img_extension: Extension of the images.
    """
    ## Getting all the directories of imagenet ##
    batch_size = x.size()[0]
    unique_labels = torch.unique(y)
    final_imgs = []
    final_labels = []
    for index in range(batch_size):
        base_image = x[index]
        base_lable = y[index].item()

        other_class_images = []
        other_class_labels = []
        for label in unique_labels:
            if label != base_lable:
                other_class_images.extend(x[y ==label ])
                other_class_labels.extend(y[y == label ])

        if len(other_class_images) < dataset_times:
            raise ValueError("Not enough images from other classes to perform CutMix.")

        cutmix_img_indices = random.sample(range(len(other_class_images)), dataset_times)
        cutmix_imgs = [other_class_images[i] for i in cutmix_img_indices]
        cutmix_labels = [other_class_labels[i] for i in cutmix_img_indices]  # 获取对应的标签

        for cutmix_img,cutmix_label in zip(cutmix_imgs,cutmix_labels):
            final_img,final_y = prepare_images(
                base_image, cutmix_img, torch.tensor(base_lable), cutmix_label,num_classes, device,base_image_size, setting
            )
        final_imgs.append(final_img)
        final_labels.append(final_y)
    return torch.stack(final_imgs),torch.stack(final_labels)