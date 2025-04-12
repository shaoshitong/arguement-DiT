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
from collections import defaultdict

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


def prepare_images(pil_base_image, pil_cutmix_image, base_image_size=256, n=5):
    """Prepares the cutmix images.

    :param pil_base_image: PIL image of the base image.
    :param pil_cutmix_image: PIL image of the cutmix image.
    :param base_image_size: Size of the base image. Default: 256.
    :param n: Setting for the cutmix image. Default: 5.
    """

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

    ## Scale = same as base --> Setting 1 ##
    if n == 1:
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

    return final_img, x_start, y_start, x_end, y_end


def create_all_images(
    img_dir, base_image_size, setting, dataset_times, out_dir, img_extension
):
    # 获取所有类别目录
    class_directories = sorted(glob(f"{img_dir}/*"))
    all_classes = [os.path.basename(d) for d in class_directories]
    class_to_index = {cls: idx for idx, cls in enumerate(all_classes)}
    num_classes = len(class_to_index)

    print(f"[INFO] Total classes: {num_classes}")

    # 为每个类保存一个标签列表
    class_label_dict = defaultdict(list)

    # 遍历每一个 base 类
    for idx, class_directory in enumerate(class_directories):
        base_class = os.path.basename(class_directory)
        base_class_index = class_to_index[base_class]

        current_class_image_paths = sorted(
            glob(f"{class_directory}/**/*.{img_extension}", recursive=True)
        )
        if not current_class_image_paths:
            continue

        print(f"[DEBUG] Processing class: {base_class} - {len(current_class_image_paths)} images")

        # 获取所有其他类图像路径
        other_class_image_paths = []
        for other_class_dir in class_directories:
            if other_class_dir == class_directory:
                continue
            other_class_image_paths.extend(
                glob(f"{other_class_dir}/**/*.{img_extension}", recursive=True)
            )

        # 创建输出路径
        save_img_dir = os.path.join(out_dir, base_class, "images")
        os.makedirs(save_img_dir, exist_ok=True)

        # 遍历当前类下所有图像
        for idx, img_path in tqdm(
            enumerate(current_class_image_paths), total=len(current_class_image_paths)
        ):
            base_img_name = os.path.basename(img_path).split(".")[0]
            base_img = Image.open(img_path).convert("RGB")

            # 随机采样 cutmix 图像
            cutmix_img_paths = random.sample(other_class_image_paths, dataset_times)

            for cutmix_img_path in cutmix_img_paths:
                cutmix_img_name = os.path.basename(cutmix_img_path).split(".")[0]
                cutmix_img = Image.open(cutmix_img_path).convert("RGB")

                # 调用外部函数生成 cutmix 图像（你已有）
                final_img, x_start, y_start, x_end, y_end = prepare_images(
                    base_img, cutmix_img, base_image_size, setting
                )

                # 构造保存名
                save_name = f"{base_img_name}_{cutmix_img_name}_{x_start}_{y_start}_{x_end}_{y_end}.{img_extension}"
                save_path = os.path.join(save_img_dir, save_name)
                save_image(final_img, save_path)

                # 生成标签：base + cutmix 的 one-hot 平均
                cutmix_class = os.path.basename(os.path.dirname(os.path.dirname(cutmix_img_path)))

                cutmix_class_index = class_to_index[cutmix_class]

                label = np.zeros(num_classes, dtype=np.float32)
                label[base_class_index] = 0.75
                label[cutmix_class_index] = 0.25
                class_label_dict[base_class].append(label)

    # 保存每个类下的 labels.npy
    for cls, labels in class_label_dict.items():
        label_array = np.stack(labels)
        label_path = os.path.join(out_dir, cls, "labels.npy")
        np.save(label_path, label_array)
        print(f"[✓] Saved labels for class {cls}: {label_array.shape} → {label_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_image_size", type=int, default=256)
    parser.add_argument("--setting", type=int, default=2)
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/shaoshitong/project/argument-DiT/dataset/tiny-imagenet-200/train",
    )
    parser.add_argument(
        "--dataset_times",
        type=int,
        default=1,
        help="How many times of imagenet is the cutmix dataset?",
    )
    parser.add_argument("--out_dir", type=str, default="/home/shaoshitong/project/argument-DiT/dataset/tiny-imagenet-200/cutmix_train")
    parser.add_argument("--img_extension", type=str, default="JPEG")
    args = parser.parse_args()

    create_all_images(
        args.input_dir,
        args.base_image_size,
        args.setting,
        args.dataset_times,
        args.out_dir,
        args.img_extension,
    )

defaultdict