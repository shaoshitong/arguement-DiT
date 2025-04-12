import os
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm

def generate_onehot_labels(train_dir, image_ext="JPEG"):
    train_dir = Path(train_dir)
    class_dirs = sorted([p for p in train_dir.iterdir() if p.is_dir()])
    class_names = [p.name for p in class_dirs]
    class_to_index = {cls: idx for idx, cls in enumerate(class_names)}
    num_classes = len(class_names)

    for class_dir in tqdm(class_dirs, desc="Generating labels"):
        image_dir = class_dir / "images"
        image_paths = sorted(image_dir.glob(f"*.{image_ext}"))
        if not image_paths:
            continue

        onehot_labels = np.zeros((len(image_paths), num_classes), dtype=np.float32)
        class_index = class_to_index[class_dir.name]
        onehot_labels[:, class_index] = 1.0

        np.save(class_dir / "labels.npy", onehot_labels)

        print(f"[✓] Saved {len(image_paths)} labels → {class_dir/'labels.npy'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="/home/shaoshitong/project/argument-DiT/dataset/tiny-imagenet-200/train")
    parser.add_argument("--image_ext", default="JPEG")
    args = parser.parse_args()

    generate_onehot_labels(args.train_dir, args.image_ext)
