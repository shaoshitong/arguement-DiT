

from pathlib import Path
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

class CustomDataset(Dataset):
    def __init__(self, root,cutmix_root, image_size, transform=None):
        self.root = root
        self.image_size = image_size
        self.transform = transform
        
        # 使用 ImageFolder 加载图像数据
        self.image_folder = ImageFolder(root=root, transform=self.transform)
        self.cutmix_folder = ImageFolder(root=cutmix_root, transform=self.transform)

        # 获取所有类的索引
        self.class_to_idx = self.image_folder.class_to_idx
        self.num_classes = len(self.class_to_idx)
        
        # 获取类别名称
        self.class_names = list(self.class_to_idx.keys())
        
        self.samples = []
        
        for class_name in self.class_names:
            caption_file = Path(root) / class_name / "captions.csv"
            with open(caption_file, 'r', encoding='utf-8') as f:
                captions = [line.strip() for line in f if line.strip()]
            self.samples.append(captions)

    def __len__(self):
        return len(self.image_folder.samples)

    def __getitem__(self, index):
        img_path, _ = self.image_folder.samples[index]  # 获取图像路径
        image = self.image_folder.loader(img_path)  # 使用 ImageFolder 中的加载方法加载图像
        
        cutmix_path, _ = self.cutmix_folder.samples[index]
        cutmix_image = self.cutmix_folder.loader(cutmix_path)


        if self.transform:
            image = self.transform(image)  # 执行预处理操作
            cutmix_image = self.transform(cutmix_image)
        
        # 获取图像所属的类别索引
        class_name = self.image_folder.classes[self.image_folder.samples[index][1]]
        captions = self.samples[self.class_to_idx[class_name]]  # 获取该类的所有 captions

        # 随机选择一个 caption
        caption = random.choice(captions)
        
        parts = cutmix_path.split('/')
        cutmix_label1 = parts[-1].split('_')[0]  # 第一个类标签
        cutmix_label2 = parts[-1].split('_')[2]  # 第二个类标签
        cutmix_caption1 = random.choice(self.samples[self.class_to_idx[cutmix_label1]])
        cutmix_caption2 = random.choice(self.samples[self.class_to_idx[cutmix_label2]])


        return image, caption,cutmix_image,(cutmix_caption1,cutmix_caption2)

class CutmixDataset(Dataset):
    def __init__(self, root, raw_root, image_size, transform=None):
        self.root = root
        self.raw_root = root
        self.image_size = image_size
        self.transform = transform
        
        # 使用 ImageFolder 加载图像数据
        self.image_folder = ImageFolder(root=root, transform=self.transform)
        
        # 获取所有类的索引
        self.class_to_idx = self.image_folder.class_to_idx
        self.num_classes = len(self.class_to_idx)
        
        # 获取类别名称
        self.class_names = list(self.class_to_idx.keys())
        
        self.samples = []
        
        for class_name in self.class_names:
            caption_file = Path(root) / class_name / "captions.csv"
            with open(caption_file, 'r', encoding='utf-8') as f:
                captions = [line.strip() for line in f if line.strip()]
            self.samples.append(captions)

    def __len__(self):
        return len(self.image_folder.samples)

    def __getitem__(self, index):
        img_path, _ = self.image_folder.samples[index]  # 获取图像路径
        image = self.image_folder.loader(img_path)  # 使用 ImageFolder 中的加载方法加载图像
        
        if self.transform:
            image = self.transform(image)  # 执行预处理操作

        # 获取图像所属的类别索引
        class_name = self.image_folder.classes[self.image_folder.samples[index][1]]
        captions = self.samples[self.class_to_idx[class_name]]  # 获取该类的所有 captions

        # 随机选择一个 caption
        caption = random.choice(captions)

        return image, caption