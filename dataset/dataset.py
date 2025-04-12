

from pathlib import Path
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np
class CustomDataset(Dataset):
    def __init__(self, root, image_size, transform=None):
        self.root = root
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
            caption_embedding_file = Path(root) / class_name / "captions_embedding.csv"
            embeddings = np.loadtxt(caption_embedding_file, delimiter=',', skiprows=1)  # 跳过第一行
            self.samples.append(embeddings)

    def __len__(self):
        return len(self.image_folder.samples)

    def __getitem__(self, index):
        img_path, _ = self.image_folder.samples[index]  # 获取图像路径
        image = self.image_folder.loader(img_path)  # 使用 ImageFolder 中的加载方法加载图像
        


        if self.transform:
            image = self.transform(image)  # 执行预处理操作
        
        # 获取图像所属的类别索引
        class_name = self.image_folder.classes[self.image_folder.samples[index][1]]
        embeddings = self.samples[self.class_to_idx[class_name]]  # 获取该类的所有 embedding

        # 随机选择一个 embedding
        embedding = random.choice(embeddings)  # 选择该类别中的一个 caption embedding
        
        return image, embedding  # 返回图像和对应的 embedding

