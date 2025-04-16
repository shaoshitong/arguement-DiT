from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import numpy as np
import torch
import os
import json
from tqdm import tqdm
import random

def flatten(nest_list:list):
    return [j for i in nest_list for j in flatten(i)] if isinstance(nest_list, list) else [nest_list]

class CustomDataset(Dataset):
    def __init__(self, root, image_size, transform=None, 
                 select_info=None, select_type="random", 
                 select_num=10000, interval=1, cpu_load=False):
        self.root = root
        self.cpu_load = cpu_load
        if select_info is not None:
            print(type(select_info))
            assert isinstance(select_info, dict), "select_info must be a dict"
            assert select_type in ["random", "min", "max", "dynamic"], "select_type must be 'random' or 'min' or 'max' or 'dynamic'"
            _tmp = [[key, value] for key, value in select_info.items()][::interval]
            if select_type == "dynamic":
                print("="*50)
                print("dynamic selection is only used for full dataset")
                print("="*50)
                
        self.image_size = image_size
        self.interval = interval
        self.select_type = select_type
        self.select_num = select_num
        self.transform = transform

        # 使用 ImageFolder 加载图像路径和类信息
        self.image_folder = ImageFolder(root=root, transform=self.transform)
        self.class_to_idx = self.image_folder.class_to_idx
        self.num_classes = len(self.class_to_idx)
        self.class_names = list(self.class_to_idx.keys())
        
        self.class_images = []
        for i in range(1000):
            self.class_images.append(self.get_class_images(i))

        if select_info is not None:
            pre_select_num = int(select_num / self.num_classes)
            if select_type == "random":
                json_path = "./baseline_dataset_list"
                if select_num == 10000:
                    json_path = json_path + "/baseline_image_paths_10k_new.json"
                elif select_num == 50000:
                    json_path = json_path + "/baseline_image_paths_50k_new.json"
                elif select_num == 100000:
                    json_path = json_path + "/baseline_image_paths_100k_new.json"
                else:
                    raise ValueError("select_num must be 10000 or 50000 or 100000")
                with open(json_path, "r") as f:
                    select_training_samples = json.load(f)
                for i in range(len(select_training_samples)):
                    select_training_samples[i] = select_training_samples[i].replace("/data/shared_data/ILSVRC2012", "/mnt/weka/st_workspace/DDDM/ILSVRC/ILSVRC2012")
                dict_path2class = {self.image_folder.samples[i][0]: self.image_folder.samples[i][1] for i in range(len(self.image_folder.samples))}
                new_samples = []
                for d in select_training_samples:
                    _cls = dict_path2class[d]
                    new_samples.append((d, _cls))
                self.image_folder.samples = new_samples
                print("Random select", len(self.image_folder.samples))
            else:
                if select_type == "dynamic":
                    self.select_info = sorted(_tmp, key=lambda x: x[1], reverse=False)
                else:
                    self.select_info = sorted(_tmp, key=lambda x: x[1], reverse= (select_type == "max"))
                for i in range(len(self.select_info)):
                    self.select_info[i][0] = self.select_info[i][0].replace("/data/shared_data/ILSVRC2012", "/mnt/weka/st_workspace/DDDM/ILSVRC/ILSVRC2012")
                self.dict_path2class = dict_path2class = {self.image_folder.samples[i][0]: self.image_folder.samples[i][1] for i in range(len(self.image_folder.samples))}
                final_dataset = [[] for _ in range(1000)]
                mean_value = 0
                for _select_info in self.select_info:
                    _cls = dict_path2class[_select_info[0]]
                    if len(final_dataset[_cls]) < pre_select_num:
                        final_dataset[_cls].append((_select_info[0], _cls))
                        mean_value += _select_info[1]
                    else:
                        continue
                self.image_folder.samples = flatten(final_dataset)
                print("Mean value", mean_value / len(self.image_folder.samples))
        
        if self.cpu_load:
            print("CPU loading ...")
            new_image_folder = []
            for img_path, class_idx in tqdm(self.image_folder.samples):
                image = self.image_folder.loader(img_path)
                if self.transform:
                    image = self.transform(image)
                new_image_folder.append((image, class_idx))
            self.image_folder.samples = new_image_folder
            

    def __len__(self):
        return len(self.image_folder.samples)

    def __getitem__(self, index):
        if self.cpu_load:
            image, class_idx = self.image_folder.samples[index]
        else:
            img_path, class_idx = self.image_folder.samples[index]
            image = self.image_folder.loader(img_path)
            if self.transform:
                image = self.transform(image)


        return image, class_idx

    def update_dataset(self, iteration, total_iteration=200000):
        assert self.select_type == "dynamic", "update_dataset is only used for dynamic selection"
        if iteration > total_iteration:
            iteration = total_iteration
            tag = int(random.random() > 0.5)
        else:
            import random
            tag = 1
            
        ratio = float(iteration/total_iteration)
        middle_point = len(self.select_info) * (1/2)
        start_point = 0
        current_point = int((middle_point - start_point) * ratio + start_point)
        cur_begin_point = int(max(0, current_point - self.select_num // 2))
        cur_end_point = int(min(len(self.select_info), current_point + self.select_num // 2))

        final_dataset = []
        mean_value = 0
        if tag == 1:
            for _select_info in self.select_info[cur_begin_point:cur_end_point]:
                _cls = self.dict_path2class[_select_info[0]]
                final_dataset.append((_select_info[0], _cls))
        else:
            for _zz in np.random.choice(list(range(len(self.select_info))), size=self.select_num, replace=False):
                _cls = self.dict_path2class[self.select_info[_zz][0]]
                final_dataset.append((self.select_info[_zz][0], _cls))
        self.image_folder.samples = final_dataset
        print("Mean value", mean_value / len(self.image_folder.samples))

    def get_class_images(self, class_idx):
        class_images = []
        for img_path, _class_idx in self.image_folder.samples:
            if _class_idx == class_idx:
                class_images.append((img_path, class_idx))
        return class_images

class CustomDataset_OLD(Dataset):
    def __init__(self, root, image_size, transform=None, 
                 select_info=None, select_type="random", select_num=10000, cpu_load=True):
        self.root = root
        self.cpu_load = cpu_load
        if select_info is not None:
            print(type(select_info))
            assert isinstance(select_info, dict), "select_info must be a dict"
            assert select_type in ["random", "min", "max"], "select_type must be 'random' or 'min' or 'max'"
            _tmp = [[key, value] for key, value in select_info.items()]

        self.image_size = image_size
        self.transform = transform

        # 使用 ImageFolder 加载图像路径和类信息
        self.image_folder = ImageFolder(root=root, transform=self.transform)
        self.class_to_idx = self.image_folder.class_to_idx
        self.num_classes = len(self.class_to_idx)
        self.class_names = list(self.class_to_idx.keys())
        
        self.class_images = []
        for i in range(1000):
            self.class_images.append(self.get_class_images(i))

        if select_info is not None:
            pre_select_num = int(select_num / self.num_classes)
            if select_type == "random":
                json_path = "./baseline_dataset_list"
                if select_num == 10000:
                    json_path = json_path + "/baseline_image_paths_10k_new.json"
                    print(json_path)
                elif select_num == 50000:
                    json_path = json_path + "/baseline_image_paths_50k_new.json"
                    print(json_path)
                elif select_num == 100000:
                    json_path = json_path + "/baseline_image_paths_100k_new.json"
                    print(json_path)
                else:
                    raise ValueError("select_num must be 10000 or 50000 or 100000")
                with open(json_path, "r") as f:
                    select_training_samples = json.load(f)
                for i in range(len(select_training_samples)):
                    select_training_samples[i] = select_training_samples[i].replace("/data/shared_data/ILSVRC2012", "/mnt/weka/st_workspace/DDDM/ILSVRC/ILSVRC2012")
                dict_path2class = {self.image_folder.samples[i][0]: self.image_folder.samples[i][1] for i in range(len(self.image_folder.samples))}
                new_samples = []
                for d in select_training_samples:
                    _cls = dict_path2class[d]
                    new_samples.append((d, _cls))
                self.image_folder.samples = new_samples
                print("Random select", len(self.image_folder.samples))
            else:
                self.select_info = sorted(_tmp, key=lambda x: x[1], reverse= (select_type == "max"))
                print("="*50)
                print(self.select_info[:40])
                print("="*50)
                dict_path2class = {self.image_folder.samples[i][0]: self.image_folder.samples[i][1] for i in range(len(self.image_folder.samples))}
                final_dataset = [[] for _ in range(1000)]
                for _select_info in self.select_info[::8]:
                    _cls = dict_path2class[_select_info[0]]
                    if len(final_dataset[_cls]) < pre_select_num:
                        final_dataset[_cls].append((_select_info[0], _cls))
                    else:
                        continue
                self.image_folder.samples = flatten(final_dataset)
        
        if self.cpu_load:
            print("CPU loading ...")
            new_image_folder = []
            for img_path, class_idx in tqdm(self.image_folder.samples):
                image = self.image_folder.loader(img_path)
                if self.transform:
                    image = self.transform(image)
                new_image_folder.append((image, class_idx))
            self.image_folder.samples = new_image_folder

    def __len__(self):
        return len(self.image_folder.samples)

    def __getitem__(self, index):
        if self.cpu_load:
            image, class_idx = self.image_folder.samples[index]
        else:
            img_path, class_idx = self.image_folder.samples[index]
            image = self.image_folder.loader(img_path)
            if self.transform:
                image = self.transform(image)

        return image, class_idx

    def get_class_images(self, class_idx):
        class_images = []
        for img_path, _class_idx in self.image_folder.samples:
            if _class_idx == class_idx:
                class_images.append((img_path, class_idx))
        return class_images
