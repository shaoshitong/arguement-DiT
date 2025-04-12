import os
import json
import numpy as np

# 路径配置
train_root = '/data/shared_data/ILSVRC2012/train'
json_path = '/home/shaoshitong/project/argument-DiT/dataset/imagenet.json'

# 加载类别 index → 名称
with open(json_path, 'r') as f:
    index_to_label = json.load(f)

# 获取所有类别文件夹，并排序（index 顺序）
folders = sorted(os.listdir(train_root))

# 遍历每个类别
for idx, folder in enumerate(folders):
    folder_path = os.path.join(train_root, folder)
    
    if not os.path.isdir(folder_path):
        continue

    class_name = index_to_label.get(str(idx), "unknown")
    caption = f"the image of {class_name.replace('_', ' ')}"

    # 保存为 npy 文件（只包含一句话）
    save_path = os.path.join(folder_path, 'single_caption.npy')
    np.save(save_path, caption)
    print(f"✅ 写入: {folder} → {caption}")
