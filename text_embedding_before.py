from dataset.dataset_wocutmix import CustomDataset
from torchvision import transforms
from PIL import Image
import numpy as np

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


transform = transforms.Compose([
    transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
])

import json
with open("/mnt/weka/st_workspace/DDDM/arguement-DiT/baseline_dataset_list/merged.json", "r") as f:
    select_info = json.load(f) # list[dict, dict, ...]
merged_dict = {}
for d in select_info:
    merged_dict.update(d)

dataset = CustomDataset(
    root="/mnt/weka/st_workspace/DDDM/ILSVRC/ILSVRC2012/train",
    image_size=256,
    transform=transform,
    select_info=merged_dict,
    select_type="min",
    select_num=10000,
    interval=64,
)

samples = dataset.image_folder.samples

losses = []
for i in range(len(samples)):
    try:
        loss = dataset._tmp[samples[i][0]]
        losses.append(loss)
    except:
        print(samples[i][0], "not found")

z = np.array(losses)
# Mean, Variance
# 0.4006641214704132 0.03149787888274977, 10000 random
# 0.3359007159821689 0.04558494286699252, 10000 min interval 8
# 0.35155708361417054 0.041304655875590325, 10000 min interval 16
# 0.36683580516427755 0.03712526658556503, 10000 min interval 32
# 0.3819912439838052 0.033171012512993876, 10000 min interval 64

# 0.4002776589280248 0.03162617037407476, 50000 random
# 0.37156111012622717 0.03460542526707638, 50000 min interval 8
# 0.3562120408923924 0.039105937783951716, 50000 min interval 4
print(np.mean(z,axis=0), np.std(z,axis=0))


