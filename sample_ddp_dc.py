# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

# 确保 samples 文件夹存在


"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusion.gaussian_diffusion import _extract_into_tensor
from diffusers.models import AutoencoderKL
from download import find_model
from models_old import DiT_models
import argparse
from PIL import Image
import numpy as np
from torchvision.datasets.folder import default_loader
from dataset.generate_argument import prepare_images

from torchvision import transforms
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

def get_image(img_path,transform):
    lodder = default_loader
    if img_path:
        img = lodder(img_path)
        image = transform(img).unsqueeze(0)
    return image





def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000


    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    image_path = "/home/shaoshitong/project/argument-DiT/images/n01614925/images/n01614925_13.JPEG"
    mix_image_path = "/data/shared_data/ILSVRC2012/train/n03598930/images/n03598930_7342.JPEG"
    # Labels to condition the model with (feel free to change):
    class_labels = [22]
    prepare_img = prepare_images(Image.open(image_path).convert("RGB"), Image.open(mix_image_path).convert("RGB"), 256, "cutmix")[0]
    prepare_img = prepare_img * 2 - 1
    mix_img = torch.clamp(127.5 * prepare_img + 128.0, 0, 255).to("cpu", dtype=torch.uint8).permute(0, 2, 3, 1).numpy()[0]
    print(mix_img.shape)
    Image.fromarray(mix_img).save("mix_img.png")
    x = vae.encode(prepare_img.to(device)).latent_dist.sample().mul_(0.18215)
    save_t = 50
    # begin refine
    t = torch.tensor([0] * x.shape[0], device=device)
    n = len(class_labels)
    y_null = torch.tensor([1000] * n, device=device)
    model_output = model.forward(x, t, y_null)
    model_output, model_var_values = torch.split(model_output, 4, dim=1)
    # min_log = _extract_into_tensor(diffusion.posterior_log_variance_clipped, t, x.shape)
    # max_log = _extract_into_tensor(np.log(diffusion.betas), t, x.shape)
    # # The model_var_values is [-1, 1] for [min_var, max_var].
    # frac = (model_var_values + 1) / 2
    # model_log_variance = frac * max_log + (1 - frac) * min_log
    # model_variance = torch.exp(model_log_variance)
    x_t = diffusion.q_sample(x,torch.tensor([save_t]*x.shape[0], device=device),noise=model_output)
    z = x_t
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop_with_t(
        model.forward_with_cfg, z.shape, z,
        clip_denoised=False, model_kwargs=model_kwargs, progress=True,
        device=device, begin_t=save_t
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample
    samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    os.makedirs("samples_refine", exist_ok=True)

    # Save samples to disk as   individual .png files
    for i, sample in enumerate(samples):
        image_filename = os.path.join("samples_refine", f"{int(save_t):06d}.png")
        Image.fromarray(sample).save(image_filename)
        print(f"Saved image: {image_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default="/home/shaoshitong/project/argument-DiT/DiT-XL-2-256x256.pt",
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
