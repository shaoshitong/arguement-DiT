# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
from datetime import datetime
import torch
import torchvision.utils as vutils
from pathlib import Path
import pandas as pd
import random
from dataset.dataset_wocutmix import CustomDataset_REPA as CustomDataset
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import torch.nn.functional as F
from sit import SiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import torchvision.transforms.functional as Fun
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def setup_environment():
    """检查并设置分布式训练环境"""
    if "RANK" not in os.environ:
        print("🚨 Warning: RANK is not set, assuming single GPU mode.")
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29533"

    # 仅当需要多 GPU 训练时，初始化分布式环境
    if int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")
        print(f"✅ Distributed initialized: rank {os.environ['RANK']} / {os.environ['WORLD_SIZE']}")
    else:
        print("✅ Running in single GPU mode. Distributed training is disabled.")

# 运行环境设置
setup_environment()
#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def build_dataset(args, select_num, merged_dict, transform, rank):
    assert select_num in [10000, 50000, 100000], "select_num must be 10000, 50000, or 100000"
    
    if select_num == 10000:
        dataset = CustomDataset(args.data_path,image_size=args.image_size,transform=transform,
                                select_info=merged_dict, select_type=args.select_type, 
                                select_num=10000, interval=100,
                                embedding_root="../REPA-E/svd_feat_10k_min_interval_100",
                                embedding_json="../REPA-E/svd_feat_10k_min_interval_100/total_json.json",
                                target_mean=args.target_mean,
                                cpu_load=False)
    elif select_num == 50000:
        dataset = CustomDataset(args.data_path,image_size=args.image_size,transform=transform,
                                select_info=merged_dict, select_type=args.select_type, 
                                select_num=50000, interval=16,
                                embedding_root="../REPA-E/svd_feat_50k_min_interval_16",
                                embedding_json="../REPA-E/svd_feat_50k_min_interval_16/total_json.json",
                                target_mean=args.target_mean,
                                cpu_load=False)
        print("50000 dataset")
    elif select_num == 100000:
        dataset = CustomDataset(args.data_path,image_size=args.image_size,transform=transform,
                                select_info=merged_dict, select_type=args.select_type, 
                                select_num=100000, interval=10,
                                embedding_root="../REPA-E/svd_feat_100k_min_interval_10",
                                embedding_json="../REPA-E/svd_feat_100k_min_interval_10/total_json.json",
                                target_mean=args.target_mean,
                                cpu_load=False)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoaderX(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True
    )
    return dataset, sampler, loader

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


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


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # Setup DDP:
    if not dist.is_initialized():
        dist.init_process_group("nccl")

    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    image_size = args.image_size
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank

    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Get current timestamp
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        if args.resume_ckpt is not None and os.path.isdir(args.resume_ckpt):
            experiment_dir = args.resume_ckpt.split("/checkpoints")[0]
            print(f"Resuming from checkpoint: {experiment_dir}")
            checkpoint_dir = f"{experiment_dir}/checkpoints"
        else:
            experiment_dir = f"{args.results_dir}/{timestamp}-{model_string_name}-{args.select_type}-all"  # Create an experiment folder
            checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
            os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        # 保存配置信息到config.json
        config_path = os.path.join(checkpoint_dir, "config.json")
        config_dict = vars(args)  # 将参数对象转换为字典
        import json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        logger.info(f"配置信息已保存到 {config_path}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    block_kwargs = {"fused_attn": True, "qk_norm": False}
    model = SiT_models[args.model](
        input_size=latent_size,
        in_channels=4,
        num_classes=1000,
        class_dropout_prob=0.1,
        z_dims=[768],
        encoder_depth=8,
        bn_momentum=0.1,
        **block_kwargs
    )
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    if args.resume_ckpt is not None:
        if os.path.isdir(args.resume_ckpt):
            # Find the checkpoint with the largest step number
            checkpoint_files = glob(os.path.join(args.resume_ckpt, "*.pt"))
            if not checkpoint_files:
                raise ValueError(f"No checkpoint files found in {args.resume_ckpt}")
            # Extract step numbers and find the largest one
            steps = [int(os.path.basename(f).split('.')[0]) for f in checkpoint_files]
            latest_checkpoint = checkpoint_files[steps.index(max(steps))]
            checkpoint = torch.load(latest_checkpoint, map_location=torch.device('cuda'))
            model.load_state_dict(checkpoint["model"])
            ema.load_state_dict(checkpoint["ema"])
            model = DDP(model.to(device), device_ids=[rank])
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
            # opt.load_state_dict(checkpoint["opt"])
            if rank == 0:
                logger.info(f"Resumed from checkpoint: {latest_checkpoint}")
        else:
            checkpoint = torch.load(args.resume_ckpt, map_location=torch.device('cuda'))
            latest_checkpoint = args.resume_ckpt
            model.load_state_dict(checkpoint["model"])
            ema.load_state_dict(checkpoint["ema"])
            model = DDP(model.to(device), device_ids=[rank])
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
            # opt.load_state_dict(checkpoint["opt"])
    else:
        model = DDP(model.to(device), device_ids=[rank])
        # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
        latest_checkpoint = None
    # Note that parameter initialization is done within the DiT constructor
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    import json
    with open(args.select_info, "r") as f:
        select_info = json.load(f) # list[dict, dict, ...]
    merged_dict = {}
    for d in select_info:
        merged_dict.update(d)

    dataset, sampler, loader = build_dataset(args, 50000, merged_dict, transform, rank)
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # 加载每个类的 caption_embedding.npy（每类一个）
    embedding_dict = {}
    embedding_mask_dict = {}
    prompt_root = "/mnt/weka/st_workspace/DDDM/ILSVRC/ILSVRC2012/caption_embeddings_tmp"

    for class_name in dataset.class_names:
        embedding_file = os.path.join(prompt_root, class_name + ".pt")
        embedding = torch.load(embedding_file, map_location=torch.device('cpu'), weights_only=True).cuda()
        embedding_dict[class_name] = embedding
        embedding_mask_file = os.path.join(prompt_root, class_name + "_mask.pt")
        embedding_mask = torch.load(embedding_mask_file, map_location=torch.device('cpu'), weights_only=True).cuda()
        embedding_mask_dict[class_name] = embedding_mask

    # Prepare models for training:
    if args.resume_ckpt is None:
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0 if latest_checkpoint is None else int(latest_checkpoint.split("checkpoints/")[1].split(".")[0])
    print(f"begin train_steps: {train_steps}")
    log_steps = 0
    running_loss = 0
    running_denoising_loss = 0
    running_proj_loss = 0
    stage2_tag = 0
    stage3_tag = 0
    start_time = time()
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, class_idx, zs in loader:
            # print(x.min(), x.max(), x.mean() ,"x")
            zs = [zs.to(device)]
            text_embedding = []
            text_embedding_mask = []
            for _i in class_idx:
                class_name = dataset.image_folder.classes[_i]
                text_embedding.append(embedding_dict[class_name].squeeze(0))
                text_embedding_mask.append(embedding_mask_dict[class_name].squeeze(0))
            text_embedding = torch.stack(text_embedding, dim=0)
            text_embedding_mask = torch.stack(text_embedding_mask, dim=0)

            with torch.no_grad():
                x = vae.encode(x.to(device)).latent_dist.sample().mul_(0.18215)
                # print(x.min(), x.max(), x.mean() ,"vae_x")
            loss_kwargs = dict(
                path_type="linear",
                prediction="v",
                weighting="uniform",
                align_only=False,
            )
            denoising_loss, proj_loss, predicted_x, time_input = model(
                x = x, y = text_embedding.to(torch.float32).to(device), ccls = class_idx.to(device),
                loss_kwargs = loss_kwargs,
                attn_mask=text_embedding_mask.to(torch.float32).to(device),
                zs=zs,
            )
            # print(predicted_x.min(), predicted_x.max(), predicted_x.mean() ,"predicted_x")
            with torch.no_grad():
                if rank == 0 and train_steps % 5000 == 0:
                    if not os.path.exists(f"{args.results_dir}/samples"):
                        os.makedirs(f"{args.results_dir}/samples")
                    predicted_x = vae.decode(predicted_x / 0.18215).sample
                    predicted_x = torch.clamp(127.5 * predicted_x + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                    for i, sample in enumerate(predicted_x):
                        Image.fromarray(sample).save(f"{args.results_dir}/samples/{train_steps:07d}_{i:06d}_{time_input[i].view(-1).item()}.png")

                    gt_x = vae.decode(x / 0.18215).sample
                    gt_x = torch.clamp(127.5 * gt_x + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                    for i, sample in enumerate(gt_x):
                        Image.fromarray(sample).save(f"{args.results_dir}/samples/{train_steps:07d}_{i:06d}_{time_input[i].view(-1).item()}_gt.png")

            loss = denoising_loss + proj_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            running_denoising_loss += denoising_loss.item()
            running_proj_loss += proj_loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_denoising_loss = torch.tensor(running_denoising_loss / log_steps, device=device)
                avg_proj_loss = torch.tensor(running_proj_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_denoising_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_proj_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_denoising_loss = avg_denoising_loss.item() / dist.get_world_size()
                avg_proj_loss = avg_proj_loss.item() / dist.get_world_size()
                if rank == 0:
                    print(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Denoising Loss: {avg_denoising_loss:.4f}, Train Proj Loss: {avg_proj_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                running_denoising_loss = 0
                running_proj_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

        # if train_steps >= 100000 and train_steps < 200000 and stage2_tag == 0:
        #     print("build 50000 dataset")
        #     dataset, sampler, loader = build_dataset(args, 50000, merged_dict, transform, rank)
        #     stage2_tag = 1
        #     dist.barrier()
        # elif train_steps >= 200000 and stage3_tag == 0:
        #     print("build 100000 dataset")
        #     dataset, sampler, loader = build_dataset(args, 100000, merged_dict, transform, rank)
        #     stage3_tag = 1
        #     dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":



    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default = "/mnt/weka/st_workspace/DDDM/ILSVRC/ILSVRC2012/train")
    parser.add_argument("--cutmix_data_path", type=str, default = "/mnt/weka/st_workspace/DDDM/ILSVRC/ILSVRC2012/train")
    parser.add_argument("--select_info", type=str, default = "./baseline_dataset_list/merged.json")
    parser.add_argument("--select_type", type=str, default = "min")
    parser.add_argument("--interval", type=int, default = 4)
    parser.add_argument("--select_num", type=int, default = 50000)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=43)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument("--ckpt-every", type=int, default=20_000)
    parser.add_argument("--embedding-root", type=str, default="/mnt/weka/st_workspace/DDDM/REPA-E/svd_feat_10k_middle_interval_4")
    parser.add_argument("--embedding-json", type=str, default="/mnt/weka/st_workspace/DDDM/REPA-E/svd_feat_10k_middle_interval_4/total_json.json")
    parser.add_argument("--target-mean", type=float, default = 0.)
    args = parser.parse_args()
    main(args)
