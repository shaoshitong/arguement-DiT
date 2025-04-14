# python -m torch.distributed.run --nproc-per-node=8 train_caption_wocutmix.py --select_type min --select_num 50000

python -m torch.distributed.run --nproc-per-node=8 fastdit_c2i.py --select_type random --select_num 50000