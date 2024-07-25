#! /bin/bash

## waymo mamba-L
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/lion_models/lion_mamba_waymo_8x_1f_1x_one_stride_128dim.yaml \
--extra_tag lion_mamba_waymo_8x_1f_1x_one_stride_128dim \
--batch_size 16 --epochs 24 --max_ckpt_save_num 4 --workers 4 --sync_bn

## waymo mamba
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/lion_models/lion_mamba_waymo_8x_1f_1x_one_stride_64dim.yaml \
--extra_tag lion_mamba_waymo_8x_1f_1x_one_stride_64dim \
--batch_size 16 --epochs 24 --max_ckpt_save_num 4 --workers 4 --sync_bn


## waymo retnet
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/lion_models/lion_retnet_waymo_8x_1f_1x_one_stride_64dim.yaml \
--extra_tag lion_retnet_waymo_8x_1f_1x_one_stride_64dim \
--batch_size 16 --epochs 24 --max_ckpt_save_num 4 --workers 4 --sync_bn


## waymo rwkv
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/lion_models/lion_rwkv_waymo_8x_1f_1x_one_stride_64dim.yaml \
--extra_tag lion_rwkv_waymo_8x_1f_1x_one_stride_64dim \
--batch_size 16 --epochs 24 --max_ckpt_save_num 4 --workers 4 --sync_b


