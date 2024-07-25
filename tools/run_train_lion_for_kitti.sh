#! /bin/bash


## kitti mamba
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/kitti_models/second_with_lion_mamba_64dim.yaml \
--extra_tag second_with_lion_mamba_64dim \
--batch_size 16  --epochs 80 --max_ckpt_save_num 4 --workers 4 --sync_bn


## kitti renet
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/kitti_models/second_with_lion_retnet_64dim.yaml \
--extra_tag second_with_lion_retnet_64dim \
--batch_size 16  --epochs 80 --max_ckpt_save_num 4 --workers 4 --sync_bn


## kitti rwkv
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/kitti_models/second_with_lion_rwkv_64dim.yaml \
--extra_tag second_with_lion_rwkv_64dim \
--batch_size 16  --epochs 80 --max_ckpt_save_num 4 --workers 4 --sync_bn


## kitti xLSTM
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/kitti_models/second_with_lion_xLSTM_64dim.yaml \
--extra_tag second_with_lion_xLSTM_64dim \
--batch_size 16  --epochs 80 --max_ckpt_save_num 4 --workers 4 --sync_bn


## kitti TTT
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/kitti_models/second_with_lion_TTT_64dim.yaml \
--extra_tag second_with_lion_TTT_64dim \
--batch_size 16  --epochs 80 --max_ckpt_save_num 4 --workers 4 --sync_bn



