#! /bin/bash

## once mamba
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/once_models/centerpoint_with_lion_with_128dim_mamba.yaml \
--extra_tag centerpoint_with_lion_with_128dim_mamba \
--batch_size 16  --epochs 80 --max_ckpt_save_num 4 --workers 4 --sync_bn


## once renet
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/once_models/centerpoint_with_lion_with_128dim_retnet.yaml \
--extra_tag centerpoint_with_lion_with_128dim_retnet \
--batch_size 16  --epochs 80 --max_ckpt_save_num 4 --workers 4 --sync_bn


## once rwkv
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/once_models/centerpoint_with_lion_with_128dim_rwkv.yaml \
--extra_tag centerpoint_with_lion_with_128dim_rwkv \
--batch_size 16  --epochs 80 --max_ckpt_save_num 4 --workers 4 --sync_bn

