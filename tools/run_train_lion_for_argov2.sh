#! /bin/bash

## argov2 mamba
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/lion_models/lion_mamba_1f_1x_argo_128dim_sparse_v2.yaml \
--extra_tag lion_mamba_1f_1x_argo_128dim_sparse_v2 \
--batch_size 16 --epochs 12 --max_ckpt_save_num 4 --workers 4 --sync_bn


## argov2 renet
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/lion_models/lion_retnet_1f_1x_argo_128dim_sparse_v2.yaml \
--extra_tag lion_retnet_1f_1x_argo_128dim_sparse_v2 \
--batch_size 16 --epochs 12 --max_ckpt_save_num 4 --workers 4 --sync_bn


## argov2 rwkv
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/lion_models/lion_rwkv_1f_1x_argo_128dim_sparse_v2.yaml \
--extra_tag lion_rwkv_1f_1x_argo_128dim_sparse_v2 \
--batch_size 16 --epochs 12 --max_ckpt_save_num 4 --workers 4 --sync_bn


