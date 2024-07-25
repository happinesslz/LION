#! /bin/bash

## for waymo
# only for single-frame setting
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml

# # for single-frame or multi-frame setting
# python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
#     --cfg_file tools/cfgs/dataset_configs/waymo_dataset_multiframe.yaml
# # Ignore 'CUDA_ERROR_NO_DEVICE' error as this process does not require GPU.


## for nuscens
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval

## for argoverse v2
python -m pcdet.datasets.argo2.argo2_dataset --root_path data/argo2/sensor --output_dir data/argo2

python -m pcdet.datasets.argo2.argo2_dataset --root_path data/argo2 --output_dir data/argo2 \
--func create_groundtruth_database --cfg_file tools/cfgs/dataset_configs/argo2_dataset.yaml

## for kitti
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml

## for once
python -m pcdet.datasets.once.once_dataset --func create_once_infos --cfg_file tools/cfgs/dataset_configs/once_dataset.yaml

