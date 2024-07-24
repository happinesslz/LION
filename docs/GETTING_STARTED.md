# Getting Started
The LION configs are placed in [tools/cfgs/lion_models](../tools/cfgs/lion_models) for Waymo, nuScenes and Argoverse V2 datasets.
For ONCE dataset, please refer to [tools/cfgs/once_models](../tools/cfgs/once_models). For KITTI dataset, refer to [tools/cfgs/kitti_models](../tools/cfgs/kitti_models).

## Dataset Preparation
LION supports KITTI, nuScenes, Waymo, Argoverse V2 and ONCE dataset. For these dataset preparations, please refer to [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). 


## Training & Testing

### Training

* Train with multiple GPUs
```shell script
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```

### Testing
* Test with single GPU:
```shell script
python test.py --cfg_file ${CONFIG_FILE} --ckpt ${CKPT}
```

* Test with multiple GPUs:
```shell script
sh scripts/dist_test.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE} --ckpt ${CKPT}
```

