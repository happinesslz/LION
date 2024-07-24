# Getting Started
The LION configs are located within [tools/cfgs/lion](../tools/cfgs/lion) for different datasets.

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

