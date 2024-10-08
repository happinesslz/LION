[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lion-linear-group-rnn-for-3d-object-detection/3d-object-detection-on-waymo-open-dataset)](https://paperswithcode.com/sota/3d-object-detection-on-waymo-open-dataset?p=lion-linear-group-rnn-for-3d-object-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lion-linear-group-rnn-for-3d-object-detection/3d-object-detection-on-nuscenes-lidar-only)](https://paperswithcode.com/sota/3d-object-detection-on-nuscenes-lidar-only?p=lion-linear-group-rnn-for-3d-object-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lion-linear-group-rnn-for-3d-object-detection/3d-object-detection-on-argoverse2)](https://paperswithcode.com/sota/3d-object-detection-on-argoverse2?p=lion-linear-group-rnn-for-3d-object-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lion-linear-group-rnn-for-3d-object-detection/3d-object-detection-on-once)](https://paperswithcode.com/sota/3d-object-detection-on-once?p=lion-linear-group-rnn-for-3d-object-detection)

<div align="center">

### <img src="./assets/lion.jpg" alt="Image 2" width="4%" style="margin: 0 auto;" > [LION: Linear Group RNN for 3D Object Detection in Point Clouds](https://arxiv.org/abs/2407.18232)

[Zhe Liu](https://happinesslz.github.io) <sup>1,* </sup>,
[Jinghua Hou](https://github.com/AlmoonYsl) <sup>1,* </sup>,
[Xinyu Wang](https://github.com/deepinact) <sup>1,* </sup>,
[Xiaoqing Ye](https://shuluoshu.github.io)  <sup>3</sup>,
[Jingdong Wang](https://jingdongwang2017.github.io) <sup>3</sup>,
[Hengshuang Zhao](https://hszhao.github.io) <sup>2</sup>,
[Xiang Bai](https://xbai.vlrlab.net) <sup>1,✉</sup>
<br>
<sup>1</sup> Huazhong University of Science and Technology,
<sup>2</sup> The University of Hong Kong,
<sup>3</sup> Baidu Inc.
<br>
\* Equal contribution, ✉ Corresponding author.
<br>

[**Project Page**](https://happinesslz.github.io/projects/LION) | [**NeurIPS 2024**](https://arxiv.org/abs/2407.18232)

<img src="./assets/all.jpg" alt="Image 2" width="60%" style="margin: 0 auto;" >

</div>

## 🔥 Highlights

* **Strong performance**. LION achieves state-of-the-art performance on Waymo, nuScenes, Argoverse V2, and ONCE datasets. 💪

* **Strong generalization**. LION can support almost all linear RNN operators including Mamba, RWKV, RetNet, xLSTM, and TTT. Anyone is welcome to verify more linear RNN operators. 😀

* **More friendly**. LION can train all models on less 24G GPU memory (i.e., RTX 3090, RTX4090, V100 and A100 are enough to train our LION). 😀

## News
* **2024.09.26**: LION has been accepted by NeurIPS 2024. 🎉
* **2024.07.25**: LION paper released. 🔥
* **2024.07.02**: Our new works [OPEN](https://github.com/AlmoonYsl/OPEN) and [SEED](https://github.com/happinesslz/SEED) have been accepted by ECCV 2024. 🎉

## Results
* **Waymo Val Set~(100%)**

| Model        | mAP/mAPH_L1 | mAP/mAPH_L2 |  Vec_L1   |  Vec_L2   |  Ped_L1   |  Ped_L2   |  Cyc_L1   |  Cyc_L2   | Config |
|--------------|:-----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:------:|
| LION-RetNet  |  80.9/78.8  |  74.6/72.7  | 79.0/78.5 | 70.6/70.2 | 84.6/80.0 | 77.2/72.8 | 79.0/78.0 | 76.1/75.1 |[config](tools/cfgs/lion_models/lion_retnet_waymo_8x_1f_1x_one_stride_64dim.yaml)|
| LION-RWKV    |  81.0/79.0  |  74.7/72.8  | 79.7/79.3 | 71.3/71.0 | 84.6/80.0 | 77.1/72.7 | 78.7/77.7 | 75.8/74.8 |[config](tools/cfgs/lion_models/lion_rwkv_waymo_8x_1f_1x_one_stride_64dim.yaml)|
| LION-Mamba   |  81.4/79.4  |  75.1/73.2  | 79.5/79.1 | 71.1/70.7 | 84.9/80.4 | 77.5/73.2 | 79.7/78.7 | 76.7/75.8 |[config](tools/cfgs/lion_models/lion_mamba_waymo_8x_1f_1x_one_stride_64dim.yaml)|
| LION-Mamba-L |  82.1/80.1  |  75.9/74.0  | 80.3/79.9 | 72.0/71.6 | 85.8/81.4 | 78.5/74.3 | 80.1/79.0 | 77.2/76.2 |[config](tools/cfgs/lion_models/lion_mamba_waymo_8x_1f_1x_one_stride_128dim.yaml)|

Note: You could reduce the training epochs from 24 to 12~(the performance gap is within 1 mAP/mAPH) or reduce the 100% training to 20% training sets.

* **nuScenes**

|    Model    | Split | Epoch | CBGS| NDS  | mAP  | Config | Download (Baidu Pan) |                                                           Download (Google Drive)                                                           |
|:-----------:|:-----:|:----:|:----:|:----:|:----:|:------:|:--------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------:|
| LION-RetNet |  Val  | 36 | False | 71.9 | 67.3 |[config](tools/cfgs/lion_models/lion_retnet_nusc_8x_1f_1x_one_stride_128dim.yaml)|       [checkpoint_epoch_36_nus_retnet.pth]( https://pan.baidu.com/s/11m4Tka9VRoLM7dO_-CQfng)  (ksmp)       |          [checkpoint_epoch_36_nus_retnet.pth](https://drive.google.com/file/d/1LDM6Iq91muZhzFMMa0WqKtjzJwYVadF6/view?usp=sharing)           |
|  LION-RWKV  |  Val  | 36 | False | 71.7 | 66.8 |[config](tools/cfgs/lion_models/lion_rwkv_nusc_8x_1f_1x_one_stride_128dim.yaml)|                      |                                                                                                                                             |
| LION-Mamba  |  Val  | 36 | False | 72.1 | 68.0 |[config](tools/cfgs/lion_models/lion_mamba_nusc_8x_1f_1x_one_stride_128dim.yaml)|       [checkpoint_epoch_36_nus_mamba.pth](https://pan.baidu.com/s/1fkGHULD3XVwN6qgjqahWIA)  (2tvc)       |                                                    [checkpoint_epoch_36_nus_mamba.pth](https://drive.google.com/file/d/1bBzZcZUukt7B5aD3bmKlM2mT7SL8q3Fv/view?usp=sharing)                                                    |
| LION-Mamba  |  Val  | 48 | False | 72.3 | 68.2 |[config](tools/cfgs/lion_models/lion_mamba_nusc_8x_1f_1x_one_stride_128dim_ep48.yaml)|                      |                                                                                                                                             |
| LION-Mamba  | Test  | 36 | False | 73.9 | 69.8 |        |                      |                                                                                                                                             |

Note: Our model on nuScenes does not use CBGS for training more time and without any test-time augmentation or model ensembling!
For obtaining more stable and better performance, you could try to train more time~(e.g., 48 epochs)

* **Argoverse V2 Val Set**

|    Model    | mAP  | Config |                                        Download  (Baidu Pan)                                        | Download  (Google Drive) |
|:-----------:|:----:|:------:|:---------------------------------------------------------------------------------------------------:|:------------------------:|
| LION-RetNet | 40.7 |[config](tools/cfgs/lion_models/lion_retnet_1f_1x_argo_128dim_sparse_v2.yaml)|  [checkpoint_epoch_12_argov2_retnet.pth]( https://pan.baidu.com/s/1Te8HhIVJpxarFLvr1GMtWQ) (yghm)   |           [checkpoint_epoch_12_argov2_retnet.pth](https://drive.google.com/file/d/17ZTdThUvK2tnsz0ZhVUrCuJf3bvxQDTl/view?usp=sharing)           |
|  LION-RWKV  | 41.1 |[config](tools/cfgs/lion_models/lion_rwkv_1f_1x_argo_128dim_sparse_v2.yaml)|  [checkpoint_epoch_12_argov2_rwkv.pth](https://pan.baidu.com/s/14r1OKM5Eh4JMURIvZ5i2hg)    (cr4e)   |           [checkpoint_epoch_12_argov2_rwkv.pth](https://drive.google.com/file/d/15zX5OFvd0WkdE8jsqRXMJ9JdM16gfGkM/view?usp=sharing)           |
| LION-Mamba  | 41.5 |[config](tools/cfgs/lion_models/lion_mamba_1f_1x_argo_128dim_sparse_v2.yaml)|  [checkpoint_epoch_12_argov2_mamba.pth](https://pan.baidu.com/s/1IPMhAgW4Oq14-pkfEPkrSA)   (k63i)   |           [checkpoint_epoch_12_argov2_mamba.pth](https://drive.google.com/file/d/1NfgKZZZJDodhWnp-Yi3Bi-yXHWM9mW1U/view?usp=sharing)           |

* **ONCE Val Set**

|    Model    | Vehicle | Pedestrian | Cyclist | mAP  |  Config   | Download |
|:-----------:|:-------:|:----------:|:-------:|:----:|:---------:|:--------:|
| LION-RetNet |  78.1   |    52.4    |  68.3   | 66.3 |[config](tools/cfgs/once_models/centerpoint_with_lion_with_128dim_retnet.yaml)|          |
|  LION-RWKV  |  78.3   |    50.6    |  68.4   | 65.8 |[config](tools/cfgs/once_models/centerpoint_with_lion_with_128dim_rwkv.yaml)|          |
| LION-Mamba  |  78.2   |    53.2    |  68.5   | 66.6 |[config](tools/cfgs/once_models/centerpoint_with_lion_with_128dim_mamba.yaml)|          |


## Quick Validation
* We provide some examples of LION models on KITTI dataset for quick validation of any Linear RNN operators.
* Here, we provide the results of moderate difficulty for LION with RetNet, RWKV, Mamba, xLSTM, and TTT.
* Anyone is welcome to verify more linear RNN operators. 😀


|    Model    | Car  | Pedestrian | Cyclist |  Config  | Download |
|:-----------:|:----:|:----------:|:-------:|:--------:|:--------:|
|  LION-TTT   | 78.0 |    58.6    |  69.6   |[config](tools/cfgs/kitti_models/second_with_lion_TTT_64dim.yaml)|          |
| LION-xLSTM  | 77.9 |    59.3    |  67.4   |[config](tools/cfgs/kitti_models/second_with_lion_xLSTM_64dim.yaml)|        |
| LION-RetNet | 77.9 |    60.2    |  69.6   |[config](tools/cfgs/kitti_models/second_with_lion_retnet_64dim.yaml)|       |
| LION-Mamba  | 78.3 |    60.2    |  68.6   |[config](tools/cfgs/kitti_models/second_with_lion_mamba_64dim.yaml)|        |
|  LION-RWKV  | 78.3 |    62.2    |  71.2   |[config](tools/cfgs/kitti_models/second_with_lion_rwkv_64dim.yaml)|         |


## Installation
Please refer to [INSTALL.md](docs/INSTALL.md) for the installation of LION codebase.


## Getting Started
We provide all training&evaluation scripts for training our LION, please refer to [tools/](tools/)
* Train all models of LION on nuScenes
```shell script
bash run_train_lion_for_nus.sh
```

* Train all models of LION on Waymo
```shell script
bash run_train_lion_for_waymo.sh
```

* Train all models of LION on Argoverse V2
```shell script
bash run_train_lion_for_argov2.sh
```

* Train all models of LION on ONCE
```shell script
bash run_train_lion_for_once.sh
```

* Train all models of LION on KITTI
```shell script
bash run_train_lion_for_kitti.sh
```

For more details about LION, please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more usage about LION.

## TODO
- [x] Release the paper.
- [x] Release the code of LION on the Waymo.
- [x] Release the code of LION on the nuScenes.
- [x] Release the code of LION on the Argoverse V2.
- [x] Release the code of LION on the ONCE.
- [x] Release the code of LION on the KITTI.
- [x] Release some important checkpoints of LION (nuScenes and Argoverse v2).
- [ ] Support more linear RNNs.

## Citation
```
@article{liu2024lion,
  title={LION: Linear Group RNN for 3D Object Detection in Point Clouds},
  author={Zhe Liu, Jinghua Hou, Xingyu Wang, Xiaoqing Ye, Jingdong Wang, Hengshuang Zhao, Xiang Bai},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
  }
```

## Acknowledgements
We thank these great works and open-source repositories:
[OpenPCDet](https://github.com/open-mmlab/OpenPCDet), [DSVT](https://github.com/Haiyang-W/DSVT), [FlatFormer](https://github.com/mit-han-lab/flatformer), [HEDNet](https://github.com/zhanggang001/HEDNet), [Mamba](https://github.com/state-spaces/mamba), [RWKV](https://github.com/BlinkDL/RWKV-LM), [Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV), [RMT](https://github.com/qhfan/RMT), [xLSTM](https://github.com/NX-AI/xlstm),  [TTT](https://github.com/test-time-training/ttt-lm-pytorch), and [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention).
