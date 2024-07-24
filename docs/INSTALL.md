# Installation

### Requirements
All the codes are tested in the following environment:
* Python 3.8
* PyTorch 2.1.0
* CUDA 11.8
* [spconv v2.x](https://github.com/traveller59/spconv) ``spconv-cu11.8`` may contain some errors, change ``spconv-cu11.6`` can work fine.


### Install LION codebase
* ``git clone https://github.com/happinesslz/LION.git``
* ``pip install -r requirements.txt``
* ``python setup.py develop``

### Install Linear RNN operators
We provide some guidance for some difficult to install linear RNN operators.

* **Mmaba**
  * install causal-conv1d by running ``python -m pip install causal-conv1d==1.2.0.post2``
  * ``cd pcdet/ops/mamba``
  * ``python setup.py install``


* **RWKV**
  * Because the hard code of T_MAX and HEAD_SIZE in RWKV, we should change them according to the running config.
  * Given the config (maximum group_size: 4096, layer_dim: 128, nhead: 4)
  * You should modify T_MAX and HEAD_SIZE in ``pcdet/ops/wkv6/src/wkv6_cuda.cu`` and ``pcdet/models/model_utils/rwkv_cls.py``.
  * ``T_MAX = 4096`` Because the T_MAX equals with maximum group_size.
  * ``HEAD_SIZE = 128 / 4 = 32`` Because the HEAD_SIZE equals layer_dim divided by nhead.
  * Rerun ``python setup.py develop`` to install the LION codebase.