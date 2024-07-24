import torch
from pcdet.models.model_utils.vmamba import SS2D

x = torch.randn(2, 56, 1, 96).cuda()
ss2d_model = SS2D(96).cuda()
out = ss2d_model(x)
print('#####out.shape:', out.shape)
