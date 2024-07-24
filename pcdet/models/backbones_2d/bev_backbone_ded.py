import torch.nn as nn
from .base_bev_backbone import BasicBlock


class DEDBackbone(nn.Module):

    def __init__(self, model_cfg, input_channels):
        super().__init__()

        # self.model_cfg = model_cfg

        num_SBB = model_cfg.NUM_SBB
        down_strides = model_cfg.DOWN_STRIDES
        dim = model_cfg.FEATURE_DIM
        assert len(num_SBB) == len(down_strides)

        num_levels = len(down_strides)

        first_block = []
        if input_channels != dim:
            first_block.append(BasicBlock(input_channels, dim, down_strides[0], 1, True))
        first_block += [BasicBlock(dim, dim) for _ in range(num_SBB[0])]
        self.encoder = nn.ModuleList([nn.Sequential(*first_block)])

        for idx in range(1, num_levels):
            cur_layers = [BasicBlock(dim, dim, down_strides[idx], 1, True)]
            cur_layers.extend([BasicBlock(dim, dim) for _ in range(num_SBB[idx])])
            self.encoder.append(nn.Sequential(*cur_layers))

        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        for idx in range(num_levels - 1, 0, -1):
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(dim, dim, down_strides[idx], down_strides[idx], bias=False),
                    nn.BatchNorm2d(dim, eps=1e-3, momentum=0.01),
                    nn.ReLU()
                )
            )
            self.decoder_norm.append(nn.BatchNorm2d(dim, eps=1e-3, momentum=0.01))

        self.num_bev_features = dim
        self.init_weights()

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, data_dict):
        x = data_dict['spatial_features']
        x = self.encoder[0](x)

        feats = [x]
        for conv in self.encoder[1:]:
            x = conv(x)
            feats.append(x)

        for deconv, norm, up_x in zip(self.decoder, self.decoder_norm, feats[:-1][::-1]):
            x = norm(deconv(x) + up_x)

        data_dict['spatial_features_2d'] = x
        data_dict['spatial_features'] = x
        return data_dict


class CascadeDEDBackbone(nn.Module):

    def __init__(self, model_cfg, input_channels):
        super().__init__()

        self.model_cfg = model_cfg

        num_layers = model_cfg.NUM_LAYERS

        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            input_dim = input_channels if idx == 0 else model_cfg.FEATURE_DIM
            self.layers.append(DEDBackbone(model_cfg, input_dim))

        self.num_bev_features = model_cfg.FEATURE_DIM

    def forward(self, data_dict):
        for layer in self.layers:
            data_dict = layer(data_dict)
        data_dict['spatial_features_2d'] = data_dict['spatial_features']
        return data_dict
