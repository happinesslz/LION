from functools import partial

import math
import numpy as np
import torch
import torch.nn as nn
import torch_scatter
from mamba_ssm import Block as MambaBlock
from torch.nn import functional as F

from ..model_utils.retnet_attn import Block as RetNetBlock
from ..model_utils.rwkv_cls import Block as RWKVBlock
from ..model_utils.vision_lstm2 import xLSTM_Block
from ..model_utils.ttt import TTTBlock
from ...utils.spconv_utils import replace_feature, spconv
import torch.utils.checkpoint as cp

@torch.inference_mode()
def get_window_coors_shift_v2(coords, sparse_shape, window_shape, shift=False):
    sparse_shape_z, sparse_shape_y, sparse_shape_x = sparse_shape
    win_shape_x, win_shape_y, win_shape_z = window_shape

    if shift:
        shift_x, shift_y, shift_z = win_shape_x // 2, win_shape_y // 2, win_shape_z // 2
    else:
        shift_x, shift_y, shift_z = 0, 0, 0  # win_shape_x, win_shape_y, win_shape_z

    max_num_win_x = int(np.ceil((sparse_shape_x / win_shape_x)) + 1)  # plus one here to meet the needs of shift.
    max_num_win_y = int(np.ceil((sparse_shape_y / win_shape_y)) + 1)  # plus one here to meet the needs of shift.
    max_num_win_z = int(np.ceil((sparse_shape_z / win_shape_z)) + 1)  # plus one here to meet the needs of shift.

    max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z

    x = coords[:, 3] + shift_x
    y = coords[:, 2] + shift_y
    z = coords[:, 1] + shift_z

    win_coors_x = x // win_shape_x
    win_coors_y = y // win_shape_y
    win_coors_z = z // win_shape_z

    coors_in_win_x = x % win_shape_x
    coors_in_win_y = y % win_shape_y
    coors_in_win_z = z % win_shape_z

    batch_win_inds_x = coords[:, 0] * max_num_win_per_sample + win_coors_x * max_num_win_y * max_num_win_z + \
                       win_coors_y * max_num_win_z + win_coors_z
    batch_win_inds_y = coords[:, 0] * max_num_win_per_sample + win_coors_y * max_num_win_x * max_num_win_z + \
                       win_coors_x * max_num_win_z + win_coors_z

    coors_in_win = torch.stack([coors_in_win_z, coors_in_win_y, coors_in_win_x], dim=-1)

    return batch_win_inds_x, batch_win_inds_y, coors_in_win


def get_window_coors_shift_v1(coords, sparse_shape, window_shape):
    _, m, n = sparse_shape
    n2, m2, _ = window_shape

    n1 = int(np.ceil(n / n2) + 1)  # plus one here to meet the needs of shift.
    m1 = int(np.ceil(m / m2) + 1)  # plus one here to meet the needs of shift.

    x = coords[:, 3]
    y = coords[:, 2]

    x1 = x // n2
    y1 = y // m2
    x2 = x % n2
    y2 = y % m2

    return 2 * n2, 2 * m2, 2 * n1, 2 * m1, x1, y1, x2, y2


class FlattenedWindowMapping(nn.Module):
    def __init__(
            self,
            window_shape,
            group_size,
            shift,
            win_version='v2'
    ) -> None:
        super().__init__()
        self.window_shape = window_shape
        self.group_size = group_size
        self.win_version = win_version
        self.shift = shift

    def forward(self, coords: torch.Tensor, batch_size: int, sparse_shape: list):
        coords = coords.long()
        _, num_per_batch = torch.unique(coords[:, 0], sorted=False, return_counts=True)
        batch_start_indices = F.pad(torch.cumsum(num_per_batch, dim=0), (1, 0))
        num_per_batch_p = (
                torch.div(
                    batch_start_indices[1:] - batch_start_indices[:-1] + self.group_size - 1,
                    self.group_size,
                    rounding_mode="trunc",
                )
                * self.group_size
        )

        batch_start_indices_p = F.pad(torch.cumsum(num_per_batch_p, dim=0), (1, 0))
        flat2win = torch.arange(batch_start_indices_p[-1], device=coords.device)  # .to(coords.device)
        win2flat = torch.arange(batch_start_indices[-1], device=coords.device)  # .to(coords.device)

        for i in range(batch_size):
            if num_per_batch[i] != num_per_batch_p[i]:
                
                bias_index = batch_start_indices_p[i] - batch_start_indices[i]
                flat2win[
                    batch_start_indices_p[i + 1] - self.group_size + (num_per_batch[i] % self.group_size):
                    batch_start_indices_p[i + 1]
                    ] = flat2win[
                        batch_start_indices_p[i + 1]
                        - 2 * self.group_size
                        + (num_per_batch[i] % self.group_size): batch_start_indices_p[i + 1] - self.group_size
                        ] if (batch_start_indices_p[i + 1] - batch_start_indices_p[i]) - self.group_size != 0 else \
                        win2flat[batch_start_indices[i]: batch_start_indices[i + 1]].repeat(
                            (batch_start_indices_p[i + 1] - batch_start_indices_p[i]) // num_per_batch[i] + 1)[
                        : self.group_size - (num_per_batch[i] % self.group_size)] + bias_index


            win2flat[batch_start_indices[i]: batch_start_indices[i + 1]] += (
                    batch_start_indices_p[i] - batch_start_indices[i]
            )

            flat2win[batch_start_indices_p[i]: batch_start_indices_p[i + 1]] -= (
                    batch_start_indices_p[i] - batch_start_indices[i]
            )

        mappings = {"flat2win": flat2win, "win2flat": win2flat}

        get_win = self.win_version

        if get_win == 'v1':
            for shifted in [False]:
                (
                    n2,
                    m2,
                    n1,
                    m1,
                    x1,
                    y1,
                    x2,
                    y2,
                ) = get_window_coors_shift_v1(coords, sparse_shape, self.window_shape)
                vx = (n1 * y1 + (-1) ** y1 * x1) * n2 * m2 + (-1) ** y1 * (m2 * x2 + (-1) ** x2 * y2)
                vx += coords[:, 0] * sparse_shape[2] * sparse_shape[1] * sparse_shape[0]
                vy = (m1 * x1 + (-1) ** x1 * y1) * m2 * n2 + (-1) ** x1 * (n2 * y2 + (-1) ** y2 * x2)
                vy += coords[:, 0] * sparse_shape[2] * sparse_shape[1] * sparse_shape[0]
                _, mappings["x" + ("_shift" if shifted else "")] = torch.sort(vx)
                _, mappings["y" + ("_shift" if shifted else "")] = torch.sort(vy)

        elif get_win == 'v2':
            batch_win_inds_x, batch_win_inds_y, coors_in_win = get_window_coors_shift_v2(coords, sparse_shape,
                                                                                         self.window_shape, self.shift)
            vx = batch_win_inds_x * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
            vx += coors_in_win[..., 2] * self.window_shape[1] * self.window_shape[2] + coors_in_win[..., 1] * \
                  self.window_shape[2] + coors_in_win[..., 0]

            vy = batch_win_inds_y * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
            vy += coors_in_win[..., 1] * self.window_shape[0] * self.window_shape[2] + coors_in_win[..., 2] * \
                  self.window_shape[2] + coors_in_win[..., 0]

            _, mappings["x"] = torch.sort(vx)
            _, mappings["y"] = torch.sort(vy)

        elif get_win == 'v3':
            batch_win_inds_x, batch_win_inds_y, coors_in_win = get_window_coors_shift_v2(coords, sparse_shape,
                                                                                         self.window_shape)
            vx = batch_win_inds_x * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
            vx_xy = vx + coors_in_win[..., 2] * self.window_shape[1] * self.window_shape[2] + coors_in_win[..., 1] * \
                    self.window_shape[2] + coors_in_win[..., 0]
            vx_yx = vx + coors_in_win[..., 1] * self.window_shape[0] * self.window_shape[2] + coors_in_win[..., 2] * \
                    self.window_shape[2] + coors_in_win[..., 0]

            vy = batch_win_inds_y * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
            vy_xy = vy + coors_in_win[..., 2] * self.window_shape[1] * self.window_shape[2] + coors_in_win[..., 1] * \
                    self.window_shape[2] + coors_in_win[..., 0]
            vy_yx = vy + coors_in_win[..., 1] * self.window_shape[0] * self.window_shape[2] + coors_in_win[..., 2] * \
                    self.window_shape[2] + coors_in_win[..., 0]

            _, mappings["x_xy"] = torch.sort(vx_xy)
            _, mappings["y_xy"] = torch.sort(vy_xy)
            _, mappings["x_yx"] = torch.sort(vx_yx)
            _, mappings["y_yx"] = torch.sort(vy_yx)

        return mappings


class PatchMerging3D(nn.Module):
    def __init__(self, dim, out_dim=-1, down_scale=[2, 2, 2], norm_layer=nn.LayerNorm, diffusion=False, diff_scale=0.2):
        super().__init__()
        self.dim = dim

        self.sub_conv = spconv.SparseSequential(
            spconv.SubMConv3d(dim, dim, 3, bias=False, indice_key='subm'),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        if out_dim == -1:
            self.norm = norm_layer(dim)
        else:
            self.norm = norm_layer(out_dim)

        self.sigmoid = nn.Sigmoid()
        self.down_scale = down_scale
        self.diffusion = diffusion
        self.diff_scale = diff_scale

        self.num_points = 6 #3

    def forward(self, x, coords_shift=1, diffusion_scale=4):
        assert diffusion_scale==4 or diffusion_scale==2
        x = self.sub_conv(x)

        d, h, w = x.spatial_shape
        down_scale = self.down_scale

        if self.diffusion:
            x_feat_att = x.features.mean(-1)
            batch_size = x.indices[:, 0].max() + 1
            selected_diffusion_feats_list = [x.features.clone()]
            selected_diffusion_coords_list = [x.indices.clone()]
            for i in range(batch_size):
                mask = x.indices[:, 0] == i
                valid_num = mask.sum()
                K = int(valid_num * self.diff_scale)
                _, indices = torch.topk(x_feat_att[mask], K)

                selected_coords_copy = x.indices[mask][indices].clone()
                selected_coords_num = selected_coords_copy.shape[0]
                selected_coords_expand = selected_coords_copy.repeat(diffusion_scale, 1)
                selected_feats_expand = x.features[mask][indices].repeat(diffusion_scale, 1) * 0.0


                selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 3:4] = (
                            selected_coords_copy[:, 3:4] - coords_shift).clamp(min=0, max=w - 1)
                selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 2:3] = (
                            selected_coords_copy[:, 2:3] + coords_shift).clamp(min=0, max=h - 1)
                selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 1:2] = (
                        selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                selected_coords_expand[selected_coords_num:selected_coords_num * 2, 3:4] = (
                        selected_coords_copy[:, 3:4] + coords_shift).clamp(min=0, max=w - 1)
                selected_coords_expand[selected_coords_num:selected_coords_num * 2, 2:3] = (
                        selected_coords_copy[:, 2:3] + coords_shift).clamp(min=0, max=h - 1)
                selected_coords_expand[selected_coords_num:selected_coords_num * 2, 1:2] = (
                    selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                if diffusion_scale==4:
#                         print('####diffusion_scale==4')
                    selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 3:4] = (
                        selected_coords_copy[:, 3:4] - coords_shift).clamp(min=0, max=w - 1)
                    selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 2:3] = (
                        selected_coords_copy[:, 2:3] - coords_shift).clamp(min=0, max=h - 1)
                    selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 1:2] = (
                    selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                    selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 3:4] = (
                            selected_coords_copy[:, 3:4] + coords_shift).clamp(min=0, max=w - 1)
                    selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 2:3] = (
                            selected_coords_copy[:, 2:3] - coords_shift).clamp(min=0, max=h - 1)
                    selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 1:2] = (
                        selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                selected_diffusion_coords_list.append(selected_coords_expand)
                selected_diffusion_feats_list.append(selected_feats_expand)

            coords = torch.cat(selected_diffusion_coords_list)
            final_diffusion_feats = torch.cat(selected_diffusion_feats_list)

        else:
            coords = x.indices.clone()
            final_diffusion_feats = x.features.clone()

        coords[:, 3:4] = coords[:, 3:4] // down_scale[0]
        coords[:, 2:3] = coords[:, 2:3] // down_scale[1]
        coords[:, 1:2] = coords[:, 1:2] // down_scale[2]

        scale_xyz = (x.spatial_shape[0] // down_scale[2]) * (x.spatial_shape[1] // down_scale[1]) * (
                x.spatial_shape[2] // down_scale[0])
        scale_yz = (x.spatial_shape[0] // down_scale[2]) * (x.spatial_shape[1] // down_scale[1])
        scale_z = (x.spatial_shape[0] // down_scale[2])


        merge_coords = coords[:, 0].int() * scale_xyz + coords[:, 3] * scale_yz + coords[:, 2] * scale_z + coords[:, 1]

        features_expand = final_diffusion_feats

        new_sparse_shape = [math.ceil(x.spatial_shape[i] / down_scale[2 - i]) for i in range(3)]
        unq_coords, unq_inv = torch.unique(merge_coords, return_inverse=True, return_counts=False, dim=0)

        x_merge = torch_scatter.scatter_add(features_expand, unq_inv, dim=0)

        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // scale_xyz,
                                    (unq_coords % scale_xyz) // scale_yz,
                                    (unq_coords % scale_yz) // scale_z,
                                    unq_coords % scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        x_merge = self.norm(x_merge)

        x_merge = spconv.SparseConvTensor(
            features=x_merge,
            indices=voxel_coords.int(),
            spatial_shape=new_sparse_shape,
            batch_size=x.batch_size
        )
        return x_merge, unq_inv


class PatchExpanding3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, up_x, unq_inv):
        # z, y, x
        n, c = x.features.shape

        x_copy = torch.gather(x.features, 0, unq_inv.unsqueeze(1).repeat(1, c))
        up_x = up_x.replace_feature(up_x.features + x_copy)
        return up_x


LinearOperatorMap = {
    'Mamba': MambaBlock,
    'RWKV': RWKVBlock,
    'RetNet': RetNetBlock,
    'xLSTM': xLSTM_Block,
    'TTT': TTTBlock,
}


class LIONLayer(nn.Module):
    def __init__(self, dim, nums, window_shape, group_size, direction, shift, operator=None, layer_id=0, n_layer=0):
        super(LIONLayer, self).__init__()

        self.window_shape = window_shape
        self.group_size = group_size
        self.dim = dim
        self.direction = direction

        operator_cfg = operator.CFG
        operator_cfg['d_model'] = dim

        block_list = []
        for i in range(len(direction)):
            operator_cfg['layer_id'] = i + layer_id
            operator_cfg['n_layer'] = n_layer
            # operator_cfg['with_cp'] = layer_id >= 16
            operator_cfg['with_cp'] = layer_id >= 0 ## all lion layer use checkpoint to save GPU memory!! (less 24G for training all models!!!)
            print('### use part of checkpoint!!')
            block_list.append(LinearOperatorMap[operator.NAME](**operator_cfg))

        self.blocks = nn.ModuleList(block_list)
        self.window_partition = FlattenedWindowMapping(self.window_shape, self.group_size, shift)

    def forward(self, x):
        mappings = self.window_partition(x.indices, x.batch_size, x.spatial_shape)

        for i, block in enumerate(self.blocks):
            indices = mappings[self.direction[i]]
            x_features = x.features[indices][mappings["flat2win"]]
            x_features = x_features.view(-1, self.group_size, x.features.shape[-1])

            x_features = block(x_features)

            x.features[indices] = x_features.view(-1, x_features.shape[-1])[mappings["win2flat"]]

        return x


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Linear(input_channel, num_pos_feats),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Linear(num_pos_feats, num_pos_feats))

    def forward(self, xyz):
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class LIONBlock(nn.Module):
    def __init__(self, dim: int, depth: int, down_scales: list, window_shape, group_size, direction, shift=False,
                 operator=None, layer_id=0, n_layer=0):
        super().__init__()

        self.down_scales = down_scales

        self.encoder = nn.ModuleList()
        self.downsample_list = nn.ModuleList()
        self.pos_emb_list = nn.ModuleList()

        norm_fn = partial(nn.LayerNorm)

        shift = [False, shift]
        for idx in range(depth):
            self.encoder.append(LIONLayer(dim, 1, window_shape, group_size, direction, shift[idx], operator, layer_id + idx * 2, n_layer))
            self.pos_emb_list.append(PositionEmbeddingLearned(input_channel=3, num_pos_feats=dim))
            self.downsample_list.append(PatchMerging3D(dim, dim, down_scale=down_scales[idx], norm_layer=norm_fn))

        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        self.upsample_list = nn.ModuleList()
        for idx in range(depth):
            self.decoder.append(LIONLayer(dim, 1, window_shape, group_size, direction, shift[idx], operator, layer_id + 2 * (idx + depth), n_layer))
            self.decoder_norm.append(norm_fn(dim))
            
            self.upsample_list.append(PatchExpanding3D(dim))
            

    def forward(self, x):
        features = []
        index = []

        for idx, enc in enumerate(self.encoder):
            pos_emb = self.get_pos_embed(spatial_shape=x.spatial_shape, coors=x.indices[:, 1:],
                                         embed_layer=self.pos_emb_list[idx])

            x = replace_feature(x, pos_emb + x.features)  # x + pos_emb
            x = enc(x)
            features.append(x)
            x, unq_inv = self.downsample_list[idx](x)
            index.append(unq_inv)

        i = 0
        for dec, norm, up_x, unq_inv, up_scale in zip(self.decoder, self.decoder_norm, features[::-1],
                                                      index[::-1], self.down_scales[::-1]):
            x = dec(x)
            x = self.upsample_list[i](x, up_x, unq_inv)
            x = replace_feature(x, norm(x.features))
            i = i + 1
        return x

    def get_pos_embed(self, spatial_shape, coors, embed_layer, normalize_pos=True):
        '''
        Args:
        coors_in_win: shape=[N, 3], order: z, y, x
        '''
        # [N,]
        window_shape = spatial_shape[::-1]  # spatial_shape:   win_z, win_y, win_x ---> win_x, win_y, win_z

        embed_layer = embed_layer
        if len(window_shape) == 2:
            ndim = 2
            win_x, win_y = window_shape
            win_z = 0
        elif window_shape[-1] == 1:
            ndim = 2
            win_x, win_y = window_shape[:2]
            win_z = 0
        else:
            win_x, win_y, win_z = window_shape
            ndim = 3

        z, y, x = coors[:, 0] - win_z / 2, coors[:, 1] - win_y / 2, coors[:, 2] - win_x / 2

        if normalize_pos:
            x = x / win_x * 2 * 3.1415  # [-pi, pi]
            y = y / win_y * 2 * 3.1415  # [-pi, pi]
            z = z / win_z * 2 * 3.1415  # [-pi, pi]

        if ndim == 2:
            location = torch.stack((x, y), dim=-1)
        else:
            location = torch.stack((x, y, z), dim=-1)
        pos_embed = embed_layer(location)

        return pos_embed
    
    

class MLPBlock(nn.Module):
    def __init__(self, input_channel, out_channel, norm_fn):
        super().__init__()
        self.mlp_layer = nn.Sequential(
            nn.Linear(input_channel, out_channel),
            norm_fn(out_channel),
            nn.GELU())

    def forward(self, x):
        mpl_feats = self.mlp_layer(x)
        return mpl_feats

#for waymo and nuscenes, kitti, once
class LION3DBackboneOneStride(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg

        self.sparse_shape = grid_size[::-1]  # + [1, 0, 0]
        norm_fn = partial(nn.LayerNorm)

        dim = model_cfg.FEATURE_DIM
        num_layers = model_cfg.NUM_LAYERS
        depths = model_cfg.DEPTHS
        layer_down_scales = model_cfg.LAYER_DOWN_SCALES
        direction = model_cfg.DIRECTION
        diffusion = model_cfg.DIFFUSION
        shift = model_cfg.SHIFT
        diff_scale = model_cfg.DIFF_SCALE
        self.window_shape = model_cfg.WINDOW_SHAPE
        self.group_size = model_cfg.GROUP_SIZE
        self.layer_dim = model_cfg.LAYER_DIM
        self.linear_operator = model_cfg.OPERATOR
        
        self.n_layer = len(depths) * depths[0] * 2 * 2 + 2

        down_scale_list = [[2, 2, 2],
                           [2, 2, 2],
                           [2, 2, 1],
                           [1, 1, 2],
                           [1, 1, 2]
                           ]
        total_down_scale_list = [down_scale_list[0]]
        for i in range(len(down_scale_list) - 1):
            tmp_dow_scale = [x * y for x, y in zip(total_down_scale_list[i], down_scale_list[i + 1])]
            total_down_scale_list.append(tmp_dow_scale)

        assert num_layers == len(depths)
        assert len(layer_down_scales) == len(depths)
        assert len(layer_down_scales[0]) == depths[0]
        assert len(self.layer_dim) == len(depths)

        
        self.linear_1 = LIONBlock(self.layer_dim[0], depths[0], layer_down_scales[0], self.window_shape[0],
                                    self.group_size[0], direction, shift=shift, operator=self.linear_operator, layer_id=0, n_layer=self.n_layer)  ##[27, 27, 32] --》 [13, 13, 32]

        self.dow1 = PatchMerging3D(self.layer_dim[0], self.layer_dim[0], down_scale=[1, 1, 2],
                                     norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)
        

        # [944, 944, 16] -> [472, 472, 8]
        self.linear_2 = LIONBlock(self.layer_dim[1], depths[1], layer_down_scales[1], self.window_shape[1],
                                    self.group_size[1], direction, shift=shift, operator=self.linear_operator, layer_id=8, n_layer=self.n_layer)

        self.dow2 = PatchMerging3D(self.layer_dim[1], self.layer_dim[1], down_scale=[1, 1, 2],
                                     norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)


        #  [236, 236, 8] -> [236, 236, 4]
        self.linear_3 = LIONBlock(self.layer_dim[2], depths[2], layer_down_scales[2], self.window_shape[2],
                                    self.group_size[2], direction, shift=shift, operator=self.linear_operator, layer_id=16, n_layer=self.n_layer)

        self.dow3 = PatchMerging3D(self.layer_dim[2], self.layer_dim[3], down_scale=[1, 1, 2],
                                     norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)

        #  [236, 236, 4] -> [236, 236, 2]
        self.linear_4 = LIONBlock(self.layer_dim[3], depths[3], layer_down_scales[3], self.window_shape[3],
                                    self.group_size[3], direction, shift=shift, operator=self.linear_operator, layer_id=24, n_layer=self.n_layer)

        self.dow4 = PatchMerging3D(self.layer_dim[3], self.layer_dim[3], down_scale=[1, 1, 2],
                                     norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)

        self.linear_out = LIONLayer(self.layer_dim[3], 1, [13, 13, 2], 256, direction=['x', 'y'], shift=shift,
                                      operator=self.linear_operator, layer_id=32, n_layer=self.n_layer)

        self.num_point_features = dim

        self.backbone_channels = {
            'x_conv1': 128,
            'x_conv2': 128,
            'x_conv3': 128,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.linear_1(x)
        x1, _ = self.dow1(x)  ## 14.0k --> 16.9k  [32, 1000, 1000]-->[16, 1000, 1000]
        x = self.linear_2(x1)
        x2, _ = self.dow2(x)  ## 16.9k --> 18.8k  [16, 1000, 1000]-->[8, 1000, 1000]
        x = self.linear_3(x2)
        x3, _ = self.dow3(x)   ## 18.8k --> 19.1k  [8, 1000, 1000]-->[4, 1000, 1000]
        x = self.linear_4(x3)
        x4, _ = self.dow4(x)  ## 19.1k --> 18.5k  [4, 1000, 1000]-->[2, 1000, 1000]
        x = self.linear_out(x4)

        batch_dict.update({
            'encoded_spconv_tensor': x,
            'encoded_spconv_tensor_stride': 1
        })

        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x1,
                'x_conv2': x2,
                'x_conv3': x3,
                'x_conv4': x4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': torch.tensor([1,1,2], device=x1.features.device).float(),
                'x_conv2': torch.tensor([1,1,4], device=x1.features.device).float(),
                'x_conv3': torch.tensor([1,1,8], device=x1.features.device).float(),
                'x_conv4': torch.tensor([1,1,16], device=x1.features.device).float(),
            }
        })

        return batch_dict



#for argoverse
class LION3DBackboneOneStride_Sparse(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg

        self.sparse_shape = grid_size[::-1]  # + [1, 0, 0]
        norm_fn = partial(nn.LayerNorm)

        dim = model_cfg.FEATURE_DIM
        num_layers = model_cfg.NUM_LAYERS
        depths = model_cfg.DEPTHS
        layer_down_scales = model_cfg.LAYER_DOWN_SCALES
        direction = model_cfg.DIRECTION
        diffusion = model_cfg.DIFFUSION
        shift = model_cfg.SHIFT
        diff_scale = model_cfg.DIFF_SCALE
        self.window_shape = model_cfg.WINDOW_SHAPE
        self.group_size = model_cfg.GROUP_SIZE
        self.layer_dim = model_cfg.LAYER_DIM
        self.linear_operator = model_cfg.OPERATOR
        
        self.n_layer = len(depths) * depths[0] * 2 * 2 + 2 + 2*3

        down_scale_list = [[2, 2, 2],
                           [2, 2, 2],
                           [2, 2, 1],
                           [1, 1, 2],
                           [1, 1, 2]
                           ]
        total_down_scale_list = [down_scale_list[0]]
        for i in range(len(down_scale_list) - 1):
            tmp_dow_scale = [x * y for x, y in zip(total_down_scale_list[i], down_scale_list[i + 1])]
            total_down_scale_list.append(tmp_dow_scale)

        assert num_layers == len(depths)
        assert len(layer_down_scales) == len(depths)
        assert len(layer_down_scales[0]) == depths[0]
        assert len(self.layer_dim) == len(depths)

        
        self.linear_1 = LIONBlock(self.layer_dim[0], depths[0], layer_down_scales[0], self.window_shape[0],
                                    self.group_size[0], direction, shift=shift, operator=self.linear_operator, layer_id=0, n_layer=self.n_layer)  ##[27, 27, 32] --》 [13, 13, 32]

        self.dow1 = PatchMerging3D(self.layer_dim[0], self.layer_dim[0], down_scale=[1, 1, 2],
                                     norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)
        

        # [944, 944, 16] -> [472, 472, 8]
        self.linear_2 = LIONBlock(self.layer_dim[1], depths[1], layer_down_scales[1], self.window_shape[1],
                                    self.group_size[1], direction, shift=shift, operator=self.linear_operator, layer_id=8, n_layer=self.n_layer)

        self.dow2 = PatchMerging3D(self.layer_dim[1], self.layer_dim[1], down_scale=[1, 1, 2],
                                     norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)


        #  [236, 236, 8] -> [236, 236, 4]
        self.linear_3 = LIONBlock(self.layer_dim[2], depths[2], layer_down_scales[2], self.window_shape[2],
                                    self.group_size[2], direction, shift=shift, operator=self.linear_operator, layer_id=16, n_layer=self.n_layer)

        self.dow3 = PatchMerging3D(self.layer_dim[2], self.layer_dim[3], down_scale=[1, 1, 2],
                                     norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)

        #  [236, 236, 4] -> [236, 236, 2]
        self.linear_4 = LIONBlock(self.layer_dim[3], depths[3], layer_down_scales[3], self.window_shape[3],
                                    self.group_size[3], direction, shift=shift, operator=self.linear_operator, layer_id=24, n_layer=self.n_layer)

        self.dow4 = PatchMerging3D(self.layer_dim[3], self.layer_dim[3], down_scale=[1, 1, 2],
                                     norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)

        self.linear_out = LIONLayer(self.layer_dim[3], 1, [13, 13, 2], 256, direction=['x', 'y'], shift=shift,
                                      operator=self.linear_operator, layer_id=32, n_layer=self.n_layer)
        
        self.dow_out = PatchMerging3D(self.layer_dim[3], self.layer_dim[3], down_scale=[1, 1, 2],
                                        norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)

        self.linear_bev1 = LIONLayer(self.layer_dim[3], 1, [25, 25, 1], 512, direction=['x', 'y'], shift=shift,
                                      operator=self.linear_operator, layer_id=34, n_layer=self.n_layer)
        self.linear_bev2 = LIONLayer(self.layer_dim[3], 1, [37, 37, 1], 1024, direction=['x', 'y'], shift=shift,
                                       operator=self.linear_operator, layer_id=36, n_layer=self.n_layer)
        self.linear_bev3 = LIONLayer(self.layer_dim[3], 1, [51, 51, 1], 2048, direction=['x', 'y'], shift=shift,
                                       operator=self.linear_operator, layer_id=38, n_layer=self.n_layer)
        

        self.num_point_features = dim

    def forward(self, batch_dict):
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.linear_1(x)
        x, _ = self.dow1(x)
        x = self.linear_2(x)
        x, _ = self.dow2(x)
        x = self.linear_3(x)
        x, _ = self.dow3(x)
        x = self.linear_4(x)
        x, _ = self.dow4(x)
        x = self.linear_out(x)
        
        
        x, _ = self.dow_out(x)

        x = self.linear_bev1(x)
        x = self.linear_bev2(x)
        x = self.linear_bev3(x)

        x_new = spconv.SparseConvTensor(
            features=x.features,
            indices=x.indices[:, [0, 2, 3]].type(torch.int32), #x.indices,
            spatial_shape=x.spatial_shape[1:],
            batch_size=x.batch_size
        )

        batch_dict.update({
            'encoded_spconv_tensor': x_new,
            'encoded_spconv_tensor_stride': 1
        })

        batch_dict.update({'spatial_features_2d': x_new})

        return batch_dict
