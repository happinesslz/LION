import torch.nn as nn
from functools import partial
from pcdet.utils.spconv_utils import spconv


norm_fn_1d = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
norm_fn_2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)


def post_act_block_dense_1d(input_dim, output_dim, kernel_size, stride, padding, dilation=1, norm_fn=norm_fn_1d, conv_type='conv'):

    if conv_type == 'conv':
        conv = nn.Conv1d(input_dim, output_dim, kernel_size, stride, padding, dilation, bias=False)

    elif conv_type == 'deconv':
        conv = nn.ConvTranspose1d(input_dim, output_dim, stride, stride, bias=False)

    else:
        raise NotImplementedError

    return nn.Sequential(conv, norm_fn(output_dim), nn.ReLU())


def post_act_block_dense_2d(input_dim, output_dim, kernel_size, stride, padding, dilation=1, groups=1, norm_fn=norm_fn_2d, conv_type='conv'):

    if conv_type == 'conv':
        conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, dilation, groups=groups, bias=False)

    elif conv_type == 'deconv':
        conv = nn.ConvTranspose2d(input_dim, output_dim, stride, stride, groups=groups, bias=False)

    else:
        raise NotImplementedError

    return nn.Sequential(conv, norm_fn(output_dim), nn.ReLU())


def post_act_block_sparse_2d(input_dim, output_dim, kernel_size, stride=1, padding=0, norm_fn=norm_fn_1d, conv_type='subm', indice_key=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv2d(input_dim, output_dim, kernel_size, bias=False, indice_key=indice_key)

    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(input_dim, output_dim, kernel_size, stride=stride, padding=padding, bias=False, indice_key=indice_key)

    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv2d(input_dim, output_dim, kernel_size, indice_key=indice_key, bias=False)

    else:
        raise NotImplementedError

    return spconv.SparseSequential(conv, norm_fn(output_dim), nn.ReLU())


def post_act_block_sparse_3d(input_dim, output_dim, kernel_size, stride=1, padding=0, norm_fn=norm_fn_1d, conv_type='subm', indice_key=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(input_dim, output_dim, kernel_size, bias=False, indice_key=indice_key)

    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(input_dim, output_dim, kernel_size, stride=stride, padding=padding, bias=False, indice_key=indice_key)

    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(input_dim, output_dim, kernel_size, indice_key=indice_key, bias=False)

    else:
        raise NotImplementedError

    return spconv.SparseSequential(conv, norm_fn(output_dim), nn.ReLU())


class SparseBasicBlock2D(spconv.SparseModule):

    def __init__(self, dim, indice_key, norm_fn=norm_fn_1d):
        super(SparseBasicBlock2D, self).__init__()

        self.conv1 = spconv.SubMConv2d(dim, dim, 3, 1, 1, bias=False, indice_key=indice_key)
        self.bn1 = norm_fn(dim)

        self.conv2 = spconv.SubMConv2d(dim, dim, 3, 1, 1, bias=False, indice_key=indice_key)
        self.bn2 = norm_fn(dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        out = out.replace_feature(self.relu(out.features + x.features))
        return out


class SparseBasicBlock3D(spconv.SparseModule):

    def __init__(self, dim, indice_key, norm_fn=norm_fn_1d):
        super(SparseBasicBlock3D, self).__init__()

        self.conv1 = spconv.SubMConv3d(dim, dim, 3, 1, 1, bias=False, indice_key=indice_key)
        self.bn1 = norm_fn(dim)

        self.conv2 = spconv.SubMConv3d(dim, dim, 3, 1, 1, bias=False, indice_key=indice_key)
        self.bn2 = norm_fn(dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        out = out.replace_feature(self.relu(out.features + x.features))
        return out


class BasicBlock(nn.Module):

    def __init__(self, dim, groups=1, norm_fn=norm_fn_2d, downsample=False):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(dim, dim, 3, 1 + int(downsample), 1, groups=groups, bias=False)
        self.bn1 = norm_fn(dim)

        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, groups=groups, bias=False)
        self.bn2 = norm_fn(dim)
        self.relu = nn.ReLU()

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(dim, dim, 1, 2, 0, groups=groups, bias=False),
                norm_fn(dim),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if hasattr(self, 'downsample'):
            x = self.downsample(x)

        out = self.relu(out + x)
        return out
