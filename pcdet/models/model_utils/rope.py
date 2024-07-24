from math import pi

import torch
from torch import nn

from einops import rearrange, repeat

from pcdet.utils.spconv_utils import replace_feature


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
            self,
            dim,
            theta=10000,
    ):
        super().__init__()
        assert dim % 6 == 0
        self.freq = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))


    def forward(self, x):
        batch_size = x.batch_size
        fs = list()

        for b in range(batch_size):
            t = x.features[x.indices[:, 0] == b]

            freqs = list()
            for axis in range(3):
                ind = x.indices[x.indices[:, 0] == b][:, 3 - axis]
                # seq_len = x.spatial_shape[2 - axis]
                s = ind

                freq = torch.einsum('..., f -> ... f', s, self.freq.to(s.device))
                freq = repeat(freq, '... n -> ... (n r)', r=2)
                freqs.append(freq)
            freqs = torch.cat(freqs, dim=-1)
            freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
            freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

            t = t * freqs_cos + rotate_half(t) * freqs_sin
            fs.append(t)
        x = replace_feature(x, torch.cat(fs, dim=0))
        return x
