import torch
import torch.nn as nn 
import torch.nn.functional as F
from einops import rearrange
import torch.utils.checkpoint as cp
from .retention import (chunk_retention, fused_chunk_retention,
                               fused_recurrent_retention, parallel_retention)
from timm.models.layers import DropPath


class MultiScaleRetention(nn.Module):

    def __init__(self, 
                 mode: str = 'fused_chunk',
                 hidden_size: int=1024, 
                 num_heads: int=8, 
                 expand_k: int=1.0, 
                 expand_v: int=1.0, 
                 used_MLP: bool=True,
                 with_cp=False,
                 **kwargs):
        super().__init__()
        self.mode = mode
        
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.num_heads = num_heads
        
        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        
        self.use_gate = not used_MLP
        if self.use_gate:
            self.g_norm = nn.LayerNorm(hidden_size, eps=1e-6)
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.g_act = nn.SiLU()
        
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.apply(self._initialize_weights)
        self.with_cp = with_cp

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(self, hidden_states: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b l d)
        rel_pos: mask: (n l l)
        '''
        def _inner_forward(hidden_states, rel_pos, chunkwise_recurrent, incremental_state):
            b, l, d = hidden_states.size()
            # mask = rel_pos
            
            # assert h*w == mask.size(1)

            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

            # seqlen_offset = 0
            # if past_key_values is not None:
            #     seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', h=self.num_heads), (q, k))
            # q, k = self.rotary(q, k, seqlen_offset)
            q, k = q.transpose(1, 2), k.transpose(1, 2)
            v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

            if self.mode == 'chunk':
                o, _ = chunk_retention(q, k, v)
            elif self.mode == 'fused_chunk':
                o, _ = fused_chunk_retention(q, k, v)
            elif self.mode == 'parallel':
                o, _ = parallel_retention(q, k, v)
            elif self.mode == 'fused_recurrent':
                o, _ = fused_recurrent_retention(q, k, v)
            else:
                raise NotImplementedError(f"Not supported mode `{self.mode}`.")

            o = rearrange(o, 'b h l d -> b l h d')
            if self.use_gate:
                g = self.g_proj(hidden_states)
                o = self.g_norm(o)
                o = rearrange(o, 'b l h d -> b l (h d)')
                o = o * self.g_act(g)
            else:
                o = rearrange(o, 'b l h d -> b l (h d)')
            
            o = self.o_proj(o)
            return o
        if self.with_cp and hidden_states.requires_grad:
            hidden_states = cp.checkpoint(_inner_forward, hidden_states, rel_pos, chunkwise_recurrent, incremental_state)
        else:
            hidden_states = _inner_forward(hidden_states, rel_pos, chunkwise_recurrent, incremental_state)
        return hidden_states

class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn=F.gelu,
        dropout=0.0,
        activation_dropout=0.0,
        layernorm_eps=1e-6,
        subln=False,
        with_cp=False,
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else nn.Identity()
        self.with_cp = with_cp

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.ffn_layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        '''
        x: (b l d)
        '''
        def _inner_forward(x):
            x = self.fc1(x)
            x = self.activation_fn(x)
            x = self.activation_dropout_module(x)
            x = self.ffn_layernorm(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x
    
class Block(nn.Module):

    def __init__(self, mode: str, d_model: int, n_head: int, expand_k: int=1, expand_v: int=2, hidden_rate: int=4, used_MLP: bool=True, drop_path=0., layerscale=False, layer_init_values=1e-5, with_cp=False, **kwargs):
        super().__init__()
        self.layerscale = layerscale
        embed_dim = d_model
        self.embed_dim = embed_dim
        self.retention_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        assert mode in ['chunk', 'fused_chunk', 'fused_recurrent', 'parallel']
        self.multi_scale_retention = MultiScaleRetention(mode, embed_dim, n_head, expand_k, expand_v, used_MLP, with_cp=with_cp)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        ffn_dim = hidden_rate * embed_dim
        self.used_MLP = used_MLP
        if self.used_MLP:
            self.ffn = FeedForwardNetwork(embed_dim, ffn_dim, with_cp=with_cp)

        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim),requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim),requires_grad=True)
            
        self.with_cp = False # with_cp

    def forward(
            self,
            x: torch.Tensor, 
            incremental_state=None,
            chunkwise_recurrent=False,
            retention_rel_pos=None
        ):
        def _inner_forward(x, incremental_state, chunkwise_recurrent, retention_rel_pos):
            if self.layerscale:
                x = x + self.drop_path(self.gamma_1 * self.multi_scale_retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
                if self.used_MLP:
                    x = x + self.drop_path(self.gamma_2 * self.ffn(self.final_layer_norm(x)))
            else:
                x = x + self.drop_path(self.multi_scale_retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
                if self.used_MLP:
                    x = x + self.drop_path(self.ffn(self.final_layer_norm(x)))
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x, incremental_state, chunkwise_recurrent, retention_rel_pos)
        else:
            x = _inner_forward(x, incremental_state, chunkwise_recurrent, retention_rel_pos)
        return x