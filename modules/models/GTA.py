import functools

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from einops import rearrange

from .common_modules import(
        Linear,
        LayerNorm,
        apply_dropout,)

import pdb
import logging
import functools as fn
import math

logger = logging.getLogger(__name__)

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    """
    From Fairseq.Build sinusoidal embeddings.This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1: # Zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb  


class Attention(nn.Module):
    def __init__(self, input_dim, key_dim, value_dim, output_dim, num_head,split_first=True, gating=True,inp_kernels=None, config=None):
        super().__init__()
        assert key_dim % num_head == 0
        assert value_dim % num_head == 0

        self.key_dim, self.value_dim = key_dim, value_dim

        self.num_head = num_head
        
        self.split_first = split_first

        if self.split_first:
            self.proj_q = Linear(input_dim, key_dim, init='attn', bias=False, config=config)
            self.proj_k = Linear(input_dim, key_dim, init='attn', bias=False, config=config)
            self.proj_v = Linear(input_dim, value_dim, init='attn', bias=False, config=config)
        else:
            assert (key_dim == value_dim)
            self.proj_in = Linear(input_dim, key_dim * 3, init='attn', bias=False, config=config)
        
        self.gating = gating
        if gating:
            self.gate= Linear(input_dim, value_dim, init='gate', config=config)

        self.proj_out = Linear(value_dim, output_dim, init='final', config=config)
         
        self.inp_kernels = inp_kernels
        if inp_kernels:
            self.inp_q = SpatialDepthWiseInception(key_dim // num_head, inp_kernels)
            self.inp_k = SpatialDepthWiseInception(key_dim // num_head, inp_kernels)
            self.inp_v = SpatialDepthWiseInception(value_dim // num_head, inp_kernels)

    def forward(self, q_data, k_data=None, bias=None, k_mask=None):
        """
        Arguments:
            q_data: (batch_size, N_seqs, N_queries, q_channel)
            k_data: (batch_size, N_seqs, N_keys, k_channel)
            k_mask: (batch_size, N_seqs, N_keys)
            bias  : (batch_size, N_queries, N_keys). shared by all seqs
        Returns:
            (b s l c)
        """
        key_dim, value_dim = self.key_dim // self.num_head, self.value_dim // self.num_head
        
        if self.split_first:
            assert (k_data is not None)
            q = self.proj_q(q_data) 
            k = self.proj_k(k_data)
            v = self.proj_v(k_data)
            q, k, v = map(lambda t: rearrange(t, 'b s l (h d) -> b s h l d', h = self.num_head), (q, k, v))
        else:
            assert (k_data is None)
            t = rearrange(self.proj_in(q_data), "... l (h d) -> ... h l d", h=self.num_head)
            q, k, v = torch.chunk(t, 3, dim=-1)
        
        if self.inp_kernels:
            q, k, v = map(lambda t: rearrange(t, 'b s h l d-> b (s h) l d'), (q, k, v))
            q = self.inp_q(q)
            k = self.inp_k(k)
            v = self.inp_v(v)
            q, k, v = map(lambda t: rearrange(t, 'b (s h) l d-> b s h l d', h = self.num_head), (q, k, v))
        
        q = q* key_dim**(-0.5)

        logits = torch.einsum('... h q d, ... h k d -> ... h q k', q, k)

        if bias is not None:
            logits = logits + rearrange(bias,  'b h q k -> b () h q k')

        if k_mask is not None:
            mask_value = torch.finfo(logits.dtype).min
            k_mask = rearrange(k_mask, 'b s k -> b s () () k')
            logits = logits.masked_fill(~k_mask.bool(), mask_value)

        weights = F.softmax(logits, dim = -1)
        weighted_avg = torch.einsum('b s h q k, b s h k d -> b s h q d', weights, v)
        weighted_avg = rearrange(weighted_avg, 'b s h q d -> b s q (h d)')
        
        if self.gating:
            gate_values = torch.sigmoid(self.gate(q_data))
            weighted_avg = weighted_avg * gate_values

        output = self.proj_out(weighted_avg)

        return output


class SeqAttentionWithPairBias(nn.Module):
    def __init__(self, config, num_in_seq_channel, num_in_pair_channel):
        super().__init__()
        c = config
        try:
            LoRA_conf = c.LoRA
        except:
            LoRA_conf = None
        self.seq_norm = LayerNorm(num_in_seq_channel)
        self.pair_norm = LayerNorm(num_in_pair_channel)
        self.proj_pair = Linear(num_in_pair_channel, c.num_head, init='linear', bias = False, config=LoRA_conf)

        self.attn = Attention(
                input_dim=num_in_seq_channel,
                key_dim=num_in_seq_channel,
                value_dim=num_in_seq_channel,
                output_dim=num_in_seq_channel,
                num_head=c.num_head,
                split_first=False,
                inp_kernels=c.inp_kernels,
                config=LoRA_conf)

        self.config = config

    def forward(self, seq_act, pair_act, mask):
        """
        Arguments:
            seq_act: (b l c)
            pair_act: (b l l c)
            mask: (b l), padding mask
        Returns:
            (b l c)
        """
        mask = rearrange(mask, 'b l -> b () l')
        seq_act = self.seq_norm(seq_act)
        
        pair_act = self.pair_norm(pair_act)
        bias = rearrange(self.proj_pair(pair_act), 'b i j h -> b h i j')
        
        seq_act = rearrange(seq_act, 'b l c -> b () l c')
        seq_act = self.attn(q_data=seq_act, bias=bias, k_mask=mask)
        seq_act = rearrange(seq_act, 'b s l c -> (b s) l c')
        return seq_act


class Transition(nn.Module):
    def __init__(self, config, num_in_channel, num_out_channel=None):
        super().__init__()

        c = config
        try:
            LoRA_conf = c.LoRA
        except:
            LoRA_conf = None

        if num_out_channel is None:
            num_out_channel = num_in_channel

        intermediate_channel = num_in_channel * c.num_intermediate_factor
        self.transition = nn.Sequential(
                LayerNorm(num_in_channel),
                Linear(num_in_channel, intermediate_channel, init='linear', config=LoRA_conf),
                nn.ReLU(),
                Linear(intermediate_channel, num_out_channel, init='final', config=LoRA_conf),
                )

    def forward(self, act, mask):
        return self.transition(act)

# AF2 and ESM-FOLD have different implementations
# Here we just follow ESMFOLD
class OuterProductMean(nn.Module):
    def __init__(self, config, num_in_channel, num_out_channel):
        super().__init__()

        c = config
        try:
            LoRA_conf = c.LoRA
        except:
            LoRA_conf = None
        self.norm = LayerNorm(num_in_channel)
        self.left_proj = Linear(num_in_channel, c.num_outer_channel, init='linear', config=LoRA_conf)
        self.right_proj = Linear(num_in_channel, c.num_outer_channel, init='linear', config=LoRA_conf)

        self.out_proj = Linear(2 * c.num_outer_channel, num_out_channel, init='final', config=LoRA_conf)

    def forward(self, act, mask):
        """
        act: (b l c)
        mask: (b l)
        """
        mask = rearrange(mask, 'b l -> b l ()')
        act = self.norm(act)
        left_act = mask * self.left_proj(act)
        right_act = mask * self.right_proj(act)

        #act = rearrange(left_act, 'b l c -> b l () c ()') * rearrange(right_act, 'b l c -> b  () l () c')
        #act = torch.einsum('b i c, b j d -> b i j c d', left_act, right_act)
        #act = rearrange(act, 'b i j c d -> b i j (c d)')
        
        prod = left_act[:, None, :, :] * right_act[:, :, None, :]
        diff = left_act[:, None, :, :] - right_act[:, :, None, :]

        act = torch.cat([prod, diff], dim=-1)
        act = self.out_proj(act)

        return act


class TriangleMultiplication(nn.Module): # pair_mask
    def __init__(self, config, num_in_channel):
        super().__init__()
        c = config
        assert c.orientation in ['per_row', 'per_column']
        try:
            LoRA_conf = c.LoRA
        except:
            LoRA_conf = None
        self.norm = LayerNorm(num_in_channel)

        self.left_proj = Linear(num_in_channel, c.num_intermediate_channel, init='linear', config=LoRA_conf)
        self.right_proj = Linear(num_in_channel, c.num_intermediate_channel, init='linear', config=LoRA_conf)

        self.final_norm = LayerNorm(c.num_intermediate_channel)
        
        if c.gating:
            self.left_gate = Linear(num_in_channel, c.num_intermediate_channel, init='gate', config=LoRA_conf)
            self.right_gate = Linear(num_in_channel, c.num_intermediate_channel, init='gate', config=LoRA_conf)
            self.final_gate = Linear(num_in_channel, num_in_channel, init='gate', config=LoRA_conf)
        
        self.proj_out = Linear(c.num_intermediate_channel, num_in_channel, init='final', config=LoRA_conf)

        
        if c.inp_kernels:
            self.inp_left = SpatialDepthWiseInception(c.num_intermediate_channel // c.num_head, c.inp_kernels)
            self.inp_right = SpatialDepthWiseInception(c.num_intermediate_channel // c.num_head, c.inp_kernels)

        self.config = c

    def forward(self, act, pair_mask):
        """
        act: (b l l c)
        mask: (b l)
        """
        c = self.config

        #pair_mask = rearrange(mask, 'b l -> b l () ()') * rearrange(mask, 'b l -> b () l ()')
        # TODO： change pair_mask
        #pair_mask = mask[:,:,None,None] * mask[:,None,:,None]
        
        act = self.norm(act)

        input_act = act

        left_proj_act = self.left_proj(act)
        right_proj_act = self.right_proj(act)
        
        if c.inp_kernels:
            if c.orientation == 'per_row':
                equation = 'b i j (h d) -> b (i h) j d'
            else:
                equation = 'b i j (h d) -> b (j h) i d'

            left_proj_act, right_proj_act = map(
                    lambda t: rearrange(t, equation, h = c.num_head), (left_proj_act, right_proj_act))

            left_proj_act = self.inp_left(left_proj_act)
            right_proj_act = self.inp_right(right_proj_act)
            
            if c.orientation == 'per_row':
                equation = 'b (i h) j d -> b i j (h d)'
            else:
                equation = 'b (j h) i d -> b i j (h d)'
            
            left_proj_act, right_proj_act = map(
                    lambda t: rearrange(t, equation, h = c.num_head), (left_proj_act, right_proj_act))
        
        left_proj_act = pair_mask * left_proj_act
        right_proj_act = pair_mask * right_proj_act
        
        if c.gating:
            left_gate_values = torch.sigmoid(self.left_gate(act))
            right_gate_values = torch.sigmoid(self.right_gate(act))

            left_proj_act = left_proj_act * left_gate_values
            right_proj_act = right_proj_act * right_gate_values

        if c.orientation == 'per_row':
            act = torch.einsum('b i k c, b j k c -> b i j c', left_proj_act, right_proj_act)
        elif c.orientation == 'per_column':
            act = torch.einsum('b k i c, b k j c -> b i j c', left_proj_act, right_proj_act)
        else:
            raise NotImplementedError(f'{self.orientation} not Implemented')

        act = self.final_norm(act)
        act = self.proj_out(act)
        
        if c.gating:
            gate_values = torch.sigmoid(self.final_gate(input_act))
            act = act * gate_values

        return act


class TriangleAttention(nn.Module): # seq_mask
    def __init__(self, config, num_in_pair_channel):
        super().__init__()
        c = config

        assert c.orientation in ['per_row', 'per_column']
        try:
            LoRA_conf = c.LoRA
        except:
            LoRA_conf = None

        self.norm = LayerNorm(num_in_pair_channel)
        self.proj_pair = Linear(num_in_pair_channel, c.num_head, init='linear', bias = False, config=LoRA_conf)
        self.attn = Attention(
                input_dim=num_in_pair_channel,
                key_dim=num_in_pair_channel,
                value_dim=num_in_pair_channel,
                output_dim=num_in_pair_channel,
                num_head=c.num_head,
                gating=c.gating,
                inp_kernels=c.inp_kernels,
                config=LoRA_conf)

        self.config = config

    def forward(self, pair_act, seq_mask):
        '''
        pair_act: (b l l c)
        seq_mask: (b l)
        '''
        c = self.config
        if c.orientation == 'per_column':
            pair_act = rearrange(pair_act, 'b i j c -> b j i c')

        pair_act = self.norm(pair_act)
        seq_mask = rearrange(seq_mask, 'b l -> b () l')

        bias = rearrange(self.proj_pair(pair_act), 'b i j h -> b h i j')

        pair_act = self.attn(q_data=pair_act, k_data=pair_act, bias=bias, k_mask=seq_mask)

        if c.orientation == 'per_column':
            pair_act = rearrange(pair_act, 'b i j c -> b j i c')

        return pair_act


class gtaformerIteration(nn.Module):
    def __init__(self, config, seq_channel, pair_channel):
        super().__init__()
        c = config

        self.seq_attn = SeqAttentionWithPairBias(c.seq_attention_with_pair_bias, seq_channel, pair_channel)
        self.seq_transition = Transition(c.seq_transition, seq_channel)
        self.outer_product_mean = OuterProductMean(c.outer_product_mean, seq_channel, pair_channel)
        
        self.triangle_multiplication_outgoing = TriangleMultiplication(c.triangle_multiplication_outgoing, pair_channel)
        self.triangle_multiplication_incoming = TriangleMultiplication(c.triangle_multiplication_incoming, pair_channel)
        self.triangle_attention_starting_node = TriangleAttention(c.triangle_attention_starting_node, pair_channel)
        self.triangle_attention_ending_node = TriangleAttention(c.triangle_attention_ending_node, pair_channel)
        self.pair_transition = Transition(c.pair_transition, pair_channel)

        self.seq_left_transition = Transition(c.seq_transition, pair_channel, seq_channel)
        self.seq_right_transition = Transition(c.seq_transition, pair_channel, seq_channel)
        self.config = config

    def forward(self, seq_act, pair_act, seq_mask, pair_mask):
        """
        seq_act: (b l c)
        pair_act: (b l l c)
        seq_mask: (b l)
        """
        c = self.config

        def dropout_fn(input_act, act, config):
            if self.training and config.dropout_rate > 0.:
                if config.shared_dropout:
                    if config.orientation == 'per_row':
                        broadcast_dim = 1
                    else:
                        broadcast_dim = 2
                else:
                    broadcast_dim = None
                act = apply_dropout(act, config.dropout_rate,
                        is_training=True, broadcast_dim=broadcast_dim)
            return input_act + act
        
        seq_act = dropout_fn(
                seq_act, self.seq_attn(seq_act, pair_act, seq_mask), c.seq_attention_with_pair_bias)
        seq_act = seq_act + self.seq_transition(seq_act, seq_mask)
        
        pair_act = pair_act + self.outer_product_mean(seq_act, seq_mask)
        
        pair_act = dropout_fn(
                pair_act, self.triangle_multiplication_outgoing(pair_act, pair_mask), c.triangle_multiplication_outgoing)
        pair_act = dropout_fn(
                pair_act, self.triangle_multiplication_incoming(pair_act, pair_mask), c.triangle_multiplication_incoming)
        pair_act = dropout_fn(
                pair_act, self.triangle_attention_starting_node(pair_act, seq_mask), c.triangle_attention_starting_node)
        pair_act = dropout_fn(
                pair_act, self.triangle_attention_ending_node(pair_act, seq_mask), c.triangle_attention_ending_node)
        pair_act = pair_act + self.pair_transition(pair_act, seq_mask)
        # pdb.set_trace()
        seq_act = dropout_fn(seq_act, torch.sum(self.seq_left_transition(pair_act, pair_mask) * pair_mask, dim=1), c.seq_transition)
        seq_act = dropout_fn(seq_act, torch.sum(self.seq_right_transition(pair_act, pair_mask) * pair_mask, dim=2), c.seq_transition)


        return seq_act, pair_act


class gtaformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config

        self.blocks = nn.ModuleList([gtaformerIteration(c.gtaformer, c.seq_channel, c.pair_channel) for _ in range(c.gtaformer_num_block)])

    def forward(self, seq_act, pair_act, seq_mask, pair_mask, is_recycling=True):
        for it, block in enumerate(self.blocks):
            block_fn = functools.partial(block, seq_mask=seq_mask, pair_mask=pair_mask)
            if self.training and not is_recycling and it > 0:
                seq_act, pair_act = checkpoint(block_fn, seq_act, pair_act)
                # for name, param in block.named_parameters():
                #     if param.requires_grad:
                #         print(f"Parameter name: {name}")
                #         print("Parameter value: ", param.data)
                #         print("Parameter gradient: ", param.grad)
                # print(f"seq_act: {seq_act}")
                # print(f"pair_act: {pair_act}")
                #seq_act, pair_act = block_fn(seq_act, pair_act)
            else:
                seq_act, pair_act = block_fn(seq_act, pair_act)
        return seq_act, pair_act


class SpatialDepthWiseConvolution(nn.Module):
    def __init__(self, head_dim: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels=head_dim, out_channels=head_dim,
                kernel_size=(kernel_size,),
                # padding=(kernel_size - 1,),
                padding=kernel_size//2,
                groups=head_dim)
    
    def forward(self, x: torch.Tensor):
        batch_size, heads, seq_len, head_dim = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size * heads, head_dim, seq_len)
        x = self.conv(x)
        #if self.kernel_size>1:
        #    x = x[:, :, :-(self.kernel_size - 1)]
        x = x.view(batch_size, heads, head_dim, seq_len)
        x = x.permute(0, 1, 3, 2)
        return x


class SpatialDepthWiseInception(nn.Module):
    def __init__(self, head_dim, kernels):
        super().__init__()
       
        assert len(kernels) > 1 and  kernels[0] == 1

        self.convs = torch.nn.ModuleList([SpatialDepthWiseConvolution(head_dim, kernel_size=k) for k in kernels[1:]])
        self.kernels = kernels
    def forward(self, x):
        # x: (batch, num_heads, len, head_dim)
        
        assert x.shape[1] % len(self.kernels) == 0
        group_num_head = x.shape[1] // len(self.kernels)
        
        outputs = [x[:,:group_num_head]]

        for i, conv in enumerate(self.convs):
            outputs.append(conv(x[:,group_num_head*(i+1):group_num_head*(i+2)]))

        outputs = torch.cat(outputs, dim=1)

        return outputs
