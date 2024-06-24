import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import LayerNorm as LayerNorm

from einops import rearrange 

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class Linear_lora(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result = result + (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

         
class Linear_common(nn.Linear):
    def __init__(self, input_dim, output_dim, init, bias=True):
        super().__init__(input_dim, output_dim, bias=bias)

        assert init in ['gate', 'final', 'attn', 'relu', 'linear']

        if init in ['gate', 'final']:
            nn.init.constant_(self.weight, 0.)
        elif init == 'attn':
            # GlorotUniform
            torch.nn.init.xavier_uniform_(self.weight)
        elif init in ['relu', 'linear']:
            # Relu, He
            # linear, Le cun
            distribution_stddev = 0.87962566103423978
            scale = 2. if init == 'relu' else 1.
            stddev = np.sqrt(scale / input_dim) / distribution_stddev
            nn.init.trunc_normal_(self.weight, mean=0., std=stddev)
        else:
            raise NotImplementedError(f'{init} not Implemented')

        if bias:
            if init == 'gate':
                nn.init.constant_(self.bias, 1.)
            else:
                nn.init.constant_(self.bias, 0.)


def Linear(input_dim, output_dim, init, bias=True, config=None):
    assert init in ['gate', 'final', 'attn', 'relu', 'linear']  
    if config is not None:
        if config.enabled:
            return Linear_lora(input_dim, output_dim, config.rank, config.dropout_rate)
        else:
            return Linear_common(input_dim, output_dim, init, bias)            
    else:
        return Linear_common(input_dim, output_dim, init, bias)
    

def apply_dropout(tensor, rate, is_training, broadcast_dim=None):
    if is_training and rate > 0.0:
        if broadcast_dim is not None:
            shape = list(tensor.shape)
            shape[broadcast_dim] = 1
            with torch.no_grad():
                scale = 1. / (1. - rate)
                keep_rate = torch.full(shape, 1. - rate, dtype=tensor.dtype, device=tensor.device)
                keep = torch.bernoulli(keep_rate)
            return scale * keep * tensor
        else:
            return F.dropout(tensor, rate)
    else:
        return tensor
