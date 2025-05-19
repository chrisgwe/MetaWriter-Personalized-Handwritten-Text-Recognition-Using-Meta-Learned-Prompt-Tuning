import math
import warnings
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple

from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union
import torchvision as tv
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
__all__ = ['Conv2d']



class _ConvNd(Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        ...

    in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    padding_init: str
    num_tokens: int
    data_crop_size: int
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 ## new added aruguments
                 padding_init: str,
                 num_tokens: int,
                 data_crop_size: int,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_ConvNd, self).__init__()
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding, valid_padding_strings))
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular', 'trainable'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.padding_init = padding_init
        if num_tokens <= 0:
            raise ValueError(f"Invalid num_tokens: {num_tokens}. Must be greater than 0.")
        self.num_tokens = num_tokens
        self.data_crop_size = data_crop_size
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, kernel_size,
                                   range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        if transposed:
            self.weight = Parameter(torch.empty(
                (in_channels, out_channels // groups, *kernel_size), **factory_kwargs))
        else:
            self.weight = Parameter(torch.empty(
                (out_channels, in_channels // groups, *kernel_size), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'



class Conv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        ## new added aruguments
        padding_init: str = 'gaussian',#'random',
        num_tokens: int = 1,
        data_crop_size: int = 1,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, padding_init, num_tokens,data_crop_size, **factory_kwargs)
        if self.padding_mode == 'trainable':
            self._setup_prompt_pad()
    def make_padding_trainable(self):
        self.padding_mode = 'trainable'
        self._setup_prompt_pad()
    def _setup_prompt_pad(self):
        if self.padding_init == "random":
            self.prompt_embeddings_tb = nn.Parameter(torch.zeros(
                    1, self.in_channels, 2 * self.num_tokens,
                    self.data_crop_size + 2 * self.num_tokens
            ))
            self.prompt_embeddings_lr = nn.Parameter(torch.zeros(
                    1, self.in_channels, self.data_crop_size, 2 * self.num_tokens
            ))

            nn.init.uniform_(self.prompt_embeddings_tb.data, 0.0, 1.0)
            nn.init.uniform_(self.prompt_embeddings_lr.data, 0.0, 1.0)

            self.prompt_norm = tv.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

        elif self.padding_init == "gaussian":
            if self.data_crop_size <= 0:
                raise ValueError(f"Invalid data_crop_size: {self.data_crop_size}. Must be greater than 0.")
            if self.num_tokens <= 0:
                raise ValueError(f"Invalid num_tokens: {self.num_tokens}. Must be greater than 0.")

            self.prompt_embeddings_tb = nn.Parameter(torch.zeros(
                    1, self.in_channels, 2 * self.num_tokens,
                    self.data_crop_size + 2 * self.num_tokens
            ))
            self.prompt_embeddings_lr = nn.Parameter(torch.zeros(
                    1, self.in_channels, self.data_crop_size, 2 * self.num_tokens
            ))

            nn.init.normal_(self.prompt_embeddings_tb.data)
            nn.init.normal_(self.prompt_embeddings_lr.data)

            self.prompt_norm = nn.Identity()
                    # Validation check
            if self.prompt_embeddings_lr.shape[-1] == 0:
                raise ValueError(f"Invalid width after initialization: {self.prompt_embeddings_lr.shape}")
            if self.prompt_embeddings_tb.shape[-2] == 0:
                raise ValueError(f"Invalid height after initialization: {self.prompt_embeddings_tb.shape}")
        elif self.padding_init == "zero":
            self.prompt_embeddings_tb = nn.Parameter(torch.zeros(
                    1, self.in_channels, 2 * self.num_tokens,
                    self.data_crop_size + 2 * self.num_tokens
            ))
            self.prompt_embeddings_lr = nn.Parameter(torch.zeros(
                    1, self.in_channels, self.data_crop_size, 2 * self.num_tokens
            ))

            self.prompt_norm = nn.Identity()

        else:
            raise ValueError("Other initiation scheme is not supported")

        
    
    def _incorporate_prompt(self, x):
        B, C, H, W = x.shape
        #print("x",x.shape)
        if W <= 0 or H <= 0:
            raise ValueError(f"Invalid tensor dimensions: Height (H) = {H}, Width (W) = {W}. Both must be greater than 0.")

        prompt_width = max(2 * self.num_tokens, 1)
        prompt_emb_lr_resized = F.interpolate(self.prompt_norm(self.prompt_embeddings_lr), size=(H, prompt_width), mode='bilinear', align_corners=False)
        prompt_emb_tb_resized = F.interpolate(self.prompt_norm(self.prompt_embeddings_tb), size=(prompt_width, W), mode='bilinear', align_corners=False)

        prompt_emb_lr_resized = prompt_emb_lr_resized.expand(B, -1, -1, -1).to(x.device)
        prompt_emb_tb_resized = prompt_emb_tb_resized.expand(B, -1, -1, -1).to(x.device)
        
        x = torch.cat((
            prompt_emb_lr_resized[:, :, :, :self.num_tokens],  
            x,                                                
            prompt_emb_lr_resized[:, :, :, self.num_tokens:]   
        ), dim=-1)  
        if x.shape[-1] <= 0:
            raise ValueError("Concatenated tensor width is zero or negative after left-right concatenation.")
    
        prompt_emb_tb_resized = F.interpolate(prompt_emb_tb_resized, size=(2 * self.num_tokens, x.shape[-1]), mode='bilinear', align_corners=False)
        
        x = torch.cat((
            prompt_emb_tb_resized[:, :, :self.num_tokens, :],  
            x,                                                 
            prompt_emb_tb_resized[:, :, self.num_tokens:, :]  
        ), dim=-2) 
        if x.shape[-2] <= 0:
            raise ValueError("Concatenated tensor height is zero or negative after top-bottom concatenation.")
        return x
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode == 'trainable':
            x = self._incorporate_prompt(input)
            return F.conv2d(x,
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        if self.padding_mode != 'zeros' and self.padding_mode != 'trainable':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)