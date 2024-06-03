# export PYTHONPATH= $PYTHONPATH:/mnt/workspace/RetinexMamba/basicsr/models/archs/vmamba_arch.py

#将IGMSA替换为IFA进行特征融合
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
#from .SS2D_arch import SS2D
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import numbers
from einops import rearrange

import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """
    
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """
    
    return flops


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
    

class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim*2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x
    

class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,   ## 原来是3
        expand=2,  ## 原来是2
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)

        return out


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x
    


class VSSLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        upsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None


    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x
    


class VSSM(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        # WASTED absolute position embedding ======================
        self.ape = False
        # self.ape = False
        # drop_rate = 0.0
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer)

        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1]//4, num_classes, 1)

        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        skip_list = []
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)
        return x, skip_list
    
    def forward_features_up(self, x, skip_list):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x+skip_list[-inx])

        return x
    
    def forward_final(self, x):
        x = self.final_up(x)
        x = x.permute(0,3,1,2)
        x = self.final_conv(x)
        return x

    def forward_backbone(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        x, skip_list = self.forward_features(x)
        x = self.forward_features_up(x, skip_list)
        x = self.forward_final(x)
        
        return x




    
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(y))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class IFA(nn.Module):
    def __init__(self, dim_2, dim, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(IFA, self).__init__()

        self.conv1 = nn.Conv2d(dim_2, dim, (1, 1))
        # self.conv2 = nn.Conv2d(dim, dim_2, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, input_R, input_S):
        # input_ch = input_R.size()[1]
        input_S = F.interpolate(input_S, [input_R.shape[2], input_R.shape[3]])
        input_S = self.conv1(input_S)
        # input_S = F.interpolate(input_S, size=input_size, mode='bilinear', align_corners=True)
        input_R = self.norm1(input_R)
        input_S = self.norm1(input_S)
        input_R = input_R + self.attn(input_R, input_S)
        input_R = input_R + self.ffn(self.norm2(input_R))
        return input_R



import sys
print(sys.path)
# 这两个函数 _no_grad_trunc_normal_ 和 trunc_normal_ 都用于初始化神经网络中的权重参数。它们将权重初始化为遵循截断正态分布的值，这种方法有助于在训练开始时改善网络的性能和稳定性。
def _no_grad_trunc_normal_(tensor, mean, std, a, b):#这是一个内部函数，用于实际执行初始化过程。
    """
    用截断正态分布初始化张量。

    在给定的范围 [a, b] 内，根据指定的均值 (mean) 和标准差 (std) 截断正态分布，
    并用这个分布来填充输入的张
    量。
    

    参数:
        tensor (Tensor): 需要被初始化的张量。
        mean (float): 截断正态分布的均值。
        std (float): 截断正态分布的标准差。
        a (float): 分布的下限。
        b (float): 分布的上限。

    返回:
        Tensor: 填充后的张量，其值遵循指定的截断正态分布。
    
    功能：
        这个函数主要用于深度学习中的权重初始化，尤其是在需要限制权重范围以避免激活值过大或过小时非常有用。通过截断正态分布进行初始化，可以帮助神经网络更快地收敛并提高模型的稳定性。
    """
    # 计算正态分布的累积分布函数（CDF）值
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    # 检查均值是否在截断区间外的两个标准差范围内
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():  # 确保以下操作不计算梯度
        # 计算截断区间的累积分布函数值
        l = norm_cdf((a - mean) / std)  # noqa: E741
        u = norm_cdf((b - mean) / std)
        # 以这个区间初始化张量
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()  # 计算逆误差函数，以获得正态分布的值
        tensor.mul_(std * math.sqrt(2.))  # 缩放
        tensor.add_(mean)  # 加上均值
        tensor.clamp_(min=a, max=b)  # 截断到 [a, b]

        return tensor  # 返回初始化后的张量

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):#这是一个公开的接口函数，通常由外部调用来初始化权重。
    """
    将输入张量初始化为截断正态分布。

    这个函数是`_no_grad_trunc_normal_`的公开接口，用于初始化张量，使其值遵循指定均值和标准差的截断正态分布。

    参数:
        tensor (torch.Tensor): 需要初始化的张量。
        mean (float, 可选): 正态分布的均值，默认为 0。
        std (float, 可选): 正态分布的标准差，默认为 1。
        a (float, 可选): 分布的下限，默认为 -2。
        b (float, 可选): 分布的上限，默认为 2。

    返回:
        torch.Tensor: 初始化后的张量，其值遵循指定的截断正态分布。
    
    功能：
        这个函数通常用于神经网络的权重初始化过程中，特别是当我们需要确保权重在特定范围内且遵循正态分布时。通过截断分布，我们可以避免极端值的影响，从而帮助提高模型训练的稳定性和效率。
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    """
    对张量进行变量缩放初始化。

    这个函数根据张量的维度(fan_in, fan_out)和给定的参数，对张量进行初始化，
    以确保初始化的张量具有一定的方差，这有助于控制网络层输出的方差在训练初期保持稳定。

    参数:
        tensor (torch.Tensor): 需要初始化的张量。
        scale (float): 缩放因子,用于调整方差,默认为1.0。
        mode (str): 指定使用张量的哪部分维度来计算方差。
                    'fan_in'：使用输入维度，
                    'fan_out'：使用输出维度，
                    'fan_avg'：使用输入和输出维度的平均值。
                    默认为'fan_in'。
        distribution (str): 初始化的分布类型，可以是
                    'truncated_normal'：截断正态分布，
                    'normal'：正态分布，
                    'uniform'：均匀分布。
                    默认为'normal'。

    返回:
        None，直接在输入的张量上进行修改。
    
    功能：
        这种初始化方法通常用于深度神经网络中，可以帮助保持激活函数输入的方差在训练过程中保持稳定，从而有助于避免梯度消失或爆炸问题。通过调整scale、mode和distribution参数，可以进一步优化模型的初始化过程。
    """

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)  # 计算张量的输入和输出维度大小 

    # 根据mode确定分母
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom  # 计算方差

    # 根据指定的分布进行初始化
    if distribution == "truncated_normal":
        # 使用截断正态分布进行初始化，std是标准差
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        # 使用正态分布进行初始化
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        # 使用均匀分布进行初始化
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        # 如果提供了无效的分布类型，抛出异常
        raise ValueError(f"invalid distribution {distribution}")

def lecun_normal_(tensor):
    """
    使用 LeCun 正态初始化方法对张量进行初始化。

    LeCun 初始化是一种变量缩放方法，特别适用于带有S型激活函数（如sigmoid或tanh）的深度学习模型。
    它根据层的输入数量（fan_in）来调整权重的缩放，从而使网络的训练更加稳定。

    参数:
        tensor (torch.Tensor): 需要初始化的张量。

    返回:
        None，直接在输入的张量上进行修改。
    """
    # 调用 variance_scaling_ 函数，使用'fan_in'模式和'normal'分布进行初始化
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')

class PreNorm(nn.Module):
    """
    预归一化模块，通常用于Transformer架构中。

    在执行具体的功能（如自注意力或前馈网络）之前先进行层归一化，
    这有助于稳定训练过程并提高模型性能。

    属性:
        dim: 输入特征的维度。
        fn: 要在归一化后应用的模块或函数。
    """

    def __init__(self, dim, fn):
        """
        初始化预归一化模块。

        参数:
            dim (int): 输入特征的维度，也是层归一化的维度。
            fn (callable): 在归一化之后应用的模块或函数。
        """
        super().__init__()  # 初始化基类 nn.Module
        self.fn = fn  # 存储要应用的函数或模块
        self.norm = nn.LayerNorm(dim)  # 创建层归一化模块

    def forward(self, x, *args, **kwargs):
        """
        对输入数据进行前向传播。

        参数:
            x (Tensor): 输入到模块的数据。
            *args, **kwargs: 传递给self.fn的额外参数。

        返回:
            Tensor: self.fn的输出，其输入是归一化后的x。
        """
        x = self.norm(x)  # 首先对输入x进行层归一化
        return self.fn(x, *args, **kwargs)  # 将归一化的数据传递给self.fn，并执行

class GELU(nn.Module):
    """
    GELU激活函数的封装。

    GELU (Gaussian Error Linear Unit) 是一种非线性激活函数，
    它被广泛用于自然语言处理和深度学习中的其他领域。
    这个函数结合了ReLU和正态分布的性质。
    """

    def forward(self, x):
        """
        在输入数据上应用GELU激活函数。

        参数:
            x (Tensor): 输入到激活函数的数据。

        返回:
            Tensor: 经过GELU激活函数处理后的数据。
        """
        return F.gelu(x)  # 使用PyTorch的函数实现GELU激活

def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    """
    创建并返回一个二维卷积层。

    参数:
    in_channels (int): 输入特征图的通道数。
    out_channels (int): 输出特征图的通道数。
    kernel_size (int): 卷积核的大小，卷积核是正方形的。
    bias (bool, 可选): 是否在卷积层中添加偏置项。默认为 False。
    padding (int, 可选): 在输入特征图周围添加的零填充的层数。默认为 1。
    stride (int, 可选): 卷积的步长。默认为 1。

    返回:
    nn.Conv2d: 配置好的二维卷积层。
    """
    # 创建并返回一个二维卷积层，具体配置参数包括输入通道数、输出通道数、卷积核大小等。
    # padding 参数设置为卷积核大小的一半，确保输出特征图的大小与步长和填充有关，但如果步长为 1，则与输入相同。
    return nn.Conv2d(
        in_channels,  # 输入特征的通道数
        out_channels,  # 输出特征的通道数
        kernel_size,  # 卷积核的大小
        padding=(kernel_size // 2),  # 自动计算填充大小以保持特征图的空间尺寸
        bias=bias,  # 是否添加偏置项
        stride=stride  # 卷积的步长
    )

def shift_back(inputs, step=2):
    """
    对输入张量进行列向移动裁剪，以使输出张量的列数减少。

    参数:
    inputs (Tensor): 输入的四维张量，形状为 [batch_size, channels, rows, cols]。
    step (int, 可选): 每个通道移动的基础步长，默认为2。

    返回:
    Tensor: 列数被裁剪的输出张量，形状为 [batch_size, channels, rows, 256]。
    """
    [bs, nC, row, col] = inputs.shape  # 提取输入张量的形状维度

    # 计算下采样比率，此处特定用途为256固定列输出
    down_sample = 256 // row

    # 调整步长以反映下采样比率的平方影响
    step = float(step) / float(down_sample * down_sample)

    # 设置输出列数，此处为输入行数，这里假设行数等于期望的列数
    out_col = row

    # 遍历每个通道，对每个通道的列进行移动
    for i in range(nC):
        # 对每个通道的图像数据进行列向的裁剪，移动确定的步长
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]

    # 返回裁剪后的张量，仅包含需要的列数
    return inputs[:, :, :, :out_col]

class Illumination_Estimator(nn.Module):
    """
    一个用于估计图像中的照明条件的神经网络模型。
    
    参数:
    - n_fea_middle: 中间特征层的特征数量。
    - n_fea_in: 输入特征的数量，默认为4。
    - n_fea_out: 输出特征的数量，默认为3。
    """

    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        """
        初始化Illumination_Estimator网络结构。
        """
        super(Illumination_Estimator, self).__init__()

        # 第一个卷积层，用于将输入特征映射到中间特征空间。
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        # 深度可分离卷积层，用于在中间特征空间内部进行空间特征提取。
        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        # 第二个卷积层，用于将中间特征映射到输出特征。
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

        # print("Illumination_Estimator的三个卷积模块已经建立完成！")

        # time.sleep(2)

    def forward(self, img):
        """
        前向传播函数定义。
        
        参数:
        - img: 输入图像，形状为 (b, c=3, h, w)，其中 b 是批量大小, c 是颜色通道数, h 和 w 是图像的高度和宽度。
        
        返回:
        - illu_fea: 照明特征图。
        - illu_map: 输出的照明映射，形状为 (b, c=3, h, w)。
        """
        
        # 计算输入图像每个像素点在所有颜色通道上的平均值。
        mean_c = img.mean(dim=1).unsqueeze(1)  # 形状为 (b, 1, h, w) 对应公式中的Lp，也就是照明先验prior
        # print(f"照明先验的图片大小：{mean_c.shape}")

        # 将原始图像和其平均通道合并作为网络输入。
        input = torch.cat([img, mean_c], dim=1)#对应
        # print("原始图像和其平均通道合并图片大小:",input.shape)

        # 通过第一个卷积层处理。
        x_1 = self.conv1(input)

        # 应用深度可分离卷积提取特征。
        illu_fea = self.depth_conv(x_1)
        # print("照明特征图大小:",illu_fea.shape)

        # 通过第二个卷积层得到最终的照明映射。
        illu_map = self.conv2(illu_fea)
        # print("照明图片大小:",illu_map.shape)
    
        return illu_fea, illu_map

class FeedForward2(nn.Module):
    """
    实现一个基于卷积的前馈网络模块，通常用于视觉Transformer结构中。
    这个模块使用1x1卷积扩展特征维度，然后通过3x3卷积在这个扩展的维度上进行处理，最后使用1x1卷积将特征维度降回原来的大小。

    参数:
        dim (int): 输入和输出特征的维度。
        mult (int): 特征维度扩展的倍数，默认为4。
    """
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),  # 使用1x1卷积提升特征维度
            GELU(),  # 使用GELU激活函数增加非线性
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),  # 分组卷积处理，维持特征维度不变，增加特征的局部相关性
            GELU(),  # 再次使用GELU激活函数增加非线性
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),  # 使用1x1卷积降低特征维度回到原始大小
        )

    def forward(self, x):
        """
        前向传播函数。
        
        参数:
        x (tensor): 输入特征，形状为 [b, h, w, c]，其中b是批次大小，h和w是空间维度，c是通道数。

        返回:
        out (tensor): 输出特征，形状与输入相同。
        """
        # 由于PyTorch的卷积期望的输入形状为[b, c, h, w]，需要将通道数从最后一个维度移到第二个维度
        out = self.net(x.permute(0, 3, 1, 2).contiguous())  # 调整输入张量的维度
        return out.permute(0, 2, 3, 1)  # 将输出张量的维度调整回[b, h, w, c]格式

class IGAB(nn.Module):
    """
    实现一个交错组注意力块（Interleaved Group Attention Block，IGAB），该块包含多个注意力和前馈网络层。
    每个块循环地执行自注意力和前馈网络操作。

    参数:
        dim (int): 特征维度，对应于每个输入/输出通道的数量。
        dim_head (int): 每个注意力头的维度。
        heads (int): 注意力机制的头数。
        num_blocks (int): 该层中重复模块的数量。
    """
    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2,d_state = 16):

        super().__init__()
        self.blocks = nn.ModuleList([])
        self.device = None
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IFA(dim_2=dim,dim = dim,num_heads = heads,ffn_expansion_factor=2.66 ,bias=True, LayerNorm_type='WithBias'),
                SS2D(d_model=dim,dropout=0,d_state =d_state),   # 加入Mamba的SS2D模块，输入参数都为源代码指定参数
                PreNorm(dim, FeedForward2(dim=dim))  # 预归一化层，后跟前馈网络,这一部分相当于LN+FFN
            ]))

    def forward(self, x, illu_fea):
        """
        前向传播函数处理输入特征和光照特征。

        参数:
            x (Tensor): 输入特征张量，形状为 [b, c, h, w]，其中 b 是批次大小，c 是通道数，h 和 w 是高度和宽度。
            illu_fea (Tensor): 光照特征张量，形状为 [b, c, h, w]。

        返回:
            Tensor: 输出特征张量，形状为 [b, c, h, w]。
        """
        # x = x.permute(0, 2, 3, 1)  # 调整张量维度以匹配预期的输入格式[b, h, w, c]

        for (trans,ss2d,ff) in self.blocks:
            y=trans(x,illu_fea).permute(0, 2, 3, 1)
            #应用SS2D模块并进行残差连接
            x = x.permute(0, 2, 3, 1)
            x=ss2d(y) + x  #当我创建了一个类之后,如果继承nn并且自己定义了一个forward,那么nn会把hook挂到对象中,直接用对象(forward的参数)就能调用forward函数
            # print("经过ss2d的特征大小",x.shape)
            # 应用前馈网络并进行残差连接
            x = ff(x) + x# bhwc
            x = x.permute(0, 3, 1, 2) #bchw
            # print("模块输出的特征大小",x.shape)
        #print("\n")
        return x


class Denoiser(nn.Module):
    """
    Denoiser 类是一个用于图像去噪的深度神经网络模型。该模型利用编码器-解码器架构进行特征提取和重建，同时结合了
    自注意力机制来增强模型对图像中不同区域的关注能力。

    参数:
    in_dim (int): 输入图像的通道数，默认为3（彩色图像）。
    out_dim (int): 输出图像的通道数，通常与输入通道数相同。
    dim (int): 第一层卷积后的特征维度。
    level (int): 编码器和解码器中的层数。将level设置成2的原因是在论文结构中，维数的变化由C变到2C，再变到4C，总共进行两级的下采样，所以level为2
    num_blocks (list): 每一层中IGAB模块的数量。
    """
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4],d_state= 16):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level

        # 输入投影层，使用卷积层将输入图像的通道数转换为更高的维度以进行后续处理。
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)#第一个卷积，得到的特征向量是F0

        # 编码器，逐层增加特征维度的同时降低空间分辨率，用于捕获更高层次的抽象特征。
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim

       
        for i in range(level):#每一级由于对应的特征维数不同，所以IGAB中的块数也不同，也要采用不同的头数
            self.encoder_layers.append(nn.ModuleList([
                IGAB(dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim, d_state=d_state),#确定每一次调用IGAB时所需要的块数
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),#由于特征维数加深，所以经过卷积要减少H和W的维数，相当于进行下采样,对应前馈中的FeaDownSample
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)#由于x的特征维数加深，导致分辨率下降，而x后面要进MSA，所以为了保持一直，fea也要进行对应的采样。对应前馈中的IlluFeaDownsample
            ]))
            dim_level *= 2#让其特征维数dim不断加深
            d_state *= 2
        
        # print(f"目前已经建立好的编码器：{self.encoder_layers}")

        # time.sleep(3)

        # neck，处理最深层的特征，使用IGAB模块进行自注意力计算。

        
        self.bottleneck = IGAB(dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1], d_state=d_state)#这一个IGAB对应4C特征维度的

        # print(f"4C特征维度IGAB的建立：{self.bottleneck}，其对应的block为：{num_blocks[-1]}")

        # print(f"目前最大的特征维数：{dim_level}")

        # time.sleep(3)

        # 解码器，逐层减少特征维度的同时增加空间分辨率，用于重建图像。
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2, kernel_size=2, padding=0, output_padding=0),#反卷积，进行上采样，分辨率变大，特征维度变小，对应前馈中的FeaUpSample
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),#与之前编码时候的IGAB输出concatenate后所经过的卷积层
                IGAB(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim, heads=(dim_level // 2) // dim, d_state=d_state)
            ]))
            dim_level //= 2
            d_state //= 2

        # 输出投影层，将解码器的输出特征转换回原始图像的通道数。
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # 使用LeakyReLU激活函数提供非线性处理。
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        # 对所有的可训练参数进行权重初始化。
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 初始化权重，为线性层使用截断正态分布，为层归一化设置常数。
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        """
        前向传播函数定义了如何处理输入x及光照特征illu_fea。

        参数:
        x (torch.Tensor): 输入特征张量，形状为 [b, 3, h, w]。
        illu_fea (torch.Tensor): 光照特征张量，大小为(H,W,C)。

        返回:
        torch.Tensor: 去噪后的输出特征张量，形状与x相同。
        """
        # 特征嵌入
        fea = self.embedding(x)#对应图中的F0特征向量

        # print(f"F0的大小：{fea.shape},此时对应illu_fea的大小为：{illu_fea.shape}")#这个大小应该与illu_fea的相同

        count = 0

        # time.sleep(3)

        # print("------------------------------------------开始编码---------------------------------------------")

        # 编码过程
        fea_encoder = []#为了存入每次编码时产生的特征向量，为了后续解码时与解码的特征向量进行concatenate
        illu_fea_list = []#为了存入每次编码时产生的特征向量，为后续解码时候直接调用对应维度的fea使用
        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            count+=1
            # print(f"第{count}次进行编码！")
            fea = IGAB(fea, illu_fea)#调用IGAB的前馈网络,此时fea的大小不变
            illu_fea_list.append(illu_fea)#将当前的illu_fea添加到列表中
            fea_encoder.append(fea)#将经过IGAB的特征向量添加到列表中，此时大小还是不变
            fea = FeaDownSample(fea)#进行下采样，此时特征向量分辨率变为原来的一般，特征维度变为原来的两倍
            # print(f"F{count}的大小为：{fea.shape}")
            illu_fea = IlluFeaDownsample(illu_fea)#调整illu_fea的大小，为了使其下次输入的时候能够和特征向量大小对齐

        # 处理最深层的特征
        fea = self.bottleneck(fea, illu_fea)

        # print(f"F'2的大小：{fea.shape}")

        #print("------------------------------------------编码完成---------------------------------------------")

        #time.sleep(3)
        
        #print("------------------------------------------开始解码---------------------------------------------")
        
        count = 0

        # 解码过程
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            count+=1
            #print(f"第{count}次进行解码！")
            fea = FeaUpSample(fea)#进行反卷积，使特征向量分辨率变大，特征维度下降为原来的一半，此时illu_fea还没有变
            #print("")
            fea = Fution(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))#在这里直接将编码时对应的解码向量concatenate，然后送入卷积层
            illu_fea = illu_fea_list[self.level - 1 - i]#取出这一次所需要的illu_fea
            fea = LeWinBlcok(fea, illu_fea)#送入IGAB模块
            #print(f"F'{2-count}的大小：{fea.shape}")
        
        #print("------------------------------------------解码完成---------------------------------------------")

        # time.sleep(3)

        # 映射到输出维度，并添加原始输入以实现残差连接
        out = self.mapping(fea) + x

        #print(f"最终输出图片Ien的大小为：{out.shape}")    

        return out


class RetinexMamba_Single_Stage(nn.Module):
    """
    定义 Retinex 网络的单一阶段模型，该模型将照明估计和去噪/图像增强任务结合在一个统一的框架中。
    """
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1],d_state = 16):
        """
        初始化 RetinexMamba 单阶段模型。
        
        参数:
            in_channels (int): 输入通道数，通常是3（对于RGB图像）。
            out_channels (int): 输出通道数，通常是3（对于RGB图像）。
            n_feat (int): 特征数，表示网络中间层的特征深度。
            level (int): 网络的深度或级别，影响网络的复杂度和性能。
            num_blocks (list): 每个级别的块数量，用于定义每个级别的重复模块数。
        """
        super(RetinexMamba_Single_Stage, self).__init__()
        self.estimator = Illumination_Estimator(n_feat)  # 照明估计器，估计图像的照明成分。
        self.denoiser = Denoiser(in_dim=in_channels, out_dim=out_channels, dim=n_feat, level=level, num_blocks=num_blocks, d_state=d_state)  # 编解码获取最终结果
    
    def forward(self, img):
        """
        定义模型的前向传播路径。

        参数:
            img (Tensor): 输入的图像张量，尺寸为 [batch_size, channels, height, width]

        返回:
            output_img (Tensor): 增强后的输出图像张量。
        """
        # 从输入图像中估计照明特征和照明图
        illu_fea, illu_map = self.estimator(img)  # illu_fea: 照明特征; illu_map: 照明图
        
        # 计算输入图像与其照明图的组合，以进行后续增强
        input_img = img * illu_map + img  # 增加照明影响后的图像，为图中的Ilu

        # print(f"Ilu的大小为：{input_img.shape}")
        
        # 使用去噪器进行图像增强，同时利用照明特征
        output_img = self.denoiser(input_img, illu_fea)

        return output_img

class GetGradientNopadding(nn.Module):
    def __init__(self):
        super(GetGradientNopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)

        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, inp_feat):
        x_list = []
        for i in range(inp_feat.shape[1]):
            x_i = inp_feat[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        res = torch.cat(x_list, dim=1)

        return res
    
class RetinexMamba(nn.Module):
    """
    多阶段 Retinex 图像处理网络，每个阶段都通过 RetinexMamba_Single_Stage 来实现，
    进行图像的照明估计和增强。
    """
    def __init__(self, in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2], d_state=16):
        """
        初始化 Retinex 图像处理网络。

        参数:
            in_channels (int): 输入图像的通道数，通常为3（RGB图像）。
            out_channels (int): 输出图像的通道数，通常为3（RGB图像）。
            n_feat (int): 特征层数，表示中间特征的深度。
            stage (int): 网络包含的阶段数，每个阶段都使用一个 RetinexMamba_Single_Stage 模块。
            num_blocks (list): 每个阶段的块数，指定每个单阶段中的块数量。
        """
        super(RetinexMamba, self).__init__()
        self.stage = stage  # 网络的阶段数

        # 创建多个 RetinexMamba_Single_Stage 实例，每个实例都作为网络的一个阶段
        modules_body = [RetinexMamba_Single_Stage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2, num_blocks=num_blocks, d_state=d_state)
                        for _ in range(stage)]
        
        # 将所有阶段模块封装成一个顺序模型
        self.body = nn.Sequential(*modules_body)
    
        self.get_gradient = GetGradientNopadding()

    def forward(self, x):
        """
        定义网络的前向传播路径。

        参数:
            x (Tensor): 输入的图像张量，尺寸为 [batch_size, channels, height, width]

        返回:
            out (Tensor): 经过多个阶段处理后的输出图像张量。
        """
        # 通过网络体进行图像处理
        out = self.body(x)
        res_grad = self.get_gradient(out)

        return out, res_grad


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    model = RetinexMamba(stage=1,n_feat=40,num_blocks=[1,2,2]).cuda()
    print(model)
    inputs = torch.randn((1, 3, 64, 64)).cuda()
    output, res_grad = model(inputs)
    print(output.shape, res_grad.shape)
    flops = FlopCountAnalysis(model,inputs)
    n_param = sum([p.nelement() for p in model.parameters()])  # 所有参数数量
    print(f'GMac:{flops.total()/(1024*1024*1024)}')
    print(f'Params:{n_param}')