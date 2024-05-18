# --- Imports --- #
from utils import *


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





"""

if __name__ == "__main__":
    model = AIMnet()
    x = torch.ones([1, 3, 256, 256])
    x1 = torch.ones([1, 3, 256, 256])
    y,gradient_map = model(x, x1)
    print(y.shape, gradient_map.shape)
    print('model params: %d' % count_parameters(model))  

"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
# import cv2
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# input [bs,28,256,310]  output [bs, 28, 256, 256]
def shift_back(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]



class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  #__init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        
        mean_c = img.mean(dim=1).unsqueeze(1)
        # stx()
        input = torch.cat([img,mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map



class IG_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans # illu_fea: b,c,h,w -> b,h,w,c
        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
        v = v * illu_attn
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True, isuseBN=False):
        super(ConvBlock, self).__init__()
        self.isuseBN = isuseBN
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        if self.isuseBN:
            self.bn = nn.BatchNorm2d(output_size)
        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        if self.isuseBN:
            out = self.bn(out)
        out = self.act(out)
        return out


import torch
import torch.nn.parallel
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class DownsampleOp(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(DownsampleOp, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        # print('Filter size [%i]'%filt_size)
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer

class ResidualDownSample(nn.Module):
    def __init__(self,in_channel,bias=False):
        super(ResidualDownSample,self).__init__()
        self.prelu = nn.PReLU()
        
        self.conv1 = nn.Conv2d(in_channel,in_channel,3,1,1,bias=bias)
        self.downsamp = DownsampleOp(channels=in_channel,filt_size=3,stride=2)
        self.conv2 = nn.Conv2d(in_channel,2*in_channel,1,stride=1,padding=0,bias=bias)
    def forward(self, x):
        out = self.prelu(self.conv1(x))
        out = self.downsamp(out)
        out = self.conv2(out)
        return out

class DownSample(nn.Module):
    def __init__(self, in_channel,scale_factor=2, stride=2,kernel_size=3):
        super(DownSample,self).__init__()
        self.scale_factor=scale_factor
        self.residualdownsample=ResidualDownSample(in_channel)

    def forward(self, x):
        out = self.residualdownsample(x)
        return out

class ResidualUpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualUpSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
                                nn.PReLU(),
                                nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1,bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(in_channels, in_channels//2, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
                                nn.Conv2d(in_channels, in_channels//2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top+bot
        return out

class UpSample(nn.Module):
    def __init__(self, in_channel, scale_factor=2,stride=2,kernel_size=3):
        super(UpSample,self).__init__()
        self.scale_factor=scale_factor
        self.residualupsample=ResidualUpSample(in_channel)

    def forward(self, x):
        out = self.residualupsample(x)
        return out


class EncoderBlock(nn.Module):  
    def __init__(self, input_dim, out_dim,):
        super(EncoderBlock, self).__init__()
        hidden = input_dim // 4  # 2021-3-30 8->4
        self.prelu = nn.PReLU()
        
        self.SGblock = nn.Sequential(
                        ConvBlock(input_dim,input_dim,3,1,1,isuseBN=False),
                        nn.Conv2d(input_dim,hidden,1,1,0),
                        nn.Conv2d(hidden,out_dim,1,1,0,),
                        ConvBlock(out_dim,out_dim,3,1,1,isuseBN=False))
    def forward(self, x):
        out = self.SGblock(x)
        out = out + x
        return out

class FusionLayer(nn.Module):
    def __init__(self, inchannel, outchannel, reduction=16):
        super(FusionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inchannel // reduction, inchannel, bias=False),
            nn.Sigmoid()
        )
        self.fusion   = ConvBlock(inchannel, inchannel, 1,1,0,bias=True)
        self.outlayer = ConvBlock(inchannel, outchannel, 1, 1, 0, bias=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        avg = self.fc(avg).view(b, c, 1, 1)
        max = self.max_pool(x).view(b, c)
        max = self.fc(max).view(b, c, 1, 1)
        fusion = self.fusion(avg+max) 
        fusion = x * fusion.expand_as(x)
        fusion = fusion + x
        fusion = self.outlayer(fusion)
        return fusion
        
class FeatureEnhancer(nn.Module):  
    def __init__(self, dim=32):
        super(FeatureEnhancer, self).__init__()

        self.downsample   = DownSample(in_channel=dim,scale_factor=2)
        self.downsample2x = DownSample(in_channel=2*dim,scale_factor=2)
        self.upsample     = UpSample(in_channel=dim,scale_factor=2)
        self.upsample2x   = UpSample(in_channel=2*dim,scale_factor=2)
        self.upsample4x   = UpSample(in_channel=4*dim,scale_factor=2)
        self.endecoder1x  = EncoderBlock(input_dim=dim,out_dim=dim)
        self.endecoder2x  = EncoderBlock(2*dim,2*dim)
        self.endecoder4x  = EncoderBlock(4*dim,4*dim)        
        self.prelu   = torch.nn.PReLU()
        self.out_fushion = FusionLayer(3*dim, 3*dim)  
        self.out_conv2 = nn.Conv2d(3*dim, dim, 3, 1, 1)

    def forward(self, feats):

        fullres = feats
        halfres = self.downsample(fullres)
        quarres = self.downsample2x(halfres)  

        ende_quarres = self.endecoder4x(quarres)
        ende_quarres_up = self.upsample4x(ende_quarres)
        ende_quarres_up_up = self.upsample2x(ende_quarres_up)
        ende_halfres = self.endecoder2x(self.prelu(halfres)+ende_quarres_up)
        ende_halfres_up = self.upsample2x(ende_halfres)
        ende_fullres_out = self.endecoder1x(self.prelu(fullres)+ende_halfres_up)

        cat_all = torch.cat((ende_fullres_out,ende_halfres_up,ende_quarres_up_up),dim=1)
        #fusion_all = self.out_fushion(cat_all) 
        fusion_out = self.out_conv2(cat_all)  
        fusion_out = fusion_out+fullres
        return fusion_out
    
class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            if i < 1:
                self.encoder_layers.append(nn.ModuleList([
                    FeatureEnhancer(dim=dim_level),
                    IGAB(
                        dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                    nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                    nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
                ]))
            else:
                self.encoder_layers.append(nn.ModuleList([
                    nn.Identity(),
                    IGAB(
                        dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                    nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                    nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
                ]))                

            dim_level *= 2

        # Bottleneck
        self.bottleneck = IGAB(
            dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            if i < 1:
                self.decoder_layers.append(nn.ModuleList([
                    FeatureEnhancer(dim=dim_level),
                    nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                    kernel_size=2, padding=0, output_padding=0),
                    nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                    IGAB(
                        dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                        heads=(dim_level // 2) // dim),
                ]))
            else:
                self.decoder_layers.append(nn.ModuleList([
                    nn.Identity(),
                    nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                    kernel_size=2, padding=0, output_padding=0),
                    nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                    IGAB(
                        dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                        heads=(dim_level // 2) // dim),
                ]))

            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        """
        x:          [b,c,h,w]         x是feature, 不是image
        illu_fea:   [b,c,h,w]
        return out: [b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)

        fullres = fea

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        for (feats_enhancer, IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = feats_enhancer(fea)
            fea = IGAB(fea,illu_fea)  # bchw
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        fea = self.bottleneck(fea,illu_fea)

        # Decoder
        for i, (feats_enhancer, FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = feats_enhancer(fea)
            fea = FeaUpSample(fea)
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level-1-i]
            fea = LeWinBlcok(fea,illu_fea)

        # Mapping
        out = self.mapping(fea) + x

        return out


class RetinexFormer_Single_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1]):
        super(RetinexFormer_Single_Stage, self).__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(in_dim=in_channels,out_dim=out_channels,dim=n_feat,level=level,num_blocks=num_blocks)  #### 将 Denoiser 改为 img2img
    
    def forward(self, img):
        # img:        b,c=3,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img
        output_img = self.denoiser(input_img,illu_fea)

        return output_img


class RetinexFormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1,2,2]):
        super(RetinexFormer, self).__init__()
        self.stage = stage

        modules_body = [RetinexFormer_Single_Stage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2, num_blocks=num_blocks)
                        for _ in range(stage)]
        
        self.body = nn.Sequential(*modules_body)
    
        #self.get_gradient = GetGradientNopadding()

    def forward(self, x, x2=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        out = self.body(x)

        #res_grad = self.get_gradient(out)

        return out

class CMTNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1,2,2]):
        super(CMTNet, self).__init__()
        self.stage = stage

        modules_body = [RetinexFormer_Single_Stage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2, num_blocks=num_blocks)
                        for _ in range(stage)]
        
        self.body = nn.Sequential(*modules_body)
    
        #self.get_gradient = GetGradientNopadding()

    def forward(self, x, x2=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        out = self.body(x)

        #res_grad = self.get_gradient(out)

        return out
    
class CMTNetWithGrad(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1,2,2]):
        super(CMTNetWithGrad, self).__init__()
        self.stage = stage

        modules_body = [RetinexFormer_Single_Stage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2, num_blocks=num_blocks)
                        for _ in range(stage)]
        
        self.body = nn.Sequential(*modules_body)
    
        self.get_gradient = GetGradientNopadding()

    def forward(self, x, x2=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        out = self.body(x)

        res_grad = self.get_gradient(out)

        return out, res_grad
    
if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis
    
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    model = CMTNetWithGrad(stage=1,n_feat=40,num_blocks=[1,2,2]).to(device=device)
    print(model)
    inputs = torch.randn((1, 3, 256, 256)).to(device=device)

    outputs = model(inputs)
    print(outputs[0].shape,outputs[1].shape)
    flops = FlopCountAnalysis(model,inputs)
    n_param = sum([p.nelement() for p in model.parameters()])  # 所有参数数量
    print(f'GMac:{flops.total()/(1024*1024*1024)}')
    print(f'Params:{n_param}')

    feats_enhancer = FeatureEnhancer().to(device=device)
    input_feats = torch.randn((1, 32, 64, 64)).to(device=device)

    enhanced_feats = feats_enhancer(input_feats)
    print(enhanced_feats.shape)