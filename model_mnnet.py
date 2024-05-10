import torch
import torch.nn as nn
from antialias import Downsample as downsamp
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


class NewEncoderBlock(nn.Module):  
    def __init__(self, input_dim, out_dim, kernel_size, stride, padding):
        super(NewEncoderBlock, self).__init__()
        self.firstconv = ConvBlock(input_size=4,output_size=input_dim,kernel_size=3,stride=1,padding=1)
        self.prelu = nn.PReLU()
        codeim = out_dim // 2
        self.conv_Encoder = ConvBlock(input_dim, codeim, kernel_size, stride, padding, isuseBN=False)
        self.conv_Offset  = ConvBlock(codeim, codeim, kernel_size, stride, padding, isuseBN=False)
        self.conv_Decoder = ConvBlock(codeim, out_dim, kernel_size, stride, padding, isuseBN=False)

    def forward(self, x):
        firstconv = self.prelu(self.firstconv(x))
        code   = self.conv_Encoder(firstconv)
        offset = self.conv_Offset(code)
        code_add = code + offset
        out    = self.conv_Decoder(code_add)
        return out



class ResidualDownSample(nn.Module):
    def __init__(self,in_channel,bias=False):
        super(ResidualDownSample,self).__init__()
        self.prelu = nn.PReLU()
        
        self.conv1 = nn.Conv2d(in_channel,in_channel,3,1,1,bias=bias)
        self.downsamp = downsamp(channels=in_channel,filt_size=3,stride=2)
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

class lowlightnet3(nn.Module):
    def __init__(self, input_dim=3, dim=64):
        super(lowlightnet3, self).__init__()
        inNet_dim = input_dim + 1

        self.prelu   = torch.nn.PReLU()
        self.sigmoid = torch.nn.Sigmoid()


        self.out_fushion = FusionLayer(3*dim, 3*dim)  
        self.out_conv2 = nn.Conv2d(3*dim, dim, 3, 1, 1)
        self.out_conv3 = nn.Conv2d(dim, 4, 3, 1, 1)
        self.out_conv4 = nn.Conv2d(4,3,1,1,0)

        self.firstconv = nn.Sequential(ConvBlock(input_size=4,output_size=dim,kernel_size=3,stride=1,padding=1),
                                       EncoderBlock(input_dim=dim,out_dim=dim))
        self.downsample   = DownSample(in_channel=dim,scale_factor=2)
        self.downsample2x = DownSample(in_channel=2*dim,scale_factor=2)
        self.upsample     = UpSample(in_channel=dim,scale_factor=2)
        self.upsample2x   = UpSample(in_channel=2*dim,scale_factor=2)
        self.upsample4x   = UpSample(in_channel=4*dim,scale_factor=2)
        self.endecoder1x  = EncoderBlock(input_dim=dim,out_dim=dim)
        self.endecoder2x  = EncoderBlock(2*dim,2*dim)
        self.endecoder4x  = EncoderBlock(4*dim,4*dim)
        self.get_gradient = GetGradientNopadding()

    def forward(self, x_ori, tar=None):
        x = x_ori
        x_bright, _ = torch.max(x_ori, dim=1, keepdim=True)

        x_in = torch.cat((x, x_bright), 1)

       
        f_endecoder = self.firstconv(x_in)

        # channel=3dim
        # here is 1st block out
        fullres = f_endecoder
        halfres = self.downsample(fullres)
        quarres = self.downsample2x(halfres)  

        ende_quarres = self.endecoder4x(quarres)
        ende_quarres_up = self.upsample4x(ende_quarres)
        ende_quarres_up_up = self.upsample2x(ende_quarres_up)
        ende_halfres = self.endecoder2x(self.prelu(halfres)+ende_quarres_up)
        ende_halfres_up = self.upsample2x(ende_halfres)
        ende_fullres_out = self.endecoder1x(self.prelu(fullres)+ende_halfres_up)


        cat_all = torch.cat((ende_fullres_out,ende_halfres_up,ende_quarres_up_up),dim=1)
        fusion_all = self.out_fushion(cat_all) 
        fusion_out = self.out_conv2(fusion_all)  
        fusion_out = fusion_out+fullres

        # here is 2nd block out
        fullres2 = fusion_out
        halfres2 = self.downsample(fullres2)
        quarres2 = self.downsample2x(halfres2)

        ende_quarres2 = self.endecoder4x(quarres2)
        ende_quarres_up2 = self.upsample4x(ende_quarres2)
        ende_quarres_up_up2 = self.upsample2x(ende_quarres_up2)
        ende_halfres2 = self.endecoder2x(self.prelu(halfres2)+ende_quarres_up2)
        ende_halfres_up2 = self.upsample2x(ende_halfres2)
        ende_fullres_out2 = self.endecoder1x(self.prelu(fullres2)+ende_halfres_up2)

        cat_all2 = torch.cat((ende_fullres_out2,ende_halfres_up2,ende_quarres_up_up2),dim=1)
        fusion_all2 = self.out_fushion(cat_all2)
        fusion_out2 = self.out_conv2(fusion_all2)
        fusion_out2 = fusion_out2+fullres2

        # here is 3rd block out
        fullres3 = fusion_out2
        halfres3 = self.downsample(fullres3)
        quarres3 = self.downsample2x(halfres3)

        ende_quarres3 = self.endecoder4x(quarres3)
        ende_quarres_up3 = self.upsample4x(ende_quarres3)
        ende_quarres_up_up3 = self.upsample2x(ende_quarres_up3)
        ende_halfres3 = self.endecoder2x(self.prelu(halfres3) + ende_quarres_up3)
        ende_halfres_up3 = self.upsample2x(ende_halfres3)
        ende_fullres_out3 = self.endecoder1x(self.prelu(fullres3) + ende_halfres_up3)

        cat_all3 = torch.cat((ende_fullres_out3, ende_halfres_up3, ende_quarres_up_up3), dim=1)
        fusion_all3 = self.out_fushion(cat_all3)
        fusion_out3 = self.out_conv2(fusion_all3)
        fusion_out3 = fusion_out3 + fullres3

        # real out
        out = self.prelu(fusion_out3) 
        out = self.out_conv3(out)  
        out = out  +  x_bright
        out = self.out_conv4(out)

        res_grad = self.get_gradient(out)

        return out, res_grad


############################################################################################
# Base models
############################################################################################

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

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.deconv(x)
        return self.act(out)


class UpBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(UpBlock, self).__init__()

        self.conv1 = DeconvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = ConvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1 = ConvBlock(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        hr = self.conv1(x)
        lr = self.conv2(hr)
        residue = self.local_weight1(x) - lr
        h_residue = self.conv3(residue)
        hr_weight = self.local_weight2(hr)
        return hr_weight + h_residue


class DownBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(DownBlock, self).__init__()

        self.conv1 = ConvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = ConvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1 = ConvBlock(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        lr = self.conv1(x)
        hr = self.conv2(lr)
        residue = self.local_weight1(x) - hr
        l_residue = self.conv3(residue)
        lr_weight = self.local_weight2(lr)
        return lr_weight + l_residue


class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(num_filter)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(num_filter)

        self.act1 = torch.nn.PReLU()
        self.act2 = torch.nn.PReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        out = self.act2(out)

        return out
if __name__=='__main__':
    data=torch.randn(8,3,128,128).cuda()
    model = lowlightnet3().cuda()
    out, res_grad = model(data)
    print(out.shape, res_grad.shape)