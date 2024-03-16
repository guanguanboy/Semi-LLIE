from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import torch.nn
import os
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from mobile_sam import sam_model_registry as mobile_sam_model_registry
import torch.nn.functional as F
import cv2
import requests
"""
below are the implementation of RAM
"""
from ram.models.ram import ram as ram_fix
from ram.models.ram_lora import ram as ram
from ram import get_transform,get_resize_transform


def rescale(X):
    X = 2 * X - 1.0
    return X


def inverse_rescale(X):
    X = (X + 1.0) / 2.0
    return torch.clamp(X, 0.0, 1.0)


def imageresize2tensor(path, image_size):
    img = Image.open(path)
    convert = transforms.Compose(
        [transforms.Resize(image_size, interpolation=Image.BICUBIC), transforms.ToTensor()]
    )
    return convert(img)


def image2tensor(path):
    img = Image.open(path)
    convert_tensor = transforms.ToTensor()
    return convert_tensor(img)


def ram_generate_embedding_torch(sam_model, image,device):

    #resize_transform = get_resize_transform(image_size=384)
    #image = resize_transform(image)
    #print('image shape = ', image.shape)
    image_rezied = image.clone()
    new_height = 384
    new_width = 384
    image_rezied = F.interpolate(image_rezied, size=(new_height, new_width), mode='bilinear', align_corners=False)
    #image_rezied = image_rezied.squeeze(0)
    #assert image.shape == (image.shape[0], 3, 384,384), 'input image should be resized to 3*384*384'

    with torch.no_grad():
        embedding, logits_gt, _ = sam_model.condition_forward(image_rezied.to(device), only_feature=False)


    #image = F.interpolate(image.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
    #image = image.squeeze(0)

    return embedding, logits_gt
    
def ram_generate_embedding(sam_model, image,device):
    if sam_model is not None:
        #ram_transform = get_transform()
        #resampled_image_tensor = ram_transform(image)
        
        resampled_image_tensor = torch.as_tensor(image.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        
        
        #print(resampled_image_tensor.shape)
        assert resampled_image_tensor.shape == (1, 3, 384,384), 'input image should be resized to 384*384'

        with torch.no_grad():
            embedding, logits_gt, _ = sam_model.condition_forward(resampled_image_tensor, only_feature=False)

        return embedding, logits_gt


class RAMContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(RAMContrastLoss, self).__init__()
        """ Initializes a perceptual loss torch.nn.Module

        Parameters (default listed first)
        ---------------------------------
        samscore : float
            The SAM score between the source image and the generated image
        model_type : str
            The type of model to use for the SAM score. Currently only supports 'vit_l,vit_b,vit_h'
        version : str
            The version of the SAM model to use. Currently only supports '1.0'
        source_image_path : str
            The path to the source image
        generated_image_path : str
            The path to the generated image
        """
        #model_type = "vit_t"
        #version='1.0'


        #self.version = version
        #self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        ram_model_path = 'pretrained/ram_swin_large_14m.pth'
        self.ram = ram_fix(pretrained=ram_model_path, image_size=384, vit='swin_l')
        
        self.ram.to(device=self.device)
        self.ram.eval()

        self.ab = ablation
        self.l1 = nn.L1Loss().to(self.device)

    def forward(self, anchor, positive, negative):
        a_vgg, a_logits = ram_generate_embedding_torch(self.ram, anchor, self.device)
        p_vgg, p_logits= ram_generate_embedding_torch(self.ram, positive, self.device)
        n_vgg, n_logtis = ram_generate_embedding_torch(self.ram, negative, self.device)

        loss = 0

        d_ap, d_an = 0, 0
        d_ap = self.l1(a_vgg, p_vgg.detach())
        if not self.ab:
            d_an = self.l1(a_vgg, n_vgg.detach())
            contrastive = d_ap / (d_an + 1e-7)
        else:
            contrastive = d_ap

        d_ap_logits, d_an_logits = 0, 0
        d_ap_logits = self.l1(a_logits, p_logits.detach())
        if not self.ab:
            d_an_logits = self.l1(a_logits, n_logtis.detach())
            contrastive_logits = d_ap_logits / (d_an_logits + 1e-7)
        else:
            contrastive_logits = d_ap_logits

        loss = contrastive + contrastive_logits
        return loss
    
    def evaluation_from_path(self, anchor_image_path=None,  positive_image_path=None, negative_image_path=None):
        anchor = cv2.imread(anchor_image_path)
        positive = cv2.imread(positive_image_path)
        negative = cv2.imread(negative_image_path)

        anchor = cv2.resize(anchor, (384, 384))
        positive = cv2.resize(positive, (384, 384))
        negative = cv2.resize(negative, (384, 384))

        a_vgg, a_logits = ram_generate_embedding(self.ram, anchor, self.device)
        p_vgg, p_logits = ram_generate_embedding(self.ram, positive, self.device)
        n_vgg, n_logtis = ram_generate_embedding(self.ram, negative, self.device)

        loss = 0

        d_ap, d_an = 0, 0
        d_ap = self.l1(a_vgg, p_vgg.detach())
        if not self.ab:
            d_an = self.l1(a_vgg, n_vgg.detach())
            contrastive = d_ap / (d_an + 1e-7)
        else:
            contrastive = d_ap

        d_ap_logits, d_an_logits = 0, 0
        d_ap_logits = self.l1(a_logits, p_logits.detach())
        if not self.ab:
            d_an_logits = self.l1(a_logits, n_logtis.detach())
            contrastive_logits = d_ap_logits / (d_an_logits + 1e-7)
        else:
            contrastive_logits = d_ap_logits

        loss = contrastive + contrastive_logits
        
        return loss

if __name__ == "__main__":
    sam_constrast_loss = RAMContrastLoss()

    anchor_image_path = 'data/val/input/0000001_02999_d_0000005.jpg'
    positive_image_path = 'data/val/GT/0000001_02999_d_0000005.jpg'
    negative_image_path = 'data/val/input/0000001_03499_d_0000006.jpg'

    loss = sam_constrast_loss.evaluation_from_path(anchor_image_path, positive_image_path, negative_image_path)
    print(loss)