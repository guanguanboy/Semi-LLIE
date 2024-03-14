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


def calculate_l2_given_paths(path1, path2):
    file_name_old = os.listdir(path1)
    total = 0
    file_name = []
    for filename in file_name_old:
        if "fake" in str(filename):
            file_name.append(filename)

    for name in file_name:
        s = imageresize2tensor(os.path.join(path1, name.replace("fake", "real")), 256)
        name_i = name.split('.')[0]
        name = name_i + '.png'
        t = imageresize2tensor(os.path.join(path2, name), 256)
        l2_i = torch.norm(s - t, p=2)
        total += l2_i
    return total / len(file_name)


def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)


def cosine_similarity(X, Y):
    '''
    compute cosine similarity for each pair of image
    Input shape: (batch,channel,H,W)
    Output shape: (batch,1)
    '''
    b, c, h, w = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
    X = X.reshape(b, c, h * w)
    Y = Y.reshape(b, c, h * w)
    corr = norm(X) * norm(Y)  # (B,C,H*W)
    similarity = corr.sum(dim=1).mean(dim=1)
    return similarity


def sam_encode(sam_model, image, image_generated,device):
    if sam_model is not None:
        sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        resampled_image = sam_transform.apply_image(image)
        # resized_shapes.append(resize_img.shape[:2])
        resampled_image_tensor = torch.as_tensor(resampled_image.transpose(2, 0, 1)).to(device)
        # model input: (1, 3, 1024, 1024)
        resampled_image = sam_model.preprocess(resampled_image_tensor[None, :, :, :])  # (1, 3, 1024, 1024)
        assert resampled_image.shape == (1, 3, sam_model.image_encoder.img_size,
                                     sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'

        resampled_image_generated = sam_transform.apply_image(image_generated)
        resampled_image_generated_tensor = torch.as_tensor(resampled_image_generated.transpose(2, 0, 1)).to(device)
        resampled_image_generated = sam_model.preprocess(resampled_image_generated_tensor[None, :, :, :])
        assert resampled_image_generated.shape == (1, 3, sam_model.image_encoder.img_size,
                                            sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'

        # input_imgs.append(input_image.cpu().numpy()[0])
        with torch.no_grad():
            embedding = sam_model.image_encoder(resampled_image)
            embedding_generated = sam_model.image_encoder(resampled_image_generated)
            samscore = cosine_similarity(embedding, embedding_generated)
        return samscore


def sam_encode_from_torch(sam_model, image, image_generated,device):
    if sam_model is not None:
        sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        resampled_image = sam_transform.apply_image_torch(image).to(device)
        resampled_image = sam_model.preprocess(resampled_image)

        resampled_image_generated = sam_transform.apply_image_torch(image_generated).to(device)
        resampled_image_generated = sam_model.preprocess(resampled_image_generated)

        assert resampled_image.shape == (resampled_image.shape[0], 3, sam_model.image_encoder.img_size,sam_model.image_encoder.img_size), 'input image should be resized to 3*1024*1024'
        assert resampled_image_generated.shape == (resampled_image.shape[0], 3, sam_model.image_encoder.img_size,
                                            sam_model.image_encoder.img_size), 'input image should be resized to 3*1024*1024'

        with torch.no_grad():
            embedding = sam_model.image_encoder(resampled_image)
            embedding_generated = sam_model.image_encoder(resampled_image_generated)
            samscore = cosine_similarity(embedding, embedding_generated)
        return samscore

def sam_generate_embedding_torch(sam_model, image,device):
    if sam_model is not None:
        sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        resampled_image = sam_transform.apply_image_torch(image).to(device)
        resampled_image = sam_model.preprocess(resampled_image)


        assert resampled_image.shape == (resampled_image.shape[0], 3, sam_model.image_encoder.img_size,sam_model.image_encoder.img_size), 'input image should be resized to 3*1024*1024'

        with torch.no_grad():
            embedding = sam_model.image_encoder(resampled_image)

        return embedding
    
def sam_generate_embedding(sam_model, image,device):
    if sam_model is not None:
        sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        resampled_image = sam_transform.apply_image(image)
        resampled_image_tensor = torch.as_tensor(resampled_image.transpose(2, 0, 1)).to(device)
        resampled_image = sam_model.preprocess(resampled_image_tensor[None, :, :, :])  # (1, 3, 1024, 1024)

        assert resampled_image.shape == (1, 3, sam_model.image_encoder.img_size,
                                     sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'

        with torch.no_grad():
            embedding = sam_model.image_encoder(resampled_image)

        return embedding
    
def download_model(url,model_name,destination):

    chunk_size = 8192  # Size of each chunk in bytes

    response = requests.get(url+model_name, stream=True)

    if response.status_code == 200:
        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)
        print("Weights downloaded successfully.")
    else:
        print("Failed to download file. Status code:", response.status_code)



class SAMContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(SAMContrastLoss, self).__init__()
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
        model_type = "vit_l"
        model_weight_path = None 
        version='1.0'
        
        download_url = "https://dl.fbaipublicfiles.com/segment_anything/"
        online_vit_b_model_name = "sam_vit_b_01ec64.pth"
        online_vit_l_model_name = "sam_vit_l_0b3195.pth"
        online_vit_h_model_name = "sam_vit_h_4b8939.pth"
        online_vit_t_model_name = "mobile_sam.pt"
      
        
        if model_weight_path is None:
            if model_type == "vit_l":
                online_model_weight_name = online_vit_l_model_name
            elif model_type == "vit_b":
                online_model_weight_name = online_vit_b_model_name
            elif model_type == "vit_h":
                online_model_weight_name = online_vit_h_model_name
            elif model_type == "vit_t":
                online_model_weight_name = online_vit_t_model_name
                download_url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/"
                
            else:
                raise ValueError("model_type must be one of 'vit_l','vit_b','vit_h'")
            
        # to download the model weights from online link
        if model_weight_path is None:
            save_path = "weights"
            destination_path = os.path.join(save_path,online_model_weight_name)
            model_weight_path = destination_path

            if not os.path.exists(destination_path):
                os.makedirs(save_path,exist_ok=True)
                download_model(url = download_url,model_name = online_model_weight_name, destination= destination_path)


        self.version = version
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_type == "vit_t":
            self.sam = mobile_sam_model_registry[model_type](checkpoint=model_weight_path)
        else:
            self.sam = sam_model_registry[self.model_type](checkpoint=model_weight_path)
        self.sam.to(device=self.device)

        self.ab = ablation
        self.l1 = nn.L1Loss().to(self.device)

    def forward(self, anchor, positive, negative):
        a_vgg = sam_generate_embedding_torch(self.sam, anchor, self.device)
        p_vgg= sam_generate_embedding_torch(self.sam, positive, self.device)
        n_vgg = sam_generate_embedding_torch(self.sam, negative, self.device)

        loss = 0

        d_ap, d_an = 0, 0
        d_ap = self.l1(a_vgg, p_vgg.detach())
        if not self.ab:
            d_an = self.l1(a_vgg, n_vgg.detach())
            contrastive = d_ap / (d_an + 1e-7)
        else:
            contrastive = d_ap

        loss = contrastive
        return loss
    
    def evaluation_from_path(self, anchor_image_path=None,  positive_image_path=None, negative_image_path=None):
        anchor = cv2.imread(anchor_image_path)
        positive = cv2.imread(positive_image_path)
        negative = cv2.imread(negative_image_path)

        a_vgg = sam_generate_embedding(self.sam, anchor, self.device)
        p_vgg= sam_generate_embedding(self.sam, positive, self.device)
        n_vgg = sam_generate_embedding(self.sam, negative, self.device)

        loss = 0

        d_ap, d_an = 0, 0
        d_ap = self.l1(a_vgg, p_vgg.detach())
        if not self.ab:
            d_an = self.l1(a_vgg, n_vgg.detach())
            contrastive = d_ap / (d_an + 1e-7)
        else:
            contrastive = d_ap

        loss = contrastive
        
        return loss

if __name__ == "__main__":
    sam_constrast_loss = SAMContrastLoss()

    anchor_image_path = 'data/val/input/0000001_02999_d_0000005.jpg'
    positive_image_path = 'data/val/GT/0000001_02999_d_0000005.jpg'
    negative_image_path = 'data/val/input/0000001_02999_d_0000006.jpg'

    loss = sam_constrast_loss.evaluation_from_path(anchor_image_path, positive_image_path, negative_image_path)
    print(loss)