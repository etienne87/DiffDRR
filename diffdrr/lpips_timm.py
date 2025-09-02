"""To circumvent SSL Error
"""
import os
import warnings
import requests
from urllib3.exceptions import InsecureRequestWarning

# Suppress only the specific InsecureRequestWarning
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

# Patch the requests library to ignore SSL verification by default
old_merge_environment_settings = requests.Session.merge_environment_settings

def merge_environment_settings(self, url, proxies, stream, verify, cert):
    settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
    settings['verify'] = False
    return settings

requests.Session.merge_environment_settings = merge_environment_settings



""" Actual Script there
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale



class SimpleLPIPS(nn.Module):
    """Simplified LPIPS with timm backbones"""

    def __init__(self, backbone: str = 'vit_base_patch14_dinov2.lvd142m'):
        super().__init__()

        self.scaling_layer = ScalingLayer()

        # Create feature extractor
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True)

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False


    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Normalize to [-1, 1]
        x1 = self.scaling_layer(x1)
        x2 = self.scaling_layer(x2)

        # Convert grayscale to RGB if needed
        if x1.shape[1] == 1:
            x1 = x1.repeat(1, 3, 1, 1)
        if x2.shape[1] == 1:
            x2 = x2.repeat(1, 3, 1, 1)

        # # Resize to 224x224 if needed (most timm models expect this)
        # if x1.shape[-1] != 224 or x1.shape[-2] != 224:
        #     x1 = F.interpolate(x1, size=(224, 224), mode='bilinear', align_corners=False)
        # if x2.shape[-1] != 224 or x2.shape[-2] != 224:
        #     x2 = F.interpolate(x2, size=(224, 224), mode='bilinear', align_corners=False)

        # Extract features
        feats1 = self.backbone(x1)
        feats2 = self.backbone(x2)

        # Compute LPIPS
        total_distance = 0.0

        for f1, f2 in zip(feats1, feats2):
            # Normalize features
            f1_norm = F.normalize(f1, dim=1)
            f2_norm = F.normalize(f2, dim=1)

            # Squared difference
            diff = (f1_norm - f2_norm) ** 2

            total_distance += diff.mean(dim=[1,2,3])

        return total_distance

# Quick factory functions for common models
def create_dinov2_lpips():
    return SimpleLPIPS('vit_base_patch14_dinov2.lvd142m')

def create_convnext_lpips():
    return SimpleLPIPS('convnext_base.fb_in22k_ft_in1k')

def create_efficientnet_lpips():
    return SimpleLPIPS('efficientnet_b3.ra_in1k')

def create_resnet_lpips():
    return SimpleLPIPS('resnet50.a1_in1k')



def example_registration_loss():
    """Example of how to use in your DiffDRR pipeline"""

    # Create the loss function
    lpips_loss = create_convnext_lpips()

    # Dummy example (replace with your actual images)
    drr_image = torch.randn(1, 1, 200, 200)  # Grayscale DRR
    target_image = torch.randn(1, 1, 200, 200)  # Grayscale X-ray

    # Compute perceptual loss
    loss = lpips_loss(drr_image, target_image)

    return loss.item()



if __name__ == '__main__':
    import fire;fire.Fire(example_registration_loss)
