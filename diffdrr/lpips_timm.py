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




class SimpleLPIPS(nn.Module):
    """Simplified LPIPS with timm backbones"""

    def __init__(self, backbone: str = 'vit_base_patch14_dinov2.lvd142m'):
        super().__init__()

        # Create feature extractor
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True)

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Cfg
        cfg = timm.data.resolve_model_data_config(self.backbone)
        self.register_buffer('mean', torch.tensor(cfg['mean']).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(cfg['std']).view(1, -1, 1, 1))

        # Compute expected size
        self.expected_size = self.backbone.default_cfg['input_size'][1:]

        # Linear Layers
        # Get feature dimensions by running a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, self.expected_size[0], self.expected_size[1])
            features = self.backbone(dummy_input)
            self.feature_dims = [f.shape[1] for f in features]
            print(f"Feature dimensions: {self.feature_dims}")

        # Create linear layers (1x1 conv) for each feature level
        self.linear_layers = nn.ModuleList([
            nn.Conv2d(dim, 1, 1, bias=False) for dim in self.feature_dims
        ])

        # Initialize to identity-like behavior
        for lin in self.linear_layers:
            nn.init.constant_(lin.weight, 1.0)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Normalize to [-1, 1]
        x1 = (x1 - self.mean) / self.std
        x2 = (x2 - self.mean) / self.std

        # Convert grayscale to RGB if needed
        if x1.shape[1] == 1:
            x1 = x1.repeat(1, 3, 1, 1)
        if x2.shape[1] == 1:
            x2 = x2.repeat(1, 3, 1, 1)

        # Resize to expected_size if needed
        target_h, target_w = self.expected_size
        if x1.shape[-1] != target_w or x1.shape[-2] != target_h:
            x1 = F.interpolate(x1, size=self.expected_size, mode='bilinear', align_corners=False)
        if x2.shape[-1] != target_w or x2.shape[-2] != target_h:
            x2 = F.interpolate(x2, size=self.expected_size, mode='bilinear', align_corners=False)

        # Extract features
        feats1 = self.backbone(x1)
        feats2 = self.backbone(x2)

        # Compute LPIPS
        total_distance = 0.0

        for f1, f2, linear in zip(feats1, feats2, self.linear_layers):
            # Normalize features
            f1_norm = F.normalize(f1, dim=1)
            f2_norm = F.normalize(f2, dim=1)

            # Squared difference
            diff = (f1_norm - f2_norm) ** 2

            weighted_diff = linear(diff)

            # Spatial average
            distance = weighted_diff.mean(dim=[2, 3])

            total_distance += distance

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
