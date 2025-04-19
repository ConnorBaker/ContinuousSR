import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import to_pixel_samples
import time
from torchvision.utils import save_image
from itertools import product

from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def generate_meshgrid(height, width):
    """
    Generate a meshgrid of coordinates for a given image dimensions.
    Args:
        height (int): Height of the image.
        width (int): Width of the image.
    Returns:
        torch.Tensor: A tensor of shape [height * width, 2] containing the (x, y) coordinates for each pixel in the image.
    """
    # Generate all pixel coordinates for the given image dimensions
    y_coords, x_coords = torch.arange(0, height), torch.arange(0, width)
    # Create a grid of coordinates
    yy, xx = torch.meshgrid(y_coords, x_coords)
    # Flatten and stack the coordinates to obtain a list of (x, y) pairs
    all_coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return all_coords


def fetching_features_from_tensor(image_tensor, input_coords):
    """
    Extracts pixel values from a tensor of images at specified coordinate locations.
    Args:
        image_tensor (torch.Tensor): A 4D tensor of shape [batch, channel, height, width] representing a batch of images.
        input_coords (torch.Tensor): A 2D tensor of shape [N, 2] containing the (x, y) coordinates at which to extract pixel values.
    Returns:
        color_values (torch.Tensor): A 3D tensor of shape [batch, N, channel] containing the pixel values at the specified coordinates.
        coords (torch.Tensor): A 2D tensor of shape [N, 2] containing the normalized coordinates in the range [-1, 1].
    """
    # Normalize pixel coordinates to [-1, 1] range
    input_coords = input_coords.to(image_tensor.device)
    coords = input_coords / torch.tensor([image_tensor.shape[-2], image_tensor.shape[-1]],
                                         device=image_tensor.device).float()
    center_coords_normalized = torch.tensor([0.5, 0.5], device=image_tensor.device).float()
    coords = (center_coords_normalized - coords) * 2.0

    # Fetching the colour of the pixels in each coordinates
    batch_size = image_tensor.shape[0]
    input_coords_expanded = input_coords.unsqueeze(0).expand(batch_size, -1, -1)

    y_coords = input_coords_expanded[..., 0].long()
    x_coords = input_coords_expanded[..., 1].long()
    batch_indices = torch.arange(batch_size).view(-1, 1).to(input_coords.device)

    color_values = image_tensor[batch_indices, :, x_coords, y_coords]

    return color_values, coords


def scale_to_range(tensor, min_value, max_value):
    min_tensor = torch.min(tensor)
    max_tensor = torch.max(tensor)
    scaled_tensor = (tensor - min_tensor) / (max_tensor - min_tensor)  
    return scaled_tensor * (max_value - min_value) + min_value


def get_coord(width, height):
    x_coords = torch.arange(width)
    y_coords = torch.arange(height)

    # Generate coordinate grid using torch.meshgrid
    x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')

    # Map coordinates to the range of -1 to 1
    x_grid = 2 * (x_grid / (width)) - 1 #+ 1/width
    y_grid = 2 * (y_grid / (height)) - 1 #+ 1/height

    # Stack the x and y coordinates to form the final coordinate tensor
    coordinates = torch.stack((y_grid, x_grid), dim=-1).reshape(-1, 2)
    
    return coordinates



@register('continuous-gaussian')
class ContinuousGaussian(nn.Module):
    """A module that applies 2D Gaussian splatting to input features."""

    def __init__(self, encoder_spec, cnn_spec, fc_spec, **kwargs):
        """
        Initialize the ContinuousGaussian module.
        
        Args:
            encoder_spec (dict): Specifications for the encoder.
            cnn_spec (dict): Specifications for the CNN layers.
            fc_spec (dict): Specifications for the fully connected layers.
            kwargs: Additional arguments.
        """
        super(ContinuousGaussian, self).__init__()
        
        # Create the encoder module based on the given specifications
        self.encoder = models.make(encoder_spec)
        
        # Initialize placeholders for various attributes
        self.feat = None  # Low-resolution (LR) features
        self.inp = None  # Input image
        self.feat_coord = None  # Feature coordinates
        self.init_num_points = None
        self.H, self.W = None, None  # Image height and width
        self.BLOCK_H, self.BLOCK_W = 16, 16  # Block size for tiling

        # Define additional convolutional and activation layers
        self.conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # Convolutional layer
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # Leaky ReLU activation
        self.ps = nn.PixelUnshuffle(2)  # Pixel unshuffle with a scaling factor of 2

        # Define an MLP for vector generation for gaussian dict
        mlp_spec = {'name': 'mlp', 'args': {'in_dim': 3, 'out_dim': 512, 'hidden_list': [256, 512, 512, 512]}}
        self.mlp_vector = models.make(mlp_spec)
        
        # Define an MLP for color prediction
        mlp_spec = {'name': 'mlp', 'args': {'in_dim': 256, 'out_dim': 3, 'hidden_list': [512, 1024, 256, 128, 64]}}
        self.mlp = models.make(mlp_spec)
        
        # Define an MLP for offset prediction
        mlp_spec = {'name': 'mlp', 'args': {'in_dim': 256, 'out_dim': 2, 'hidden_list': [512, 1024, 256, 128, 64]}}
        self.mlp_offset = models.make(mlp_spec)
        
        # Initialize pre-defined Gaussian convariance parameter dictionaries
        cho1 = torch.tensor([0, 0.41, 0.62, 0.98, 1.13, 1.29, 1.64, 1.85, 2.36]).cuda()
        cho2 = torch.tensor([-0.86, -0.36, -0.16, 0.19, 0.34, 0.49, 0.84, 1.04, 1.54]).cuda()
        cho3 = torch.tensor([0, 0.33, 0.53, 0.88, 1.03, 1.18, 1.53, 1.73, 2.23]).cuda()
        self.gau_dict = torch.tensor(list(product(cho1, cho2, cho3))).cuda()
        self.gau_dict = torch.cat((self.gau_dict, torch.zeros(1, 3).cuda()), dim=0)  # Add zeros

        self.last_size = (self.H, self.W)
        self.background = torch.ones(3).cuda()  # Default background color

    def gen_feat(self, inp):
        """
        Generate features from the input image using the encoder.
        
        Args:
            inp (torch.Tensor): Input image.
        
        Returns:
            torch.Tensor: Extracted low-resolution features.
        """
        self.inp = inp  # Store the input image
        feat = self.encoder(inp)  # Encode the input image
        self.feat = self.ps(feat)  # Apply pixel unshuffle to the encoded features
        return self.feat

    def query_output(self, inp, scale):
        """
        Generate the high-resolution image output for a given scale.
        
        Args:
            inp (torch.Tensor): Input image.
            scale (torch.Tensor): Scaling factor.
        
        Returns:
            torch.Tensor: High-resolution output image.
        """
        feat = self.feat

        # Process the scaling factors
        if scale.shape == (1, 2):  # Handle cases with two scaling factors
            scale1 = float(scale[0, 0])
            scale2 = float(scale[0, 1])
        else:  # Handle uniform scaling
            scale1 = float(scale[0])
            scale2 = float(scale[0])

        # Compute dimensions of the low-resolution and high-resolution images
        lr_h = self.inp.shape[-2]  # Low-resolution height
        lr_w = self.inp.shape[-1]  # Low-resolution width
        H = round(int(lr_h) * scale1)  # High-resolution height
        W = round(int(lr_w) * scale2)  # High-resolution width

        # Determine the number of tiles for rasterization
        self.tile_bounds = (
            (W + self.BLOCK_W - 1) // self.BLOCK_W,
            (H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )

        window_size = 1  # Window size for Gaussian position adjustments
        pred = []  # List to store predictions
        bs, _, _, _ = feat.shape  # Batch size and feature dimensions

        # Process features for color prediction
        para_c = self.feat.reshape(bs, -1, lr_h * lr_w * 4).permute(1, 0, 2)
        color = self.mlp(para_c.reshape(-1, bs * lr_h * lr_w * 4).permute(1, 0))
        color = color.reshape(bs, lr_h * lr_w * 4, -1)

        # Process features for Gaussian convariance parameter estimation
        para_c = self.leaky_relu(self.feat)
        para = self.conv1(para_c)
        vector = self.mlp_vector(self.gau_dict.to(para.device)) # Transform Gaussian covariance dictionary to increase dimensions
        para = para.reshape(bs, -1, lr_h * lr_w * 4).permute(1, 0, 2).reshape(-1, bs * lr_h * lr_w * 4)
        para = vector @ para  # This calculates the similarity between para and each element in the dictionary
        para = torch.softmax(para, dim=0)  # Normalize the similarity scores to produce weights using softmax
        para = para.permute(1, 0) @ self.gau_dict.to(para.device) # Compute the weighted sum of dictionary elements to get the final covariance
        para = para.reshape(bs, lr_h * lr_w * 4, -1)

        # Process features for offset prediction
        offset = self.mlp_offset(para_c.reshape(-1, bs * lr_h * lr_w * 4).permute(1, 0))
        offset = torch.tanh(offset).reshape(bs, lr_h * lr_w * 4, -1)

        # Generate output predictions for each image in the batch
        for i in range(bs):
            offset_ = offset[i, :, :].squeeze(0)
            color_ = color[i, :, :].squeeze(0)
            para_ = para[i, :, :].squeeze(0)

            # Generate coordinate grid for the high-resolution image
            get_xyz = torch.tensor(get_coord(lr_h * 2, lr_w * 2)).reshape(lr_h * 2, lr_w * 2, 2).cuda()
            get_xyz = get_xyz.reshape(-1, 2)

            # Adjust coordinates using offsets
            xyz1 = get_xyz[:, 0:1] + 2 * window_size * offset_[:, 0:1] / lr_w - 1 / W
            xyz2 = get_xyz[:, 1:2] + 2 * window_size * offset_[:, 1:2] / lr_h - 1 / H
            get_xyz = torch.cat((xyz1, xyz2), dim=1)

            # Adjust Gaussian parameters
            weighted_cholesky = para_ / 4
            weighted_opacity = torch.ones(color_.shape[0], 1).cuda()
            weighted_cholesky[:, 0] *= scale2
            weighted_cholesky[:, 1] *= scale1
            weighted_cholesky[:, 2] *= scale1

            # Perform Gaussian projection and rasterization
            xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
                get_xyz, weighted_cholesky, H, W, self.tile_bounds
            )
            out_img = rasterize_gaussians_sum(
                xys, depths, radii, conics, num_tiles_hit, color_, weighted_opacity,
                H, W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False
            )
            out_img = out_img.permute(2, 0, 1).unsqueeze(0)
            pred.append(out_img)

        # Combine outputs for the batch
        out_img = torch.cat(pred)
        return out_img

    def forward(self, inp, scale):
        """
        Forward pass for the ContinuousGaussian module.
        
        Args:
            inp (torch.Tensor): Input image.
            scale (torch.Tensor): Scaling factor.
        
        Returns:
            torch.Tensor: High-resolution output image.
        """
        self.gen_feat(inp)  # Generate low-resolution features
        image = self.query_output(inp, scale)  # Generate high-resolution output
        return image
