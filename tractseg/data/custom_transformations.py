
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform


import random
import torch
import torch.nn.functional as F

def augment_linear_downsampling_torch(data, zoom_range=(0.5, 1)):
    """
    Downsamples each sample (linearly) by a random factor and upsamples to original resolution again (nearest neighbor)
    Info:
    * Uses torch for resampling.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor (like isotropic=True from linear_downsampling_generator_nilearn)
    """
    zoom_range = list(zoom_range)
    zoom_range[1] += + 1e-6
    if zoom_range[0] >= zoom_range[1]:
        raise ValueError("First value of zoom_range must be smaller than second value.")

    device = data.device
    B, C, *dims = data.shape
    dim = len(dims)

    for sample_idx in range(B):
        zoom = round(random.uniform(zoom_range[0], zoom_range[1]), 2)
        zoom_reverse = round(1. / zoom, 2)

        for channel_idx in range(C):
            img = data[sample_idx, channel_idx].unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

            # Downsample
            size_down = [int(d * zoom) for d in dims]
            img_down = F.interpolate(img, size=size_down, mode='trilinear' if dim == 3 else 'bilinear', align_corners=False)

            # Upsample
            img_up = F.interpolate(img_down, size=dims, mode='nearest')

            data[sample_idx, channel_idx] = img_up.squeeze(0).squeeze(0)  # Remove batch and channel dimensions

    return data


class ResampleTransformLegacy(AbstractTransform):
    """
    This is no longer part of batchgenerators, so we have an implementation here.
    CPU always 100% when using this, but batch_time on cluster not longer (1s)

    Downsamples each sample (linearly) by a random factor and upsamples to original resolution again (nearest neighbor)
    Info:
    * Uses torch for resampling.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor
      (like isotropic=True from linear_downsampling_generator_nilearn)

    Args:
        zoom_range (tuple of float): Random downscaling factor in this range. (e.g.: 0.5 halfs the resolution)
    """

    def __init__(self, zoom_range=(0.5, 1)):
        self.zoom_range = zoom_range

    def __call__(self, **data_dict):
        data_dict['data'] = augment_linear_downsampling_torch(data_dict['data'], zoom_range=self.zoom_range)
        return data_dict

class ResampleTransformLegacy_PyTorch(AbstractTransform):
    """
    This is no longer part of batchgenerators, so we have an implementation here.
    CPU always 100% when using this, but batch_time on cluster not longer (1s)

    Downsamples each sample (linearly) by a random factor and upsamples to original resolution again (nearest neighbor)
    Info:
    * Uses torch for resampling.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor
      (like isotropic=True from linear_downsampling_generator_nilearn)

    Args:
        zoom_range (tuple of float): Random downscaling factor in this range. (e.g.: 0.5 halfs the resolution)
    """

    def __init__(self, zoom_range=(0.5, 1)):
        self.zoom_range = zoom_range

    def __call__(self, **data_dict):
        data_dict['data'] = augment_linear_downsampling_torch(data_dict['data'], zoom_range=self.zoom_range)
        return data_dict


def flip_vector_axis_numpy(data):
    data = np.copy(data)
    if (len(data.shape) != 4) and (len(data.shape) != 5) or data.shape[1] != 9:
        raise Exception("Invalid dimension for data. Data should be either [BATCH_SIZE, 9, x, y] or [BATCH_SIZE, 9, x, y, z]")
    axis = np.random.choice(["x", "y", "z"])   #chose axes to flip
    BATCH_SIZE = data.shape[0]
    for id in np.arange(BATCH_SIZE):
        if np.random.uniform() < 0.5:
            if axis == "x":
                data[id, 0] *= -1
                data[id, 3] *= -1
                data[id, 6] *= -1
            elif axis == "y":
                data[id, 1] *= -1
                data[id, 4] *= -1
                data[id, 7] *= -1
            elif axis == "z":
                data[id, 2] *= -1
                data[id, 5] *= -1
                data[id, 8] *= -1
    return data

def flip_vector_axis_torch(data):
    data = data.clone()
    if (len(data.shape) != 4) and (len(data.shape) != 5) or data.shape[1] != 9:
        raise Exception("Invalid dimension for data. Data should be either [BATCH_SIZE, 9, x, y] or [BATCH_SIZE, 9, x, y, z]")
    axis = np.random.choice(["x", "y", "z"])   #chose axes to flip
    BATCH_SIZE = data.shape[0]
    for id in np.arange(BATCH_SIZE):
        if np.random.uniform() < 0.5:
            if axis == "x":
                data[id, 0] *= -1
                data[id, 3] *= -1
                data[id, 6] *= -1
            elif axis == "y":
                data[id, 1] *= -1
                data[id, 4] *= -1
                data[id, 7] *= -1
            elif axis == "z":
                data[id, 2] *= -1
                data[id, 5] *= -1
                data[id, 8] *= -1
    return data


class FlipVectorAxisTransform_PyTorch(AbstractTransform):
    """
    Expects as input an image with 3 3D-vectors at each voxels, encoded as a nine-channel image. Will randomly
    flip sign of one dimension of all 3 vectors (x, y or z).
    """
    def __init__(self, axes=(2, 3, 4), data_key="data"):
        self.data_key = data_key
        self.axes = axes

    def __call__(self, **data_dict):
        data_dict[self.data_key] = flip_vector_axis_torch(data=data_dict[self.data_key])
        return data_dict

class FlipVectorAxisTransform(AbstractTransform):
    """
    Expects as input an image with 3 3D-vectors at each voxels, encoded as a nine-channel image. Will randomly
    flip sign of one dimension of all 3 vectors (x, y or z).
    """
    def __init__(self, axes=(2, 3, 4), data_key="data"):
        self.data_key = data_key
        self.axes = axes

    def __call__(self, **data_dict):
        data_dict[self.data_key] = flip_vector_axis_numpy(data=data_dict[self.data_key])
        return data_dict
