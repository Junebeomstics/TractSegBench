import random
from typing import Tuple

import torch
from batchgenerators.augmentations.utils import get_range_val, mask_random_squares


def augment_rician_noise(data_sample, noise_variance=(0, 0.1)):
    variance = random.uniform(noise_variance[0], noise_variance[1])
    device = data_sample.device
    noise1 = torch.normal(0.0, variance, size=data_sample.shape, device=device)
    noise2 = torch.normal(0.0, variance, size=data_sample.shape, device=device)
    data_sample = torch.sqrt((data_sample + noise1) ** 2 + noise2 ** 2) * torch.sign(data_sample)
    return data_sample


def augment_gaussian_noise(data_sample: torch.Tensor, noise_variance: Tuple[float, float] = (0, 0.1),
                           p_per_channel: float = 1, per_channel: bool = False) -> torch.Tensor:
    device = data_sample.device
    if not per_channel:
        variance = noise_variance[0] if noise_variance[0] == noise_variance[1] else \
            random.uniform(noise_variance[0], noise_variance[1])
    else:
        variance = None
    for c in range(data_sample.shape[0]):
        if torch.rand(1).item() < p_per_channel:
            variance_here = variance if variance is not None else \
                noise_variance[0] if noise_variance[0] == noise_variance[1] else \
                    random.uniform(noise_variance[0], noise_variance[1])
            data_sample[c] = data_sample[c] + torch.normal(0.0, variance_here, size=data_sample[c].shape, device=device)
    return data_sample


def gaussian_blur_3d(data_sample: torch.Tensor, sigma: float) -> torch.Tensor:
    channels, depth, height, width = data_sample.shape
    kernel_size = int(2 * (sigma * 3) + 1)
    kernel = torch.arange(kernel_size, dtype=torch.float32, device=data_sample.device) - kernel_size // 2
    kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
    kernel = kernel / kernel.sum()
    
    kernel_3d = kernel[:, None, None] * kernel[None, :, None] * kernel[None, None, :]
    kernel_3d = kernel_3d.expand(channels, 1, kernel_size, kernel_size, kernel_size)
    
    padding = kernel_size // 2
    data_sample = data_sample.unsqueeze(0)
    blurred = torch.nn.functional.conv3d(data_sample, kernel_3d, padding=padding, groups=channels)
    return blurred.squeeze(0)


def augment_gaussian_blur(data_sample: torch.Tensor, sigma_range: Tuple[float, float], per_channel: bool = True,
                          p_per_channel: float = 1, different_sigma_per_axis: bool = False,
                          p_isotropic: float = 0) -> torch.Tensor:
    if not per_channel:
        sigma = get_range_val(sigma_range) if ((not different_sigma_per_axis) or
                                               ((torch.rand(1).item() < p_isotropic) and
                                                different_sigma_per_axis)) \
            else [get_range_val(sigma_range) for _ in data_sample.shape[1:]]
    else:
        sigma = None
    for c in range(data_sample.shape[0]):
        if torch.rand(1).item() <= p_per_channel:
            if per_channel:
                sigma = get_range_val(sigma_range) if ((not different_sigma_per_axis) or
                                                       ((torch.rand(1).item() < p_isotropic) and
                                                        different_sigma_per_axis)) \
                    else [get_range_val(sigma_range) for _ in data_sample.shape[1:]]
            data_sample[c] = gaussian_blur_3d(data_sample[c].unsqueeze(0), sigma).squeeze(0)
    return data_sample


def augment_blank_square_noise(data_sample, square_size, n_squares, noise_val=(0, 0), channel_wise_n_val=False,
                               square_pos=None):
    rnd_square_size = get_range_val(square_size)
    rnd_n_squares = get_range_val(n_squares)

    data_sample = mask_random_squares(data_sample, square_size=rnd_square_size, n_squares=rnd_n_squares,
                                      n_val=noise_val, channel_wise_n_val=channel_wise_n_val,
                                      square_pos=square_pos)
    return data_sample
