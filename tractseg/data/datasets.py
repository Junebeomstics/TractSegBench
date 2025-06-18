import torch
from torch.utils.data import Dataset
import numpy as np
from scipy import ndimage
from os.path import join
import nibabel as nib
import random

from batchgenerators.transforms.resample_transforms import ResampleTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.noise_transforms import GaussianBlurTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.spatial_transforms import ZoomTransform
from batchgenerators.transforms.spatial_transforms import ResizeTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.augmentations.utils import center_crop_2D_image_batched
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.augmentations.spatial_transformations import augment_zoom

# from batchgenerators.transforms.sample_normalization_transforms import ZeroMeanUnitVarianceTransform
from tractseg.data.DLDABG_standalone import ZeroMeanUnitVarianceTransform as ZeroMeanUnitVarianceTransform_Standalone

from tractseg.data.custom_transformations import ResampleTransformLegacy
from tractseg.data.custom_transformations import FlipVectorAxisTransform
from tractseg.data.spatial_transform_peaks import SpatialTransformPeaks
from tractseg.data.spatial_transform_custom import SpatialTransformCustom

# pytorch implementation
from tractseg.data.DLDABG_standalone import ZeroMeanUnitVarianceTransform_PyTorch
from tractseg.custom_batchgenerators.transforms.spatial_transforms import SpatialTransform_PyTorch
from tractseg.custom_batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform_PyTorch
from tractseg.data.custom_transformations import ResampleTransformLegacy_PyTorch
from tractseg.custom_batchgenerators.transforms.noise_transforms import GaussianBlurTransform_PyTorch
from tractseg.custom_batchgenerators.transforms.noise_transforms import GaussianNoiseTransform_PyTorch
from tractseg.custom_batchgenerators.transforms.spatial_transforms import MirrorTransform_PyTorch
from tractseg.data.custom_transformations import FlipVectorAxisTransform_PyTorch

class MRISliceDataset(Dataset):
    def __init__(self, config, subjects, transform=None):
        self.config = config
        if config.DIM == "2D":
            self.subjects = subjects * int(self.config.INPUT_DIM[0]/self.config.NR_SLICES) # multiply by 144 to replicate original dataloader
        elif config.DIM == "3D":
            self.subjects = subjects 
        self.transform = transform

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        if self.config.DIM == "2D":
            # override idx to replicate original dataloader
            idx = int(random.uniform(0, len(self.subjects)))
            subject = self.subjects[idx] 
            data, seg = self.load_subject_data(subject)

            # Convert peaks to tensors if tensor model
            # if self.config.NR_OF_GRADIENTS == 18*self.config.NR_SLICES:
            #     data = self.peaks_to_tensors(data)

            slice_direction = self.slice_dir_to_int(self.config.TRAINING_SLICE_DIRECTION) # randomly choose on slice direction
            slice_idxs = self.get_random_slices(data, slice_direction, nr_slices=self.config.NR_SLICES) # sample as much as batch size

            if self.config.USE_CONSECUTIVE_SLICES:
                # Choose random starting index for consecutive slices
                #start_idx = np.random.randint(0, data.shape[slice_direction] - self.batch_size + 1)
                start_idx = np.random.randint(0, data.shape[slice_direction] - self.config.NR_SLICES + 1)
                slice_idxs = np.arange(start_idx, start_idx + self.config.NR_SLICES)

            # if self.config.NR_SLICES > 1:
            #     x, y = self.sample_Xslices(data, seg, slice_idxs, slice_direction=slice_direction, slice_window=self.config.NR_SLICES)
            # else:
            x, y = self.sample_slices(data, seg, slice_idxs, slice_direction=slice_direction)

            if self.config.PAD_TO_SQUARE:
                #Crop and pad to input size
                #print(x.shape,y.shape) # (NR_SLICES, 9, 109, 114) (NR_SLICES, 72, 109, 114), NR_SLICES play a role as BATCH_SIZE
                x, y = crop(x, y, crop_size=self.config.INPUT_DIM)  # does not work with img with batches and channels
                #print(x.shape,y.shape) # (NR_SLICES, 9, 144, 144) (NR_SLICES, 72, 144, 144)
            else:
                x = pad_nd_image(x, shape_must_be_divisible_by=(16, 16), mode='constant', kwargs={'constant_values': 0})
                y = pad_nd_image(y, shape_must_be_divisible_by=(16, 16), mode='constant', kwargs={'constant_values': 0})

            x = x.astype(np.float32)
            y = y.astype(np.float32)

            # Apply transformations
            if self.transform:
                output = self.transform(**{'data':x, 'seg':y})
            else:
                output = None
            
            data_dict = {"subject": subject,
                        "data": output['data'] if output else x,  # (nr_slices, channels, x, y, [z])
                        "seg": output['seg'] if output else y, # (nr_slices, channels, x, y, [z])
                        "slice_dir": slice_direction} 

        elif self.config.DIM == "3D":
            subject = self.subjects[idx]
            x, y = self.load_subject_data(subject)
            x = np.expand_dims(np.array(x).transpose(3, 0, 1, 2), axis=0) # (x,y,z, channels) -> (1, channels, x, y, z)
            y = np.expand_dims(np.array(y).transpose(3, 0, 1, 2), axis=0) # (x,y,z, channels) -> (1, channels, x, y, z)
            if self.config.PAD_TO_SQUARE:
                x, y = crop(x, y, crop_size=self.config.INPUT_DIM) 
            else:
                x = pad_nd_image(x, shape_must_be_divisible_by=(16, 16, 16), mode='constant', kwargs={'constant_values': 0})
                y = pad_nd_image(y, shape_must_be_divisible_by=(16, 16, 16), mode='constant', kwargs={'constant_values': 0})
            
        
            x = x.astype(np.float32)
            y = y.astype(np.float32)

            # Apply transformations
            if self.transform:
                output = self.transform(**{'data':x, 'seg':y})
            else:
                output = {'data':x, 'seg':y}

            
            data_dict = {"subject": subject,
                        "data": output['data'].squeeze(0),  # (channels, x, y, [z])
                        "seg": output['seg'].squeeze(0), # (channels, x, y, [z])
                        }  

        return data_dict

        #torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def load_subject_data(self, subject): # originally, load_training_data
        def load(filepath):
            return nib.load(filepath + ".nii.gz").get_fdata()

        data = load(join(self.config.DATA_PATH, self.config.DATASET_FOLDER, subject, self.config.FEATURES_FILENAME))
        
        if "|" in self.config.LABELS_FILENAME:
            parts = self.config.LABELS_FILENAME.split("|")
            seg = []  # [4, x, y, z, 54]
            for part in parts:
                seg.append(load(join(self.config.DATA_PATH, self.config.DATASET_FOLDER, subject, part)))
            seg = np.array(seg).transpose(1, 2, 3, 4, 0)
            seg = seg.reshape(data.shape[:3] + (-1,))  # [x, y, z, 54*4]
        else:
            seg = load(join(self.config.DATA_PATH, self.config.DATASET_FOLDER, subject, self.config.LABELS_FILENAME))
        return data, seg

    def get_random_slices(self, data, slice_direction, nr_slices):
        slice_dim = data.shape[slice_direction]

        if data.shape[slice_direction] <= nr_slices:
            print("INFO: Batch size bigger than nr of slices. Therefore sampling with replacement.")
            return np.random.choice(slice_dim, nr_slices, replace=True)
        else:
            return np.random.choice(slice_dim, nr_slices, replace=False)

    def sample_slices(self, data, seg, slice_idxs, slice_direction=0, labels_type=np.int16):
        if slice_direction == 0:
            x = data[slice_idxs, :, :].astype(np.float32)  # (bs, y, z, channels)
            y = seg[slice_idxs, :, :].astype(labels_type)
            # depth-channel has to be before width and height for Unet (but after batches)
            x = np.array(x).transpose(0, 3, 1, 2)
            # nr_classes channel has to be before with and height for DataAugmentation (bs, channels, x, y)
            y = np.array(y).transpose(0, 3, 1, 2)
        elif slice_direction == 1:
            x = data[:, slice_idxs, :].astype(np.float32)  # (x, bs, z, channels)
            y = seg[:, slice_idxs, :].astype(labels_type)
            x = np.array(x).transpose(1, 3, 0, 2)
            y = np.array(y).transpose(1, 3, 0, 2)
        elif slice_direction == 2:
            x = data[:, :, slice_idxs].astype(np.float32)  # (x, y, bs, channels)
            y = seg[:, :, slice_idxs].astype(labels_type)
            x = np.array(x).transpose(2, 3, 0, 1)
            y = np.array(y).transpose(2, 3, 0, 1)
        return x, y

    @staticmethod
    def slice_dir_to_int(slice_dir):
        if slice_dir == "xyz":
            slice_direction_int = int(round(random.uniform(0, 2)))
            return slice_direction_int
        elif slice_dir == "x":
            return 0
        elif slice_dir == "y":
            return 1
        elif slice_dir == "z":
            return 2
        else:
            raise ValueError(f"Invalid slice direction: {slice_dir}")

    def peaks_to_tensors(self, peaks):
        """
        Convert peak image to tensor image

        Args:
            peaks: shape: [x,y,z,nr_peaks*3]

        Returns:
            tensor with shape: [x,y,z, nr_peaks*6]
        """

        def _peak_to_tensor(peak):
            tensor = np.zeros(peak.shape[:3] + (6,), dtype=np.float32)
            tensor[..., 0] = peak[..., 0] * peak[..., 0]
            tensor[..., 1] = peak[..., 0] * peak[..., 1]
            tensor[..., 2] = peak[..., 0] * peak[..., 2]
            tensor[..., 3] = peak[..., 1] * peak[..., 1]
            tensor[..., 4] = peak[..., 1] * peak[..., 2]
            tensor[..., 5] = peak[..., 2] * peak[..., 2]
            return tensor

        nr_peaks = int(peaks.shape[3] / 3)
        tensor = np.zeros(peaks.shape[:3] + (nr_peaks * 6,), dtype=np.float32)
        for idx in range(nr_peaks):
            tensor[..., idx*6:(idx*6)+6] = _peak_to_tensor(peaks[..., idx*3:(idx*3)+3])
        return tensor

        
def transform_data(Config, type):
    tfs=[]
    if Config.NORMALIZE_DATA:
        # todo: Use original transform as soon as bug fixed in batchgenerators
        # tfs.append(ZeroMeanUnitVarianceTransform(per_channel=self.Config.NORMALIZE_PER_CHANNEL))
        tfs.append(ZeroMeanUnitVarianceTransform_Standalone(per_channel=Config.NORMALIZE_PER_CHANNEL))

    if Config.SPATIAL_TRANSFORM == "SpatialTransformPeaks":
        SpatialTransformUsed = SpatialTransformPeaks
    elif Config.SPATIAL_TRANSFORM == "SpatialTransformCustom":
        SpatialTransformUsed = SpatialTransformCustom
    else:
        SpatialTransformUsed = SpatialTransform

    if Config.DATA_AUGMENTATION:
        if type == "train":
            # patch_center_dist_from_border:
            #   if 144/2=72 -> always exactly centered; otherwise a bit off center
            #   (brain can get off image and will be cut then)
            if Config.DAUG_SCALE:
                if Config.INPUT_RESCALING:
                    source_mm = 2  # for bb
                    target_mm = float(Config.RESOLUTION[:-2])
                    scale_factor = target_mm / source_mm
                    scale = (scale_factor, scale_factor)
                else:
                    scale = (0.9, 1.5)

                if Config.PAD_TO_SQUARE:
                    patch_size = Config.INPUT_DIM
                else:
                    patch_size = None  # keeps dimensions of the data

                # spatial transform automatically crops/pads to correct size
                center_dist_from_border = int(Config.INPUT_DIM[0] / 2.) - 10  # (144,144) -> 62
                tfs.append(SpatialTransformUsed(patch_size,
                                            patch_center_dist_from_border=center_dist_from_border,
                                            do_elastic_deform=Config.DAUG_ELASTIC_DEFORM,
                                            alpha=Config.DAUG_ALPHA, sigma=Config.DAUG_SIGMA,
                                            do_rotation=Config.DAUG_ROTATE,
                                            angle_x=Config.DAUG_ROTATE_ANGLE,
                                            angle_y=Config.DAUG_ROTATE_ANGLE,
                                            angle_z=Config.DAUG_ROTATE_ANGLE,
                                            do_scale=True, scale=scale, border_mode_data='constant',
                                            border_cval_data=0,
                                            order_data=3,
                                            border_mode_seg='constant', border_cval_seg=0,
                                            order_seg=0, random_crop=True,
                                            p_el_per_sample=Config.P_SAMP,
                                            p_rot_per_sample=Config.P_SAMP,
                                            p_scale_per_sample=Config.P_SAMP))

            if Config.DAUG_RESAMPLE:
                tfs.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), p_per_sample=0.2, per_channel=False))

            if Config.DAUG_RESAMPLE_LEGACY:
                tfs.append(ResampleTransformLegacy(zoom_range=(0.5, 1)))

            if Config.DAUG_GAUSSIAN_BLUR:
                tfs.append(GaussianBlurTransform(blur_sigma=Config.DAUG_BLUR_SIGMA,
                                                 different_sigma_per_channel=False,
                                                 p_per_sample=Config.P_SAMP))

            if Config.DAUG_NOISE:
                tfs.append(GaussianNoiseTransform(noise_variance=Config.DAUG_NOISE_VARIANCE,
                                                  p_per_sample=Config.P_SAMP))

            if Config.DAUG_MIRROR:
                tfs.append(MirrorTransform())

            if Config.DAUG_FLIP_PEAKS:
                tfs.append(FlipVectorAxisTransform())
        
    tfs.append(NumpyToTensor(keys=["data", "seg"], cast_to="float"))
    return tfs

# transform_data for torch  
def augment_data(Config, batch, type):
    # X should be (bs, slices, features, w, h)
    # y should be (bs, slices, classes, w, h)
    x = batch["data"]  # (bs, slices, nr_of_channels, x, y)
    y = batch["seg"]  # (bs, slices, nr_of_classes, x, y)
    if x.ndim == 5:
        bs, slices, nr_of_channels, w, h = x.shape
        x = x.reshape(bs * slices, nr_of_channels, w, h)
        y = y.reshape(bs * slices, y.shape[-3], w, h)
    batch["data"] = x
    batch["seg"] = y

    tfs=[]
    if Config.NORMALIZE_DATA:
        ZeroMeanUnitVarianceTransform_PyTorch(per_channel=Config.NORMALIZE_PER_CHANNEL)

    if Config.DATA_AUGMENTATION:
        if type == "train":
            # patch_center_dist_from_border:
            #   if 144/2=72 -> always exactly centered; otherwise a bit off center
            #   (brain can get off image and will be cut then)
            if Config.DAUG_SCALE:
                if Config.INPUT_RESCALING:
                    source_mm = 2  # for bb
                    target_mm = float(Config.RESOLUTION[:-2])
                    scale_factor = target_mm / source_mm
                    scale = (scale_factor, scale_factor)
                else:
                    scale = (0.9, 1.5)

                if Config.PAD_TO_SQUARE:
                    patch_size = Config.INPUT_DIM
                else:
                    patch_size = None  # keeps dimensions of the data

                # spatial transform automatically crops/pads to correct size
                center_dist_from_border = int(Config.INPUT_DIM[0] / 2.) - 10  # (144,144) -> 62
                tfs.append(SpatialTransform_PyTorch(patch_size,
                                            patch_center_dist_from_border=center_dist_from_border,
                                            do_elastic_deform=Config.DAUG_ELASTIC_DEFORM,
                                            alpha=Config.DAUG_ALPHA, sigma=Config.DAUG_SIGMA,
                                            do_rotation=Config.DAUG_ROTATE,
                                            angle_x=Config.DAUG_ROTATE_ANGLE,
                                            angle_y=Config.DAUG_ROTATE_ANGLE,
                                            angle_z=Config.DAUG_ROTATE_ANGLE,
                                            do_scale=True, scale=scale, border_mode_data='constant',
                                            border_cval_data=0,
                                            order_data=3,
                                            border_mode_seg='constant', border_cval_seg=0,
                                            order_seg=0, random_crop=True,
                                            p_el_per_sample=Config.P_SAMP,
                                            p_rot_per_sample=Config.P_SAMP,
                                            p_scale_per_sample=Config.P_SAMP))

            if Config.DAUG_RESAMPLE:
                tfs.append(SimulateLowResolutionTransform_PyTorch(zoom_range=(0.5, 1), p_per_sample=0.2, per_channel=False))

            if Config.DAUG_RESAMPLE_LEGACY:
                tfs.append(ResampleTransformLegacy_PyTorch(zoom_range=(0.5, 1)))

            if Config.DAUG_GAUSSIAN_BLUR:
                tfs.append(GaussianBlurTransform_PyTorch(blur_sigma=Config.DAUG_BLUR_SIGMA,
                                                 different_sigma_per_channel=False,
                                                 p_per_sample=Config.P_SAMP))

            if Config.DAUG_NOISE:
                tfs.append(GaussianNoiseTransform_PyTorch(noise_variance=Config.DAUG_NOISE_VARIANCE,
                                                  p_per_sample=Config.P_SAMP))

            if Config.DAUG_MIRROR:
                tfs.append(MirrorTransform_PyTorch())

            if Config.DAUG_FLIP_PEAKS:
                tfs.append(FlipVectorAxisTransform_PyTorch())
    transforms = Compose(tfs)
    return transforms(**batch)