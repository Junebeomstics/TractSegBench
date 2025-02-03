import torch
from torch.utils.data import Dataset
import numpy as np
from scipy import ndimage
from os.path import join
import nibabel as nib

class MRISliceDataset(Dataset):
    def __init__(self, config, subjects, transform=None):
        self.config = config
        self.subjects = subjects
        self.transform = transform

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx] # randomly choose one subject
        data, seg = self.load_subject_data(subject)

        # Convert peaks to tensors if tensor model
        if self.config.NR_OF_GRADIENTS == 18*self.config.NR_SLICES:
            data = self.peaks_to_tensors(data)

        slice_direction = self.slice_dir_to_int(self.config.TRAINING_SLICE_DIRECTION) # randomly choose on slice direction
        slice_idxs = self.get_random_slices(data, slice_direction, batch_size=self.config.BATCH_SIZE) # sample as much as batch size

        if self.config.NR_SLICES > 1:
            x, y = self.sample_Xslices(data, seg, slice_idxs, slice_direction=slice_direction, slice_window=self.config.NR_SLICES)
        else:
            x, y = self.sample_slices(data, seg, slice_idxs, slice_direction=slice_direction)

        if self.config.PAD_TO_SQUARE:
            #Crop and pad to input size
            x, y = crop(x, y, crop_size=self.Config.INPUT_DIM)  # does not work with img with batches and channels
        else:
            # Works -> results as good?
            # Will pad each axis to be multiple of 16. (Each sample can end up having different dimensions. Also x and y
            # can be different)
            # This is needed for Schizo dataset
            x = pad_nd_image(x, shape_must_be_divisible_by=(16, 16), mode='constant', kwargs={'constant_values': 0})
            y = pad_nd_image(y, shape_must_be_divisible_by=(16, 16), mode='constant', kwargs={'constant_values': 0})

        x = x.astype(np.float32)
        y = y.astype(np.float32)

        # Apply transformations
        if self.transform:
            x,y = self._augment_data(x, y)

        data_dict = {"data": x,  # (batch_size, channels, x, y, [z])
                     "seg": y,
                     "slice_dir": self.slice_direction}  # (batch_size, channels, x, y, [z])

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

    def get_random_slices(self, data, slice_direction, batch_size):
        slice_dim = data.shape[slice_direction]

        if data.shape[slice_direction] <= batch_size:
            print("INFO: Batch size bigger than nr of slices. Therefore sampling with replacement.")
            return np.random.choice(slice_dim, batch_size, replace=True)
        else:
            return np.random.choice(slice_dim, batch_size, replace=False)

    def sample_Xslices(self, data, seg, slice_idxs, slice_direction, slice_window, labels_type=np.int16):
        """
        Sample slices but add slices_window/2 above and below.
        """
        sw = slice_window  # slice_window (only odd numbers allowed)
        assert sw % 2 == 1, "Slice_window has to be an odd number"
        pad = int((sw - 1) / 2)

        if slice_direction == 0:
            y = seg[slice_idxs, :, :].astype(labels_type)
            y = np.array(y).transpose(0, 3, 1, 2)  # nr_classes channel has to be before with and height for DataAugmentation (bs, nr_of_classes, x, y)
        elif slice_direction == 1:
            y = seg[:, slice_idxs, :].astype(labels_type)
            y = np.array(y).transpose(1, 3, 0, 2)
        elif slice_direction == 2:
            y = seg[:, :, slice_idxs].astype(labels_type)
            y = np.array(y).transpose(2, 3, 0, 1)

        data_pad = np.zeros((data.shape[0] + sw - 1, data.shape[1] + sw - 1, data.shape[2] + sw - 1, data.shape[3])).astype(
            data.dtype)
        data_pad[pad:-pad, pad:-pad, pad:-pad, :] = data  # padded with two slices of zeros on all sides
        batch = []
        for s_idx in slice_idxs:
            if slice_direction == 0:
                # (s_idx+2)-2:(s_idx+2)+3 = s_idx:s_idx+5
                x = data_pad[s_idx:s_idx + sw:, pad:-pad, pad:-pad, :].astype(np.float32)  # (5, y, z, channels)
                x = np.array(x).transpose(0, 3, 1, 2)  # channels dim has to be before width and height for Unet (but after batches)
                x = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # (5*channels, y, z)
                batch.append(x)
            elif slice_direction == 1:
                x = data_pad[pad:-pad, s_idx:s_idx + sw, pad:-pad, :].astype(np.float32)  # (5, y, z, channels)
                x = np.array(x).transpose(1, 3, 0, 2)
                x = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # (5*channels, y, z)
                batch.append(x)
            elif slice_direction == 2:
                x = data_pad[pad:-pad, pad:-pad, s_idx:s_idx + sw, :].astype(np.float32)  # (5, y, z, channels)
                x = np.array(x).transpose(2, 3, 0, 1)
                x = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # (5*channels, y, z)
                batch.append(x)

        return np.array(batch), y  # (bs, channels, x, y)
    
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

    def _augment_data(self, batch_generator, type=None):
        tfs = []

        if self.Config.NORMALIZE_DATA:
            # todo: Use original transform as soon as bug fixed in batchgenerators
            # tfs.append(ZeroMeanUnitVarianceTransform(per_channel=self.Config.NORMALIZE_PER_CHANNEL))
            tfs.append(ZeroMeanUnitVarianceTransform_Standalone(per_channel=self.Config.NORMALIZE_PER_CHANNEL))

        if self.Config.SPATIAL_TRANSFORM == "SpatialTransformPeaks":
            SpatialTransformUsed = SpatialTransformPeaks
        elif self.Config.SPATIAL_TRANSFORM == "SpatialTransformCustom":
            SpatialTransformUsed = SpatialTransformCustom
        else:
            SpatialTransformUsed = SpatialTransform

        if self.Config.DATA_AUGMENTATION:
            if type == "train":
                # patch_center_dist_from_border:
                #   if 144/2=72 -> always exactly centered; otherwise a bit off center
                #   (brain can get off image and will be cut then)
                if self.Config.DAUG_SCALE:

                    if self.Config.INPUT_RESCALING:
                        source_mm = 2  # for bb
                        target_mm = float(self.Config.RESOLUTION[:-2])
                        scale_factor = target_mm / source_mm
                        scale = (scale_factor, scale_factor)
                    else:
                        scale = (0.9, 1.5)

                    if self.Config.PAD_TO_SQUARE:
                        patch_size = self.Config.INPUT_DIM
                    else:
                        patch_size = None  # keeps dimensions of the data

                    # spatial transform automatically crops/pads to correct size
                    center_dist_from_border = int(self.Config.INPUT_DIM[0] / 2.) - 10  # (144,144) -> 62
                    tfs.append(SpatialTransformUsed(patch_size,
                                                patch_center_dist_from_border=center_dist_from_border,
                                                do_elastic_deform=self.Config.DAUG_ELASTIC_DEFORM,
                                                alpha=self.Config.DAUG_ALPHA, sigma=self.Config.DAUG_SIGMA,
                                                do_rotation=self.Config.DAUG_ROTATE,
                                                angle_x=self.Config.DAUG_ROTATE_ANGLE,
                                                angle_y=self.Config.DAUG_ROTATE_ANGLE,
                                                angle_z=self.Config.DAUG_ROTATE_ANGLE,
                                                do_scale=True, scale=scale, border_mode_data='constant',
                                                border_cval_data=0,
                                                order_data=3,
                                                border_mode_seg='constant', border_cval_seg=0,
                                                order_seg=0, random_crop=True,
                                                p_el_per_sample=self.Config.P_SAMP,
                                                p_rot_per_sample=self.Config.P_SAMP,
                                                p_scale_per_sample=self.Config.P_SAMP))

                if self.Config.DAUG_RESAMPLE:
                    tfs.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), p_per_sample=0.2, per_channel=False))

                if self.Config.DAUG_RESAMPLE_LEGACY:
                    tfs.append(ResampleTransformLegacy(zoom_range=(0.5, 1)))

                if self.Config.DAUG_GAUSSIAN_BLUR:
                    tfs.append(GaussianBlurTransform(blur_sigma=self.Config.DAUG_BLUR_SIGMA,
                                                     different_sigma_per_channel=False,
                                                     p_per_sample=self.Config.P_SAMP))

                if self.Config.DAUG_NOISE:
                    tfs.append(GaussianNoiseTransform(noise_variance=self.Config.DAUG_NOISE_VARIANCE,
                                                      p_per_sample=self.Config.P_SAMP))

                if self.Config.DAUG_MIRROR:
                    tfs.append(MirrorTransform())

                if self.Config.DAUG_FLIP_PEAKS:
                    tfs.append(FlipVectorAxisTransform())

        tfs.append(NumpyToTensor(keys=["data", "seg"], cast_to="float"))

        #num_cached_per_queue 1 or 2 does not really make a difference
        batch_gen = MultiThreadedAugmenter(batch_generator, Compose(tfs), num_processes=num_processes,
                                           num_cached_per_queue=1, seeds=None, pin_memory=True)
        return batch_gen  # data: (batch_size, channels, x, y), seg: (batch_size, channels, x, y)