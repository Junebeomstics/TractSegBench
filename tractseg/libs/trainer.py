from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import time
import pickle
import socket
import datetime
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from pprint import pprint
import nibabel as nib

from tractseg.libs import exp_utils
from tractseg.libs import metric_utils
from tractseg.libs import plot_utils
from tractseg.libs import utils
from tractseg.data.data_loader_inference import DataLoaderInference
from tractseg.data import dataset_specific_utils, datasets
from tractseg.data.data_loader_training import DataLoaderTraining as DataLoaderTraining2D
from tractseg.data.data_loader_training_3D import DataLoaderTraining as DataLoaderTraining3D
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

#from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
import torch

# new
from tractseg.data.datasets import MRISliceDataset, transform_data
from torch.utils.data import Dataset, DataLoader
from batchgenerators.transforms.abstract_transforms import Compose

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler


def _get_weights_for_this_epoch(Config, epoch_nr):
    if Config.LOSS_WEIGHT is None:
        weight_factor = None
    elif Config.LOSS_WEIGHT_LEN == -1:
        weight_factor = float(Config.LOSS_WEIGHT)
    else:
        # Linearly decrease from LOSS_WEIGHT to 1 over LOSS_WEIGHT_LEN epochs
        if epoch_nr < Config.LOSS_WEIGHT_LEN:
            weight_factor = -((Config.LOSS_WEIGHT - 1) /
                              float(Config.LOSS_WEIGHT_LEN)) * epoch_nr + float(Config.LOSS_WEIGHT)
        else:
            weight_factor = 1.
        exp_utils.print_and_save(Config.EXP_PATH, "Current weight_factor: {}".format(weight_factor))
    return weight_factor

def train_model(Config, model, run, scheduler=None):
    epoch_times = []
    nr_of_updates = 0

    metrics = {}
    for type in ["train","validate", "test"]:
        for metric in Config.METRIC_TYPES:
            metrics[metric + "_" + type] = [0]
        # initialize f1 macro metrics for each bundle
        if Config.LOG_PER_BUNDLE:
            for i, bundle_name in enumerate(dataset_specific_utils.get_bundle_names(Config.CLASSES)[1:]):
                metrics[f'bundle_{bundle_name}_f1_{type}' ] = [0]
                if f'bundle_{bundle_name}_f1' not in Config.METRIC_TYPES:
                    Config.METRIC_TYPES.append(f'bundle_{bundle_name}_f1') # avoid duplicates
    
    if Config.USE_NEW_DATALOADER:
        # Define dataloaders 
        tfs_train = Compose(transform_data(Config, type='train'))
        tfs_val = Compose(transform_data(Config, type='val'))
        
        train_dataset = MRISliceDataset(Config, subjects=getattr(Config, "TRAIN_SUBJECTS"), transform=tfs_train)
        val_dataset = MRISliceDataset(Config, subjects=getattr(Config, "VALIDATE_SUBJECTS"), transform=tfs_val)
        test_dataset = MRISliceDataset(Config, subjects=getattr(Config, "TEST_SUBJECTS"), transform=tfs_val)

        if Config.distributed:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
            test_sampler = DistributedSampler(test_dataset, shuffle=False)
        else:
            train_sampler = RandomSampler(train_dataset)
            val_sampler = None
            test_sampler = None
        
        batch_gen_train = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_PROCESSES, pin_memory=True, sampler=train_sampler, persistent_workers=True)
        batch_gen_val = DataLoader(val_dataset, batch_size=Config.VAL_BATCH_SIZE, shuffle=(val_sampler is None), num_workers=Config.NUM_PROCESSES, pin_memory=True, sampler=val_sampler, persistent_workers=True)
        batch_gen_test = DataLoader(test_dataset, batch_size=Config.VAL_BATCH_SIZE, shuffle=(test_sampler is None), num_workers=Config.NUM_PROCESSES, pin_memory=True, sampler=test_sampler, persistent_workers=True)
    else:
        if Config.DIM == "2D":
            data_loader = DataLoaderTraining2D(Config) 
        else:
            data_loader = DataLoaderTraining3D(Config)
        batch_gen_train = data_loader.get_batch_generator(batch_size=Config.BATCH_SIZE, type="train",
                                                        subjects=getattr(Config, "TRAIN_SUBJECTS")) # BATCH_SIZE is not used in real.
        batch_gen_val = data_loader.get_batch_generator(batch_size=Config.BATCH_SIZE, type="validate",
                                                        subjects=getattr(Config, "VALIDATE_SUBJECTS"))
        batch_gen_test = data_loader.get_batch_generator(batch_size=Config.BATCH_SIZE, type="test",
                                                        subjects=getattr(Config, "TEST_SUBJECTS"))

    types = ["validate"] if Config.ONLY_VAL else ["train", "validate", "test"]
    if Config.ONLY_VAL:
        Config.NUM_EPOCHS = 1

    for epoch_nr in range(Config.BEST_EPOCH,Config.NUM_EPOCHS):
        start_time = time.time()

        timings = defaultdict(lambda: 0) 
        batch_nr = defaultdict(lambda: 0) # 매 epoch마다 초기화
        weight_factor = _get_weights_for_this_epoch(Config, epoch_nr) # weight_factor = None

        if Config.distributed:
            all_predictions = []
            all_labels = []
        
        for type in types:
            print(f"Start looping {type} batches in epoch {epoch_nr}...")
            print_loss = []
            start_time_batch_part = time.time()
            if Config.USE_NEW_DATALOADER:
                if type == "train":
                    loader = batch_gen_train
                    if Config.distributed:
                        # Set epoch for distributed sampler
                        loader.sampler.set_epoch(epoch_nr)
                elif type == "validate":
                    loader = batch_gen_val
                    if Config.distributed:
                        # Also set epoch for validation sampler to ensure proper shard distribution
                        loader.sampler.set_epoch(epoch_nr)
                elif type == "test":
                    loader = batch_gen_test  
                    if Config.distributed:
                        # Set epoch for test sampler too
                        loader.sampler.set_epoch(epoch_nr) 

                for batch in loader:
                    start_time_data_preparation = time.time()
                    batch_nr[type] += 1
                    subject = batch["subject"]
                    x = batch["data"]
                    y = batch["seg"]  # (bs, nr_of_classes, x, y)
                    
                    utils.check_tensor_values(subject, x, y)
                
                    if Config.MODEL == "LatentDiffusionModel":
                        # Sample random timesteps
                        timesteps = torch.randint(0, scheduler.num_train_timesteps, (x.size(0),), device=x.device)

                        # Add noise to inputs (forward diffusion)
                        noise = torch.randn_like(x)
                        x = scheduler.add_noise(x, noise, timesteps)
                        y = noise
                    else:
                        noise = None
                        timesteps = None

                    timings["data_preparation_time"] += time.time() - start_time_data_preparation
                    start_time_network = time.time()
                    #nr_of_updates += 1
                    
                    # base_model
                    probs, metr_batch = model.train(x, y, weight_factor=weight_factor, timesteps=timesteps, type=type)
                    timings["network_time"] += time.time() - start_time_network
                    start_time_metrics = time.time()
                    # if Config.distributed:
                    #     # Store the predictions and labels for later metric calculation
                    #     predictions_list.append(probs.detach())  
                    #     labels_list.append(y.detach())
                    # else:
                    metrics = metric_utils._update_metrics(Config.CALC_F1, Config.EXPERIMENT_TYPE, Config.METRIC_TYPES,
                                            metrics, metr_batch, type)
                    timings["metrics_time"] += time.time() - start_time_metrics

                    print_loss.append(metr_batch["loss"])
                    if batch_nr[type] % Config.PRINT_FREQ == 0:
                        time_batch_part = time.time() - start_time_batch_part
                        start_time_batch_part = time.time()
                        exp_utils.print_and_save(Config.EXP_PATH, "{} Ep {}, Sp {}, loss {}, t print {}s, t batch {}s".format(
                            type, epoch_nr, batch_nr[type] * Config.BATCH_SIZE * Config.NR_SLICES, round(np.array(print_loss).mean(), 6),
                            round(time_batch_part, 3), round( time_batch_part / Config.PRINT_FREQ, 3)))
                        print_loss = []     
            else: # batchgenerator
                #*Config.EPOCH_MULTIPLIER needed to have roughly same number of updates/batches as with 2D U-Net (only valid for 3D)   
                if Config.DIM == "2D":
                    nr_of_samples = len(getattr(Config, type.upper() + "_SUBJECTS")) * Config.INPUT_DIM[0] # INPUT_DIM[0] = 144 for HCP data 1.25mm
                    nr_batches = int(int(nr_of_samples / Config.NR_SLICES) * Config.EPOCH_MULTIPLIER)
                else:
                    nr_of_samples = len(getattr(Config, type.upper() + "_SUBJECTS"))
                    if type == "train":
                        nr_batches = int(int(nr_of_samples / Config.NR_SLICES) * Config.EPOCH_MULTIPLIER) # 일반적인 1 epoch 세팅에 비해 2배 * 3배 = 6배 많은 iteration.
                        #nr_batches = int(int(nr_of_samples / Config.BATCH_SIZE) * Config.EPOCH_MULTIPLIER) #<- 이게 올바른 형태 (Config.EPOCH_MULTIPLIER = 1)
                    else:
                        nr_batches = int(int(nr_of_samples / Config.BATCH_SIZE)) # 실제 sample 만큼의 숫자
                for i in range(nr_batches):  
                    if type == "train":
                        batch = next(batch_gen_train)
                    elif type == "validate":
                        batch = next(batch_gen_val)
                    elif type == "test":
                        batch = next(batch_gen_test)

                    start_time_data_preparation = time.time()
                    batch_nr[type] += 1

                    subject = batch["subject"]
                    x = batch["data"]  # (nr_slices, nr_of_channels, x, y) for 2D and (batch_size, nr_of_channels, x, y, z) for 3D 
                    y = batch["seg"]  # (nr_slices, nr_of_classes, x, y) for 2D and (batch_size, nr_of_classes, x, y, z) for 3D 
                    utils.check_tensor_values(subject, x, y)

                    if Config.DIM == "2D":
                        # currently, multiple subjects are not implemented for 2D batchgenerator, so please add batch dimension for subject
                        x = torch.unsqueeze(x,0) # (1, nr_slices, nr_of_channels, x, y) # batch size = 1
                        y = torch.unsqueeze(y,0)  # (1, nr_slices, nr_classes, x, y) # batch size = 1


                    if Config.MODEL == "LatentDiffusionModel":
                        # Sample random timesteps
                        timesteps = torch.randint(0, scheduler.num_train_timesteps, (x.size(0),), device=x.device)

                        # Add noise to inputs (forward diffusion)
                        noise = torch.randn_like(x)
                        x = scheduler.add_noise(x, noise, timesteps)
                        y = noise
                    else:
                        noise = None
                        timesteps = None

                    timings["data_preparation_time"] += time.time() - start_time_data_preparation
                    start_time_network = time.time()
                    #nr_of_updates += 1
                    
                    # base_model
                    probs, metr_batch = model.train(x, y, weight_factor=weight_factor, timesteps=timesteps, type=type)
                    timings["network_time"] += time.time() - start_time_network
                    start_time_metrics = time.time()
                    metrics = metric_utils._update_metrics(Config.CALC_F1, Config.EXPERIMENT_TYPE, Config.METRIC_TYPES,
                                            metrics, metr_batch, type)
                    timings["metrics_time"] += time.time() - start_time_metrics

                    print_loss.append(metr_batch["loss"])
                    if batch_nr[type] % Config.PRINT_FREQ == 0:
                        time_batch_part = time.time() - start_time_batch_part
                        start_time_batch_part = time.time()
                        exp_utils.print_and_save(Config.EXP_PATH, "{} Ep {}, Sp {}, loss {}, t print {}s, t batch {}s".format(
                            type, epoch_nr, batch_nr[type] * Config.BATCH_SIZE * Config.NR_SLICES, round(np.array(print_loss).mean(), 6),
                            round(time_batch_part, 3), round( time_batch_part / Config.PRINT_FREQ, 3)))
                        print_loss = []     

            
        ################################### Post Training tasks (each epoch) ###################################

        if Config.ONLY_VAL:
            metrics = metric_utils.normalize_last_element(metrics, batch_nr["validate"], type="validate")
            print("f1 macro validate: {}".format(round(metrics["f1_macro_validate"][0], 4)))
            return model


        # if Config.distributed:
        #     # Concatenate all predictions and labels from this process
        #     if predictions_list and labels_list:  # Check if lists are not empty
        #         all_predictions = torch.cat(predictions_list, dim=0)
        #         all_labels = torch.cat(labels_list, dim=0)
                
        #         # Calculate metrics using the new function
        #         metr_batch = metric_utils.calculate_metrics_for_ddp(Config, model, all_predictions, all_labels, model.device)
                
        #         # Update metrics
        #         metrics = metric_utils._update_metrics_ddp(Config, metrics, metr_batch, type)

        # Average losses of batches in the epoch
        # -1 element of metrics is the summed loss/f1 for batches in the current epoch
        # by normalizing it with the number of batches, we get the average loss/f1 for the epoch
        metrics = metric_utils.normalize_last_element(metrics, batch_nr["train"], type="train")
        metrics = metric_utils.normalize_last_element(metrics, batch_nr["validate"], type="validate")
        metrics = metric_utils.normalize_last_element(metrics, batch_nr["test"], type="test")

        # gather metrics across all processes for validation results
        # if Config.distributed:
        #     # Make sure metrics are properly synchronized across processes
        #     # First ensure all processes have finished their computations
        #     torch.distributed.barrier()

        #     # Synchronize metrics across processes if using distributed training
        #     for key in metrics:
        #         if isinstance(metrics[key][-1], (int, float, np.int32, np.int64, np.float32, np.float64)):
        #             # Convert to tensor for all-reduce operation
        #             tensor = torch.tensor(metrics[key][-1]).to(model.device)
        #             torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        #             # Average the metric across all processes
        #             metrics[key][-1] = tensor.item() / torch.distributed.get_world_size()
            
        #     # Add another barrier to ensure all processes have updated metrics
        #     torch.distributed.barrier()

        # Log metrics
        if run is not None and (not Config.distributed or (Config.distributed and torch.distributed.get_rank() == 0)):
            for key,value in metrics.items():
                run[key].log(metrics[key][-1])
        
            # Log learning rate
            run['learning_rate'].log(model.optimizer.param_groups[0]['lr'])
        print("  Epoch {}, Average Epoch loss = {}".format(epoch_nr, metrics["loss_train"][-1]))
        # exp_utils.print_and_save(Config.EXP_PATH, "  Epoch {}, nr_of_updates {}".format(epoch_nr, nr_of_updates))

        if Config.distributed:
            torch.distributed.barrier(device_ids=[model.device.index])


        # Adapt LR
        if Config.LR_SCHEDULE and Config.LR_SCHEDULE_TYPE == "ReduceLROnPlateau":
            if Config.LR_SCHEDULE_MODE == "min":
                model.scheduler.step(metrics["loss_validate"][-1])
            else:
                model.scheduler.step(metrics["f1_macro_validate"][-1])
            model.print_current_lr()
        elif Config.LR_SCHEDULE and Config.LR_SCHEDULE_TYPE == "CosineAnnealingLR":
            model.scheduler.step()
            model.print_current_lr()

        if Config.distributed:
            torch.distributed.barrier(device_ids=[model.device.index])

        # Save Weights
        start_time_saving = time.time()
        if Config.SAVE_WEIGHTS:
            # save model and save optimizer state
            if not Config.distributed or (Config.distributed and torch.distributed.get_rank() == 0):
                model.save_model(metrics, epoch_nr, mode=Config.BEST_EPOCH_SELECTION)
        timings["saving_time"] += time.time() - start_time_saving

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        exp_utils.print_and_save(Config.EXP_PATH, "  Epoch {}, time total {}s".format(epoch_nr, epoch_time))
        exp_utils.print_and_save(Config.EXP_PATH, "  Epoch {}, time UNet: {}s".format(epoch_nr, timings["network_time"]))
        exp_utils.print_and_save(Config.EXP_PATH, "  Epoch {}, time metrics: {}s".format(epoch_nr, timings["metrics_time"]))
        exp_utils.print_and_save(Config.EXP_PATH, "  Epoch {}, time saving files: {}s".format(epoch_nr, timings["saving_time"]))
        exp_utils.print_and_save(Config.EXP_PATH, str(datetime.datetime.now()))

        # Adding next Epoch
        if epoch_nr < Config.NUM_EPOCHS-1:
            metrics = metric_utils.add_empty_element(metrics)

def predict_img(Config, model, data_loader, probs=False, scale_to_world_shape=True, only_prediction=False,
                batch_size=1, unit_test=False):
    """
    Return predictions for one 3D image.

    Runtime on CPU
    - python 2 + pytorch 0.4:
          bs=1  -> 9min      ~7GB RAM
          bs=48 -> 6.5min    ~30GB RAM
    - python 3 + pytorch 1.0:
          bs=1  -> 2.7min    ~7GB RAM
    """
    def _finalize_data(layers):
        layers = np.array(layers)

        if Config.DIM == "2D":
            # Get in right order (x,y,z) and
            if Config.SLICE_DIRECTION == "x":
                layers = layers.transpose(0, 1, 2, 3)

            elif Config.SLICE_DIRECTION == "y":
                layers = layers.transpose(1, 0, 2, 3)

            elif Config.SLICE_DIRECTION == "z":
                layers = layers.transpose(1, 2, 0, 3)

        if scale_to_world_shape:
            layers = dataset_specific_utils.scale_input_to_original_shape(layers, Config.DATASET, Config.RESOLUTION)
        # assert (layers.dtype == np.float32)
        return layers

    img_shape = [Config.INPUT_DIM[0], Config.INPUT_DIM[0], Config.INPUT_DIM[0], Config.NR_OF_CLASSES]
    layers_seg = np.empty(img_shape).astype(np.float32)
    layers_y = None if only_prediction else np.empty(img_shape).astype(np.float32)

    if unit_test:
        # Return some mockup data to test different input arguments end 2 end and to test the postprocessing of the
        # segmentations (using real segmentations on DWI test image would take too much time if we run it for
        # different configurations)
        probs = np.zeros(img_shape).astype(np.float32)

        # CA (bundle specific postprocessing)
        probs[10:30, 10:30, 10:30, 4] = 0.7  # big blob 1
        probs[10:30, 10:30, 40:50, 4] = 0.7  # big blob 2
        probs[20:25, 20:25, 30:34, 4] = 0.4  # incomplete bridge between blobs with lower probability
        probs[20:25, 20:25, 36:40, 4] = 0.4  # incomplete bridge between blobs with lower probability
        probs[50:55, 50:55, 50:55, 4] = 0.2  # below threshold
        probs[60:63, 60:63, 60:63, 4] = 0.9  # small blob -> will get removed by postprocessing
        # should restore the bridge

        # CC_1
        probs[10:30, 10:30, 10:30, 5] = 0.7  # big blob 1
        probs[10:30, 10:30, 40:50, 5] = 0.7  # big blob 2
        probs[20:25, 20:25, 30:34, 5] = 0.4  # incomplete bridge between blobs with lower probability
        probs[20:25, 20:25, 36:40, 5] = 0.4  # incomplete bridge between blobs with lower probability
        probs[50:55, 50:55, 50:55, 5] = 0.2  # below threshold
        probs[60:63, 60:63, 60:63, 5] = 0.9  # small blob -> will get removed by postprocessing
        # should not restore the bridge

        return probs, layers_y

    batch_generator = data_loader.get_batch_generator(batch_size=batch_size)
    batch_generator = list(batch_generator)
    idx = 0
    for batch in tqdm(batch_generator):
        x = batch["data"]   # (bs, nr_channels, x, y, (z))
        y = batch["seg"]    # (bs, nr_classes, x, y, (z))
        y = y.numpy()

        if not only_prediction:
            y = y.astype(Config.LABELS_TYPE)
            if Config.DIM == "2D":
                y = y.transpose(0, 2, 3, 1) # (bs, x, y, nr_classes)
            else:
                y = y.transpose(0, 2, 3, 4, 1) # (bs, x, y, z, nr_classes)

        if Config.DROPOUT_SAMPLING:
            # For Dropout Sampling (must set deterministic=False in model)
            NR_SAMPLING = 30
            samples = []
            for i in range(NR_SAMPLING):
                layer_probs = model.predict(x)  # (bs, x, y, nr_classes)
                samples.append(layer_probs)

            samples = np.array(samples)  # (NR_SAMPLING, bs, x, y, nr_classes)
            layer_probs = np.std(samples, axis=0)    # (bs, x, y, nr_classes)
        else:
            # For normal prediction
            layer_probs = model.predict(x)  # (bs, x, y, nr_classes)

        if probs:
            seg = layer_probs   # (x, y, nr_classes)
        else:
            seg = layer_probs
            seg[seg >= Config.THRESHOLD] = 1
            seg[seg < Config.THRESHOLD] = 0
            seg = seg.astype(np.uint8)

        if Config.DIM == "2D":
            layers_seg[idx*batch_size:(idx+1)*batch_size, :, :, :] = seg
            if not only_prediction:
                layers_y[idx*batch_size:(idx+1)*batch_size, :, :, :] = y
        else:
            layers_seg = np.squeeze(seg)
            if not only_prediction:
                layers_y = np.squeeze(y)

        idx += 1

    layers_seg = _finalize_data(layers_seg)
    if not only_prediction:
        layers_y = _finalize_data(layers_y)
    return layers_seg, layers_y


def test_whole_subject(Config, model, run, subjects, type):

    metrics = {
        "loss_" + type: [0],
        "f1_macro_" + type: [0],
    }

    metrics_bundles = defaultdict(lambda: [0])
    f1_scores = []
    for subject in subjects:
        print("{} subject {}".format(type, subject))
        start_time = time.time()

        data_loader = DataLoaderInference(Config, subject=subject)
        img_probs, img_y = predict_img(Config, model, data_loader, probs=True, scale_to_world_shape=False)
        # img_probs_xyz, img_y = DirectionMerger.get_seg_single_img_3_directions(Config, model, subject=subject)
        # img_probs = DirectionMerger.mean_fusion(Config.THRESHOLD, img_probs_xyz, probs=True)

        print("Took {}s".format(round(time.time() - start_time, 2)))
        img_probs = np.reshape(img_probs, (-1, img_probs.shape[-1]))  # Flatten all dims except nr_classes dim
        img_y = np.reshape(img_y, (-1, img_y.shape[-1]))
        metrics, subj_f1_score = metric_utils.calculate_metrics(metrics, img_y, img_probs, 0,
                                                    type=type, threshold=Config.THRESHOLD, return_subj_f1=True)
        f1_scores.append(subj_f1_score)
        metrics_bundles = metric_utils.calculate_metrics_each_bundle(metrics_bundles, img_y, img_probs,
                                                                        dataset_specific_utils.get_bundle_names(Config.CLASSES)[1:],
                                                                         threshold=Config.THRESHOLD)
        

    metrics = metric_utils.normalize_last_element(metrics, len(subjects), type=type)
    metrics_bundles = metric_utils.normalize_last_element_general(metrics_bundles, len(subjects))

    print("WHOLE SUBJECT:")
    pprint(metrics)
    for key,value in metrics.items():
        run['whole_'+key].log(metrics[key][-1])

    print("WHOLE SUBJECT BUNDLES:")
    pprint(metrics_bundles)
    for key, value in metrics_bundles.items():
        run['whole_' + key].log(metrics_bundles[key][-1])

    # Only write files on rank 0 to avoid conflicts
    if not Config.distributed or (Config.distributed and torch.distributed.get_rank() == 0):
        with open(join(Config.EXP_PATH, "score_" + type + "-set.txt"), "w") as f:
            pprint(metrics, f)
            f.write("\n\nWeights: {}\n".format(Config.WEIGHTS_PATH))
            f.write("type: {}\n\n".format(type))
            pprint(metrics_bundles, f)
        pickle.dump(metrics, open(join(Config.EXP_PATH, "score_" + type + str(Config.CV_FOLD)+ ".pkl"), "wb"))
        pickle.dump(f1_scores, open(join(Config.EXP_PATH, "subj_f1_scores_" + type + str(Config.CV_FOLD) + ".pkl"), "wb"))
    return metrics
