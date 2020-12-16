#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 06:08:37 2020

@author: siddhesh
"""


import os
import time
import sys
import torch
import pandas as pd
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from histoseg.trainer.lightning_networks import Segment
from histoseg.data_module import SegDataModule

# Gotta set it to 42
seed_everything(42)

def train_network(hparams):
    """
    Receiving a configuration file and a device, the training is pushed through this file

    Parameters
    ----------
    cfg : TYPE
        DESCRIPTION.
    device : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    training_start_time = time.asctime()
    startstamp = time.time()
    print("\nHostname   :" + str(os.getenv("HOSTNAME")))
    print("\nStart Time :" + str(training_start_time))
    print("\nStart Stamp:" + str(startstamp))
    sys.stdout.flush()
    # READING FROM A CFG FILE and check if file exists or not
    
    # Although uneccessary, we still do this
    if not os.path.isdir(str(hparams.model_dir)):
        os.mkdir(hparams.model_dir)

    # PRINT PARSED ARGS
    print("\n\n")
    print("Model Dir               :", hparams.model_dir)
    print("Training CSV            :", hparams.train_csv)
    print("Validation CSV          :", hparams.validation_csv)
    print("Number of channels      :", hparams.num_channels)
    print("Modalities              :", hparams.modalities)
    print("Encoder                 :", hparams.encoder)
    print("Decoder                 :", hparams.decoder)
    print("Decoder Base Filters    :", hparams.decoder_base_filters)
    print("Number of classes       :", hparams.num_classes)
    print("Use ImageNet Pretrained :", hparams.use_imagenet)
    print("Freeze Encoder epochs   :", hparams.encoder_freeze_epochs)
    print("Max. Number of epochs   :", hparams.max_epochs)
    print("Batch Size              :", hparams.batch_size)
    print("Optimizer               :", hparams.optimizer)
    print("Learning Rate           :", hparams.learning_rate)
    print("Learning Rate Milestones:", hparams.lr_milestones)
    print("Patience to decay       :", hparams.decay_milestones)
    print("Early Stopping Patience :", hparams.early_stop_patience)
    print("Encoder Depth           :", hparams.layers)
    print("Save top k              :", hparams.save_top_k)
    sys.stdout.flush()
    print("Device Given :", hparams.device)
    sys.stdout.flush()
    # Although uneccessary, we still do this
    print("Current Device : ", torch.cuda.current_device())
    print("Device Count on Machine : ", torch.cuda.device_count())
    print("Device Name : ", torch.cuda.get_device_name())
    print("Cuda Availibility : ", torch.cuda.is_available())
    print('Using device:', hparams.device, type(hparams.device))
    if hparams.device != 'cpu':
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1),
              'GB')
        print('Cached: ', round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')
    sys.stdout.flush()

    try:
        log_dir = sorted(os.listdir(hparams.model_dir))[-1]
    except IndexError:
        log_dir = os.path.join(hparams.model_dir, 'version_0')
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(log_dir,
                                                                'checkpoints'),
                                          monitor='val_loss',
                                          verbose=True,
                                          save_top_k=int(hparams.save_top_k),
                                          mode='auto',
                                          save_weights_only=False,
                                          prefix=str(hparams.encoder+'_'+\
                                                     hparams.decoder+'_'+\
                                                     str(hparams.decoder_base_filters)+'_')
                                          )
    stop_callback = EarlyStopping(monitor='val_loss', mode='auto',
                                  patience=int(hparams.early_stop_patience),
                                  verbose=True)
    segmentation_model = Segment(hparams)
    segmentation_datamodule = SegDataModule(hparams)

    trainer = Trainer(
                      checkpoint_callback=True,
                      callbacks=[checkpoint_callback, stop_callback],
                      default_root_dir=hparams.model_dir,
                      gpus=str(hparams.device),
                      max_epochs=int(hparams.max_epochs),
                      min_epochs=int(hparams.min_epochs),
                      accelerator='ddp',
                      precision=16,  # Do you need 16 bit?
                      weights_summary='full',
                      weights_save_path=hparams.model_dir,
                      amp_level='O1',
                      num_sanity_val_steps=5,
                      resume_from_checkpoint=hparams.weights,
                      benchmark=True
                      )

    trainer.fit(model=segmentation_model, datamodule=segmentation_datamodule)
