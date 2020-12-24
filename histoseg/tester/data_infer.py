#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 20:03:35 2019

@author: siddhesh
"""

from __future__ import print_function, division, absolute_import
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from skimage.io import imsave
from torch.utils.data import DataLoader
from histoseg.models.utils import check_encoder, check_decoder, fetch_model
from histoseg.tester.test_dataloader import InferTumorSegDataset
from histoseg.tester.test_dataloader_jpg import InferTumorSegDatasetJpg
from openslide import OpenSlide
from tqdm import tqdm


def test_network(hparams):
    # Report the time stamp
    training_start_time = time.asctime()
    startstamp = time.time()
    print("\nHostname   :" + str(os.getenv("HOSTNAME")))
    print("\nStart Time :" + str(training_start_time))
    print("\nStart Stamp:" + str(startstamp))
    sys.stdout.flush()

    # Get weights from the network
    hparams = get_hparams_from_weights(hparams)

    # PRINT PARSED ARGS
    print("\n\n")
    print("Output directory        :", hparams["output_dir"])
    print("Encoder                 :", hparams["models"]["encoder"])
    print("Decoder                 :", hparams["models"]["decoder"])
    print("Number of channels      :", hparams["models"]["num_channels"])
    print("Number of classes       :", hparams["models"]["num_classes"])
    print("Modalities              :", hparams["slide"]["stain"])
    print("Site                    :", hparams["slide"]["site"])
    print("Encoder Depth           :", hparams["slide"]["layers"])
    print("Batch Size              :", hparams["slide"]["batch_size"])
    print("Patch Size              :", hparams["slide"]["patch_size"])
    print("Sampling Stride         :", hparams["slide"]["stride_size"])
    print("Decoder Base Filters    :", hparams["models"]["decoder_base_filters"])
    print("Load Weights            :", hparams["load_weights"])
    sys.stdout.flush()
    # We generate CSV for training if not provided
    print("Reading CSV Files")

    test_csv = hparams.test_csv

    # Check if encoder exists
    if check_encoder(hparams.encoder):
        print("Found Encoder!")
    else:
        print("Cannot find that encoder!")
        # sys.exit(0)

    # Check if decoder exists
    if check_decoder(hparams.decoder):
        print("Found Decoder!")
    else:
        print("Cannot find that decoder!")
        # sys.exit(0)

    # Fetching the model
    decoder_channels = [
        int(hparams.decoder_base_filters) * (2 ** i)
        for i in range(int(hparams.layers), 0, -1)
    ]
    model = fetch_model(
        decoder=hparams.decoder,
        encoder=hparams.encoder,
        encoder_depth=int(hparams.layers),
        encoder_weights=None,
        decoder_channels=decoder_channels,
        decoder_use_batchnorm="none",
        decoder_attention_type="scse",
        in_channels=int(hparams.num_channels),
        classes=int(hparams.num_classes) - 1,
        activation="sigmoid",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Loading weights
    weights = torch.load(hparams.load_weights)["model_state_dict"]
    model.load_state_dict(weights)
    print("Loaded Weights successfully.")
    sys.stdout.flush()

    # If cuda is required, the pushing the model to cuda
    if device != "cpu":
        model.cuda()
    model.eval()

    test_df = pd.read_csv(test_csv)
    # Patch blocks
    patch_dims = int(hparams.patch_size)
    # Patch Stride
    stride = int(hparams.stride_size)

    for index, row in test_df.iterrows():
        subject_name = row["PID"]
        print("Patient Slide       : ", row["PID"])
        print("Patient Location    : ", row["wsi_path"])
        os_image = OpenSlide(row["wsi_path"])
        level_2_width, level_2_height = os_image.level_dimensions[2]
        subject_dest_dir = os.path.join(hparams.output_dir, subject_name)
        os.makedirs(subject_dest_dir, exist_ok=True)

        probs_map = np.zeros((level_2_height, level_2_width), dtype=np.float16)
        count_map = np.zeros((level_2_height, level_2_width), dtype=np.uint8)

        if row["wsi_path"].endswith(".jpg"):
            patient_dataset_obj = InferTumorSegDatasetJpg(
                row["wsi_path"], patch_size=patch_dims, stride_size=stride
            )

        else:
            patient_dataset_obj = InferTumorSegDataset(
                row["wsi_path"],
                patch_size=patch_dims,
                stride_size=stride,
                selected_level=2,
                mask_level=4,
            )

        dataloader = DataLoader(
            patient_dataset_obj,
            batch_size=int(hparams.batch_size),
            shuffle=False,
            num_workers=2,
        )
        for image_patches, (x_coords, y_coords) in tqdm(dataloader):
            x_coords, y_coords = y_coords.numpy(), x_coords.numpy()
            with autocast():
                output = model(image_patches.half().cuda())
            output = output.cpu().detach().numpy()
            for i in range(int(output.shape[0])):
                count_map[
                    x_coords[i] : x_coords[i] + patch_dims,
                    y_coords[i] : y_coords[i] + patch_dims,
                ] += 1
                probs_map[
                    x_coords[i] : x_coords[i] + patch_dims,
                    y_coords[i] : y_coords[i] + patch_dims,
                ] += output[i][0]
        np.place(count_map, count_map == 0, 1)
        out_map = probs_map / count_map
        count_map = np.array(count_map * 255, dtype=np.uint16)
        out_thresh = np.array((out_map > 0.25) * 255, dtype=np.uint16)
        imsave(os.path.join(subject_dest_dir, row["PID"] + "_prob.png"), out_map)
        imsave(os.path.join(subject_dest_dir, row["PID"] + "_seg.png"), out_thresh)
        imsave(os.path.join(subject_dest_dir, row["PID"] + "_count.png"), count_map)
