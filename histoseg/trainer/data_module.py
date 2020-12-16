#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 18:58:38 2020

@author: siddhesh
"""

import os
from skimage.io import imread
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as ptl
from albumentations import (
    RandomBrightnessContrast,
    HueSaturationValue,
    RandomGamma,
    GaussNoise,
    GaussianBlur,
    HorizontalFlip,
    VerticalFlip,
    Compose,
)


class SegDataset(Dataset):
    """
    """

    def __init__(self, csv_file, valid=False):
        self.data_frame = pd.read_csv(csv_file, header=0)
        self.valid = valid
        self.aug = Compose(
            [
                HorizontalFlip(),
                VerticalFlip(),
                RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4),
                HueSaturationValue(
                    hue_shift_limit=30, sat_shift_limit=45, val_shift_limit=30
                ),
                RandomGamma(gamma_limit=(80, 120)),
                GaussNoise(var_limit=(10, 200)),
                GaussianBlur(blur_limit=11),
            ]
        )
        self.valid_aug = Compose([HorizontalFlip(), VerticalFlip()])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, patient_id):
        image_path = os.path.join(self.data_frame.iloc[patient_id, 0])
        gt_path = os.path.join(self.data_frame.iloc[patient_id, 1])
        image = imread(image_path)
        gt_data = imread(gt_path)
        gt_data = gt_data.astype(np.uint8)
        gt_data = gt_data[np.newaxis, ...]
        gt_data = gt_data.transpose([1, 2, 0])
        if not self.valid:
            augmented = self.aug(image=image, mask=gt_data)
            image = augmented["image"]
            gt_data = augmented["mask"]
        else:
            valid_augmented = self.valid_aug(image=image, mask=gt_data)
            image = valid_augmented["image"]
            gt_data = valid_augmented["mask"]
        image = image.transpose([2, 0, 1])
        gt_data = gt_data.transpose([2, 0, 1])
        image = (image) / 255
        image = torch.FloatTensor(image)
        gt_data = torch.FloatTensor(gt_data)

        return image, gt_data


class SegDataModule(ptl.LightningDataModule):
    """
    """

    def __init__(self, hparams):
        super().__init__()
        self.batch_size = hparams.batch_size
        self.hparams = hparams

    def train_dataloader(self):
        dataset_train = SegDataset(self.hparams.train_csv)
        return DataLoader(
            dataset_train, batch_size=int(self.batch_size), shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        dataset_valid = SegDataset(self.hparams.validation_csv, valid=True)
        return DataLoader(
            dataset_valid,
            batch_size=int(self.batch_size) * 2,
            shuffle=False,
            num_workers=4,
        )
