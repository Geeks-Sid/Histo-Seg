#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 06:12:55 2020

@author: siddhesh
"""

import torch
import pytorch_lightning as ptl
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from histoseg.models.utils import FetchModel
from histoseg.utils.losses import focal_tversky_loss, dice
from histoseg.utils.optimizers import fetch_optimizer


class Segment(ptl.LightningModule):
    def __init__(self, hparams):
        super(Segment, self).__init__()
        self.hparams = hparams
        # Fetching the model
        decoder_channels = [
            int(hparams.decoder_base_filters) * (2 ** i)
            for i in range(int(hparams.layers), 0, -1)
        ]
        if hparams.use_imagenet == "True":
            encoder_weights = "imagenet"
        else:
            encoder_weights = None
        # Fetch a frozen model
        self.model = FetchModel(
            decoder=hparams.decoder,
            encoder=hparams.encoder,
            encoder_depth=int(hparams.layers),
            encoder_weights=encoder_weights,
            decoder_channels=decoder_channels,
            decoder_use_batchnorm="inplace",
            in_channels=int(hparams.num_channels),
            classes=int(hparams.num_classes),
            decoder_attention_type="scse",
            activation="sigmoid",
            encoder_freeze=False,
        )

    def forward(self, x):
        return self.model(x)

    def my_loss(self, output, mask):
        loss = focal_tversky_loss(output, mask)
        return loss

    def training_step(self, batch, batch_nb):
        image, mask = batch
        output = self.forward(image)
        loss = self.my_loss(output, mask)
        dice_score = dice(output, mask)
        return {"loss": loss, "dice": dice_score}

    def validation_step(self, batch, batch_nb):
        image, mask = batch
        output = self.forward(image)
        loss = self.my_loss(output, mask)
        dice_score = dice(output, mask)
        return {"val_loss": loss, "val_dice": dice_score}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_dice = torch.stack([x["val_dice"] for x in outputs]).mean()
        logs = {"avg_val_loss": avg_loss, "avg_val_dice": avg_dice}
        print("\nAverage validation loss :", avg_loss)
        print("Average validation dice", avg_dice)
        return {
            "val_loss": avg_loss,
            "val_dice": avg_dice,
            "progress_bar": logs,
            "log": logs,
        }

    def configure_optimizers(self):
        self.optimizer = fetch_optimizer(
            self.hparams.optimizer, self.hparams.learning_rate, self.model
        )
        self.lr_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        return [self.optimizer], [self.lr_scheduler]
