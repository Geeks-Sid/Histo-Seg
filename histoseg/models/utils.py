# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 17:36:02 2020

@author: Admin
"""

import sys
import segmentation_models_pytorch as smp


def _check_encoder(modelname):
    model_list = [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "resnext101_32x16d",
        "resnext101_32x32d",
        "resnext101_32x48d",
        "dpn68",
        "dpn68b",
        "dpn92",
        "dpn98",
        "dpn107",
        "dpn131",
        "vgg11",
        "vgg11_bn",
        "vgg13",
        "vgg13_bn",
        "vgg16",
        "vgg16_bn",
        "vgg19",
        "vgg19_bn",
        "senet154",
        "se_resnet50",
        "se_resnet101",
        "se_resnet152",
        "se_resnext50_32x4d",
        "se_resnext101_32x4d",
        "densenet121",
        "densenet169",
        "densenet201",
        "densenet161",
        "inceptionresnetv2",
        "inceptionv4",
        "efficientnet-b0",
        "efficientnet-b1",
        "efficientnet-b2",
        "efficientnet-b3",
        "efficientnet-b4",
        "efficientnet-b5",
        "efficientnet-b6",
        "efficientnet-b7",
        "timm-efficientnet-b0",
        "timm-efficientnet-b1",
        "timm-efficientnet-b2",
        "timm-efficientnet-b3",
        "timm-efficientnet-b4",
        "timm-efficientnet-b5",
        "timm-efficientnet-b6",
        "timm-efficientnet-b7",
        "mobilenet_v2",
        "xception",
    ]
    print("Checking encoders!", modelname)
    if modelname in model_list:
        return True
    else:
        print("Fail")
        return False


def _check_decoder(modelname):
    model_list = ["unet", "linknet", "fpn", "pspnet", "pan"]
    if modelname.lower() in model_list:
        return True
    else:
        return False


def FetchModel(
    decoder="resnet34",
    encoder="unet",
    encoder_depth=5,
    encoder_weights="imagenet",
    decoder_channels=[256, 128, 64, 32, 16],
    decoder_use_batchnorm="inplace",
    decoder_attention_type="scse",
    in_channels=3,
    classes=2,
    activation="softmax",
    encoder_freeze=False,
):
    if _check_encoder(encoder) and _check_decoder(decoder):
        if decoder.lower() == "unet":
            model = smp.Unet(
                encoder,
                encoder_depth=encoder_depth,
                encoder_weights=encoder_weights,
                decoder_use_batchnorm=decoder_use_batchnorm,
                decoder_channels=decoder_channels,
                decoder_attention_type=decoder_attention_type,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
            )
        elif decoder.lower() == "linknet":
            model = smp.Linknet(
                encoder,
                encoder_depth=encoder_depth,
                encoder_weights=encoder_weights,
                decoder_use_batchnorm=decoder_use_batchnorm,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
            )
        elif decoder.lower() == "fpn":
            model = smp.FPN(
                encoder,
                encoder_depth=encoder_depth,
                encoder_weights=encoder_weights,
                decoder_pyramid_channels=decoder_channels[0],
                decoder_segmentation_channels=decoder_channels[1],
                decoder_merge_policy="add",
                decoder_dropout=0.2,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
            )
        elif decoder.lower() == "pspnet":
            model = smp.PSPNet(
                encoder,
                encoder_depth=encoder_depth,
                encoder_weights=encoder_weights,
                psp_out_channels=decoder_channels[0] * 2,
                psp_use_batchnorm=True,
                psp_dropout=0.2,
                in_channels=in_channels,
                classes=classes,
                upsampling=2 ** encoder_depth,
                activation=activation,
            )
        elif decoder.lower() == "pan":
            model = smp.PAN(
                encoder,
                encoder_weights=encoder_weights,
                encoder_dilation=True,
                decoder_channels=decoder_channels[-1],
                in_channels=in_channels,
                classes=classes,
                activation=activation,
            )
        else:
            print("Model Does not exist! Check Log.")
            sys.exit(0)
        if encoder_freeze:
            for param in model.encoder.parameters():
                param.requires_grad = False
            return model
        else:
            return model
