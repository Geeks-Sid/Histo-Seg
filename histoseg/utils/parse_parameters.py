#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:45:05 2020

@author: siddhesh
"""

import os
import sys
import numpy as np
import yaml
import pkg_resources
from histoseg.models.utils import check_encoder, check_decoder


def parse_version(version_string):
    """
    Parses version string, discards last identifier (NR/alpha/beta) and returns an integer for comparison
    """
    version_string_split = version_string.split(".")
    if len(version_string_split) > 3:
        del version_string_split[-1]
    return int("".join(version_string_split))


def initialize_keys(parameters, key):
    """
    This function will initialize the key in the parameters dict to 'None' if it is absent or length is zero
    """
    if key in parameters:
        if len(parameters[key]) == 0:  # if key is present but not defined
            parameters[key] = None
        else:
            parameters[key] = None  # if key is absent

    return parameters


def parseTrainConfig(config_file_path):
    """
    This function parses the configuration file and returns a dictionary of parameters
    """
    print("***** PARSING THE CONFIG FILE *****")

    print("\n Step 1 : Checking for the Version")

    with open(config_file_path) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    if not ("version" in params):
        sys.exit(
            "The 'version' key needs to be defined in config with 'minimum' and 'maximum' fields to determine the compatibility of configuration with code base."
        )
    else:
        histoseg_version = pkg_resources.require("histoseg")[0].version
        histoseg_version_int = parse_version(histoseg_version)
        min = parse_version(params["version"]["minimum"])
        max = parse_version(params["version"]["maximum"])
        if (min > histoseg_version_int) or (max < histoseg_version_int):
            sys.exit(
                "Incompatible version of histoseg detected (" + histoseg_version + ")"
            )
    print("\n Version checks out for the current configuration : ", histoseg_version)

    print("\n Step 2 : Checking the model parameters")

    print("\n Step 2 - A : Checking the Encoder")

    # Checking for the encoder and if assignment is possible
    if "encoder" in params["model"]:
        if check_encoder(params["model"]["encoder"]):
            params["model"]["encoder"] = params["model"]["encoder"]
            print("Encoder : ", params["model"]["encoder"])
        else:
            print("Encoder : ", params["model"]["encoder"])
            sys.exit(
                "There seems to be a spelling mistake in your encoder or we aren't currently supporting it. Please try with a different one."
            )
    else:
        params["model"]["encoder"] = "resnet18"
        print(
            "Since no 'encoder' was selected, defaulting to :",
            params["model"]["encoder"],
        )

    print("\n Step 2 - B : Checking the Decoder")

    # Checking for the decoder and if assignment is possible
    if "decoder" in params["model"]:
        if check_decoder(params["model"]["decoder"]):
            params["model"]["decoder"] = params["model"]["decoder"]
            print("Decoder : ", params["model"]["decoder"])
        else:
            print("Decoder : ", params["model"]["decoder"])
            sys.exit(
                "There seems to be a spelling mistake in your decoder or we aren't currently supporting it. Please try with a different one."
            )
    else:
        params["model"]["decoder"] = "unet"
        print(
            "Since no 'decoder' was selected, defaulting to :",
            params["model"]["decoder"],
        )

    print("\n Step 2 - C : Checking the Number of channels")

    # Checking for the number of channels
    if "num_channels" in params["models"]:
        params["models"]["num_channels"] = params["models"]["num_channels"]
        if num_channels != 3:
            print(
                "Since Number of channels are not set to 3, the imagenet weights in the first layer will be randomized."
            )
    else:
        sys.exit(
            "num_channels is a required parameter. It is used to define the number of input channels for your network."
            + "It should usually be 3 for an RGB image. Please add it in the model in training.yaml and try again."
        )

    print("\n Step 2 - D : Checking the Number of classes")

    # Checking for the number of classes
    if "num_classes" in params["models"]:
        params["models"]["num_classes"] = params["models"]["num_classes"]
    else:
        sys.exit(
            "num_classes is a required parameter. It is given as number of output classes for your network."
            + "Please add it in the model in training.yaml and try again."
        )

    print("\n Step 2 - E : Checking the Decoder Base Filters")

    # Checking for the number of classes
    if "decoder_base_filters" in params["models"]:
        params["models"]["decoder_base_filters"] = params["models"][
            "decoder_base_filters"
        ]
    else:
        params["models"]["decoder_base_filters"] = 16
        print(
            "decoder_base_filters were not provided, so defaulting to 16 base filters!"
        )

    print("\n Step 2 - F : Checking the Final Activation function")

    # Checking for the number of classes
    if "activation" in params["models"]:
        params["models"]["activation"] = params["models"]["activation"]
    else:
        params["models"]["activation"] = "sigmoid"
        print(
            "activation parameter was not provided, so defaulting to sigmoid activation in final layer!"
        )

    print("\n Step 2 - G : Checking if use of imagenet weights is asked in encoder")

    # Checking for the number of classes
    if "use_imagenet" in params["models"]:
        params["models"]["use_imagenet"] = params["models"]["use_imagenet"]
    else:
        params["models"]["use_imagenet"] = True
        print(
            "use_imagenet parameter was not provided, so defaulting to setting it to True."
        )

    print("\n Step 2 - H : Checking the number of depth layers in the network")

    # Checking for the number of classes
    if "layers" in params["models"]:
        params["models"]["layers"] = params["models"]["layers"]
    else:
        params["models"]["layers"] = 5
        print(
            "layers parameter was not provided, so defaulting to setting it to 5 layers."
        )

    print("\n Step 2 - I : Checking whether the encoder was to be frozen")

    # Checking for the number of classes
    if "encoder_freeze" in params["models"]:
        params["models"]["encoder_freeze"] = params["models"]["encoder_freeze"]
    else:
        params["models"]["encoder_freeze"] = False
        print("encoder_freeze was not provided, so defaulting to setting it to False.")

    print(
        "\n Step 2 - J : Checking how many number of epochs was encoder supposed to be frozen"
    )

    # Checking for the number of classes
    if params["models"]["encoder_freeze"]:
        if "encoder_freeze_epochs" in params["models"]:
            params["models"]["encoder_freeze_epochs"] = params["models"][
                "encoder_freeze_epochs"
            ]
        else:
            print(
                "encoder_freeze was set to True, but encoder_freeze_epochs was not set. Defaulting to 5 encoder_freeze_epochs."
            )
    else:
        print(
            "Since encoder_freeze was not set to True, encoder_freeze_epochs cannot be used. Please set it to True"
        )

    print("\n Step 3 : Checking the Slide parameters")

    print("\n Step 3 - A : Checking whether the stain was mentioned or not")

    if "stain" in params["slide"]:
        params["stain"]["slide"] = params["stain"]["slide"]
    else:
        params["stain"]["slide"] = "H&E"
        print("Since modality of the stain was not mentioned, defaulting to H&E.")

    print("\n Step 3 - B : Checking whether the level was mentioned or not.")

    if "level" in params["slide"]:
        params["level"]["slide"] = params["level"]["slide"]
    else:
        params["level"]["slide"] = 2
        print(
            "Since level of the stain was not mentioned, defaulting to level-2 or 10x for a slide with 40x magnification."
        )

    print("\n Step 3 - C : Checking whether level patch_size was given or not")

    if "patch_size" in params["slide"]:
        params["slide"]["patch_size"] = params["slide"]["patch_size"]
    else:
        params["slide"]["patch_size"] = 512
        print(
            "The 'patch_size' parameter was not found in the config file. Defaulting the patch size to 512x512."
        )

    print("\n Step 4 : Checking the optimize parameters")

    print("\n Step 4 - A : Checking the batch size")

    if "batch_size" in params["optimize"]:
        params["batch_size"]["optimize"] = params["batch_size"]["optimize"]
    else:
        sys.exit(
            "Please set a batch_size parameter in optimize. It is a required parameter."
        )

    print("\n Step 4 - B : Checking the max number of epochs")

    if "max_epochs" in params["optimize"]:
        params["max_epochs"]["optimize"] = params["max_epochs"]["optimize"]
    else:
        params["max_epochs"]["optimize"] = 100
        print(
            "The 'max_epochs' parameter was not found in the config file. Defaulting the max_epochs to 100"
        )

    print("\n Step 4 - C : Checking the min number of epochs")

    if "min_epochs" in params["optimize"]:
        params["min_epochs"]["optimize"] = params["min_epochs"]["optimize"]
    else:
        params["min_epochs"]["optimize"] = 10
        print(
            "The 'min_epochs' parameter was not found in the config file. Defaulting the min_epochs to 100"
        )

    print("\n Step 4 - D : Checking the loss function")

    if "loss_function" in params["optimize"]:
        params["loss_function"]["optimize"] = params["loss_function"]["optimize"]
    else:
        params["loss_function"]["optimize"] = "dice"
        print(
            "The 'loss_function' parameter was not found in the config file. Defaulting the loss_function to dice."
        )

    print("\n Step 4 - E : Checking the optimizer")

    if "optimizer" in params["optimize"]:
        params["optimizer"]["optimize"] = params["optimizer"]["optimize"]
    else:
        params["optimizer"]["optimize"] = "dice"
        print(
            "The 'optimizer' parameter was not found in the config file. Defaulting the optimizer to 'adam'."
        )

    print("\n Step 4 - F : Checking the Learning Rate")

    if "learning_rate" in params["optimize"]:
        params["learning_rate"]["optimize"] = params["learning_rate"]["optimize"]
    else:
        params["learning_rate"]["optimize"] = 0.01
        print(
            "The 'learning_rate' parameter was not found in the config file. Defaulting the learning_rate to 1e-2."
        )

    print("\n Step 4 - F : Checking the Precision - either 16 or 32 bit")

    if "precision" in params["optimize"]:
        params["precision"]["optimize"] = params["precision"]["optimize"]
    else:
        params["precision"]["optimize"] = 16
        print(
            "The 'precision' parameter was not found in the config file. Defaulting the precision to 16."
        )

    print("\n Step 4 - H : Checking the Learning Rate Decay Patience")

    if "lr_decay_patience" in params["optimize"]:
        params["lr_decay_patience"]["optimize"] = params["lr_decay_patience"][
            "optimize"
        ]
    else:
        params["lr_decay_patience"]["optimize"] = 5
        print(
            "The 'lr_decay_patience' parameter was not found in the config file. Defaulting the lr_decay_patience to 5."
        )

    print("\n Step 5 : Checking the callback parameters")

    print(
        "\n Step 5 - A : Checking the number of epochs before validation loss before training is stopped."
    )

    if "patience" in params["callbacks"]:
        params["patience"]["callbacks"] = params["patience"]["callbacks"]
    else:
        params["patience"]["callbacks"] = 10
        print(
            "The 'patience' parameter was not found in the config file. Defaulting the patience to 10 epochs."
        )

    print(
        "\n Step 5 - B : Checking whether Saving the save_top_k for number of models to be saved was set."
    )

    if "patience" in params["save_top_k"]:
        params["patience"]["save_top_k"] = params["patience"]["save_top_k"]
    else:
        params["patience"]["save_top_k"] = 1
        print(
            "The 'save_top_k' parameter was not found in the config file. Saving only 1 model by default."
        )

    print("***** DONE PARSING THE CONFIG FILE *****")

    return params
