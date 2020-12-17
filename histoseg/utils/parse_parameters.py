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


def parseConfig(config_file_path):
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
		print("Since no 'encoder' was selected, defaulting to :", params["model"]["encoder"])

	print("\n Step 2 - A : Checking the Decoder")

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
		print("Since no 'decoder' was selected, defaulting to :", params["model"]["decoder"])

	print()
    if "patch_size" in params["slide"]:
        params["slide"]["patch_size"] = params["slide"]["patch_size"]
    else:
        sys.exit(
            "The 'patch_size' parameter was not found in the config file. Defaulting the patch size to 512x512."
        )

