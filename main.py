import argparse
from DataGenerator import DataGenerator
import matplotlib.pyplot as plt
from Network import Network
import numpy as np
from Parameters import Parameters
import tensorflow as tf
from utils import load_settings_yaml


# Define ArgumentParser
parser = argparse.ArgumentParser()
parser.add_argument("--run", help="Location/path of the run.yaml file. This is usually structured as a path.", default="runs/run.yaml", required=False)
args = parser.parse_args()

# Unpack args
yaml_path = args.run

# Load all settings from .yaml file
settings_yaml = load_settings_yaml(yaml_path)                                           # Returns a dictionary object.
params = Parameters(settings_yaml, yaml_path)
params.data_type = np.float32 if params.data_type == "np.float32" else np.float32       # This must be done here, due to the json, not accepting this kind of if statement in the parameter class.

# Create Custom Data Generator
datagen = DataGenerator(params)

# Create Neural Network - Resnet18
network = Network(params, datagen) # The network needs to know hyper-paramters from params, and needs to know how to generate data with a datagenerator object.

# Train the network
network.train()
