import numpy as np
from Network import *
from Parameters import *
from DataGenerator import *

###### Step 1.0: Load all settings from .yaml file
settings_yaml = load_settings_yaml("runs/run.yaml")
params = Parameters(settings_yaml)
params.data_type = np.float32 if params.data_type == "np.float32" else np.float32       # must be done here, due to the json, not accepting this kind of if statement in the parameter class.


###### Step 2.0: Create a Data Generator class that reads all data and holds all data.
data_generator = DataGenerator(params)


###### Step 3.0 - Create Neural Network - Resnet18
resnet18   = Network(params, data_generator)


###### Step 4.0 - Train Neural Network - Resnet18
resnet18.train_resnet18()


x=5