import os
import glob
from utils import load_settings_yaml
from Parameters import *
import numpy as np
from DataGenerator import *
from Network import *


########################################## Functions ##########################################




########################################## Params ##########################################
names           = ["binary_crossentropy", "f_beta"]
metric_interest = ["loss", "metric"]
root = os.path.join("models", "test_exp")
fraction_to_load_sources_val = 0.15


########################################## Script ##########################################
# for each member in the experiment keep a list of models that are of the same type and hyper-params
model_collections = list()

# create collections of models - models are grouped in lists
for name in names:
    model_list = glob.glob(root + "/*{}".format(name))
    model_collections.append(model_list)


for collection_idx, model_collection in enumerate(model_collections):
    for model_folder in model_collection:

        # Step 1.0 - Load settings of the model
        yaml_path = glob.glob(os.path.join(model_folder) + "/*.yaml")[0]
        settings_yaml = load_settings_yaml(yaml_path)

        # Step 2.0 - Set Parameters - and overload fraction to load sources - because not all are needed and it will just slow things down for now.
        params = Parameters(settings_yaml, yaml_path, mode="no_training")
        params.fraction_to_load_sources_vali = fraction_to_load_sources_val

        params.data_type = np.float32 if params.data_type == "np.float32" else np.float32       # must be done here, due to the json, not accepting this kind of if statement in the parameter class.
        if params.model_name == "Baseline_Enrico":
            params.img_dims = (101,101,3)

        # Step 3.0 - Define a DataGenerator that can generate validation chunks based on validation data.
        dg = DataGenerator(params, mode="no_training", do_shuffle_data=False)     #do not shuffle the data in the data generator
        
        # Step 4.0 - Construct a neural network with the same architecture as that it was trained with.
        network = Network(params, dg, training=False) # The network needs to know hyper-paramters from params, and needs to know how to generate data with a datagenerator object.
        network.model.trainable = False

        # Step 5.0 - Load weights of the neural network
        h5_path = glob.glob(model_folder + "/checkpoints/*{}.h5".format(metric_interest[collection_idx]))[0]
        network.model.load_weights(h5_path)


        X_validation_chunk, y_validation_chunk = dg.load_chunk_test(params.data_type, params.mock_lens_alpha_scaling)