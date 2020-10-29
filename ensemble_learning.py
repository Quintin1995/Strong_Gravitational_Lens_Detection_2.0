from DataGenerator import DataGenerator
import glob
from Network import Network
import numpy as np
import os
from Parameters import Parameters
import tensorflow as tf
from utils import get_model_paths, get_h5_path_dialog, load_settings_yaml, binary_dialog, hms, load_normalize_img, get_samples, compute_PSF_r
import matplotlib.pyplot as plt
import time
import random


############################################################ Functions ############################################################
    
    
# Merge a single lens and source together into a mock lens.
def merge_lens_and_source(lens, source, mock_lens_alpha_scaling = (0.02, 0.30), show_imgs = False):

    # Add lens and source together | We rescale the brightness of the simulated source to the peak brightness of the LRG in the r-band multiplied by a factor of alpha randomly drawn from the interval [0.02,0.3]
    mock_lens = lens + source / np.amax(source) * np.amax(lens) * np.random.uniform(mock_lens_alpha_scaling[0], mock_lens_alpha_scaling[1])
    
    # Take a square root stretch to emphesize lower luminosity features.
    mock_lens = np.sqrt(mock_lens)
    
    # Basically removes negative values - should not be necessary, because all input data should be normalized anyway. (I will leave it for now, but should be removed soon.)
    # mock_lens_sqrt = mock_lens_sqrt.clip(min=0.0, max=1.0)
    mock_lens = mock_lens.clip(min=0.0, max=1.0)

    # if True:
    #     show2Imgs(lens, source, "Lens max pixel: {0:.3f}".format(np.amax(lens)), "Source max pixel: {0:.3f}".format(np.amax(source)))
        # show2Imgs(mock_lens, mock_lens_sqrt, "mock_lens max pixel: {0:.3f}".format(np.amax(mock_lens)), "mock_lens_sqrt max pixel: {0:.3f}".format(np.amax(mock_lens_sqrt)))

    return mock_lens


# This function should read images from the lenses- and sources data array,
# and merge them together into a lensing system, further described as 'mock lens'.
# These mock lenses represent a strong gravitational lensing system that should 
# get the label 1.0 (positive label). 
def merge_lenses_and_sources(lenses_array, sources_array, num_mock_lenses, data_type, mock_lens_alpha_scaling = (0.02, 0.30), do_deterministic=False):
    num_lenses  = lenses_array.shape[0]
    num_sources = sources_array.shape[0]

    X_train_positive = np.empty((num_mock_lenses, lenses_array.shape[1], lenses_array.shape[2], lenses_array.shape[3]), dtype=data_type)
    Y_train_positive = np.ones(num_mock_lenses, dtype=data_type)
    
    # Which indexes to load from sources and lenses (these will be merged together)
    if do_deterministic:
        idxs_lenses = list(range(num_lenses))       # if deterministic we always want the same set
        idxs_sources = random.choices(list(range(num_sources)), k=num_mock_lenses)
    else:
        idxs_lenses = random.choices(list(range(num_lenses)), k=num_mock_lenses)
        idxs_sources = random.choices(list(range(num_sources)), k=num_mock_lenses)

    for i in range(num_mock_lenses):
        lens   = lenses_array[idxs_lenses[i]]
        source = sources_array[idxs_sources[i]]
        mock_lens = merge_lens_and_source(lens, source, mock_lens_alpha_scaling)

        # Uncomment this code if you want to inspect how a lens, source and mock lens look before they are merged.
        # import matplotlib.pyplot as plt
        # l = np.squeeze(lens)
        # s = np.squeeze(source)
        # m = np.squeeze(mock_lens)
        # plt.imshow(l, origin='lower', interpolation='none', cmap='gray', vmin=0.0, vmax=1.0)
        # plt.title("lens")
        # plt.show()
        # plt.imshow(s, origin='lower', interpolation='none', cmap='gray', vmin=0.0, vmax=1.0)
        # plt.title("source")
        # plt.show()
        # plt.imshow(m, origin='lower', interpolation='none', cmap='gray', vmin=0.0, vmax=1.0)
        # plt.title("mock lens")
        # plt.show()

        X_train_positive[i] = mock_lens

    return X_train_positive, Y_train_positive


# Loading a valiation chunk into memory
def _load_chunk_val(Xlenses_validation, Xsources_validation, Xnegatives_validation, mock_lens_alpha_scaling):
    start_time = time.time()
    num_positive = Xlenses_validation.shape[0]
    # num_negative = Xnegatives_validation.shape[0] + Xlenses_validation.shape[0]   #also num positive, because the unmerged lenses with sources are also deemed a negative sample.
    
    # Lets create a balanced validation set.
    num_negative = num_positive

    # Get mock lenses data and labels
    X_pos, y_pos = merge_lenses_and_sources(Xlenses_validation, Xsources_validation, num_positive, np.float32, mock_lens_alpha_scaling, do_deterministic=True)
        
    # Store Negative data in numpy array and create label vector
    X_neg = np.empty((num_negative, X_pos.shape[1], X_pos.shape[2], X_pos.shape[3]), dtype=np.float32)    
    y_neg = np.zeros(X_neg.shape[0], dtype=np.float32)

    # Negatives consist of the negatives set and the unmerged lenses set. A lens unmerged with a source is basically a negative.
    n = int(num_negative // 2)  # number samples to take from lenses-, and negatives set.
    
    indexes_lenses = np.random.choice(Xlenses_validation.shape[0], n, replace=False)  
    X_neg[0:int(num_negative//2)] = Xlenses_validation[indexes_lenses] # first half of negative chunk is a random selection from lenses without replacement

    indexes_negatives = np.random.choice(Xlenses_validation.shape[0], n+1, replace=False)  
    X_neg[int(num_negative//2):num_negative] = Xnegatives_validation[indexes_negatives] # second half of negatives are a random selection from the negatives without replacement
    
    # The negatives need a square root stretch, just like the positives.
    X_neg = np.sqrt(X_neg)

    # Concatenate the positive and negative examples into one chunk (also the labels)
    X_chunk = np.concatenate((X_pos, X_neg))
    y_chunk = np.concatenate((y_pos, y_neg))

    print("Creating validation chunk took: {}, chunksize: {}".format(hms(time.time() - start_time), num_positive+num_negative), flush=True)
    return X_chunk, y_chunk


# Choose multiple models as Ensemble members.
def choose_ensemble_members():
    model_paths = list()
    do_select_more_models = True
    while do_select_more_models:
        model_paths += get_model_paths()
        do_select_more_models = binary_dialog("\nDo you want to select more members for the Ensemble?")
    print("\nChoosen models: ")
    for model_path in model_paths:
        print("model path = {}".format(model_path))
    return model_paths


############################################################ Script ############################################################

# 1.0 - Fix memory leaks if running on tensorflow 2
tf.compat.v1.disable_eager_execution()


# 2.0 - Model Selection from directory - Select multiple models
model_paths = choose_ensemble_members()


# 3.0 - Select a weights file. There are 2 for each model. Selected based on either validation loss or validation metric. The metric can differ per model.
h5_paths = get_h5_path_dialog(model_paths)

# 4.0 - Select random sample from the data (with replacement)
sample_size = 25
# sample_size = int(input("How many samples do you want to create and run (int): "))
sources_fnames, lenses_fnames, negatives_fnames = get_samples(size=sample_size, deterministic=False)


# 5.1 - Load unnormalized data in order to calculate the amount of noise in a lens.
# Create a chunk of data that each neural network understands (preprocessed quite identically)
PSF_r = compute_PSF_r()  # Used for sources only
lenses      = load_normalize_img(np.float32, are_sources=False, normalize_dat="per_image", PSF_r=PSF_r, filenames=lenses_fnames)
sources     = load_normalize_img(np.float32, are_sources=True, normalize_dat="per_image", PSF_r=PSF_r, filenames=sources_fnames)
negatives   = load_normalize_img(np.float32, are_sources=False, normalize_dat="per_image", PSF_r=PSF_r, filenames=negatives_fnames)

X_chunk, y_chunk = _load_chunk_val(lenses, sources, negatives, mock_lens_alpha_scaling=(0.02, 0.30))

print(lenses.shape)
print(sources.shape)
print(negatives.shape)
print(X_chunk.shape)

if False:
    for i in range(3):
        plt.figure()
        plt.imshow(np.squeeze(lenses[i]), cmap='Greys_r')
        plt.title("lens")

        plt.figure()
        plt.imshow(np.squeeze(sources[i]), cmap='Greys_r')
        plt.title("source")

        plt.figure()
        plt.imshow(np.squeeze(X_chunk[i]), cmap='Greys_r')
        plt.title("mock lens")

        plt.figure()
        plt.imshow(np.squeeze(negatives[i]), cmap='Greys_r')
        plt.title("negative")

        plt.show()


model_names = list()
all_predictions = list()
# 4.0 - Load params - used for normalization etc -
for model_idx, model_path in enumerate(model_paths):

    # 5.0 - Set model parameters
    yaml_path = glob.glob(os.path.join(model_paths[model_idx], "run.yaml"))[0]              
    settings_yaml = load_settings_yaml(yaml_path, verbatim=False)                                           # Returns a dictionary object.
    params = Parameters(settings_yaml, yaml_path, mode="no_training")                       # Create Parameter object
    params.data_type = np.float32 if params.data_type == "np.float32" else np.float32       # This must be done here, due to the json, not accepting this kind of if statement in the parameter class.
    
    # keep track of model name
    model_names.append(params.model_name)

    # 6.0 - Create a dataGenerator object, because the network class wants it
    dg = DataGenerator(params, mode="no_training", do_shuffle_data=False, do_load_validation=False)

    # 7.0 - Construct a Network object that has a model as property.
    network = Network(params, dg, training=False, verbatim=False)
    network.model.load_weights(h5_paths[model_idx])

    # 8.0 - Evaluate on validation chunk
    results = network.model.evaluate(X_chunk, y_chunk, verbose=0)
    for met_idx in range(len(results)):
        print("\n{} = {}".format(network.model.metrics_names[met_idx], results[met_idx]))

    # 9.0 - Predict on validation chunk
    predictions = network.model.predict(X_chunk)
    all_predictions.append(predictions)


for img_idx in range(X_chunk.shape[0]):
    print("\nImage #{}".format(img_idx))
    for model_idx, model_name in enumerate(model_names):
        model_preds = all_predictions[model_idx]
        print("Model({0})\tpredicts: {1:.3f}, truth = {2}".format(model_name, model_preds[img_idx][0], y_chunk[img_idx]))
