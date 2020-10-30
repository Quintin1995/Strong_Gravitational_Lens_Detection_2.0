from DataGenerator import DataGenerator
import glob
from Network import Network
import numpy as np
import os
from Parameters import Parameters
import tensorflow as tf
from utils import get_model_paths, get_h5_path_dialog, load_settings_yaml, binary_dialog, hms, load_normalize_img, get_samples, compute_PSF_r, normalize_img
import matplotlib.pyplot as plt
import time
import random
from scipy.optimize import minimize
from scipy.special import softmax
from functools import partial
from sklearn.metrics import confusion_matrix

############################################################ Functions ############################################################


# Calculate the Mean Square Error for a matrix of datapoints given a ground truth label
# and weight vector.
def mean_square_error(y, Xs, w0):

    # Prediction of the ensemble.
    y_hat = np.dot(Xs, w0)

    # Square the error in order to get rid of negative values (common practice)
    error = np.square(y - y_hat)

    # Take average error
    mean = np.mean(error)
    return mean

    
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



# Given 4D numpy image data and corresponding labels, model paths and weights,
# this function will return a prediction matrix, containing a prediction on each 
# example for each model. It will also return names of the models.
def load_models_and_predict(X_chunk, y_chunk, model_paths, h5_paths):

    # Construct a matrix that holds a prediction value for each image and each model in the ensemble
    prediction_matrix = np.zeros((X_chunk.shape[0], len(model_paths)))
    model_names       = list()      # Keep track of model names
    
    # Load each model and perform prediction with it.
    for model_idx, model_path in enumerate(model_paths):

        # 1.0 - Set model parameters
        yaml_path = glob.glob(os.path.join(model_path, "run.yaml"))[0]              
        settings_yaml = load_settings_yaml(yaml_path, verbatim=False)                           # Returns a dictionary object.
        params = Parameters(settings_yaml, yaml_path, mode="no_training")                       # Create Parameter object
        params.data_type = np.float32 if params.data_type == "np.float32" else np.float32       # This must be done here, due to the json, not accepting this kind of if statement in the parameter class.
        
        # 2.0 - Create a dataGenerator object, because the network class wants it
        dg = DataGenerator(params, mode="no_training", do_shuffle_data=False, do_load_validation=False)

        # 3.0 - Construct a Network object that has a model as property.
        network = Network(params, dg, training=False, verbatim=False)
        network.model.load_weights(h5_paths[model_idx])

        # Keep track of model name
        model_names.append(network.params.model_name)

        # 4.0 - Evaluate on validation chunk
        evals = network.model.evaluate(X_chunk, y_chunk, verbose=0)
        print("\n\nIndividual Evaluations:")
        print("Model name: {}".format(network.params.model_name))
        for met_idx in range(len(evals)):
            print("{} = {}".format(network.model.metrics_names[met_idx], evals[met_idx]))
        print("")

        # 5.0 - Predict on validation chunk
        predictions = network.model.predict(X_chunk)
        prediction_matrix[:,model_idx] = np.squeeze(predictions)
    # Networks are also need later on, therefore we need to add them to a list
    return prediction_matrix, model_names


# Given a prediction matrix it will try to find weights for each model.
# Via a minimization problem.
def find_ensemble_weights(prediction_matrix, y_chunk, model_names, method="Nelder-Mead", verbatim=False):
    
    # Randomly initialize the initial weight vector
    w0 = np.random.random_sample((len(model_names),))

    # We want to pass arguments to the minimization function. This is done with a partial function.
    f_partial = partial(mean_square_error, y_chunk, prediction_matrix)

    # Perform the minimization with the scipy package
    res = minimize(f_partial, w0, method='Nelder-Mead', tol=1e-6, options={'disp' : True})

    # Collect model weights and print to user. (Softmax the weights so that they sum to 1.)
    model_weights = res.x
    model_weights = softmax(model_weights)
    if verbatim:
        print("Model weights\t = {}".format(model_weights))
        print("Weights Sum\t = {}".format(np.sum(model_weights)))
        print("Model names\t = {}".format(model_names))
    return model_weights


############################################################ Script ############################################################

# 1.0 - Fix memory leaks if running on tensorflow 2
tf.compat.v1.disable_eager_execution()

# 2.0 - Model Selection from directory - Select multiple models
model_paths = choose_ensemble_members()

# 3.0 - Select a weights file. There are 2 for each model. Selected based on either validation loss or validation metric. The metric can differ per model.
h5_paths = get_h5_path_dialog(model_paths)

# 4.0 - Select random sample from the data (with replacement)
sample_size = 551
# sample_size = int(input("How many samples do you want to create and run (int): "))
sources_fnames, lenses_fnames, negatives_fnames = get_samples(size=sample_size, deterministic=False)

# 5.0 - Load unnormalized data in order to calculate the amount of noise in a lens.
# Create a chunk of data that each neural network understands (preprocessed quite identically)
PSF_r = compute_PSF_r()  # Used for sources only
lenses      = load_normalize_img(np.float32, are_sources=False, normalize_dat="per_image", PSF_r=PSF_r, filenames=lenses_fnames)
sources     = load_normalize_img(np.float32, are_sources=True, normalize_dat="per_image", PSF_r=PSF_r, filenames=sources_fnames)
negatives   = load_normalize_img(np.float32, are_sources=False, normalize_dat="per_image", PSF_r=PSF_r, filenames=negatives_fnames)

# 6.0 - Load a 50/50 positive/negative chunk into memory
X_chunk, y_chunk = _load_chunk_val(lenses, sources, negatives, mock_lens_alpha_scaling=(0.02, 0.30))

# 7.0 - Load ensemble members and perform prediction with it.
prediction_matrix, model_names = load_models_and_predict(X_chunk, y_chunk, model_paths, h5_paths)

# 8.0 - Find ensemble model weights as a vector
model_weights = find_ensemble_weights(prediction_matrix, y_chunk, model_names, method="Nelder-Mead", verbatim=True)


# Ensemble Evaluation
print("\n\nEnsemble Evaluation:")
y_hat = np.dot(prediction_matrix, model_weights)
print(y_hat)
threshold = 0.5
y_hat_copy = np.copy(y_hat)
y_hat_copy[y_hat < threshold] = 0.0
y_hat_copy[y_hat >= threshold] = 1.0
error_count = np.sum(np.absolute(y_chunk - y_hat_copy))
print("mistakes count: {}".format(error_count))
error_percentage = error_count/y_hat_copy.shape[0]
acc = 1.0 - error_percentage
print(acc)
x=4