from utils import create_dir_if_not_exists, choose_ensemble_members, get_h5_path_dialog, get_samples, get_fnames_from_disk, compute_PSF_r, load_settings_yaml, dstack_data, count_TP_TN_FP_FN_and_FB
import argparse
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import RMSprop, Adam
import os
import time
import random
import numpy as np
from create_ensemble import load_chunk_test, load_chunk_val, load_chunk_train
import functools
from multiprocessing import Pool
from astropy.io import fits
import scipy
from DataGenerator import DataGenerator
import glob
from Network import Network
from Parameters import Parameters
from skimage import exposure
from ModelCheckpointYaml import *
import matplotlib.pyplot as plt
import csv
from resnet import *
import math
from scipy.optimize import minimize
from scipy.special import softmax
from functools import partial
from create_input_dep_ensemble import load_and_normalize_img, load_networks, predict_on_networks


# Deal with input arguments
def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="Location/path of the run.yaml file. This is usually structured as a path.", default="runs/ensemble_runs/3mems_NM/run_default.yaml", required=False)
    args = parser.parse_args()
    return args


def _set_and_create_dirs(settings_dict):
    root_dir_ensembles = "ensembles"
    ensemble_dir = os.path.join(root_dir_ensembles, settings_dict["ens_name"])
    create_dir_if_not_exists(root_dir_ensembles, verbatim=False)
    create_dir_if_not_exists(ensemble_dir)
    return ensemble_dir


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


# Given a prediction matrix it will try to find weights for each model.
# Via a minimization problem.
# The prediction matrix should have number of item in chunk as rows and the number of members as columns (NxM)
def find_ensemble_weights_NM(prediction_matrix, y_chunk, model_names, verbatim=False):
    
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
        print("Model weights data type\t = {}".format(type(model_weights)))
    return model_weights


def main():
    # Parse input arguments
    args = _get_arguments()

    # load setting yaml
    settings_dict = load_settings_yaml(args.run)

    # Set root directory, ensemble directory and create them.
    ensemble_dir = _set_and_create_dirs(settings_dict)

    # Fix memory leaks if running on tensorflow 2
    tf.compat.v1.disable_eager_execution()

    # Model Selection from directory - Select multiple models
    model_paths = list(settings_dict["models"])

    # Load the individual networks/models into a list and keep track of their names.
    networks, model_names = load_networks(model_paths)

    # Get filenames of the train and testdata
    sources_fnames_train, lenses_fnames_train, negatives_fnames_train = get_fnames_from_disk(settings_dict["lens_frac_train"], settings_dict["source_frac_train"], settings_dict["negative_frac_train"], type_data="train", deterministic=False)
    # sources_fnames_val, lenses_fnames_val, negatives_fnames_val       = get_fnames_from_disk(settings_dict["lens_frac_val"], settings_dict["source_frac_val"], settings_dict["negative_frac_val"], type_data="validation", deterministic=False)
    sources_fnames_test, lenses_fnames_test, negatives_fnames_test  = get_fnames_from_disk(settings_dict["lens_frac_test"], settings_dict["source_frac_test"], settings_dict["negative_frac_test"], type_data="test", deterministic=False)

    # Load all the data into memory using multi-threading
    PSF_r = compute_PSF_r()  # Used for sources only
    with Pool(24) as p:
        lenses_train_f    = functools.partial(load_and_normalize_img, np.float32, False, "per_image", PSF_r)
        lenses_train      = np.asarray(p.map(lenses_train_f, enumerate(lenses_fnames_train), chunksize=128), np.float32)
        sources_train_f   = functools.partial(load_and_normalize_img, np.float32, True, "per_image", PSF_r)
        sources_train     = np.asarray(p.map(sources_train_f, enumerate(sources_fnames_train), chunksize=128), np.float32)
        negatives_train_f = functools.partial(load_and_normalize_img, np.float32, False, "per_image", PSF_r)
        negatives_train   = np.asarray(p.map(negatives_train_f, enumerate(negatives_fnames_train), chunksize=128), np.float32)
        # lenses_val_f      = functools.partial(load_and_normalize_img, np.float32, False, "per_image", PSF_r)
        # lenses_val        = np.asarray(p.map(lenses_val_f, enumerate(lenses_fnames_val), chunksize=128), np.float32)
        # sources_val_f     = functools.partial(load_and_normalize_img, np.float32, True, "per_image", PSF_r)
        # sources_val       = np.asarray(p.map(sources_val_f, enumerate(sources_fnames_val), chunksize=128), np.float32)
        # negatives_val_f   = functools.partial(load_and_normalize_img, np.float32, False, "per_image", PSF_r)
        # negatives_val     = np.asarray(p.map(negatives_val_f, enumerate(negatives_fnames_val), chunksize=128), np.float32)
        lenses_test_f     = functools.partial(load_and_normalize_img, np.float32, False, "per_image", PSF_r)
        lenses_test       = np.asarray(p.map(lenses_test_f, enumerate(lenses_fnames_test), chunksize=128), np.float32)
        sources_test_f    = functools.partial(load_and_normalize_img, np.float32, True, "per_image", PSF_r)
        sources_test      = np.asarray(p.map(sources_test_f, enumerate(sources_fnames_test), chunksize=128), np.float32)
        negatives_test_f  = functools.partial(load_and_normalize_img, np.float32, False, "per_image", PSF_r)
        negatives_test    = np.asarray(p.map(negatives_test_f, enumerate(negatives_fnames_test), chunksize=128), np.float32)

    # Use the test data for {accuracy, f_beta, precision, recall} performance metric evaluations.
    # Load a train and validation chunk
    X_chunk_train, y_chunk_train = load_chunk_train(settings_dict["chunk_size"], lenses_train, negatives_train, sources_train, np.float32, mock_lens_alpha_scaling=(0.02, 0.30))
    # X_chunk_val, y_chunk_val     = load_chunk_val(lenses_val, sources_val, negatives_val, mock_lens_alpha_scaling=(0.02, 0.30))
    X_chunk_test, y_chunk_test   = load_chunk_test(np.float32, (0.02, 0.30), lenses_test, sources_test, negatives_test)

    if False:   # for debugging
        for i in range(X_chunk_test.shape[0]):
            plt.imshow(np.squeeze(X_chunk_test[i]), cmap='Greys_r')
            plt.title("Test Image with label: {}".format(y_chunk_test[i]))
            plt.show()

    # Prediction matrix train - Loop over individual models and predict
    prediction_matrix_train = np.zeros((X_chunk_train.shape[0], len(model_names)))
    print("Creating prediction matrix with shape: {}...".format(prediction_matrix_train.shape))
    for net_idx, network in enumerate(networks):
        prediction_matrix_train[:,net_idx] = np.squeeze(network.model.predict(X_chunk_train))
    print("")
    for idx, pred in enumerate(prediction_matrix_train):
        if idx < 10:
            print(pred)
    print("")

    # Now that we have a prediction matrix (prediction_matrix_train) and labels we can optimize the ensemble
    # member weights with the Nelder-Mead algorithm
    mem_weight_vec = find_ensemble_weights_NM(prediction_matrix_train, y_chunk_train, model_names, verbatim=True)

    # Prediction matrix test - Loop over individual models and predict
    prediction_matrix_test = np.zeros((X_chunk_test.shape[0], len(model_names)))
    print("Creating prediction matrix with shape: {}...".format(prediction_matrix_test.shape))
    for net_idx, network in enumerate(networks):
        prediction_matrix_test[:,net_idx] = np.squeeze(network.model.predict(X_chunk_test))
    print("")
    for idx, pred in enumerate(prediction_matrix_test):
        if idx < 10:
            print(pred)
    print("")

    # Now we should obtained member weight vector as weights for the individual predictions of its members.
    ens_y_hat = np.zeros(y_chunk_test.shape)
    for i in range(prediction_matrix_test.shape[0]):
        members_row     = prediction_matrix_test[i]
        ens_y_hat[i]    = np.dot(mem_weight_vec, members_row)
        if i < 10:
            print(ens_y_hat[i])

    # Piece of code that can be used if the final prediction should be performed by the member with the highest weight.
    # for i in range(individual_predictions.shape[0]):
    #     winning_idx = np.argmax(ens_preds[i])
    #     ens_y_hat[i] = individual_predictions[i,winning_idx]
    #     if i < 10:
    #         print(ens_y_hat[i])
    
    ### f_beta graph and its paramters
    beta_squarred           = 0.03                                  # For f-beta calculation
    stepsize                = 0.01                                  # For f-beta calculation
    threshold_range         = np.arange(stepsize, 1.0, stepsize)    # For f-beta calculation
    colors = ['r', 'c', 'green', 'orange', 'lawngreen', 'b', 'plum', 'darkturquoise', 'm']

    # calculate different performance metrics on current model
    f_betas, precision_data, recall_data = [], [], []
    for p_threshold in threshold_range:
        (TP, TN, FP, FN, precision, recall, fp_rate, accuracy, F_beta) = count_TP_TN_FP_FN_and_FB(ens_y_hat, y_chunk_test, p_threshold, beta_squarred)
        print("acc: {0:.3f} on threshold: {1:.3f}".format(accuracy, p_threshold))
        f_betas.append(F_beta)
        precision_data.append(precision)
        recall_data.append(recall)

    # Plotting all lines
    plt.plot(list(threshold_range), f_betas, colors[0], label = "f_beta")
    plt.plot(list(threshold_range), precision_data, ":", color=colors[0], alpha=0.9, linewidth=3, label="Precision")
    plt.plot(list(threshold_range), recall_data, "--", color=colors[0], alpha=0.9, linewidth=3, label="Recall")
    
    # set up plot aesthetics
    plt.xlabel("p threshold", fontsize=40)
    plt.ylabel("F", fontsize=40)
    plt.title("Ensemble F_beta, where Beta = {0:.2f}".format(math.sqrt(beta_squarred)), fontsize=40)
    figure = plt.gcf() # get current figure
    axes = plt.gca()
    axes.set_ylim([0.63, 1.00])
    figure.set_size_inches(12, 8)       # (12,8), seems quite fine
    plt.grid(color='grey', linestyle='dashed', linewidth=1)
    plt.legend()
    plt.savefig('{}.png'.format(os.path.join(ensemble_dir, "f_beta_plot")))
    plt.show()


############################################################ Script ############################################################

if __name__ == "__main__":
    main()