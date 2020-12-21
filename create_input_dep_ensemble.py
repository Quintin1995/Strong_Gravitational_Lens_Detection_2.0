from utils import create_dir_if_not_exists, choose_ensemble_members, get_h5_path_dialog, get_samples, get_fnames_from_disk, compute_PSF_r, load_settings_yaml, dstack_data
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
import csv

########################### Description ###########################
## Take user through dialog that lets the user select trained models.
## Construct a model/network that takes in training data and outputs a vector of length=#members in enemble
## We use softmax on the output vector and determine which member/model of the ensemble gets to predict on the input image.
## Via this method, a network selects which network gets to predict on the input image, based on information within the image.
########################### End Description #######################




############################################################ Functions ############################################################



# Given 4D numpy image data and corresponding labels, model paths and weights,
# this function will return a prediction matrix, containing a prediction on each 
# example for each model. It will also return names of the models.
def _load_models_and_predict(X_chunk, y_chunk, model_weights_paths):

    # Construct a matrix that holds a prediction value for each image and each model in the ensemble
    prediction_matrix = np.zeros((X_chunk.shape[0], len(model_weights_paths)))
    model_names       = list()      # Keep track of model names
    individual_scores = list()      # Keep track of model acc on an individual basis
    
    # Load each model and perform prediction with it.
    for model_idx, model_path in enumerate(model_weights_paths):

        # 1.0 - Set model parameters
        model_path, _ = os.path.split(model_path)        # Remove last part of path
        model_path, _ = os.path.split(model_path)        # Remove last part of path again.
        system_specific_path = os.getcwd()
        path = os.path.join(system_specific_path, model_path, "run.yaml")
        yaml_path = glob.glob(path)[0]
        settings_yaml = load_settings_yaml(yaml_path, verbatim=False)                           # Returns a dictionary object.
        params = Parameters(settings_yaml, yaml_path, mode="no_training")                       # Create Parameter object
        params.data_type = np.float32 if params.data_type == "np.float32" else np.float32       # This must be done here, due to the json, not accepting this kind of if statement in the parameter class.
        
        # 2.0 - Create a dataGenerator object, because the network class wants it
        dg = DataGenerator(params, mode="no_training", do_shuffle_data=False, do_load_validation=False)

        # 3.0 - Construct a Network object that has a model as property.
        network = Network(params, dg, training=False, verbatim=False)
        network.model.load_weights(model_weights_paths[model_idx])

        # Keep track of model name
        model_names.append(network.params.model_name)

        # 4.0 - Evaluate on validation chunk
        if "resnet_single_newtr_last_last_weights_only" in model_path:
            X_chunk = dstack_data(X_chunk)
        evals = network.model.evaluate(X_chunk, y_chunk, verbose=0)
        print("\nIndividual Evaluations:")
        print("Model name: {}".format(network.params.model_name))
        for met_idx in range(len(evals)):
            print("{} = {}".format(network.model.metrics_names[met_idx], evals[met_idx]))
            individual_scores.append((network.model.metrics_names[met_idx], evals[met_idx]))
        print("-----")

        # 5.0 - Predict on validation chunk
        predictions = network.model.predict(X_chunk)
        prediction_matrix[:,model_idx] = np.squeeze(predictions)
    # Networks are also need later on, therefore we need to add them to a list
    return prediction_matrix, model_names, individual_scores


# Normalization per image
def normalize_img(numpy_img):
    return ((numpy_img - np.amin(numpy_img)) / (np.amax(numpy_img) - np.amin(numpy_img)))


# Simple case function to reduce line count in other function
def normalize_function(img, norm_type, data_type):
    if norm_type == "per_image":
        img = normalize_img(img)
    if norm_type == "adapt_hist_eq":
        img = normalize_img(img)
        img = exposure.equalize_adapthist(img).astype(data_type)
    return img


# If the data array contains sources, then a PSF_r convolution needs to be performed over the image.
# There is also a check on whether the loaded data already has a color channel dimension, if not create it.
def _load_and_normalize_img(data_type, are_sources, normalize_dat, PSF_r, idx_filename):
    idx, filename = idx_filename
    if idx % 1000 == 0:
        print("loaded {} images".format(idx), flush=True)
    if are_sources:
        img = fits.getdata(filename).astype(data_type)
        img = scipy.signal.fftconvolve(img, PSF_r, mode="same")                                # Convolve with psf_r, has to do with camara point spread function.
        return np.expand_dims(normalize_function(img, normalize_dat, data_type), axis=2)       # Expand color channel and normalize
    else:
        img = fits.getdata(filename).astype(data_type)
        if img.ndim == 3:                                                                      # Some images are stored with color channel
            return normalize_function(img, normalize_dat, data_type)
        elif img.ndim == 2:                                                                    # Some images are stored without color channel
            return np.expand_dims(normalize_function(img, normalize_dat, data_type), axis=2)


# Return a neural network, considered to be the ensemble model. 
# This model should predict which network in the ensemble gets
# to predict the final prediction on the input image.
def get_ensemble_model(setting_dict, input_shape=(101,101,1), num_outputs=3):

    # Define input
    inp = Input(shape=input_shape)
    
    # Define architecture
    x = Conv2D(64, kernel_size=3, padding='valid', kernel_initializer=glorot_uniform(seed=0))(inp)
    x = BatchNormalization(axis=3)(x)   # Axis three is color channel
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, kernel_size=3, padding='valid', kernel_initializer=glorot_uniform(seed=0))(inp)
    x = BatchNormalization(axis=3)(x)   # Axis three is color channel
    x = Activation('relu')(x)

    x = Conv2D(128, kernel_size=3, padding='valid', kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)   # Axis three is color channel
    x = Activation('relu')(x)

    x = Conv2D(128, kernel_size=3, padding='valid', kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)   # Axis three is color channel
    x = Activation('relu')(x)

    x = Conv2D(256, kernel_size=3, padding='valid', kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)   # Axis three is color channel
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    flat = Flatten()(x)
    flat = Dense(10, activation='relu')(flat)
    out = Dense(num_outputs, activation='softmax')(flat)

    # Construct the model based on the defined architecture and compile it.
    model = Model(inputs=inp, outputs=out)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=float(setting_dict["learning_rate"])),
                  metrics = ['binary_accuracy'])

    model.summary()
    return model


# Deal with input arguments
def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="Location/path of the run.yaml file. This is usually structured as a path.", default="runs/ensembles/run1.yaml", required=False)
    args = parser.parse_args()
    return args


def _set_and_create_dirs(settings_dict):
    root_dir_ensembles = "ensembles"
    ensemble_dir = os.path.join(root_dir_ensembles, settings_dict["ens_name"])
    create_dir_if_not_exists(root_dir_ensembles, verbatim=False)
    create_dir_if_not_exists(ensemble_dir)
    return ensemble_dir


# Converts an 2D array of floats to a one hot encoding. Where a row represent the predictions of all members of the ensemble.
def convert_pred_matrix_to_one_hot_encoding(pred_matrix):
    one_hot = np.zeros(pred_matrix.shape)
    for row, column in enumerate(np.argmax(pred_matrix, axis=1)):
        one_hot[row][column] = 1.0
    return one_hot


def main():
    # Parse input arguments
    args = _get_arguments()

    settings_dict = load_settings_yaml(args.run)

    # Set root directory, ensemble directory and create them.
    ensemble_dir = _set_and_create_dirs(settings_dict)

    # Fix memory leaks if running on tensorflow 2
    tf.compat.v1.disable_eager_execution()

    # Model Selection from directory - Select multiple models
    model_paths = list(settings_dict["models"])

    sources_fnames_train, lenses_fnames_train, negatives_fnames_train = get_fnames_from_disk(settings_dict["lens_frac_train"], settings_dict["source_frac_train"], settings_dict["negative_frac_train"], type_data="train", deterministic=False)
    sources_fnames_val, lenses_fnames_val, negatives_fnames_val       = get_fnames_from_disk(settings_dict["lens_frac_val"], settings_dict["source_frac_val"], settings_dict["negative_frac_val"], type_data="validation", deterministic=False)
    sources_fnames_test, lenses_fnames_test, negatives_fnames_test    = get_fnames_from_disk(settings_dict["lens_frac_test"], settings_dict["source_frac_test"], settings_dict["negative_frac_test"], type_data="test", deterministic=False)
    
    # Load all the data into memory
    PSF_r = compute_PSF_r()  # Used for sources only
    with Pool(24) as p:
        lenses_train_f    = functools.partial(_load_and_normalize_img, np.float32, False, "per_image", PSF_r)
        lenses_train      = np.asarray(p.map(lenses_train_f, enumerate(lenses_fnames_train), chunksize=128), np.float32)

        sources_train_f   = functools.partial(_load_and_normalize_img, np.float32, True, "per_image", PSF_r)
        sources_train     = np.asarray(p.map(sources_train_f, enumerate(sources_fnames_train), chunksize=128), np.float32)
        
        negatives_train_f = functools.partial(_load_and_normalize_img, np.float32, False, "per_image", PSF_r)
        negatives_train   = np.asarray(p.map(negatives_train_f, enumerate(negatives_fnames_train), chunksize=128), np.float32)
        
        lenses_val_f      = functools.partial(_load_and_normalize_img, np.float32, False, "per_image", PSF_r)
        lenses_val        = np.asarray(p.map(lenses_val_f, enumerate(lenses_fnames_val), chunksize=128), np.float32)

        sources_val_f     = functools.partial(_load_and_normalize_img, np.float32, True, "per_image", PSF_r)
        sources_val       = np.asarray(p.map(sources_val_f, enumerate(sources_fnames_val), chunksize=128), np.float32)

        negatives_val_f   = functools.partial(_load_and_normalize_img, np.float32, False, "per_image", PSF_r)
        negatives_val     = np.asarray(p.map(negatives_val_f, enumerate(negatives_fnames_val), chunksize=128), np.float32)

        lenses_test_f     = functools.partial(_load_and_normalize_img, np.float32, False, "per_image", PSF_r)
        lenses_test       = np.asarray(p.map(lenses_test_f, enumerate(lenses_fnames_test), chunksize=128), np.float32)

        sources_test_f    = functools.partial(_load_and_normalize_img, np.float32, True, "per_image", PSF_r)
        sources_test      = np.asarray(p.map(sources_test_f, enumerate(sources_fnames_test), chunksize=128), np.float32)

        negatives_test_f  = functools.partial(_load_and_normalize_img, np.float32, False, "per_image", PSF_r)
        negatives_test    = np.asarray(p.map(negatives_test_f, enumerate(negatives_fnames_test), chunksize=128), np.float32)

    # 1 Construct input dependent ensemble model
    if settings_dict["network_name"] == "simple_net":
        ens_model = get_ensemble_model(settings_dict, input_shape=(101,101,1), num_outputs=len(model_paths))

    # Create a callback that will store the model if the validation binary accuracy is better than it was.
    mc_loss = ModelCheckpointYaml(
        os.path.join(ensemble_dir, "ensemble_model_val_bin_acc.h5"), 
        monitor="val_binary_accuracy",
        verbose=1, save_best_only=True,
        mode='min',
        save_weights_only=False,
        mc_dict_filename=os.path.join(ensemble_dir, "mc_best_val_bin_acc.yaml"))

    # Keep track of training history with a .csv file
    f = open(os.path.join(ensemble_dir, "training_history.csv"), "w", 1)
    writer = csv.writer(f)
    writer.writerow(["chunk", "loss", "binary_accuracy", "val_loss", "val_binary_accuracy"])    #csv headers

    # Loop over training chunks
    for chunk_idx in range(int(settings_dict["num_chunks"])):
        print("Chunk: {}".format(chunk_idx))
        
        # 1 Load a train and validation chunk
        X_chunk_train, y_chunk_train = load_chunk_train(settings_dict["chunk_size"], lenses_train, negatives_train, sources_train, np.float32, mock_lens_alpha_scaling=(0.02, 0.30))
        X_chunk_val, y_chunk_val     = load_chunk_val(lenses_val, sources_val, negatives_val, mock_lens_alpha_scaling=(0.02, 0.30))

        # 2 Load the individual networks and predict on the train chunk
        prediction_matrix_train, model_names_train, individual_scores_train = _load_models_and_predict(X_chunk_train, y_chunk_train, model_paths)
        prediction_matrix_val, model_names_val, individual_scores_val       = _load_models_and_predict(X_chunk_val, y_chunk_val, model_paths)

        # 3 Convert prediction matrix to one hot encoding
        y_true_train = np.zeros(prediction_matrix_train.shape)
        for i in range(prediction_matrix_train.shape[0]):
            y_true_train[i] = 1 - np.absolute(y_chunk_train[i] - prediction_matrix_train[i])

        y_true_val = np.zeros(prediction_matrix_val.shape)
        for i in range(prediction_matrix_val.shape[0]):
            y_true_val[i] = 1 - np.absolute(y_chunk_val[i] - prediction_matrix_val[i])

        history = ens_model.fit(x=X_chunk_train,
                                y=y_true_train,
                                batch_size=None,
                                epochs=1,
                                verbose=1,
                                validation_data=(X_chunk_val, y_true_val),
                                shuffle=True,
                                callbacks = [mc_loss]
                                )
        writer.writerow([str(chunk_idx),
                        str(history.history["loss"][0]),
                        str(history.history["binary_accuracy"][0]),
                        str(history.history["val_loss"][0]),
                        str(history.history["val_binary_accuracy"][0])]) 

    # Outside the training loop we want to use the test data for performance metric evaluation.
    X_chunk_test, y_chunk_test   = load_chunk_test(np.float32, (0.02, 0.30), lenses_test, sources_test, negatives_test)



 

############################################################ Script    ############################################################




if __name__ == "__main__":
    main()