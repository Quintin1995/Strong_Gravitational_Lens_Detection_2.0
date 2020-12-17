from utils import create_dir_if_not_exists, choose_ensemble_members, get_h5_path_dialog, get_samples, get_fnames_from_disk, load_normalize_img, compute_PSF_r
import argparse
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation
from tensorflow.keras.initializers import glorot_uniform
import os
import time
import random
import numpy as np
from create_ensemble import load_chunk_test, load_chunk_val, load_chunk_train

########################### Description ###########################
## Take user through dialog that lets the user select trained models.
## Construct a model/network that takes in training data and outputs a vector of length=#members in enemble
## We use softmax on the output vector and determine which member/model of the ensemble gets to predict on the input image.
## Via this method, a network selects which network gets to predict on the input image, based on information within the image.
########################### End Description #######################






############################################################ Functions ############################################################

# Return a neural network, considered to be the ensemble model. 
# This model should predict which network in the ensemble gets
# to predict the final prediction on the input image.
def get_ensemble_model(input_shape=(101,101,1), num_outputs=3):

    inp = Input(shape=input_shape)
    
    x = Conv2D(64, kernel_size=3, padding='valid', kernel_initializer=glorot_uniform(seed=0))(inp)
    x = BatchNormalization(axis=3)(x)   # Axis three is color channel
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

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

    model = Model(inputs=inp, outputs=out)

    model.summary()

    return model


# Deal with input arguments
def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_name", help="Name of the ensemble. There will be a folder with the given name.", default="test_ensemble", required=False)
    parser.add_argument("--n_chunks", help="The amount of chunks that the ensemble will be trained for.", default=1, required=False)
    # parser.add_argument("--method", help="What ensemble method should be used? To determine model weights? For example: Nelder-Mead.", default="Nelder-Mead", required=False)
    parser.add_argument("--network", help="Name of the network used, in order to load the right architecture", default="simple_net", required=False)
    args = parser.parse_args()
    return args


def _set_and_create_dirs(args):
    root_dir_ensembles = "ensembles"
    ensemble_dir = os.path.join(root_dir_ensembles, args.ensemble_name)
    create_dir_if_not_exists(root_dir_ensembles, verbatim=False)
    create_dir_if_not_exists(ensemble_dir)
    return ensemble_dir




def main():
    # Parse input arguments
    args = _get_arguments()

    # Set root directory, ensemble directory and create them.
    ensemble_dir = _set_and_create_dirs(args)

    # Fix memory leaks if running on tensorflow 2
    tf.compat.v1.disable_eager_execution()

    # Model Selection from directory - Select multiple models
    model_paths = choose_ensemble_members()

    # Select a weights file. There are 2 for each model. Selected based on either validation loss or validation metric. The metric can differ per model.
    h5_paths = get_h5_path_dialog(model_paths)

    sources_fnames_train, lenses_fnames_train, negatives_fnames_train = get_fnames_from_disk(lens_frac=1.0, source_frac=0.085, negative_frac=1.0, type_data="train", deterministic=False)
    sources_fnames_val, lenses_fnames_val, negatives_fnames_val       = get_fnames_from_disk(lens_frac=1.0, source_frac=0.5, negative_frac=1.0, type_data="validation", deterministic=False)
    sources_fnames_test, lenses_fnames_test, negatives_fnames_test    = get_fnames_from_disk(lens_frac=1.0, source_frac=0.5, negative_frac=1.0, type_data="test", deterministic=False)
    
    # Load all the data into memory
    PSF_r = compute_PSF_r()  # Used for sources only
    lenses_train      = load_normalize_img(np.float32, are_sources=False, normalize_dat="per_image", PSF_r=PSF_r, filenames=lenses_fnames_train)
    sources_train     = load_normalize_img(np.float32, are_sources=True, normalize_dat="per_image", PSF_r=PSF_r, filenames=sources_fnames_train)
    negatives_train   = load_normalize_img(np.float32, are_sources=False, normalize_dat="per_image", PSF_r=PSF_r, filenames=negatives_fnames_train)
    
    lenses_val       = load_normalize_img(np.float32, are_sources=False, normalize_dat="per_image", PSF_r=PSF_r, filenames=lenses_fnames_val)
    sources_val      = load_normalize_img(np.float32, are_sources=True, normalize_dat="per_image", PSF_r=PSF_r, filenames=sources_fnames_val)
    negatives_val    = load_normalize_img(np.float32, are_sources=False, normalize_dat="per_image", PSF_r=PSF_r, filenames=negatives_fnames_val)

    lenses_test      = load_normalize_img(np.float32, are_sources=False, normalize_dat="per_image", PSF_r=PSF_r, filenames=lenses_fnames_test)
    sources_test     = load_normalize_img(np.float32, are_sources=True, normalize_dat="per_image", PSF_r=PSF_r, filenames=sources_fnames_test)
    negatives_test   = load_normalize_img(np.float32, are_sources=False, normalize_dat="per_image", PSF_r=PSF_r, filenames=negatives_fnames_test)

    for i in range(args.n_chunk):
        X_chunk_train, y_chunk_train = load_chunk_train(int(args.sample_size), lenses_train, negatives_train, sources_train, np.float32, mock_lens_alpha_scaling=(0.02, 0.30))
        X_chunk_val, y_chunk_val     = load_chunk_val(lenses_val, sources_val, negatives_val, mock_lens_alpha_scaling=(0.02, 0.30))

        #1. load the ensemble network and predict on train chunk
        # # Construct input dependent ensemble model
        ens_model = get_ensemble_model(input_shape=(101,101,1), num_outputs=len(model_paths))

        #2. Load the individual networks and predict on the train chunk

        #2. Create prediction matrix (y_hat_matrix), where each column corresponds to the prediction of a single model in the ensemble
    
        #3. 

    X_chunk_test, y_chunk_test   = load_chunk_test(np.float32, (0.02, 0.30), lenses_test, sources_test, negatives_test)



 

############################################################ Script    ############################################################




if __name__ == "__main__":
    main()