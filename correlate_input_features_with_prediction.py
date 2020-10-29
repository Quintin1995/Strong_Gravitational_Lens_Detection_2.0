from astropy.stats import sigma_clipped_stats
from astropy.io import fits
import cv2
from DataGenerator import DataGenerator
from functools import reduce, partial
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Network import Network
import numpy as np
import os
from Parameters import Parameters
from photutils import make_source_mask
import pyfits
import pandas as pd
import random
import scipy
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from utils import show2Imgs, calc_RMS, get_model_paths, get_h5_path_dialog, binary_dialog, set_experiment_folder, set_models_folders, load_settings_yaml, normalize_img, get_samples, normalize_function, compute_PSF_r

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# Make a simple 3D plot based on 3 continuous variables/features.
def plot_3D(con_fea1, con_fea2, con_fea3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(einstein_radii, alpha_scalings, predictions, cmap='viridis')
    plt.xlabel("Einstein Radius")
    plt.ylabel("Source Intensity Scaling")
    plt.title("Influence of brightness intensity scaling of Source and Einstein Radius")
    plt.show()



# This plot approximates a 3D plot. A 2D plane gets hexogonal tesselation.
# A datapoint in a 2D space (so based on 2 continuous variables/features)
# Gets assigned a color based on model average model certainty in that tile, or
# assigned a color based on the True Positive Rate in that tile.
def hexbin_plot(con_fea1, con_fea2, predictions, gridsize=10, calc_TPR=True):
    if calc_TPR:
        threshold = float(input("\n\nWhat should the model threshold be? (float): "))
    # Define a partial function so that we can pass parameters to the reduce_C_function.
    fig, ax = plt.subplots()
    if calc_TPR:
        reduce_function = partial(reduce_C_function_TPR, threshold=threshold)
        plt.hexbin(x=con_fea1, y=con_fea2, C=predictions, gridsize=gridsize, cmap='copper', reduce_C_function=reduce_function)
    else:
        plt.hexbin(x=con_fea1, y=con_fea2, C=predictions, gridsize=gridsize, cmap='copper')
    plt.xlabel("Einstein Radius")
    plt.ylabel("Source Brighness Scaling")
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 15
    if calc_TPR:
        plt.title("Einstein Radius versus Brighness Scaling of Source - TPR (th={})".format(threshold))
        cbar.ax.set_ylabel('True Positive Ratio per bin', rotation=270)
    else:
        plt.title("Einstein Radius versus Brighness Scaling of Source and Model Certainty")
        cbar.ax.set_ylabel('Model Prediction', rotation=270)
    plt.show()


# Based on Einstein Radius and pixel intensity, assign a color based on the model prediction.
def show_scatterplot_ER_vs_intensity_and_certainty(einstein_radii, alpha_scalings, predictions):
    fig, ax = plt.subplots()
    plt.scatter(x=einstein_radii, y=alpha_scalings, c=predictions, cmap='copper')   #cmap {'winter', 'cool', 'copper'}
    plt.xlabel("Einstein Radius")
    plt.ylabel("Source Brighness Scaling")
    plt.title("Einstein Radius and Brighness Scaling of Source versus Model Prediction")
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Prediction value Model', rotation=270)


# Plot Signal to Noise Ratio (binned) versus the true positive rate of that bin.
def plot_SNR_vs_TPR(predictions, SNRs, threshold, num_bins=10):
    
    # Define the bins as an array with floats.
    bins = np.arange(0.0, 1.0, 1.0/num_bins)

    # Array that keeps track of the count of True Positives per bin.
    TPs = np.zeros((num_bins))
    # Array that keeps track of the count of False Negatives per bin.
    FNs = np.zeros((num_bins))
    
    # The SNRs and predictions are still in the same order.
    for pred, SNR in zip(predictions, SNRs):
        if pred < threshold:
            FNs[int(SNR*num_bins)] += 1.0       # The max of SNR = 1.0, therefore the SNR decides in which bin it ends up.
        else:
            TPs[int(SNR*num_bins)] += 1.0

    # Calculate the True Positive Rate per bin, set it to zero, if we divide by 0.
    TPRs = [tp/(tp+fn) if (tp+fn) != 0 else 0 for tp, fn in zip(list(TPs), list(FNs))]

    # Plot the results
    plt.clf()
    plt.plot(bins, TPRs)
    plt.title("SNR versus TPR")
    plt.xlabel("SNR")
    plt.ylabel("TPR")
    plt.grid(color='grey', linestyle='dashed', linewidth=1)
    plt.show()



# Function used by hexbin in order to reduce all datapoints to a singular TPR value.
def reduce_C_function_TPR(predictions, threshold):
    #We can calculate Recall or True Positive Rate
    TP_count, FN_count = 0, 0
    for pred in predictions:
        if pred >= threshold:
            TP_count += 1
        else:
            FN_count += 1
    TPR = TP_count / (TP_count + FN_count)
    print("TPR = {}, sample size={}".format(TPR, len(predictions)))
    return TPR


def get_empty_dataframe():
    # Define which parameter to collect
    column_names = ["LENSER", "LENSAR", "LENSAA", "LENSSH", "LENSSA", "SRCER", "SRCX", "SRCY", "SRCAR", "SRCAA", "SRCSI", "path"]

    # Create a dataframe for parameter storage
    return pd.DataFrame(columns = column_names)


# Fill the dataframe with SIE parameters and Sersic parameters
# use numpy arrays for speed
def fill_dataframe(df, paths):

    # Define numpy arrays to temporarily hold the parameters
    # LENS PARAMETERS
    LENSER = np.zeros((len(paths),), dtype=np.float32)
    LENSAR = np.zeros((len(paths),), dtype=np.float32)
    LENSAA = np.zeros((len(paths),), dtype=np.float32)
    LENSSH = np.zeros((len(paths),), dtype=np.float32)
    LENSSA = np.zeros((len(paths),), dtype=np.float32)
    # SERSIC PARAMETERS
    SRCER = np.zeros((len(paths),), dtype=np.float32)
    SRCX = np.zeros((len(paths),), dtype=np.float32)
    SRCY = np.zeros((len(paths),), dtype=np.float32)
    SRCAR = np.zeros((len(paths),), dtype=np.float32)
    SRCAA = np.zeros((len(paths),), dtype=np.float32)
    SRCSI = np.zeros((len(paths),), dtype=np.float32)

    # Loop over all sources files
    for idx, filename in enumerate(paths):
        if idx % 1000 == 0:
            print("processing source idx: {}".format(idx))
        hdul = fits.open(filename)

        LENSER[idx] = hdul[0].header["LENSER"]
        LENSAR[idx] = hdul[0].header["LENSAR"] 
        LENSAA[idx] = hdul[0].header["LENSAA"] 
        LENSSH[idx] = hdul[0].header["LENSSH"] 
        LENSSA[idx] = hdul[0].header["LENSSA"] 

        SRCER[idx] = hdul[0].header["SRCER"] 
        SRCX[idx] = hdul[0].header["SRCX"] 
        SRCY[idx] = hdul[0].header["SRCY"] 
        SRCAR[idx] = hdul[0].header["SRCAR"] 
        SRCAA[idx] = hdul[0].header["SRCAA"] 
        SRCSI[idx] = hdul[0].header["SRCSI"]

    df["LENSER"] = LENSER
    df["LENSAR"] = LENSAR
    df["LENSAA"] = LENSAA
    df["LENSSH"] = LENSSH
    df["LENSSA"] = LENSSA

    df["SRCER"] = SRCER
    df["SRCX"] = SRCX
    df["SRCY"] = SRCY
    df["SRCAR"] = SRCAR
    df["SRCAA"] = SRCAA
    df["SRCSI"] = SRCSI

    df["path"] = paths
    return df


# Merge a single lens and source together into a mock lens.
def merge_lens_and_source(lens, source, mock_lens_alpha_scaling = (0.02, 0.30), show_imgs = False, do_plot=False, noise_fac=2.0):
    
    # Keep a copy of the lens end source normalized.
    lens_norm   = normalize_img(np.copy(lens))
    source_norm = normalize_img(np.copy(source))

    # Determine the noise level of the lens before merging it with the source
    rms_lens = calc_RMS(lens)

    # Determine alpha scaling drawn from the interval [0.02,0.3]
    alpha_scaling = np.random.uniform(mock_lens_alpha_scaling[0], mock_lens_alpha_scaling[1])

    # We rescale the brightness of the simulated source to the peak brightness
    source = source / np.amax(source) * np.amax(lens) * alpha_scaling
    
    # Get indexes where lensing features have pixel values below: noise_factor*noise
    idxs = np.where(source < (noise_fac * rms_lens))

    # Make a copy of the source and set all below noise_factor*noise to 0.0
    trimmed_source = np.copy(source)
    trimmed_source[idxs] = 0.0

    # Calculate surface area of visual features that are stronger than noise_fac*noise
    (x_idxs,_,_) = np.where(source >= (noise_fac * rms_lens))
    feature_area_frac = len(x_idxs) / (source.shape[0] * source.shape[1])
    
    # Add lens and source together 
    mock_lens = lens_norm + source_norm / np.amax(source_norm) * np.amax(lens_norm) * alpha_scaling
    
    # Perform a square root stretch to emphesize lower luminosity features.
    mock_lens = np.sqrt(mock_lens)
    
    # Basically removes negative values - should not be necessary, because all input data should be normalized anyway. (I will leave it for now, but should be removed soon.)
    # mock_lens_sqrt = mock_lens_sqrt.clip(min=0.0, max=1.0)
    mock_lens = mock_lens.clip(min=0.0, max=1.0)

    if do_plot:
        show2Imgs(source, trimmed_source, "Lens max pixel: {0:.3f}".format(np.amax(source)), "Source max pixel: {0:.3f}".format(np.amax(trimmed_source)))
        # show2Imgs(mock_lens, mock_lens_sqrt, "mock_lens max pixel: {0:.3f}".format(np.amax(mock_lens)), "mock_lens_sqrt max pixel: {0:.3f}".format(np.amax(mock_lens_sqrt)))

    return mock_lens, alpha_scaling, feature_area_frac


# This function should read images from the lenses- and sources data array,
# and merge them together into a lensing system, further described as 'mock lens'.
# These mock lenses represent a strong gravitational lensing system that should 
# get the label 1.0 (positive label). 
def merge_lenses_and_sources(lenses_array, sources_array, mock_lens_alpha_scaling = (0.02, 0.30), noise_fac=2.0):
    X_train_positive = np.empty((lenses_array.shape[0], lenses_array.shape[1], lenses_array.shape[2], lenses_array.shape[3]), dtype=np.float32)
    Y_train_positive = np.ones(lenses_array.shape[0], dtype=np.float32)
    
    # For correlating input features with prediction, we also want to keep track of alpha scaling.
    # This is the ratio between peak brighntess of the lens versus that of the source.
    alpha_scalings = list()
    feature_areas_fracs  = list()   # We want to keep track of feature area size in order to correlate it with neural network prediction values

    for i in range(lenses_array.shape[0]):
        lens   = lenses_array[i]
        source = sources_array[i]
        mock_lens, alpha_scaling, feature_area_frac = merge_lens_and_source(lens, source, mock_lens_alpha_scaling, noise_fac=noise_fac)

        # Uncomment this code if you want to inspect how a lens, source and mock lens look before they are merged.
        # l = np.squeeze(normalize_img(lens))
        # s = np.squeeze(normalize_img(source))
        # m = np.squeeze(normalize_img(mock_lens))
        # plt.imshow(l, cmap='Greys_r')
        # plt.title("lens")
        # plt.show()
        # plt.imshow(s, cmap='Greys_r')
        # plt.title("source")
        # plt.show()
        # plt.imshow(m, cmap='Greys_r')
        # plt.title("mock lens")
        # plt.show()
        X_train_positive[i] = mock_lens
        alpha_scalings.append(alpha_scaling)
        feature_areas_fracs.append(feature_area_frac)

    return X_train_positive, Y_train_positive, alpha_scalings, feature_areas_fracs


# If the data array contains sources, then a PSF_r convolution needs to be performed over the image.
# There is also a check on whether the loaded data already has a color channel dimension, if not create it.
def load_normalize_img(data_type, are_sources, normalize_dat, PSF_r, filenames):
    data_array = np.zeros((len(filenames), 101, 101, 1))

    for idx, filename in enumerate(filenames):
        if idx % 100 == 0:
            print("loaded {} images".format(idx), flush=True)
        if are_sources:
            img = fits.getdata(filename).astype(data_type)
            img = scipy.signal.fftconvolve(img, PSF_r, mode="same")                                # Convolve with psf_r, has to do with camara point spread function.
            img = np.expand_dims(normalize_function(img, normalize_dat, data_type), axis=2)       # Expand color channel and normalize
        else:
            img = fits.getdata(filename).astype(data_type)
            if img.ndim == 3:                                                                      # Some images are stored with color channel
                img = normalize_function(img, normalize_dat, data_type)
            elif img.ndim == 2:                                                                    # Some images are stored without color channel
                img = np.expand_dims(normalize_function(img, normalize_dat, data_type), axis=2)
        data_array[idx] = img
    return data_array


# Makes a plot of a feature versus model prediction in a 2D plot. The plot is seperated into
# two classes, positive and negative, based on a given threshold.
def plot_feature_versus_prediction(predictions, feature_list, threshold=None, title=""):
    if threshold == None:
        threshold = float(input("What model threshold do you want to set (float): "))
    idx_positives  = [(idx, pred) for idx, pred in enumerate(predictions) if pred>=threshold]
    idx_negatives  = [(idx, pred) for idx, pred in enumerate(predictions) if pred<threshold]

    print("Number positive: {}".format(len(idx_positives)))
    print("Number negative: {}".format(len(idx_negatives)))
    print("Minimum {} value = {}".format(title, min(feature_list)))
    print("Maximum {} value = {}".format(title, max(feature_list)))

    positives, preds_positives = list(), list()
    for idx, pred in idx_positives:
        positives.append(feature_list[idx])
        preds_positives.append(pred)

    negatives, preds_negatives = list(), list()
    for idx, pred in idx_negatives:
        negatives.append(feature_list[idx])
        preds_negatives.append(pred)

    plt.plot(positives, preds_positives, 'o', color='blue', label="positives {}".format(len(preds_positives)))
    plt.plot(negatives, preds_negatives, 'o', color='red', label="negatives {}".format(len(preds_negatives)))

    plt.title("{} and network certainty. FNR: {:.2f}".format(title, len(preds_negatives)/sample_size))
    plt.xlabel("{} of source".format(title))
    plt.ylabel("Model prediction")
    plt.legend()
    plt.show()




############################################################ script ############################################################

# 1.0 - Fix memory leaks if running on tensorflow 2
tf.compat.v1.disable_eager_execution()


# 2.0 - Model Selection from directory
model_paths = get_model_paths()


# 2.1 - Select a weights file. There are 2 for each model. Selected based on either validation loss or validation metric. The metric can differ per model.
h5_path = get_h5_path_dialog(model_paths)


# 3.0 - Load params - used for normalization etc -
yaml_path = glob.glob(os.path.join(model_paths[0], "run.yaml"))[0]                      # Only choose the first one for now
settings_yaml = load_settings_yaml(yaml_path)                                           # Returns a dictionary object.
params = Parameters(settings_yaml, yaml_path, mode="no_training")                       # Create Parameter object
params.data_type = np.float32 if params.data_type == "np.float32" else np.float32       # This must be done here, due to the json, not accepting this kind of if statement in the parameter class.


# 4.0 - Select random sample from the data (with replacement)
sample_size = int(input("How many samples do you want to create and run (int): "))
sources_fnames, lenses_fnames, negatives_fnames = get_samples(size=sample_size, deterministic=False)


# 5.0 - Load lenses and sources in 4D numpy arrays
PSF_r = compute_PSF_r()  # Used for sources only
# lenses  = load_normalize_img(params.data_type, are_sources=False, normalize_dat="per_image", PSF_r=PSF_r, filenames=lenses_fnames)
# sources = load_normalize_img(params.data_type, are_sources=True, normalize_dat="per_image", PSF_r=PSF_r, filenames=sources_fnames)

# 5.1 - Load unnormalized data in order to calculate the amount of noise in a lens. 
lenses_unnormalized    = load_normalize_img(params.data_type, are_sources=False, normalize_dat="None", PSF_r=PSF_r, filenames=lenses_fnames)
sources_unnormalized   = load_normalize_img(params.data_type, are_sources=True, normalize_dat="None", PSF_r=PSF_r, filenames=sources_fnames)


# 6.0 - Create mock lenses based on the sample
noise_fac = 2.0
# mock_lenses, pos_y, alpha_scalings, SNRs = merge_lenses_and_sources(lenses, sources, noise_fac=noise_fac)
mock_lenses, pos_y, alpha_scalings, SNRs = merge_lenses_and_sources(lenses_unnormalized, sources_unnormalized, noise_fac=noise_fac)


# 7.0 - Initialize and fill a pandas dataframe to store Source parameters
df = get_empty_dataframe()
df = fill_dataframe(df, sources_fnames)


# 8.0 - Create a dataGenerator object, because the network class wants it
dg = DataGenerator(params, mode="no_training", do_shuffle_data=True, do_load_validation=False)


# 9.0 - Construct a Network object that has a model as property.
network = Network(params, dg, training=False)
network.model.load_weights(h5_path)


# 9.5 - Create a heatmap - Gradient Class Activation Map (Grad_CAM) of a given a positive image.
if binary_dialog("Do Grad-CAM?"):
    another_list = ["batch_normalization_16", "activation_12", "activation_8"]
    another_list = ["add", "add_1", "add_2", "add_3", "add_4", "add_5", "add_6", "add_7"]
    Grad_CAM_plot(mock_lenses, network.model, layer_list=another_list, plot_title="Positive Example", labels=pos_y)
    negatives = load_normalize_img(params.data_type, are_sources=False, normalize_dat="per_image", PSF_r=PSF_r, filenames=negatives_fnames)
    Grad_CAM_plot(negatives, network.model, layer_list=another_list, plot_title="Negative Example", labels=pos_y*0.0)


# 10.0 - Use the network to predict on the sample
einstein_radii = list(df["LENSER"])
predictions = network.model.predict(mock_lenses)
predictions = list(np.squeeze(predictions))


# 11.0 - Make a plot of feature area size versus network certainty.
if binary_dialog("Do you want to plot SNR versus prediction?"):
    plot_feature_versus_prediction(predictions, SNRs, threshold=0.5, title="Image Ratio of Source above {}x noise level".format(noise_fac))


# 12.0 - Make a plot of einstein radius and network certainty
if binary_dialog("Do you want to plot Einstein Radii versus prediction?"):
    plot_feature_versus_prediction(predictions, einstein_radii, threshold=0.5, title="Einstein Radii")


# 13.0 - Lets create a 2D matrix with x-axis and y-axis being Einstein radius and alpha scaling.
if binary_dialog("Do scatter plot and hexbin plot?"):
    show_scatterplot_ER_vs_intensity_and_certainty(einstein_radii, alpha_scalings, predictions)
    hexbin_plot(einstein_radii, alpha_scalings, predictions, gridsize=10, calc_TPR=False)
    hexbin_plot(einstein_radii, alpha_scalings, predictions, gridsize=10, calc_TPR=True)
    

# Plot a binned SNR on the x-axis versus TPR on the y-axis.
if binary_dialog("Do SNR vs TPR plot?"):
    threshold = float(input("\n\nWhat model threshold should be used? (float): "))
    plot_SNR_vs_TPR(predictions, SNRs, threshold=threshold, num_bins=50)


# 14.0 - Lets try to make a 3D plot, with:
# x-axis is Einstein Radius, # y-axis is alpha_scaling, # z-axis is prediction of model.
if binary_dialog("Make 3D plot?"):
    plot_3D(con_fea1 = einstein_radii, con_fea2 = alpha_scalings, con_fea3 = predictions)