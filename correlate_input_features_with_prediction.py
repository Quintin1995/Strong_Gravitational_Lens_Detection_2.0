from compare_results import set_experiment_folder, set_models_folders, load_settings_yaml
from DataGenerator import DataGenerator
import glob
from Network import Network
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage import exposure
import os
import scipy
import pyfits
import random
from Parameters import Parameters
import pandas as pd


def get_h5_path_dialog(model_paths):
    h5_choice = int(input("\n\nWhich model do you want? A model selected on validation loss (1) or validation metric (2)? (int): "))
    if h5_choice == 1:
        h5_paths = glob.glob(os.path.join(model_paths[0], "checkpoints/*loss.h5"))
    elif h5_choice == 2:
        h5_paths = glob.glob(os.path.join(model_paths[0], "checkpoints/*metric.h5"))
    else:
        h5_paths = glob.glob(os.path.join(model_paths[0], "*.h5"))

    print("Choice h5 path: {}".format(h5_paths[0]))
    return h5_paths[0]


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
def merge_lenses_and_sources(lenses_array, sources_array, mock_lens_alpha_scaling = (0.02, 0.30)):
    X_train_positive = np.empty((lenses_array.shape[0], lenses_array.shape[1], lenses_array.shape[2], lenses_array.shape[3]), dtype=np.float32)
    Y_train_positive = np.ones(lenses_array.shape[0], dtype=np.float32)
    
    for i in range(lenses_array.shape[0]):
        lens   = lenses_array[i]
        source = sources_array[i]
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


def compute_PSF_r():
        ## This piece of code is needed for some reason that i will try to find out later.
        nx = 101
        ny = 101
        f1 = pyfits.open("data/PSF_KIDS_175.0_-0.5_r.fits")  # PSF
        d1 = f1[0].data
        d1 = np.asarray(d1)
        nx_, ny_ = np.shape(d1)
        PSF_r = np.zeros((nx, ny))  # output
        dx = (nx - nx_) // 2  # shift in x
        dy = (ny - ny_) // 2  # shift in y
        for ii in range(nx_):  # iterating over input array
            for jj in range(ny_):
                PSF_r[ii + dx][jj + dy] = d1[ii][jj]

        return PSF_r


# Opens dialog with the user to select a folder that contains models.
def get_model_paths(root_dir="models"):

    #Select which experiment path to take in directory structure
    experiment_folder = set_experiment_folder(root_folder=root_dir)

    # Can select 1 or multiple models.
    models_paths = set_models_folders(experiment_folder)
    return models_paths


# Select a random sample with replacement from all files.
def get_sample_lenses_and_sources(size=1000):
    lenses_path_train    = os.path.join("data", "train", "lenses")
    sources_path_train   = os.path.join("data", "train", "sources")
    # negatives_path_train = os.path.join("data", "train", "negatives")

    # Try to glob files in the given path
    sources_fnames = glob.glob(os.path.join(sources_path_train, "*/*.fits"))
    lenses_fnames  = glob.glob(os.path.join(lenses_path_train, "*_r_*.fits"))
    print("\nsources count {}".format(len(sources_fnames)))
    print("lenses count {}".format(len(lenses_fnames)))

    sources_fnames = random.sample(sources_fnames, size)
    lenses_fnames  = random.sample(lenses_fnames, size)
    return sources_fnames, lenses_fnames


# Normalizationp per image
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
def load_normalize_img(data_type, are_sources, normalize_dat, PSF_r, filenames):
    data_array = np.zeros((len(filenames), 101, 101, 1))

    for idx, filename in enumerate(filenames):
        if idx % 10 == 0:
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
sources_fnames, lenses_fnames = get_sample_lenses_and_sources(size=1000)

# 5.0 - Load lenses and sources in 4D numpy arrays
PSF_r = compute_PSF_r()  # Used for sources
lenses  = load_normalize_img(params.data_type, are_sources=False, normalize_dat="per_image", PSF_r=PSF_r, filenames=lenses_fnames)
sources = load_normalize_img(params.data_type, are_sources=True, normalize_dat="per_image", PSF_r=PSF_r, filenames=sources_fnames)

# 6.0 - Create mock lenses based on the sample
mock_lenses, _ = merge_lenses_and_sources(lenses, sources)

# 7.0 - Initialize and fill a pandas dataframe to store Source parameters
df = get_empty_dataframe()
df = fill_dataframe(df, sources_fnames)
print(df.head())

# 8.0 - Create a dataGenerator object, because the network class wants it
dg = DataGenerator(params, mode="no_training", do_shuffle_data=True, do_load_validation=False)

# 9.0 - Construct a Network object that has a model as property.
network = Network(params, dg, training=False)
network.model.load_weights(h5_path)


# 10.0 - Use the network to predict on the sample
preds = network.model.predict(mock_lenses)
preds = list(np.squeeze(preds))

low  = [pred for pred in preds if pred<0.5]
high = [pred for pred in preds if pred>=0.5]
print(len(low))
print(len(high))

# 11.0 - Make a plot of einstein radius and network certainty
einstein_radii = list(df["LENSER"])
plt.plot(einstein_radii, preds, 'o', color='blue')
plt.title("Einstein radius and network certainty")
plt.xlabel("Einstein Radius of source")
plt.ylabel("Model prediction")
plt.show()
s=3