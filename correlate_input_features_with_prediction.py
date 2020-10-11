import os
from compare_results import set_experiment_folder, set_models_folders, load_settings_yaml
from DataGenerator import DataGenerator
from Network import Network
import numpy as np
from Parameters import Parameters
import tensorflow as tf
import glob
from astropy.io import fits
from skimage import exposure
import scipy
import pyfits
import random
from einstein_radii_distribution import get_empty_dataframe, fill_dataframe


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

        # seds = np.loadtxt("data/SED_colours_2017-10-03.dat")

        # Rg = 3.30
        # Rr = 2.31
        # Ri = 1.71
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


############################## script ##############################

# Fix memory leaks if running on tensorflow 2
tf.compat.v1.disable_eager_execution()

# Model Selection from directory
model_paths = get_model_paths()

# Load params - used for normalization etc
# only choose the first one for now
yaml_path = glob.glob(os.path.join(model_paths[0], "run.yaml"))[0]
settings_yaml = load_settings_yaml(yaml_path)                                           # Returns a dictionary object.
params = Parameters(settings_yaml, yaml_path, mode="no_training")
params.data_type = np.float32 if params.data_type == "np.float32" else np.float32       # This must be done here, due to the json, not accepting this kind of if statement in the parameter class.

PSF_r = compute_PSF_r()

# Select random sample from the data (with replacement)
sources_fnames, lenses_fnames = get_sample_lenses_and_sources(size=10)

# Load lenses and sources in 4D numpy arrays
lenses, _ = load_normalize_img(params.data_type, are_sources=False, normalize_dat="per_image", PSF_r=PSF_r, filenames=lenses_fnames)
sources, source_params = load_normalize_img(params.data_type, are_sources=True, normalize_dat="per_image", PSF_r=PSF_r, filenames=sources_fnames)

# Initialize a pandas dataframe to store source parameters
df = get_empty_dataframe()
df = fill_dataframe(df, sources_fnames)

x=3