### This script attempts to generate lenses.
# Definitions:
#  "lenses" A simple galaxy without any lensing features.
#       It contains random Noise Galaxies. (NG(s))
#  "Noise Galaxy" - NG: A galaxy in an image, that is not
#       in the centre of the image. Therefore it is not considered 
#       the object of interest, therefore it is considered a noise object.

import glob
import time
import random
import numpy as np
import functools
from skimage import exposure
import scipy
from multiprocessing import Pool
from utils import *
from astropy.io import fits
import os
import matplotlib.pyplot as plt

# Returns a numpy array with lens images from disk
def load_img(path, data_type = np.float32, normalize_dat = "per_image"):
    return load_and_normalize_img(data_type, normalize_dat, path)


def load_and_normalize_img(data_type, normalize_dat, idx_and_filename):
    idx, filename = idx_and_filename
    if idx % 1000 == 0:
        print("Loading image #{}".format(idx))
    img = fits.getdata(filename).astype(data_type)                                         #read file and expand dims
    return np.expand_dims(normalize_function(img, normalize_dat, data_type), axis=2)       #normalize


# Simple case function to reduce line count in other function
def normalize_function(img, norm_type, data_type):
    if norm_type == "per_image":
        img = normalize_img(img)
    if norm_type == "adapt_hist_eq":
        img = normalize_img(img)
        img = exposure.equalize_adapthist(img).astype(data_type)
    return img


# Normalizationp per image
def normalize_img(numpy_img):
    return ((numpy_img - np.amin(numpy_img)) / (np.amax(numpy_img) - np.amin(numpy_img)))


# Loads all lenses into a data array, via multithreading
def load_lenses(fits_files):
    with Pool(24) as pool:
        print("\n\n\nLoading Training Data", flush=True)
        data_array = np.asarray(pool.map(load_img, enumerate(fits_files), chunksize=128), dtype=data_type)           # chunk_sizes = amount of data that will be given to one thread
    return data_array


def get_all_fits_paths():
    # Set all fits paths of the lenses
    train_path = os.path.join("data", "train", "lenses")
    val_path   = os.path.join("data", "validation", "lenses")
    test_path  = os.path.join("data", "test", "lenses")
    paths = [train_path, val_path, test_path]

    all_fits_paths = []
    for path in paths:
        all_fits_paths += glob.glob(path + "/*_r_*.fits")
    return all_fits_paths


def print_data_array_stats(data_array):
    print("------------")
    print("Dimensions: {}".format(data_array.shape), flush=True)
    print("Max array  = {}".format(np.amax(data_array)), flush=True)
    print("Min array  = {}".format(np.amin(data_array)), flush=True)
    print("Mean array = {}".format(np.mean(data_array)), flush=True)
    print("Median array = {}".format(np.median(data_array)), flush=True)
    print("Numpy nbytes = {},  GBs = {}".format(data_array.nbytes, bytes2gigabyes(data_array.nbytes)), flush=True)
    print("------------")


# Shows a random sample of the given data array, to the user.
def show_img_grid(data_array, columns=4, rows=4, seed=1245):
    random.seed(seed)
    img_count = int(columns * rows)
    rand_idxs = [random.choice(list(range(data_array.shape[0]))) for x in range(img_count)]

    fig=plt.figure(figsize=(8, 8))
    for idx, i in enumerate(range(1, columns*rows +1)):
        fig.add_subplot(rows, columns, i)
        ax = (fig.axes)[idx]
        ax.axis('off')
        ax.set_title("img idx = {}".format(rand_idxs[idx]))
        lens = np.squeeze(data_array[rand_idxs[idx]])
        plt.imshow(lens, origin='lower', interpolation='none', cmap='gray', vmin=0.0, vmax=1.0)

    plt.show()


########################################
# Parameters
data_type = np.float32
seed = 1234
########################################

# Load lenses.
all_fits_paths = get_all_fits_paths()
lenses         = load_lenses(all_fits_paths)
print_data_array_stats(lenses)


# View random sample of lenses.
show_img_grid(lenses, columns=4, rows=4, seed=seed)


