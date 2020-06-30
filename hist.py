import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from utils import *
import os
from skimage import data, img_as_float
from skimage import exposure
import time
import glob


# returns a numpy array with lens images from disk
def get_data_array(img_dims, path, fraction_to_load = 1.0, data_type = np.float32, are_sources=False, normalize = "per_image"):
    start_time = time.time()

    data_paths = []

    # Get all file names
    if are_sources:
        data_paths = glob.glob(path + "*/*.fits")
    else:
        data_paths = glob.glob(path + "*_r_*.fits")

    # Shuffle the filenames
    random.shuffle(data_paths)

    # How many are on disk?
    print("\nNumber of images on disk: {}, for {}".format(len(data_paths), path))

    # How many does the user actually want?
    num_to_actually_load = int(fraction_to_load*len(data_paths))
    data_paths = data_paths[0:num_to_actually_load]

    # Pre-allocate numpy array for the data
    data_array = np.zeros((len(data_paths),img_dims[0], img_dims[1], img_dims[2]),dtype=data_type)
    print("data array shape: {}".format(data_array.shape))

    # Load all the data in into the numpy array:
    for idx, filename in enumerate(data_paths):

        if are_sources:
            img = fits.getdata(filename).astype(data_type)
            # img = np.expand_dims(scipy.signal.fftconvolve(img, PSF_r, mode="same"), axis=2)
            if normalize == "per_image":
                img = normalize_img(img)
            data_array[idx] = img
        else:
            img = np.expand_dims(fits.getdata(filename), axis=2).astype(data_type)
            if normalize == "per_image":
                img = normalize_img(img)
            data_array[idx] = img

    if normalize == "per_array":
        return normalize_data_array(data_array)

    print("max array  = {}".format(np.amax(data_array)))
    print("min array  = {}".format(np.amin(data_array)))
    print("mean array = {}".format(np.mean(data_array)))
    print("median array = {}".format(np.median(data_array)))

    print("Loading data took: {} for folder: {}".format(hms(time.time() - start_time), path))
    return data_array


# A normlize function that normalizes a data array based on its maximum pixel value and minimum pixel value.
# So not normalization per image
def normalize_data_array(data_array):
    #at this point we are unsure what kind of value range is in the data array.
    return ((data_array - np.amin(data_array)) / (np.amax(data_array) - np.amin(data_array)))


def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.
    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


# Normalizationp per image
def normalize_img(numpy_img):
    return ((numpy_img - np.amin(numpy_img)) / (np.amax(numpy_img) - np.amin(numpy_img)))


data_type = np.float32
lenses_array = get_data_array((101,101,1), path="data/training/negatives/", fraction_to_load=0.25, data_type=data_type, are_sources=False, normalize="per_image")

for i in range(30):
    # Load img
    img = lenses_array[random.randint(0, lenses_array.shape[0])]
    img = np.squeeze(img)

    # Equalization
    img_eq = exposure.equalize_hist(img,nbins=256)

    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(img)

    # Display results
    fig = plt.figure(figsize=(8, 5))
    axes = np.zeros((2, 4), dtype=np.object)
    axes[0, 0] = fig.add_subplot(2, 4, 1)
    for i in range(1, 4):
        axes[0, i] = fig.add_subplot(2, 4, 1+i, sharex=axes[0,0], sharey=axes[0,0])
    for i in range(0, 4):
        axes[1, i] = fig.add_subplot(2, 4, 5+i)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
    ax_img.set_title('Original Image')

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel('Number of pixels')
    ax_hist.set_yticks(np.linspace(0, y_max, 5))

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 1])
    ax_img.set_title('Original Image')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
    ax_img.set_title('Histogram equalization')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
    ax_img.set_title('Adaptive equalization')

    ax_cdf.set_ylabel('Fraction of total intensity')
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    # prevent overlap of y-axis labels
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    fig.tight_layout()
    plt.show()