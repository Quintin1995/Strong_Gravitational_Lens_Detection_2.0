import numpy as np
from scipy import ndimage
import scipy
import glob
import itertools
import threading
import skimage.transform
import skimage.io
from skimage import exposure
import gzip
import os
import queue
import multiprocessing as mp
import pickle
from astropy.io import fits
import subprocess
import math
import pyfits
import time
from PIL import Image
import random
from utils import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from multiprocessing import Pool
import functools


# Normalizationp per image
def normalize_img(numpy_img):
    return ((numpy_img - np.amin(numpy_img)) / (np.amax(numpy_img) - np.amin(numpy_img)))


# Simple case function to reduce line count in other function
def normalize_function(img, norm_type, data_type):
    if norm_type == "per_image":
        img = normalize_img(img)
    if norm_type == "adapt_hist_eq":
        # img = normalize_img(img)
        img = exposure.equalize_adapthist(img).astype(data_type)
    return img


# If the data array contains sources, then a PSF_r convolution needs to be performed over the image.
# There is also a check on whether the loaded data already has a color channel dimension, if not create it.
def load_and_normalize_img(data_type, are_sources, normalize_dat, PSF_r, idx_filename):
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


class DataGenerator:
    def __init__(self, params, mode="training", do_shuffle_data=True, do_load_validation=True, *args, **kwargs):
        self.params = params
        self.PSF_r = self._compute_PSF_r()

        # We want a 80% probability of selecting from the contaminants set, and 20% probability of selecting an LRG from the lenses set.
        self.negative_sample_contaminant_prob = 0.8

        with Pool(24) as p:
            if mode == "training":
                # Load all training data
                print("\n\n\nLoading Training Data", flush=True)
                self.Xsources_train   = self.get_data_array(path=self.params.sources_path_train,
                                                            fraction_to_load=self.params.fraction_to_load_sources_train,
                                                            are_sources=True,
                                                            normalize_dat=self.params.normalize,
                                                            do_shuffle=do_shuffle_data,
                                                            pool=p)
                self.Xnegatives_train = self.get_data_array(path=self.params.negatives_path_train,
                                                            fraction_to_load=self.params.fraction_to_load_negatives_train,
                                                            are_sources=False,
                                                            normalize_dat=self.params.normalize,
                                                            do_shuffle=do_shuffle_data,
                                                            pool=p)
                self.Xlenses_train = self.get_data_array(path=self.params.lenses_path_train,
                                                            fraction_to_load=self.params.fraction_to_load_lenses_train,
                                                            are_sources=False,
                                                            normalize_dat=self.params.normalize,
                                                            do_shuffle=do_shuffle_data,
                                                            pool=p)
            if do_load_validation:
                # Load all validation data.
                print("\n\n\nLoading Validation Data", flush=True)
                self.Xsources_validation = self.get_data_array(path=self.params.sources_path_validation,
                                                            fraction_to_load=self.params.fraction_to_load_sources_vali,
                                                            are_sources=True,
                                                            normalize_dat=self.params.normalize,
                                                            do_shuffle=do_shuffle_data,
                                                            pool=p)
                self.Xnegatives_validation = self.get_data_array(path=self.params.negatives_path_validation,
                                                            fraction_to_load=self.params.fraction_to_load_negatives_vali,
                                                            are_sources=False,
                                                            normalize_dat=self.params.normalize,
                                                            do_shuffle=do_shuffle_data,
                                                            pool=p)
                self.Xlenses_validation    = self.get_data_array(path=self.params.lenses_path_validation,
                                                            fraction_to_load=self.params.fraction_to_load_lenses_vali,
                                                            are_sources=False,
                                                            normalize_dat=self.params.normalize,
                                                            do_shuffle=do_shuffle_data,
                                                            pool=p)
        p.join()

        ###### Step 5.0 - Data Augmentation - Data Generator Keras - Training Generator is based on train data array.
        self.train_generator = ImageDataGenerator(
                rotation_range=params.aug_rotation_range,
                width_shift_range=params.aug_width_shift_range,
                height_shift_range=params.aug_height_shift_range,
                zoom_range=params.aug_zoom_range,
                horizontal_flip=params.aug_do_horizontal_flip,
                fill_mode=params.aug_default_fill_mode)

        ###### Step 5.1 - Data Augmentation - Data Generator Keras - Validation Generator is based on test data for now
        self.validation_generator = ImageDataGenerator(
                horizontal_flip=params.aug_do_horizontal_flip,
                fill_mode=params.aug_default_fill_mode)


    def _compute_PSF_r(self):
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


    # Returns a numpy array with lens images from disk
    def get_data_array(self, path, pool, fraction_to_load = 1.0, data_type = np.float32, are_sources=False, normalize_dat = "per_image", do_shuffle=True):
        
        # Check if normalization is a valid entry among the acceptable possibilities.
        if normalize_dat not in ("per_image", "per_array", "adapt_hist_eq"):
            raise Exception('Normalization is not initialized, check if normalization is set correctly in run.yaml')
        start_time = time.time()

        # Try to glob files in the given path
        data_paths = glob.glob(path + "*_r_*.fits")
        
        # if the previous glob fails, then the files are probably in a sub-directory. This sub-directory structure is preferred over bulk files in one folder.
        if len(data_paths) == 0:
            data_paths = glob.glob(os.path.join(path, "*/*.fits"))

        # Shuffle the filenames
        if do_shuffle:
            random.shuffle(data_paths)

        # How many are on disk?
        print("\nNumber of images on disk: {}, for {}".format(len(data_paths), path), flush=True)

        # How many does the user actually want?
        num_to_actually_load = int(fraction_to_load*len(data_paths))
        data_paths = data_paths[0:num_to_actually_load]

        print("Loading...", flush=True)

        # Define a partial function
        f = functools.partial(load_and_normalize_img, data_type, are_sources, normalize_dat, self.PSF_r)

        # Load all the data in into the numpy array:
        data_array = np.asarray(pool.map(f, enumerate(data_paths), chunksize=128),dtype=data_type)           # Chunksize = Amount of data that will be given to one thread.

        # Normalize per data array.
        if normalize_dat == "per_array":                                                               #normalize
            return self.normalize_data_array(data_array)

        print("data array shape: {}".format(data_array.shape), flush=True)
        print("Max array  = {}".format(np.amax(data_array)), flush=True)
        print("Min array  = {}".format(np.amin(data_array)), flush=True)
        print("Mean array = {}".format(np.mean(data_array)), flush=True)
        print("Median array = {}".format(np.median(data_array)), flush=True)
        print("Numpy nbytes = {},  GBs = {}".format(data_array.nbytes, bytes2gigabyes(data_array.nbytes)), flush=True)
        print("Normalization = {}".format(normalize_dat), flush=True)
        print("Loading data took: {} for folder: {}".format(hms(time.time() - start_time), path), flush=True)

        return data_array


    # A normlize function that normalizes a data array based on its maximum pixel value and minimum pixel value.
    # So not normalization per image
    def normalize_data_array(self, data_array):
        #at this point we are unsure what kind of value range is in the data array.
        return ((data_array - np.amin(data_array)) / (np.amax(data_array) - np.amin(data_array)))
    

    # Data_rray = numpy data array that has 4 dimensions (num_imgs, img_width, img_height, num_channels)
    # Label = label that is assigned to the data array, preferably 0.0 or 1.0, set to anything else like a string or something, to not assign a label vector.
    # Test_fraction = Percentage of the data array that will be assigned to the test data_array
    def split_train_test_data(self, data_array, label, test_fraction=0.2):
        num_imgs = data_array.shape[0]

        X_train = data_array[0:int(num_imgs*(1.0-test_fraction))]
        X_test  = data_array[int(num_imgs*(1.0-test_fraction)):num_imgs]

        y_train = None
        y_test = None

        if label == 0.0:
            y_train = np.zeros(X_train.shape[0])
            y_test  = np.zeros(X_test.shape[0])

        if label == 1.0:
            y_train = np.ones(X_train.shape[0])
            y_test  = np.ones(X_test.shape[0])

        return X_train, X_test, y_train, y_test

    
    # A simple wrapper function in order to keep other functions shorter in code length
    def _fill_arguments_max_tree_segmenter(self, X_chunk):
        from max_tree_segmenter import max_tree_segmenter
        return max_tree_segmenter(X_chunk,
                    do_square_crop=self.params.do_square_crop,
                    square_crop_size=self.params.crop_size,
                    do_circular_crop=self.params.do_circular_crop,
                    do_scale=self.params.do_scale,
                    do_floodfill=self.params.do_floodfill,
                    x_scale=self.params.x_scale,
                    tolerance=self.params.tolerance,
                    area_th=self.params.area_th,
                    conv_method=self.params.conv_method,
                    ksize=(self.params.ksize,self.params.ksize),
                    use_seg_imgs=self.params.use_seg_imgs)
    

    # Loading a chunk into memory
    def load_chunk(self, chunksize, X_lenses, X_negatives, X_sources, data_type, mock_lens_alpha_scaling):
        start_time = time.time()

        # Half a chunk positive and half negative
        num_positive = int(chunksize / 2)
        num_negative = int(chunksize / 2)
        
        # Get mock lenses data and labels
        X_pos, y_pos = self.merge_lenses_and_sources(X_lenses, X_sources, num_positive, data_type, mock_lens_alpha_scaling, do_deterministic=False)
            
        # Store Negative data in numpy array and create label vector
        X_neg = np.empty((num_negative, X_pos.shape[1], X_pos.shape[2], X_pos.shape[3]), dtype=data_type)
        y_neg = np.zeros(X_neg.shape[0], dtype=data_type)

        # We want a 80% probability of selecting from the contaminants set, and 20% probability of selecting an LRG from the lenses set.
        negative_sample_contaminant_prob = 0.8
        for i in range(num_negative):
            if random.random() <= negative_sample_contaminant_prob:
                X_neg[i] = np.sqrt(X_negatives[random.randint(0, X_negatives.shape[0] - 1)])        #sqrt stretch is needed, because the positive examples are also sqrt stretched
            else:
                X_neg[i] = np.sqrt(X_lenses[random.randint(0, X_lenses.shape[0] - 1)])        #sqrt stretch is needed, because the positive examples are also sqrt stretched

        # Concatenate the positive and negative examples into one chunk (also the labels)
        X_chunk = np.concatenate((X_pos, X_neg))
        y_chunk = np.concatenate((y_pos, y_neg))

        if self.params.do_max_tree_seg:
            X_chunk = self._fill_arguments_max_tree_segmenter(X_chunk)

        print("Creating chunk took: {}, chunksize: {}".format(hms(time.time() - start_time), chunksize), flush=True)
        return X_chunk, y_chunk


    # Loading a valiation chunk into memory
    def load_chunk_val(self, data_type, mock_lens_alpha_scaling):
        start_time = time.time()
        num_positive = self.Xlenses_validation.shape[0]
        # num_negative = self.Xnegatives_validation.shape[0] + self.Xlenses_validation.shape[0]   #also num positive, because the unmerged lenses with sources are also deemed a negative sample.
        
        # Lets create a balanced validation set.
        num_negative = num_positive

        # Get mock lenses data and labels
        X_pos, y_pos = self.merge_lenses_and_sources(self.Xlenses_validation, self.Xsources_validation, num_positive, data_type, mock_lens_alpha_scaling, do_deterministic=True)
            
        # Store Negative data in numpy array and create label vector
        X_neg = np.empty((num_negative, X_pos.shape[1], X_pos.shape[2], X_pos.shape[3]), dtype=data_type)    
        y_neg = np.zeros(X_neg.shape[0], dtype=data_type)

        # Negatives consist of the negatives set and the unmerged lenses set. A lens unmerged with a source is basically a negative.
        n = int(num_negative // 2)  # number samples to take from lenses-, and negatives set.
        
        indexes_lenses = np.random.choice(self.Xlenses_validation.shape[0], n, replace=False)  
        X_neg[0:int(num_negative//2)] = self.Xlenses_validation[indexes_lenses] # first half of negative chunk is a random selection from lenses without replacement

        indexes_negatives = np.random.choice(self.Xlenses_validation.shape[0], n+1, replace=False)  
        X_neg[int(num_negative//2):num_negative] = self.Xnegatives_validation[indexes_negatives] # second half of negatives are a random selection from the negatives without replacement
        
        # The negatives need a square root stretch, just like the positives.
        X_neg = np.sqrt(X_neg)

        # Concatenate the positive and negative examples into one chunk (also the labels)
        X_chunk = np.concatenate((X_pos, X_neg))
        y_chunk = np.concatenate((y_pos, y_neg))

        if self.params.do_max_tree_seg:
            X_chunk = self._fill_arguments_max_tree_segmenter(X_chunk)

        print("Creating validation chunk took: {}, chunksize: {}".format(hms(time.time() - start_time), num_positive+num_negative), flush=True)
        return X_chunk, y_chunk
    

    # Merge a single lens and source together into a mock lens.
    def merge_lens_and_source(self, lens, source, mock_lens_alpha_scaling = (0.02, 0.30), show_imgs = False):

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
    def merge_lenses_and_sources(self, lenses_array, sources_array, num_mock_lenses, data_type, mock_lens_alpha_scaling = (0.02, 0.30), do_deterministic=False):
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
            mock_lens = self.merge_lens_and_source(lens, source, mock_lens_alpha_scaling)

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