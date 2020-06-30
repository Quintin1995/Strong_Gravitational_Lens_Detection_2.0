import numpy as np
from scipy import ndimage
import scipy
import glob
import itertools
import threading
import skimage.transform
import skimage.io
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


class DataGenerator:
    def __init__(self, params, *args, **kwargs):
        self.params = params

        self.PSF_r = self.compute_PSF_r()

        self.lenses_array    = self.get_data_array(self.params.img_dims,
                                                    path=self.params.lenses_path,
                                                    fraction_to_load=self.params.fraction_to_load_lenses,
                                                    are_sources=False,
                                                    normalize_dat=self.params.normalize)
        self.negatives_array = self.get_data_array(self.params.img_dims,
                                                    path=self.params.negatives_path,
                                                    fraction_to_load=self.params.fraction_to_load_negatives,
                                                    are_sources=False,
                                                    normalize_dat=self.params.normalize)
        self.sources_array   = self.get_data_array(self.params.img_dims,
                                                    path=self.params.sources_path,
                                                    fraction_to_load=self.params.fraction_to_load_sources,
                                                    are_sources=True,
                                                    normalize_dat=self.params.normalize)
         ###### Step 3.1: Split the data into train and test data.
        splits = self.split_train_test_data(self.lenses_array,
                                            0.0,
                                            test_fraction=self.params.test_fraction)
        self.X_train_lenses, self.X_test_lenses, self.y_train_lenses, self.y_test_lenses = splits
        splits = self.split_train_test_data(self.negatives_array,
                                            0.0,
                                            test_fraction=self.params.test_fraction)
        self.X_train_negatives, self.X_test_negatives, self.y_train_negatives, self.y_test_negatives = splits
        splits = self.split_train_test_data(self.sources_array,
                                            "no_label", 
                                            test_fraction=self.params.test_fraction)
        self.X_train_sources, self.X_test_sources, _, _ = splits


        ###### Step 5.0 - Data Augmentation - Data Generator Keras - Training Generator is based on train data array.
        self.train_generator = ImageDataGenerator(
                rotation_range=params.aug_rotation_range,
                width_shift_range=params.aug_width_shift_range,
                height_shift_range=params.aug_height_shift_range,
                zoom_range=params.aug_zoom_range,
                horizontal_flip=params.aug_do_horizontal_flip,
                fill_mode=params.aug_default_fill_mode
                )

        ###### Step 5.1 - Data Augmentation - Data Generator Keras - Validation Generator is based on test data for now
        self.validation_generator = ImageDataGenerator(
                # rotation_range=params.aug_rotation_range,
                # width_shift_range=params.aug_width_shift_range,
                # height_shift_range=params.aug_height_shift_range,
                # zoom_range=params.aug_zoom_range,
                horizontal_flip=params.aug_do_horizontal_flip,
                fill_mode=params.aug_default_fill_mode
                )

    
    
    def compute_PSF_r(self):
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

        seds = np.loadtxt("data/SED_colours_2017-10-03.dat")

        Rg = 3.30
        Rr = 2.31
        Ri = 1.71
        return PSF_r
        ### END OF IMPORTANT PIECE.
    
    # returns a numpy array with lens images from disk
    def get_data_array(self, img_dims, path, fraction_to_load = 1.0, data_type = np.float32, are_sources=False, normalize_dat = "per_image"):
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
        if fraction_to_load > 1:
            num_to_actually_load = int(fraction_to_load)
        else:
            num_to_actually_load = int(fraction_to_load*len(data_paths))
        data_paths = data_paths[0:num_to_actually_load]

        # Pre-allocate numpy array for the data
        data_array = np.zeros((len(data_paths),img_dims[0], img_dims[1], img_dims[2]),dtype=data_type)
        print("data array shape: {}".format(data_array.shape))

        # Load all the data in into the numpy array:
        for idx, filename in enumerate(data_paths):

            if are_sources:
                img = fits.getdata(filename).astype(data_type)
                img = np.expand_dims(scipy.signal.fftconvolve(img, self.PSF_r, mode="same"), axis=2)
                if normalize_dat == "per_image":
                    img = self.normalize_img(img)
                data_array[idx] = img
            else:
                img = np.expand_dims(fits.getdata(filename), axis=2).astype(data_type)
                if normalize_dat == "per_image":
                    img = self.normalize_img(img)
                data_array[idx] = img
        
        if normalize_dat == "per_array":
            return self.normalize_data_array(data_array)

        print("max array  = {}".format(np.amax(data_array)))
        print("min array  = {}".format(np.amin(data_array)))
        print("mean array = {}".format(np.mean(data_array)))
        print("median array = {}".format(np.median(data_array)))

        print("Loading data took: {} for folder: {}".format(hms(time.time() - start_time), path))
        return data_array
    
    # Normalizationp per image
    def normalize_img(self, numpy_img):
        return ((numpy_img - np.amin(numpy_img)) / (np.amax(numpy_img) - np.amin(numpy_img)))

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
    
    # Loading a chunk into memory
    def load_chunk(self, chunksize, X_lenses, X_negatives, X_sources, data_type, mock_lens_alpha_scaling):
        start_time = time.time()

        # Half a chunk positive and half negative
        num_positive = int(chunksize / 2)       
        num_negative = int(chunksize / 2)

        # Get mock lenses data and labels
        X_pos, y_pos = self.merge_lenses_and_sources(X_lenses, X_sources, num_positive, data_type, mock_lens_alpha_scaling)
        
        # Store Negative data in numpy array and create label vector
        X_neg = np.empty((num_negative, X_pos.shape[1], X_pos.shape[2], X_pos.shape[3]), dtype=data_type)
        y_neg = np.zeros(X_neg.shape[0], dtype=data_type)

        # We want a 80% probability of selecting from the contaminants set, and 20% probability of selecting an LRG from the lenses set.
        negative_sample_contaminant_prob = 0.8
        for i in range(num_negative):
            if random.random() <= negative_sample_contaminant_prob:
                X_neg[i] = np.sqrt(X_negatives[random.randint(0, X_negatives.shape[0] - 1)])
            else:
                X_neg[i] = np.sqrt(X_lenses[random.randint(0, X_lenses.shape[0] - 1)])

        # Concatenate the positive and negative examples into one chunk (also the labels)
        X_chunk = np.concatenate((X_pos, X_neg))
        y_chunk = np.concatenate((y_pos, y_neg))

        print("Creating chunk took: {}, chunksize: {}".format(hms(time.time() - start_time), chunksize))

        return X_chunk, y_chunk
    
        # Merge a single lens and source together into a mock lens.
    def merge_lens_and_source(self, lens, source, mock_lens_alpha_scaling = (0.02, 0.30), show_imgs = False):

        # Add lens and source together | We rescale the brightness of the simulated source to the peak brightness of the LRG in the r-band multiplied by a factor of alpha randomly drawn from the interval [0.02,0.3]
        mock_lens = lens + source / np.amax(source) * np.amax(lens) * np.random.uniform(mock_lens_alpha_scaling[0], mock_lens_alpha_scaling[1])

        # Take a square root stretch to emphesize lower luminosity features.
        mock_lens_sqrt = np.sqrt(mock_lens)
        
        # Basically removes negative values - should not be necessary, because all input data should be normalized anyway. (I will leave it for now, but should be removed soon.)
        mock_lens_sqrt = mock_lens_sqrt.clip(min=0.0, max=1.0)
        mock_lens = mock_lens.clip(min=0.0, max=1.0)

        if show_imgs:
            show2Imgs(lens, source, "Lens max pixel: {0:.3f}".format(np.amax(lens)), "Source max pixel: {0:.3f}".format(np.amax(source)))
            show2Imgs(mock_lens, mock_lens_sqrt, "mock_lens max pixel: {0:.3f}".format(np.amax(mock_lens)), "mock_lens_sqrt max pixel: {0:.3f}".format(np.amax(mock_lens_sqrt)))

        return mock_lens_sqrt

    # This function should read images from the lenses- and sources data array,
    # and merge them together into a lensing system, further described as 'mock lens'.
    # These mock lenses represent a strong gravitational lensing system that should 
    # get the label 1.0 (positive label). 
    def merge_lenses_and_sources(self, lenses_array, sources_array, num_mock_lenses, data_type, mock_lens_alpha_scaling = (0.02, 0.30)):
        num_lenses  = lenses_array.shape[0]
        num_sources = sources_array.shape[0]

        X_train_positive = np.empty((num_mock_lenses, lenses_array.shape[1], lenses_array.shape[2], lenses_array.shape[3]), dtype=data_type)
        Y_train_positive = np.ones(num_mock_lenses, dtype=data_type)

        # Which indexes to load from sources and lenses (these will be merged together)
        idxs_lenses = random.choices(list(range(num_lenses)), k=num_mock_lenses)
        idxs_sources = random.choices(list(range(num_sources)), k=num_mock_lenses)
        
        for i in range(num_mock_lenses):
            lens   = lenses_array[idxs_lenses[i]]
            source = sources_array[idxs_sources[i]]
            mock_lens = self.merge_lens_and_source(lens, source, mock_lens_alpha_scaling)
            X_train_positive[i] = mock_lens

        return X_train_positive, Y_train_positive