### This script attempts to generate lenses.
# Definitions:
#  "lenses" A simple galaxy without any lensing features.
#       It contains random Noise Galaxies. (NG(s))
#  "Noise Galaxy" - NG: A galaxy in an image, that is not
#       in the centre of the image. Therefore it is not considered 
#       the object of interest, therefore it is considered a noise object.
#  "Centre Galaxy" - CG: The galaxy in the middle of the image. a.k.a. the object of interest.

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
        print("Loading Training Data", flush=True)
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


def print_data_array_stats(data_array, name):
    print("------------")
    print("Name: {}".format(name), flush=True)
    print("Dimensions: {}".format(data_array.shape), flush=True)
    print("Max array  = {}".format(np.amax(data_array)), flush=True)
    print("Min array  = {}".format(np.amin(data_array)), flush=True)
    print("Mean array = {}".format(np.mean(data_array)), flush=True)
    print("Median array = {}".format(np.median(data_array)), flush=True)
    print("Numpy nbytes = {},  GBs = {}".format(data_array.nbytes, bytes2gigabyes(data_array.nbytes)), flush=True)
    print("------------")


# Shows a random sample of the given data array, to the user.
def show_img_grid(data_array, iterations=1, columns=4, rows=4, seed=None, titles=None, fig_title=""):
    if seed != None:
        random.seed(seed)
    img_count = int(columns * rows)

    for _ in range(iterations):
        rand_idxs = [random.choice(list(range(data_array.shape[0]))) for x in range(img_count)]

        fig=plt.figure(figsize=(8, 8))
        fig.suptitle(fig_title, fontsize=16)

        for idx, i in enumerate(range(1, columns*rows +1)):
            fig.add_subplot(rows, columns, i)
            ax = (fig.axes)[idx]
            ax.axis('off')
            if titles==None:
                ax.set_title("img idx = {}".format(rand_idxs[idx]))
            else:
                ax.set_title(titles[rand_idxs[idx]])
            lens = np.squeeze(data_array[rand_idxs[idx]])
            plt.imshow(lens, origin='lower', interpolation='none', cmap='gray', vmin=0.0, vmax=1.0)

        plt.show()


#----------------------------------------------------------
#   Define an exponential profile in numpy.         Taken from: https://github.com/miguel-aragon/Semantic-Autoencoder-Paper/blob/master/PAPER_Exponential_profile_3-parameters_Gaussian-noise.ipynb
#----------------------------------------------------------
def exponential_2d_np(xx,yy, center=[0.0,0.0], amplitude=1.0, scale=0.5, ellipticity=0.0, angle=0.0):

    scl_a = scale
    scl_b = scl_a * (1-ellipticity)
    
    angle = angle*180  # OJO
    theta = angle*np.pi/180 
    #--- Rotate coordinates
    xt = np.cos(theta) * (xx - center[0]) - np.sin(theta)*(yy - center[1])
    yt = np.sin(theta) * (xx - center[0]) + np.cos(theta)*(yy - center[1])
    #--- Radius
    rt = np.sqrt(np.square(xt/scl_a) + np.square(yt/scl_b))
    #return np.power(amplitude * np.exp(-rt),0.75)
    return amplitude * np.exp(-rt)


#----------------------------------------------------------
#   Really wasteful way of passing coordinates to the network 
#     but for this simple case it is ok...        Taken from: https://github.com/miguel-aragon/Semantic-Autoencoder-Paper/blob/master/PAPER_Exponential_profile_3-parameters_Gaussian-noise.ipynb
#----------------------------------------------------------
def make_coord_list(_n, _img_size, _range=(-1,1)):

    xy = make_xy_coords(_img_size, _range=_range)
    xx = xy[0]
    yy = xy[1]
    
    list_xx = []
    list_yy = []
    for i in range(_n):
        list_xx.append(xx)
        list_yy.append(yy)

    list_xx = np.asarray(list_xx)
    list_yy = np.asarray(list_yy)
    list_xx = np.expand_dims(list_xx, axis=-1)
    list_yy = np.expand_dims(list_yy, axis=-1)
    return list_xx,list_yy


#----------------------------------------------------------
#   Create x and y arrays with coordinates      Taken from: https://github.com/miguel-aragon/Semantic-Autoencoder-Paper/blob/master/PAPER_Exponential_profile_3-parameters_Gaussian-noise.ipynb
#----------------------------------------------------------
def make_xy_coords(_n, _range=(0,1)):
    x = np.linspace(_range[0],_range[1], _n)
    y = np.linspace(_range[0],_range[1], _n)
    return np.meshgrid(x, y)


#----------------------------------------------------------
#  Create Galaxie(s)      Partially taken from: https://github.com/miguel-aragon/Semantic-Autoencoder-Paper/blob/master/PAPER_Exponential_profile_3-parameters_Gaussian-noise.ipynb
#----------------------------------------------------------
def gen_galaxies(n_sam=100, n_pix=101, scale_im=4.0, half_pix=0, scale_r=(0.01,0.05), ellip_r=(0.2,0.75), g_noise_sigma=0.025, do_centre=True):

    #--- Coordinates image. We will pass this to the neural net
    xx, yy = make_coord_list(n_sam, n_pix)
    xx = xx/n_pix*scale_im + half_pix
    yy = yy/n_pix*scale_im + half_pix

    #--- Unpack input parameters
    ps_min, ps_max = scale_r
    pe_min, pe_max = ellip_r

    #--- Intermediate parameters
    par_scale = np.random.uniform(ps_min, ps_max, size=n_sam)
    par_ellip = np.random.uniform(pe_min, pe_max, size=n_sam)
    par_angle = np.random.uniform(0, 1, size=n_sam)

    #--- Generate centre profiles/galaxies
    if do_centre:
        gs = np.zeros((n_sam, n_pix, n_pix, 1))            #gs = Galaxie(s)
        for i in range(n_sam):
            gs[i,:,:,0] = exponential_2d_np(xx[i,:,:,0],yy[i,:,:,0], scale=par_scale[i], ellipticity=par_ellip[i], angle=par_angle[i])

    #--- Generate Noise profiles/galaxies - Noise galaxies are typically not in the centre of the image.
    if not do_centre:
        gs = np.zeros((n_sam, n_pix, n_pix, 1))            #gs = in this case Noise Galaxie(s)
        for i in range(n_sam):
            cx = 10#random.uniform(0,1)#random.uniform(-(n_pix/2), (n_pix/2))
            cy = 10#random.uniform(0,1)#random.uniform(-(n_pix/2), (n_pix/2))
            gs[i,:,:,0] = exponential_2d_np(xx[i,:,:,0],yy[i,:,:,0], center=[cx, cy], scale=par_scale[i], ellipticity=par_ellip[i], angle=par_angle[i])


    #--- Add Gaussian noise
    # y_true_noise = np.zeros((n_sam, n_pix, n_pix, 1))
    for i in range(n_sam):
        gs[i,:,:,0] = gs[i,:,:,0] + np.random.randn(n_pix, n_pix)*g_noise_sigma
    return gs, par_scale, par_ellip, par_angle


# Create titles that can accompany the image grid view.
def create_exponential_profile_titles(par_scale, par_ellip, par_angle):
    titles = []
    for i in range(par_scale.shape[0]):
        titles.append("A:{:.3f} e:{:.3f} t:{:.3f}".format(par_scale[i], par_ellip[i], par_angle[i]))
    return titles


########################################
# Parameters
data_type = np.float32
seed = 1234
########################################
if False:
    ### 1 - Load lenses.
    all_fits_paths = get_all_fits_paths()
    lenses         = load_lenses(all_fits_paths)
    print_data_array_stats(lenses, name="Lenses")
    show_img_grid(lenses, iterations=1, columns=4, rows=4, seed=None, titles=None, fig_title="Real Lenses")

if False:
    ### 2 - Create a Centre Galaxies as realistically as possible - Centre Galaxy = cg
    cg, par_scale, par_ellip, par_angle = gen_galaxies(n_sam=100, n_pix=101, scale_im=4.0, half_pix=0.0, scale_r=(0.001,0.004), ellip_r=(0.2,0.5), g_noise_sigma=0.025)
    fig_titles = create_exponential_profile_titles(par_scale, par_ellip, par_angle)
    print_data_array_stats(cg, name="Centre Galaxies")
    show_img_grid(cg, iterations=1, columns=4, rows=4, seed=seed, titles=fig_titles, fig_title="Centre Galaxies")

if True:
    ### 3 - Create Noise Galaxies - as realistically as possible
    ngs, par_scale, par_ellip, par_angle = gen_galaxies(n_sam=100, n_pix=101, scale_im=4.0, half_pix=0.0, scale_r=(0.001,0.004), ellip_r=(0.2,0.5), g_noise_sigma=0.025, do_centre=False)
    fig_titles = create_exponential_profile_titles(par_scale, par_ellip, par_angle)
    print_data_array_stats(ngs, name="Noise Galaxies")
    show_img_grid(ngs, iterations=1, columns=4, rows=4, seed=seed, titles=fig_titles, fig_title="Noise Galaxies")