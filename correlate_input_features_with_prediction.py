from compare_results import set_experiment_folder, set_models_folders, load_settings_yaml
from DataGenerator import DataGenerator
import glob
from Network import Network
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage import exposure
import os
import scipy
import pyfits
import random
from Parameters import Parameters
import pandas as pd
from utils import show2Imgs
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from functools import reduce
from photutils import make_source_mask
from astropy.stats import sigma_clipped_stats

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import cv2


def binary_dialog(question_string):
    print("\nType 'y' or '1' for yes, '0' or 'n' for no")
    ans = input(question_string + ": ")
    if ans in ["1", "y", "Y", "Yes", "yes"]:
        ans = True
    else:
        ans = False
    return ans


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
def merge_lens_and_source(lens, source, mock_lens_alpha_scaling = (0.02, 0.30), show_imgs = False, do_plot=False, noise_fac=2.0):
    # Determine noise level of lens - First the naive approach
    # noise_lens = np.mean(lens)
    # print("\n\nMean all mock lenses: {}".format(np.mean(mock_lenses)))
    
    mask = make_source_mask(np.squeeze(lens), nsigma=2, npixels=5, dilate_size=11)
    mean, median, std = sigma_clipped_stats(np.squeeze(lens), sigma=3.0, mask=mask)
    noise_lens = mean

    # Determine alpha scaling drawn from the interval [0.02,0.3]
    alpha_scaling = np.random.uniform(mock_lens_alpha_scaling[0], mock_lens_alpha_scaling[1])

    # We rescale the brightness of the simulated source to the peak brightness
    source = source / np.amax(source) * np.amax(lens) * alpha_scaling
    
    # Get indexes where lensing features have pixel values below: noise_factor*noise
    idxs = np.where(source < (noise_fac * noise_lens))

    # Make a copy of the source and set all below noise_factor*noise to 0.0
    trimmed_source = np.copy(source)
    trimmed_source[idxs] = 0.0

    # Calculate surface area of visual features that are stronger than noise_fac*noise
    (x_idxs,_,_) = np.where(source >= (noise_fac * noise_lens))
    feature_area_frac = len(x_idxs) / (source.shape[0] * source.shape[1])
    
    # Add lens and source together 
    mock_lens = lens + source
    
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
        alpha_scalings.append(alpha_scaling)
        feature_areas_fracs.append(feature_area_frac)

    return X_train_positive, Y_train_positive, alpha_scalings, feature_areas_fracs


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
def get_samples(size=1000, type_data="validation", deterministic=True, seed="30"):
    lenses_path_train    = os.path.join("data", type_data, "lenses")
    sources_path_train   = os.path.join("data", type_data, "sources")
    negatives_path_train = os.path.join("data", type_data, "negatives")

    # Try to glob files in the given path
    sources_fnames      = glob.glob(os.path.join(sources_path_train, "*/*.fits"))
    lenses_fnames       = glob.glob(os.path.join(lenses_path_train, "*_r_*.fits"))
    negatives_fnames    = glob.glob(os.path.join(negatives_path_train, "*_r_*.fits"))

    print("\nsources count {}".format(len(sources_fnames)))
    print("lenses count {}".format(len(lenses_fnames)))
    print("negatives count {}".format(len(negatives_fnames)))

    if deterministic:
        random.seed(seed)
    sources_fnames   = random.sample(sources_fnames, size)
    lenses_fnames    = random.sample(lenses_fnames, size)
    negatives_fnames = random.sample(negatives_fnames, size)
    return sources_fnames, lenses_fnames, negatives_fnames


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
    if norm_type == "None":
        return img
    return img


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


# Plot an image grid given a list of images.
def _plot_image_grid_GRADCAM(list_of_rows, layer_names, plot_title):

    subplot_titles = list()
    for layer_name in layer_names:
        subplot_titles.append( ["Input Image", "Grad-CAM Layer: {}".format(layer_name), "Superimposed Heatmap"] )
    
    all_imgs = reduce(lambda l1, l2: l1+l2, list_of_rows)
    all_titles = reduce(lambda l1, l2: l1+l2, subplot_titles)

    rows, cols, axes = len(layer_names), len(list_of_rows[0]), []
    fig=plt.figure(figsize=(10,3))
    for a in range(rows*cols):
        axes.append( fig.add_subplot(rows, cols, a+1) )
        axes[-1].set_title(all_titles[a], fontsize=8)
        if a%3 == 0:
            plt.imshow(np.squeeze(all_imgs[a]), cmap='Greys_r')
        else:
            plt.imshow(np.squeeze(all_imgs[a]))
        axes[-1].axis('off')
    # fig.tight_layout()
    fig.suptitle("Grad-CAM\n{}".format(plot_title), fontsize=9)
    plt.show()


def _normalize_heatmap(heatmap):
    # Normalize the heatmap - Do not normalize if the maximum of the heatmap is less than zero,
    # otherwise with Relu we would get NaN, since we would devide by zero.
    if np.max(heatmap) < 0.0:
        heatmap = np.ones((heatmap.shape[0], heatmap.shape[1])) * 0.0000001
    else:
        heatmap = np.maximum(heatmap, 0.0)      # Relu operation
        heatmap /= np.max(heatmap)              # Normalization
    return heatmap


# Create a color image from a greyscale image a heatmap and an empty color channel.
def _construct_color_img(inp_img, heatmap):
    inp_img = np.squeeze(inp_img)
    color_img = np.zeros((inp_img.shape[0], inp_img.shape[1], 3))
    color_img[:,:,2] = heatmap
    color_img[:,:,1] = np.zeros((inp_img.shape[0], inp_img.shape[1]))
    color_img[:,:,0] = np.squeeze(inp_img)
    return color_img


# Construct a Gradient based Class Activation Map of a given input image,
# for a given model and layer
def Grad_CAM(inp_img, model, layer_name):
    # Add batch dimension to the image so that we can feed it into the model
    inp_img = np.expand_dims(inp_img, axis=0)

    # Mock lens output index of the model, in the prediction vector.
    # It is still a vector, only it has one scaler. Therefore we still index
    # it, as if it is a vector(array).
    mock_lens_output = model.output[:, 0]

    # Output feature map of the block 'layer_name' layer, the last convolutional layer.
    last_conv_layer = model.get_layer(layer_name) # "add_5", "layer_name" are pretty good

    # Gradient of the mock lens class with regard to the output feature map of "layer_name
    grads = K.gradients(mock_lens_output, last_conv_layer.output)[0]

    # Vector of shape (num_featuresmaps,) where each entry is the mean intensity of the
    # gradient over a specidfic feature map channel 
    pooled_grads = K.mean(grads, axis=(0,1,2))

    # Lets you access the values of the quantities you just defined. (keras doesnt
    # calculate them, until told so.): pooled_grads and the output feature map of
    # layer_name given the sample image
    iterate = K.function([model.input],
                        [pooled_grads, last_conv_layer.output[0]])

    # Values of these two quantities as Numpy arrays, given the sample mock lens
    pooled_grads_value, conv_layer_output_value = iterate([inp_img])

    # Multiplies each channel in the feature-map array by "how important this channel is"
    # with regard to the "mock_lens" class.
    for i in range(model.get_layer(layer_name).output.shape[-1]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    #The channel-wise mean of the resulting feature map is the heat map of the class activation.
    heatmap = np.mean(conv_layer_output_value, axis=-1)

    # Normalize to bring values between 0.0 and 1.0
    return _normalize_heatmap(heatmap)


# Shows input images to the user and heatmaps of where the model looks.
def Grad_CAM_plot(image_set, model, layer_list, plot_title="", labels=None):

    list_of_rows = list()

    for i in range(image_set.shape[0]):
        
        # Set input image
        inp_img = image_set[i]
        
        # Predict input image for figure title
        prediction = model.predict(np.expand_dims(inp_img, axis=0))[0][0]

        # Finding false negatives and false positives in the sample data
        if False:   
            threshold = 0.5
            if prediction < threshold:
                continue

        # Title for plot
        plot_string = "Model Prediction: {:.3f}\n{}".format(prediction, plot_title)      

        for layer_name in layer_list:

            # Perform the Gradient Class Activation Map algorithm
            heatmap = Grad_CAM(inp_img, model, layer_name)
            
            # Show the heatmap
            low_res_heatmap = np.copy(heatmap)
            
            # Resize heatmap for sumperimposing with input image
            heatmap = cv2.resize(heatmap, (inp_img.shape[0], inp_img.shape[1]))  

            # Construct a color image, with heatmap as one channel and input image another. The third channel are zeros
            color_img = _construct_color_img(inp_img, heatmap)

            # Collect all three images into a list and their respective plot titles
            images = [inp_img, low_res_heatmap, color_img]
            list_of_rows.append(images)
            
        # Format Plotting
        _plot_image_grid_GRADCAM(list_of_rows, layer_list, plot_string)
        list_of_rows = list()
        plt.close()




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
PSF_r = compute_PSF_r()  # Used for sources
lenses  = load_normalize_img(params.data_type, are_sources=False, normalize_dat="per_image", PSF_r=PSF_r, filenames=lenses_fnames)
sources = load_normalize_img(params.data_type, are_sources=True, normalize_dat="per_image", PSF_r=PSF_r, filenames=sources_fnames)
negatives = load_normalize_img(params.data_type, are_sources=False, normalize_dat="per_image", PSF_r=PSF_r, filenames=negatives_fnames)


# 6.0 - Create mock lenses based on the sample
noise_fac = 2.0
mock_lenses, pos_y, alpha_scalings, features_areas_fracs = merge_lenses_and_sources(lenses, sources, noise_fac=noise_fac)


# 6.5 - Determine noise level of a given image.
# sources_unnorm = load_normalize_img(params.data_type, are_sources=True, normalize_dat="None", PSF_r=PSF_r, filenames=sources_fnames)
if binary_dialog("Calculate and view noise of an image?"):
    print("\n\nMean all mock lenses: {}".format(np.mean(mock_lenses)))
    for i in range(sample_size):
        img = np.squeeze(lenses[i])
        mask = make_source_mask(img, nsigma=2, npixels=5, dilate_size=11)
        mean, median, std = sigma_clipped_stats(img, sigma=3.0, mask=mask)
        print("\nImg idx: {}\nmean: {}\nmedian: {}\nstd: {}".format(i, mean, median, std))
        plt.hist(img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k') #calculating histogram
        plt.title("Histogram of image pixels")
        plt.show()
        x=5


# 7.0 - Initialize and fill a pandas dataframe to store Source parameters
df = get_empty_dataframe()
df = fill_dataframe(df, sources_fnames)
print(df.head())


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
    Grad_CAM_plot(negatives, network.model, layer_list=another_list, plot_title="Negative Example", labels=pos_y*0.0)


# 10.0 - Use the network to predict on the sample
einstein_radii = list(df["LENSER"])
predictions = network.model.predict(mock_lenses)
predictions = list(np.squeeze(predictions))


if binary_dialog("Do you want to plot feature versus prediction?"):
    # 11.0 - Lets make a plot of feature area size versus network certainty.
    plot_feature_versus_prediction(predictions, features_areas_fracs, threshold=0.5, title="Image Ratio of Source above {}x noise level".format(noise_fac))

    # 12.0 - Make a plot of einstein radius and network certainty
    plot_feature_versus_prediction(predictions, einstein_radii, threshold=0.5, title="Einstein Radii")


# 13.0 - Lets create a 2D matrix with x-axis and y-axis being Einstein radius and alpha scaling.
# For each data-point based on these features, assign a color based on the model prediction.
# SCATTER - PLOT 1
fig, ax = plt.subplots()
plt.scatter(x=einstein_radii, y=alpha_scalings, c=predictions, cmap='copper')   #cmap {'winter', 'cool', 'copper'}
plt.xlabel("Einstein Radius")
plt.ylabel("Source Intensity Scaling")
plt.title("Influence of brightness intensity scaling of Source and Einstein Radius")
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Prediction value Model', rotation=270)

# HEXBIN - PLOT 2
fig, ax = plt.subplots()
plt.hexbin(x=einstein_radii, y=alpha_scalings, C=predictions, gridsize=15, cmap='copper')
plt.xlabel("Einstein Radius")
plt.ylabel("Source Intensity Scaling")
plt.title("Influence of brightness intensity scaling of Source and Einstein Radius")
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Prediction value Model', rotation=270)
plt.show()


# 14.0 - Lets try to make a 3D plot, with:
# x-axis is Einstein Radius
# y-axis is alpha_scaling
# z-axis is prediction of model.
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(einstein_radii, alpha_scalings, predictions, cmap='viridis')
plt.xlabel("Einstein Radius")
plt.ylabel("Source Intensity Scaling")
plt.title("Influence of brightness intensity scaling of Source and Einstein Radius")
plt.show()
