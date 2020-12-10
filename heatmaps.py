import cv2
from DataGenerator import DataGenerator
import glob
from Network import Network
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import os
from Parameters import Parameters
from utils import show2Imgs, get_model_paths, get_h5_path_dialog, load_settings_yaml, get_samples, compute_PSF_r, load_normalize_img, normalize_img, calc_RMS, create_dir_if_not_exists, count_TP_TN_FP_FN_and_FB
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


############################################################ Functions ############################################################

# Plot an image grid given a list of images.
def _plot_image_grid_GRADCAM(list_of_rows, layer_names, plot_title, fname, m_path):

    subplot_titles = list()
    for layer_name in layer_names:
        subplot_titles.append( ["Input Image", "Grad-CAM Layer: {}".format(layer_name), "Superimposed Heatmap"] )
    
    img_h = list_of_rows[0][0].shape[1]
    img_w = list_of_rows[0][0].shape[0]

    for idx, img_row_list in enumerate(list_of_rows):

        # Needed for empty color channels
        empty_img = np.zeros((img_w, img_h, 1))

        # Color channel input image needs color
        img_row_list[0] = np.concatenate((img_row_list[0], img_row_list[0], empty_img), axis=2)

        # heatmap needs color and resizing
        img_row_list[1] = cv2.resize(img_row_list[1], (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        img_row_list[1] = np.expand_dims(img_row_list[1], axis=2)
        img_row_list[1] = np.concatenate((img_row_list[1], img_row_list[1], empty_img), axis=2)
        
        comb_row = np.concatenate((img_row_list[0], img_row_list[1], img_row_list[2]), axis=1)

        # We need a beginning image row to concatenate with later on.
        if idx == 0:
            comb_img = comb_row
        else:
            comb_img = np.concatenate((comb_img, comb_row), axis = 0)

    plt.title("Grad-CAM\n{}\n{}".format(plot_title, "Layers: " + ', '.join(layer_names)), fontsize=9)
    plt.xlabel("(a) Input Image        (b) Grad-CAM         (c) SuperImposed")
    plt.imshow(comb_img)
    create_dir_if_not_exists(os.path.join(m_path, "grad_cams"), verbatim=False)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.savefig('{}.png'.format(os.path.join(m_path, "grad_cams", fname)))
    plt.clf()


def _normalize_heatmap(heatmap):
    # Normalize the heatmap - Do not normalize if the maximum of the heatmap is less than zero,
    # otherwise with Relu we would get NaN, since we would devide by zero.
    if np.max(heatmap) <= 0.0:
        heatmap = np.ones((heatmap.shape[0], heatmap.shape[1])) * 0.0000001
    else:
        heatmap = np.maximum(heatmap, 0.0)      # Relu operation
        heatmap /= np.max(heatmap)              # Normalization
    return heatmap


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
    last_conv_layer = model.get_layer(layer_name)

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

    # For debugging purposes
    if False:
        print("mean heatmap before norm: {}".format(np.mean(heatmap)))
        print("min heatmap before norm: {}".format(np.amin(heatmap)))
        print("max heatmap before norm: {}".format(np.amax(heatmap)))

        if np.mean(heatmap) == 0:
            plt.imshow(heatmap, cmap='Greys_r')
            plt.show()

    # Normalize to bring values between 0.0 and 1.0
    return _normalize_heatmap(heatmap)


# Create a color image from a greyscale image a heatmap and an empty color channel.
def _construct_color_img(inp_img, heatmap):
    inp_img = np.squeeze(inp_img)
    color_img = np.zeros((inp_img.shape[0], inp_img.shape[1], 3))
    color_img[:,:,2] = heatmap
    color_img[:,:,1] = np.zeros((inp_img.shape[0], inp_img.shape[1]))
    color_img[:,:,0] = np.squeeze(inp_img)
    return color_img


# Shows input images to the user and heatmaps of where the model looks.
def Grad_CAM_plot(image_set, model, m_path, layer_list, plot_title="", labels=None, is_positive_set=True):
    list_of_rows = list()

    for i in range(image_set.shape[0]):
        print("image index: {}/{}".format(i, image_set.shape[0]))
        
        # Set input image
        inp_img = image_set[i]
        
        # Predict input image for figure title
        prediction = model.predict(np.expand_dims(inp_img, axis=0))[0][0]

        # Assigning filname string to an example
        fname = ""
        threshold = 0.5     # model.evaluate evaluates on a decision threshold of 0.5, therefore we do it to over here.
        if (prediction < threshold) and is_positive_set:
            fname += "FN{}".format(i)
        elif (prediction >= threshold) and is_positive_set:
            fname += "TP{}".format(i)
        elif (prediction < threshold) and not is_positive_set:
            fname += "TN{}".format(i)
        elif (prediction >= threshold) and not is_positive_set:
            fname += "FP{}".format(i)
            
        # Title for plot
        plot_string = "Model Prediction: {:.3f}, {}".format(prediction, plot_title)      

        for layer_name in layer_list:

            # Perform the Gradient Class Activation Map algorithm
            heatmap = Grad_CAM(inp_img, model, layer_name)
            
            # Show the heatmap
            low_res_heatmap = np.copy(heatmap)
            
            # Resize heatmap for sumperimposing with input image
            heatmap = cv2.resize(heatmap, (inp_img.shape[0], inp_img.shape[1]), interpolation=cv2.INTER_NEAREST)  

            # Construct a color image, with heatmap as one channel and input image another. The third channel are zeros
            color_img = _construct_color_img(inp_img, heatmap)

            # Collect all three images into a list and their respective plot titles
            images = [inp_img, low_res_heatmap, color_img]
            list_of_rows.append(images)
            
        # Format Plotting
        _plot_image_grid_GRADCAM(list_of_rows, layer_list, plot_string, fname, m_path)
        list_of_rows = list()
        plt.close()
        plt.clf()


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




############################################################ script ############################################################


# 1.0 - Fix memory leaks if running on tensorflow 2
tf.compat.v1.disable_eager_execution()


# 2.0 - Model Selection from directory
model_paths = get_model_paths()


# 2.1 - Select a weights file. There are 2 for each model. Selected based on either validation loss or validation metric. The metric can differ per model.
h5_paths = get_h5_path_dialog(model_paths)


# 3.0 - Load params - used for normalization etc -
yaml_path = glob.glob(os.path.join(model_paths[0], "run.yaml"))[0]                      # Only choose the first one for now
settings_yaml = load_settings_yaml(yaml_path)                                           # Returns a dictionary object.
params = Parameters(settings_yaml, yaml_path, mode="no_training")                       # Create Parameter object
params.data_type = np.float32 if params.data_type == "np.float32" else np.float32       # This must be done here, due to the json, not accepting this kind of if statement in the parameter class.


# 4.0 - Select random sample from the data (with replacement)
sample_size = int(input("How many samples do you want to create and run (int): "))
sources_fnames, lenses_fnames, negatives_fnames = get_samples(size=sample_size, type_data="test", deterministic=False)


# 5.0 - Load lenses and sources in 4D numpy arrays
PSF_r = compute_PSF_r()  # Used for sources only
# lenses    = load_normalize_img(params.data_type, are_sources=False, normalize_dat="per_image", PSF_r=PSF_r, filenames=lenses_fnames)
# sources   = load_normalize_img(params.data_type, are_sources=True, normalize_dat="per_image", PSF_r=PSF_r, filenames=sources_fnames)
lenses_unnormalized    = load_normalize_img(params.data_type, are_sources=False, normalize_dat="None", PSF_r=PSF_r, filenames=lenses_fnames)
sources_unnormalized   = load_normalize_img(params.data_type, are_sources=True, normalize_dat="None", PSF_r=PSF_r, filenames=sources_fnames)
negatives = load_normalize_img(params.data_type, are_sources=False, normalize_dat="per_image", PSF_r=PSF_r, filenames=negatives_fnames)


# 6.0 - Create mock lenses based on the sample
noise_fac = 2.0
mock_lenses, pos_y, alpha_scalings, SNRs = merge_lenses_and_sources(lenses_unnormalized, sources_unnormalized, noise_fac=noise_fac)


# 8.0 - Create a dataGenerator object, because the network class wants it
dg = DataGenerator(params, mode="no_training", do_shuffle_data=True, do_load_validation=False)


# 9.0 - Construct a Network object that has a model as property.
network = Network(params, dg, training=False)
network.model.load_weights(h5_paths[0])


# Perform Grad-CAM
another_list = ["batch_normalization_16", "activation_12", "activation_8"]
another_list = ["conv2d_2", "conv2d_4", "conv2d_6", "conv2d_9", "conv2d_11", "conv2d_14", "conv2d_16", "conv2d_19"]     # This one is poor.
another_list = ["add", "add_1", "add_2", "add_3", "add_4", "add_5", "add_6", "add_7"]                                   # This one works well
another_list = ["add_4", "add_5", "add_6", "add_7"]                                   # This one works well
Grad_CAM_plot(negatives, network.model, layer_list=another_list, plot_title="Negative Example", labels=pos_y*0.0, is_positive_set=False, m_path=model_paths[0])
Grad_CAM_plot(mock_lenses, network.model, layer_list=another_list, plot_title="Positive Example", labels=pos_y, is_positive_set=True, m_path=model_paths[0])

