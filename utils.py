from astropy.io import fits
import glob
import random
import numpy as np
import os
from skimage import exposure
from datetime import datetime
import matplotlib.pyplot as plt
import psutil
import pyfits
import yaml
import scipy




# Count the number of true positive, true negative, false positive and false negative in for a prediction vector relative to the label vector.
def count_TP_TN_FP_FN_and_FB(prediction_vector, y_test, threshold, beta_squarred, verbatim = False):
    TP = 0 #true positive
    TN = 0 #true negative
    FP = 0 #false positive
    FN = 0 #false negative

    for idx, pred in enumerate(prediction_vector):
        if pred >= threshold and y_test[idx] >= threshold:
            TP += 1
        if pred < threshold and y_test[idx] < threshold:
            TN += 1
        if pred >= threshold and y_test[idx] < threshold:
            FP += 1
        if pred < threshold and y_test[idx] >= threshold:
            FN += 1

    tot_count = TP + TN + FP + FN
    
    precision = TP/(TP + FP) if TP + FP != 0 else 0
    recall    = TP/(TP + FN) if TP + FN != 0 else 0
    fp_rate   = FP/(FP + TN) if FP + TN != 0 else 0
    accuracy  = (TP + TN) / len(prediction_vector) if len(prediction_vector) != 0 else 0
    F_beta    = (1+beta_squarred) * ((precision * recall) / ((beta_squarred * precision) + recall)) if ((beta_squarred * precision) + recall) else 0
    
    if verbatim:
        if tot_count != len(prediction_vector):
            print("Total count {} of (TP, TN, FP, FN) is not equal to the length of the prediction vector: {}".format(tot_count, len(prediction_vector)), flush=True)

        print("Total Count {}\n\tTP: {}, TN: {}, FP: {}, FN: {}".format(tot_count, TP, TN, FP, FN), flush=True)
        print("precision = {}".format(precision), flush=True)
        print("recall    = {}".format(recall), flush=True)
        print("fp_rate   = {}".format(fp_rate), flush=True)
        print("accuracy  = {}".format(accuracy), flush=True)
        print("F beta    = {}".format(F_beta), flush=True)

    return TP, TN, FP, FN, precision, recall, fp_rate, accuracy, F_beta



# dstack the data to three channels instead of one
def dstack_data(data):
    dstack_data = np.empty((data.shape[0], data.shape[1], data.shape[2], 3), dtype=np.float32)
    for i in range(data.shape[0]):
        img = data[i]
        dstack_data[i] = np.dstack((img,img,img))
    return dstack_data


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


# Simple case function to reduce line count in other function
def normalize_function(img, norm_type, data_type):
    if norm_type == "per_image":
        img = normalize_img(img)
    if norm_type == "adapt_hist_eq":
        # img = normalize_img(img)
        img = exposure.equalize_adapthist(img).astype(data_type)
    if norm_type == "None":
        return img
    return img


# Calculate Point Spread Function for the sources.
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
    numpy_img = ((numpy_img - np.amin(numpy_img)) / (np.amax(numpy_img) - np.amin(numpy_img)))
    return numpy_img


# Prompt the user which models to compare against each other in the given experiment folder
def set_models_folders(experiment_folder):
    print("------------------------------")
    print("\n\nRoot folder this experiment: {}".format(experiment_folder))

    cwd = os.path.join(os.getcwd(), experiment_folder)
    folders = sorted(os.listdir(cwd))
    local_folders = [x for x in folders if os.path.isdir(os.path.join(cwd, x))]

    print("\nSet model folders:")
    for idx, folder in enumerate(local_folders):
        print("\t{} - {}".format(idx, folder))
    folder_idxs = input("Set indexes model folder(s)\n(integer)\nOr comma seperated ints: ")
    
    str_indexes = folder_idxs.split(',')
    chosen_models = [local_folders[int(string_idx)] for string_idx in str_indexes]
    
    print("\nUser Choices: ")
    for chosen_model in chosen_models:
        print(chosen_model)

    full_paths = [os.path.join(cwd, m) for m in chosen_models]
    return full_paths


# Prompt the user to fill in which experiment folder to run.
def set_experiment_folder(root_folder):
    print("------------------------------")
    print("\n\nRoot folder of experiment: {}".format(root_folder))


    cwd = os.path.join(os.getcwd(), root_folder)
    folders = sorted(os.listdir(cwd))
    local_folders = [x for x in folders if os.path.isdir(os.path.join(cwd, x))]

    print("\nSet experiment folder:")
    for idx, exp_folder in enumerate(local_folders):
        print("\t{} - {}".format(idx, exp_folder))
    exp_idx = int(input("Set number experiment folder (integer): "))
    print("Choose index: {}, {}".format(exp_idx, os.path.join(root_folder, local_folders[exp_idx])))

    return os.path.join(root_folder, local_folders[exp_idx])


def binary_dialog(question_string):
    print("\nType 'y' or '1' for yes, '0' or 'n' for no")
    ans = input(question_string + ": ")
    if ans in ["1", "y", "Y", "Yes", "yes"]:
        ans = True
    else:
        ans = False
    return ans


# Obtain the neural network weights of the trained models.
def get_h5_path_dialog(model_paths):
    weights_paths = list()
    for model_path in model_paths:
        print("Choosing for model: {}".format(model_path))
        h5_choice = int(input("Which model do you want? A model selected on validation loss (1) or validation metric (2) or (3) for neither? (int): "))
        if h5_choice == 1:
            h5_paths = glob.glob(os.path.join(model_path, "checkpoints/*loss.h5"))
        elif h5_choice == 2:
            h5_paths = glob.glob(os.path.join(model_path, "checkpoints/*metric.h5"))
        else:
            h5_paths = glob.glob(os.path.join(model_path, "*.h5"))

        print("Choice h5 path: {}".format(h5_paths[0]))
        weights_paths.append(h5_paths[0])
    return weights_paths


# Opens dialog with the user to select a folder that contains models.
def get_model_paths(root_dir="models"):

    #Select which experiment path to take in directory structure
    experiment_folder = set_experiment_folder(root_folder=root_dir)

    # Can select 1 or multiple models.
    models_paths = set_models_folders(experiment_folder)
    return models_paths


# Calculate the Root-Mean-Square of a 4d numpy array. Considering only the negative one sided gaussian.
def calc_RMS(unnormalized_3d_numpy_array):
    # set all positive values to zero
    clipped = np.clip(unnormalized_3d_numpy_array, -1.0, 0.0)

    # Mask all zero values
    masked = np.ma.masked_equal(clipped, 0.0)

    # Remove the zero values from the array aswell, so that we truely have an array with negative values only.
    negs = masked.compressed()
    negs = np.ravel(negs)
    num_samples = negs.shape[0]

    # Calculate Root-Mean-Square
    rms_half = np.square(negs)
    rms_half = np.sum(rms_half)
    rms_half = (1/num_samples) * rms_half
    rms_half = np.sqrt(rms_half)

    rms_two_sided = rms_half / np.sqrt(1.0 - (2/np.pi))
    return rms_two_sided


def plot_first_last_stats(X, y):
    num_imgs = int(input("\nAmount of images to show to the user? integer value: "))
    columns = 2
    rows = 1
    for img_num in range(num_imgs):
        fig=plt.figure(figsize=(8, 8))
        pos_img = np.squeeze(X[0+img_num])
        neg_img = np.squeeze(X[-1-img_num])
        imgs = [pos_img, neg_img]
        print("pos image data type: {}".format(pos_img.dtype.name), flush=True)
        print("pos image shape: {}".format(pos_img.shape), flush=True)
        print("pos image min: {}".format(np.amin(pos_img)), flush=True)
        print("pos image max: {}".format(np.amax(pos_img)), flush=True)
        print("neg image data type: {}".format(neg_img.dtype.name), flush=True)
        print("neg image shape: {}".format(neg_img.shape), flush=True)
        print("neg image min: {}".format(np.amin(neg_img)), flush=True)
        print("neg image max: {}".format(np.amax(neg_img)), flush=True)
        for i in range(1, columns*rows +1):
            fig.add_subplot(rows, columns, i)
            plt.imshow(imgs[i-1], origin='lower', interpolation='none', cmap='Greys_r')
        fig.suptitle("label0: {}, label1: {}".format(y[0+img_num], y[-1-img_num]))
        plt.show()


# Show the user some random images of the given numpy array, numpy array structured like: [num_imgs, width, height, num_channels]
def show_random_img_plt_and_stats(data_array, num_imgs=1, title="title", do_plot=True, do_seed=False, seed=7846):
    for _ in range(num_imgs):
        if do_seed:
            random.seed(seed)
        random_idx = random.randint(0, data_array.shape[0]-1)
        img = np.squeeze(data_array[random_idx])                    # Remove the color channel from the image
        
        fig=plt.figure()
        print("\n")
        print(title + " image data type: {}".format(img.dtype.name), flush=True)
        print(title + " image shape: {}".format(img.shape), flush=True)
        print(title + " image min: {}".format(np.amin(img)), flush=True)
        print(title + " image max: {}".format(np.amax(img)), flush=True)
        plt.title(title)
        plt.imshow(img, origin='lower', interpolation='none', cmap='Greys_r')
        if do_plot:
            plt.show()


# Show 2 images next to each other. Squeeze out the color channel if it exist and it is equal to 1.
def show2Imgs(img1_numpy, img2_numpy, img1_title, img2_title):
    if len(img1_numpy.shape) != 2:
        img1_numpy = np.squeeze(img1_numpy)
    if len(img2_numpy.shape) != 2:
        img2_numpy = np.squeeze(img2_numpy)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img1_numpy,cmap='Greys_r')
    plt.title(img1_title)
    plt.subplot(1, 2, 2)
    plt.imshow(img2_numpy, cmap='Greys_r')
    plt.title(img2_title)
    plt.show()


# One imshow, Un-Normalized.
def show1Img(img, img_title):
    if len(img.shape) != 2:
        img = np.squeeze(img)

    plt.figure()
    plt.imshow(img, norm=None)
    plt.title(img_title)
    plt.show()


# Convert seconds to a nice string with hours, minutes and seconds.
def hms(seconds):
    seconds = np.floor(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    return "%02d:%02d:%02d" % (hours, minutes, seconds)


# Create target directory & all intermediate directories if don't exists
def create_dir_if_not_exists(dirName, verbatim=False):
    try:
        os.makedirs(dirName)
        print("Directory " , dirName ,  " Created ", flush=True)
    except FileExistsError:
        if verbatim:
            print("Directory " , dirName ,  " already exists", flush=True)


# Load all settings from a yaml file and stores it in a settings dictionary.
def load_settings_yaml(yaml_run_path, verbatim=True):
    #opens run.yaml and load all the settings into a dictionary.
    with open(yaml_run_path) as file:
        settings = yaml.load(file)
        if verbatim:
            print("\nSettings: {}".format(yaml_run_path), flush=True)
            for i in settings:
                print(str(i) + ": " + str(settings[i]), flush=True)
            print("\nAll settings loaded.\n\n", flush=True)
        return settings


# Returns a nicely formatted time string
def get_time_string():
    now = datetime.now()
    return now.strftime("%m_%d_%Y_%Hh_%Mm_%Ss")


# save a plot of binary accuracy and loss into the current model folder.
def save_loss_and_acc_figure(loss_per_chunk, bin_acc_per_chunk, params):
    x = range(1, len(bin_acc_per_chunk) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.xlabel("Chunk")
    plt.ylabel("Accuracy")
    plt.plot(x, bin_acc_per_chunk, 'b', label='Training acc')

    plt.title('Training binary accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.xlabel("Chunk")
    plt.ylabel('Loss')
    plt.plot(x, loss_per_chunk, 'r', label='Training loss')

    plt.title('Training Loss')
    plt.legend()
    plt.savefig(params.full_path_of_figure)
    print("\nsaved Loss and Accuracy figure to: {}".format(params.full_path_of_figure), flush=True)


# Define a nice plot function for the accuracy and loss over time
# History is the object returns by a model.fit()
def plot_history(acc, val_acc, loss, val_loss, params):
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training binary acc')
    plt.plot(x, val_acc[1:], 'r', label='Validation binary acc')
    plt.title('Training and validation binary accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss[1:], 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(params.full_path_of_figure)
    print("\nsaved Loss and Accuracy figure to: {}".format(params.full_path_of_figure), flush=True)
    plt.clf()
    

def bytes2gigabyes(num_bytes):
    gbs = (num_bytes/1024)/1024/1024
    # print("GBs: {}".format(gbs))
    return gbs


def print_stats_program():
    # gives a single float value
    print("\nCPU usage: {}%".format(psutil.cpu_percent()))

    # gives an object with many fields
    # print("Virtual Memory: {}".format(psutil.virtual_memory()))

    # you can convert that object to a dictionary 
    # print("Virtual Memory Properties: {}".format(dict(psutil.virtual_memory()._asdict())))

    # you can have the percentage of used RAM
    print("RAM usage: {}%".format(psutil.virtual_memory().percent))

    # you can calculate percentage of available memory
    print("Available memory: {:.01f}%".format(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total))


# Smooths a set of points according to an Exponential Moving Average
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
