import random
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import yaml


# Show the user some random images of the given numpy array, numpy array structured like: [num_imgs, width, height, num_channels]
def show_random_img_plt_and_stats(data_array, num_imgs, title):
    for _ in range(num_imgs):
        random_idx = random.randint(0, data_array.shape[0])
        img = np.squeeze(data_array[random_idx])                    #remove the color channel from the image for matplotlib
        print("\n")
        print(title + " image data type: {}".format(img.dtype.name))
        print(title + " image shape: {}".format(img.shape))
        print(title + " image min: {}".format(np.amin(img)))
        print(title + " image max: {}".format(np.amax(img)))
        plt.title(title)
        plt.imshow(img, norm=None)
        plt.show()


# Show 2 images next to each other. Squeeze out the color channel if it exist and it is equal to 1.
def show2Imgs(img1_numpy, img2_numpy, img1_title, img2_title):
    if len(img1_numpy.shape) != 2:
        img1_numpy = np.squeeze(img1_numpy)
    if len(img2_numpy.shape) != 2:
        img2_numpy = np.squeeze(img2_numpy)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img1_numpy, norm=None)
    plt.title(img1_title)
    plt.subplot(1, 2, 2)
    plt.imshow(img2_numpy, norm=None)
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
def create_dir_if_not_exists(dirName):
    try:
        os.makedirs(dirName)    
        print("Directory " , dirName ,  " Created ")
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")


# Load all settings from a yaml file and stores it in a settings dictionary.
def load_settings_yaml(yaml_run_path):
    #opens run.yaml and load all the settings into a dictionary.
    with open(yaml_run_path) as file:
        settings = yaml.load(file)
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
    plt.plot(x, val_acc, 'r', label='Validation binary acc')
    plt.title('Training and validation binary accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(params.full_path_of_figure)
    print("\nsaved Loss and Accuracy figure to: {}".format(params.full_path_of_figure), flush=True)