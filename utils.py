import random
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import yaml
import psutil



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
        print("Directory " , dirName ,  " Created ", flush=True)
    except FileExistsError:
        print("Directory " , dirName ,  " already exists", flush=True)


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


###################################3
# Piece of code that might be usefull for later:
###### Step 4.1 - Sanity check of the train and test chunk
# 1: Are both positive and negative examples within the same brightness ranges?
# 2: I have added a per image normalization, because a couple of outliers ruin the normalization per data array (That is my hypothesis at least.)
# if params.verbatim:
#     print(y_train_chunk)
#     idxs_pos = np.where(y_train_chunk == 1.0)
#     idxs_neg = np.where(y_train_chunk == 0.0)

#     for i in range(25):
#         pos_img = X_train_chunk[random.choice(list(idxs_pos[0]))]
#         neg_img = X_train_chunk[random.choice(list(idxs_neg[0]))]
#         show2Imgs(pos_img, neg_img, "pos max pixel: {0:.3f}".format(np.amax(pos_img)), "neg max pixel: {0:.3f}".format(np.amax(neg_img)))
#####################################3

#####################################3
# fig=plt.figure(figsize=(8,8))
# columns = 2
# rows = 1
# for j in range(1, columns*rows +1):
#     fig.add_subplot(rows, columns, j)
#     plt.imshow(imgs[j-1], cmap='Greys_r')
# plt.title("label = {}\nimage index={}".format(y[i], i))
# plt.show()
# #####################################3