import random
import numpy as np
import matplotlib.pyplot as plt


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
