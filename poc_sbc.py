import siamxt
from scipy import ndimage
import argparse
from DataGenerator import DataGenerator
from Network import Network
import time
import random
import numpy as np
from Parameters import Parameters
from utils import load_settings_yaml, show_random_img_plt_and_stats
import matplotlib.pyplot as plt
from skimage.segmentation import flood, flood_fill


def do_floodfill(img, tolerance=0):
    filled_img = flood_fill(img, (20, 20), 0, tolerance=tolerance)
    return filled_img


def scale_img_dist_from_centre(img, x_scale):
    nx, ny = img.shape
    x = np.arange(nx) - (nx-1)/2.  # x an y so they are distance from center, assuming array is "nx" long (as opposed to 1. which is the other common choice)
    y = np.arange(ny) - (ny-1)/2.
    X, Y = np.meshgrid(x, y)
    d = np.sqrt(X**2 + Y**2)
    return img * np.exp(-1*d/x_scale)


def apply_gaussian_kernel(numpy_array, kernel = None):
    if kernel == None:
        kernel = np.array([[1,2,1],
                            [2,4,2],
                            [1,2,1]]) * (1/16)
    if numpy_array.ndim != 2:
        for i in range(numpy_array.shape[0]):
            img = np.squeeze(numpy_array[i])
            conv_img = ndimage.convolve(img, kernel, mode='constant', cval=0.0) 
            numpy_array[i] = np.expand_dims(conv_img, axis = 2)
        return numpy_array
    if numpy_array.ndim == 2:
        return ndimage.convolve(numpy_array, kernel, mode='constant', cval=0.0)



################################# script #################################


# 1.0 - Define ArgumentParser
parser = argparse.ArgumentParser()
parser.add_argument("--run", help="Location/path of the run.yaml file. This is usually structured as a path.", default="runs/run_poc_sbc.yaml", required=False)
args = parser.parse_args()


# 2.0 - Unpack args
yaml_path = args.run


# 3.0 - Load all settings from .yaml file
settings_yaml = load_settings_yaml(yaml_path)                                           # Returns a dictionary object.
params = Parameters(settings_yaml, yaml_path)
params.data_type = np.float32 if params.data_type == "np.float32" else np.float32       # This must be done here, due to the json, not accepting this kind of if statement in the parameter class.


# 4.0 - Create Custom Data Generator
if True:
    mul = 7
    random.seed(325*mul)
    np.random.seed(789*mul)
dg = DataGenerator(params)


# 5.0 - Create Data
x, y = dg.load_chunk(params.chunksize, dg.Xlenses_train, dg.Xnegatives_train, dg.Xsources_train, params.data_type, params.mock_lens_alpha_scaling)
print(y)
print(y.shape)

# 5.1 - Visualize some data
if False:
    plt.figure()
    plt.imshow(np.squeeze(x[0]), origin='lower', interpolation='none', cmap=plt.cm.binary)
    plt.show()

#structuring element, connectivity -8
Bc8 = np.ones(shape=(3,3), dtype=bool)
Bc4 = np.array([[False,True,False],
                  [True,True,True],
                  [False,True,False]], dtype=bool)

for i in range(x.shape[0]):
    # 6.1 - Convert img format and plot
    img = np.clip(np.squeeze(x[i,:]) * 255, 0, 255)
    img = img.astype('uint16')

    t = time.time()
    mxt = siamxt.MaxTreeAlpha(img, Bc8)
    t = time.time() - t

    # Size and shape threshold
    Wmin,Wmax = 3,60
    Hmin,Hmax = 3,60
    rr = 0.45

    # Filter 1
    # Area filter
    area = 45
    mxt.areaOpen(area)

    # Computing bouding-box lengths from the
    dx = mxt.node_array[7,:] - mxt.node_array[6,:]
    dy = mxt.node_array[10,:] - mxt.node_array[9,:]
    RR = 1.0 * area / (dx*dy)

    # Filter 2
    # Selecting nodes that fit the criteria
    nodes = (dx > Hmin) & (dx<Hmax) & (dy>Wmin) & (dy<Wmax) & ((RR>1.1) | (RR<0.9))

    # Filtering
    mxt.contractDR(nodes)
    print("Max-tree build time: %fs" %t)
    print("Number of max-tree nodes: %d" %mxt.node_array.shape[1])
    print("Number of max-tree leaves: %d" %(mxt.node_array[1,:] == 0).sum())
    img_filtered = mxt.getImage()

    # Filter 3
    img_filtered = scale_img_dist_from_centre(img_filtered, x_scale=30)

    # Filter 4
    img_filtered = apply_gaussian_kernel(img_filtered)

    # Filter 5
    img_filtered = do_floodfill(img_filtered, tolerance=20)

    # Add imgs to list for plotting
    imgs = [img, img_filtered]

    fig=plt.figure(figsize=(8,8))
    columns = 2
    rows = 1
    for j in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, j)
        plt.imshow(imgs[j-1], cmap='Greys_r')
    plt.title("label = {}\nimage index={}".format(y[i], i))
    plt.show()
