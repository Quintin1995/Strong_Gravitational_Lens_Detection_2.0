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

# Proof of concept of surface brightness contouring



def apply_gaussian_kernel(numpy_array, kernel = None):
    if kernel == None:
        kernel = np.array([[1,2,1],
                            [2,4,2],
                            [1,2,1]]) * (1/16)
    for i in range(numpy_array.shape[0]):
        img = np.squeeze(numpy_array[i])
        conv_img = ndimage.convolve(img, kernel, mode='constant', cval=0.0) 
        numpy_array[i] = np.expand_dims(conv_img, axis = 2)
    return x

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
    # rr = 0.45

    # Computing bouding-box lengths from the
    # attributes stored in NA
    dx = mxt.node_array[7,:] - mxt.node_array[6,:]
    dy = mxt.node_array[10,:] - mxt.node_array[9,:]
    area = mxt.node_array[3,:]
    # RR = 1.0 * area / (dx*dy)

    # Computes the area extinction values
    area_ext = mxt.computeExtinctionValues(area,"area")

    # Applies the  area extinction filter
    n=10
    mxt.extinctionFilter(area_ext, n)
    
    # Filter based on area
    # areas = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    # for area in areas:
    # area = 25
    # mxt.areaOpen(area)

    # Selecting nodes that fit the criteria
    nodes = (dx > Hmin) & (dx<Hmax) & (dy>Wmin) & (dy<Wmax)# & (RR>rr)

    # # Filtering
    mxt.contractDR(nodes)
    print("Max-tree build time: %fs" %t)
    print("Number of max-tree nodes: %d" %mxt.node_array.shape[1])
    print("Number of max-tree leaves: %d" %(mxt.node_array[1,:] == 0).sum())
    img_filtered = mxt.getImage()

    imgs = [img, img_filtered]

    fig=plt.figure(figsize=(8,8))
    columns = 2
    rows = 1
    for j in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, j)
        plt.imshow(imgs[j-1], cmap='Greys_r')
    plt.title("label = {}".format(y[i]))
    plt.show()
