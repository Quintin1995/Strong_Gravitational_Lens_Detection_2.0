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


# Perform the floodfill operation on pixel (20,20)
def do_floodfill(img, tolerance=0):
    filled_img = flood_fill(img, (20, 20), 0, tolerance=tolerance)
    return filled_img


# Scale brightness of pixel based on the distance from centre of the image
# According to the following formula: e^(-distance/x_scale), which is exponential decay
def scale_img_dist_from_centre(img, x_scale):

    # Image dimensions
    nx, ny = img.shape

    # x and y distnace vectors from center of the image
    x = np.arange(nx) - (nx-1)/2. 
    y = np.arange(ny) - (ny-1)/2.

    # Calculate distance 2d matrix from centre
    X, Y = np.meshgrid(x, y)
    d = np.sqrt(X**2 + Y**2)

    return (img * np.exp(-1*d/x_scale)).astype(int)


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


# do_sgf:           Scale-on-distance with Gaussian kernel convolution and Floodfill
#   This basically removes a whole bunch of noise objects in the image.
# do_square_crop:   take centering square out of the image, where the centering
#   square has emperically been determined to be: (74, 74) pixels. All lensing 
#   features should fall within this range.
# do_circular_crop: Basically does the same thing as do_square_crop. However,
#   all pixels outside the centered circle with radius 74 will be set to a zero value.
# x_scale:          Parameter of brighness scaling based on distance. The higher the
#   more distant objects (from centre) their brighness is retained. The lower, the
#   more distant objects (from centre) their brightness is reduced.
# tolerance:        Refereces floodfill, where pixel values with a difference of
#   tolerance are floodfilled. The higher the value the more pixels are filled up.
# Segmentations obtained with the max-tree and post processing (sgf) are used as mask for
# the original image.
def max_tree_segmenter(numpy_array, do_square_crop=False, do_circular_crop=False, do_sgf=False, x_scale=30, tolerance=20, area_th=45):

    print("\nmax tree segmenter, incoming shape={}".format(numpy_array.shape))

    #structuring element, connectivity 8, for each pixel all 8 surrounding pixels are considered.
    connectivity_kernel = np.ones(shape=(3,3), dtype=bool)

    # Bounding-box criteria, Area size threshold, Squeare crop size
    Wmin,Wmax           = 3, 60                 # Emperically determined (ED)
    Hmin,Hmax           = 3, 60                 # ED
    area_threshold      = area_th               # ED: size of a component
    square_crop_size    = 74                    # Size of square crop. shape=(74, 74)
    r                   = square_crop_size//2   # Radius
    t                   = time.time()           # Record time

    # create array where filtered data will be stored
    if do_square_crop:
        cx = numpy_array.shape[1]//2
        cy = numpy_array.shape[2]//2
        f_dat = numpy_array[:, cx-r:cx+r, cy-r:cy+r, :]
    else:
        f_dat = np.copy(numpy_array)        # work with a copy for now

    # loop over all images
    for i in range(f_dat.shape[0]):

        # Convert image format for Max-Tree
        img = np.clip(np.squeeze(f_dat[i,:]) * 255, 0, 255).astype('uint16')

        # Construct a Max-Tree
        mxt = siamxt.MaxTreeAlpha(img, connectivity_kernel)

        # Filter - Area Filter on max-tree datastructure
        mxt.areaOpen(area_threshold)

        # Computing boudingbox
        dx = mxt.node_array[7,:] - mxt.node_array[6,:]   # These specific idxs were predetermined, by the author of the package.
        dy = mxt.node_array[10,:] - mxt.node_array[9,:]  # These specific idxs were predetermined, by the author of the package.
        RR = 1.0 * area_threshold / (dx*dy)                        # Rectangularity

        # Filter - Selecting nodes that fit the criteria
        nodes = (dx > Hmin) & (dx<Hmax) & (dy>Wmin) & (dy<Wmax) & ((RR>1.1) | (RR<0.9))  #Emperically determined
        mxt.contractDR(nodes)     # Filter out nodes in the Max-Tree that do not fit the given criteria

        # Get the image from the max-tree data structure
        img_filtered = mxt.getImage()

        # Custom filter
        if do_sgf:
            img_filtered = scale_img_dist_from_centre(img_filtered, x_scale=x_scale)
            img_filtered = apply_gaussian_kernel(img_filtered)
            img_filtered = do_floodfill(img_filtered, tolerance=tolerance)
        
        f_dat[i] = np.expand_dims(img_filtered, axis=2) 
        
    print("\nmax tree segmenter time: {:.01f}s, filtered data shape={}\n".format(time.time() - t, f_dat.shape), flush=True)
    return f_dat


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
if False:
    mul = 7
    random.seed(325*mul)
    np.random.seed(789*mul)
dg = DataGenerator(params)


# 5.0 - Create Data
x_dat, y = dg.load_chunk(params.chunksize, dg.Xlenses_train, dg.Xnegatives_train, dg.Xsources_train, params.data_type, params.mock_lens_alpha_scaling)
print(y)
copy_x_dat = np.copy(x_dat)
seg_dat = max_tree_segmenter(x_dat, do_square_crop=True, do_circular_crop=False, do_sgf=True, x_scale=30, tolerance=20, area_th=45)

for i in range(5):
    show_random_img_plt_and_stats(copy_x_dat, num_imgs=1, title="dat", do_plot=False, do_seed=True, seed=87*i)
    show_random_img_plt_and_stats(seg_dat, num_imgs=1, title="seg dat", do_plot=False, do_seed=True, seed=87*i)
    plt.show()


