import numpy as np
from skimage.segmentation import flood, flood_fill
import siamxt
from scipy import ndimage
import time


######################## functions ########################

# Normalizationp per image
def _normalize_img(numpy_img):
    return ((numpy_img - np.amin(numpy_img)) / (np.amax(numpy_img) - np.amin(numpy_img) + 0.0000001))

# Perform the floodfill operation on pixel (2,2) - I just picked a pixel in the corner
def _perform_floodfill(img, tolerance=0):
    filled_img = flood_fill(img, (2, 2), 0, tolerance=tolerance)
    return filled_img


# Crops the given numpy array around the given center coordinates cx and cy, with radius r.
# The given numpy array should be 4-dimensional: (num_imgs, widht, height, channel)
def _centre_crop_square(numpy_array, cx, cy, r):
    return numpy_array[:, cx-r:cx+r, cy-r:cy+r, :]


# Returns a predefined convolutional kernel. To be used for convolution.
# Convolutional kernels should have odd dimensions.
def _get_kernel(method="gaussian", size=(3,3)):
    if method == "gaussian":
        if size == (3,3):
            kernel = np.array([[1,2,1],
                               [2,4,2],
                               [1,2,1]]) * (1/16.)
        if size == (5,5):
            kernel = np.array([[1, 4, 6, 4,1],
                               [4,16,24,16,4],
                               [6,24,36,24,6],
                               [4,16,24,16,4],
                               [1, 4, 6, 4,1]]) * (1/256.)
    if method == "boxcar":
        if size == (3,3):
            kernel = np.array([[1,1,1],
                               [1,1,1],
                               [1,1,1]]) * (1/9.)
        if size == (5,5):
            kernel = np.array([[1,1,1,1,1],
                               [1,1,1,1,1],
                               [1,1,1,1,1],
                               [1,1,1,1,1],
                               [1,1,1,1,1]]) * (1/25.)
    return kernel


# Convolve the given 2D kernel to the numpy array.
# Either per image stack (4D) or per image (2D)
def _apply_conv_kernel(numpy_array, kernel):
    if numpy_array.ndim != 2:
        for i in range(numpy_array.shape[0]):
            img = np.squeeze(numpy_array[i])            # Squeeze out empty dimensions
            conv_img = ndimage.convolve(img, kernel, mode='constant', cval=0.0) 
            numpy_array[i] = np.expand_dims(conv_img, axis = 2) # Add empty dimension back in. (color channel)
        return numpy_array
    if numpy_array.ndim == 2:
        return ndimage.convolve(numpy_array, kernel, mode='constant', cval=0.0)


# Scale brightness of pixel based on the distance from centre of the image
# According to the following formula: e^(-distance/x_scale), which is exponential decay
def _scale_img_dist_from_centre(img, x_scale):

    # Image dimensions
    nx, ny = img.shape

    # x and y distance vectors from center of the image
    x = np.arange(nx) - (nx-1)/2. 
    y = np.arange(ny) - (ny-1)/2.

    # Calculate distance 2d matrix from centre
    X, Y = np.meshgrid(x, y)
    d = np.sqrt(X**2 + Y**2)

    # Multiply the image with 2d distance profile (brightness scaled down, when further away from centre.)
    return (img * np.exp(-1*d/x_scale)).astype(int)


# Return a tuple of 2 arrays. The x-indexes and y-indexes.
# These indexes represent the whole area of the image where the circle is NOT located.
def _get_indexes_negated_circle_crop(width, height, cx, cy, r):

    # Define zero'ed array
    x = np.arange(0, width)
    y = np.arange(0, height)
    circle_mask = np.zeros((y.size, x.size))

    # Create a circle based on cx, cy, and r
    circle = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < r**2

    # Impute the circle into the empty/nulled image.
    circle_mask[circle] = 1.      # any non-zero value will work.

    return np.where(circle_mask == 0)


# Constructs a max tree and filters on it with an area filter, and bounding box filter.
def _contruct_max_tree_and_filter(img, connectivity_kernel, area_threshold):

    # Bounding box parameters
    Wmin,Wmax           = 3, 60         # Emperically determined
    Hmin,Hmax           = 3, 60         # Emperically determined

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
    return mxt.getImage()



# Segmentations obtained with the max-tree and pre/post processing.
# do_scale:         |Whether to scale objects their brighness based on distance from
#   the centre of the image.
# do_floodfill:     |Whether to perform the floodfill operation with the given tolerance.
#   This basically removes a whole bunch of noise objects in the image.
# do_square_crop:   |Take centering do out of the image, where the centering
#   square has emperically been determined to be: (74, 74) pixels. All lensing 
#   features should fall within this range.
# do_circular_crop: |Basically does the same thing as do_square_crop. However,
#   all pixels outside the centered circle with radius 74 will be set to a zero value.
# x_scale:          Parameter of brighness scaling based on distance. The higher the
#   more distant objects (from centre) their brighness is retained. The lower, the
#   more distant objects (from centre) their brightness is reduced.
# tolerance:        |Refereces floodfill, where pixel values with a difference of
#   tolerance are floodfilled. The higher the value the more pixels are filled up.
# conv_method:      |Possibilities: "gaussian" or "boxcar". Convolves the img with the given method
# ksize:            |Convolutional kernel size.
# use_seg_imgs:     |Whether to use segmented images or masked images. If False, the original
#   images are masked with the segmented images and returned.
def max_tree_segmenter(numpy_array,
                       do_square_crop=False,
                       square_crop_size=74,
                       do_circular_crop=False,
                       do_scale=False,
                       do_floodfill=False,
                       x_scale=30,
                       tolerance=20,
                       area_th=45,
                       conv_method="gaussian",
                       ksize=(3,3),
                       use_seg_imgs=False):

    #structuring element, connectivity 8, for each pixel all 8 surrounding pixels are considered.
    connectivity_kernel = np.ones(shape=(3,3), dtype=bool)
    t                   = time.time()               # Record time
    
    # Define a circular mask based on the incoming image dimensions
    r                     = square_crop_size//2       # Radius
    cx                    = numpy_array.shape[1]//2
    cy                    = numpy_array.shape[2]//2

    if use_seg_imgs:
        copy_dat = np.copy(numpy_array)

    # Define a kernel for convolution
    kernel = _get_kernel(method=conv_method, size=ksize)

    # This only needs to be computed once per 4d numpy array
    xys_tup = _get_indexes_negated_circle_crop(numpy_array.shape[1], numpy_array.shape[2], cx, cy, r)

    for i in range(numpy_array.shape[0]):

        # Convert image format for Max-Tree
        img = np.clip(np.squeeze(numpy_array[i,:]) * 255, 0, 255).astype('uint16')

        # Apply a boxcar kernel
        img_filtered = _apply_conv_kernel(img, kernel)

        # Perform circular crop
        if do_circular_crop:
            img_filtered[xys_tup] = 0

        # Construct a max tree and filter based on area and rectangularity.
        img_filtered = _contruct_max_tree_and_filter(img, connectivity_kernel, area_th)

        # Filter on distance from centre
        if do_scale:
            img_filtered = _scale_img_dist_from_centre(img_filtered, x_scale=x_scale)

        # Filter with a floodfill
        if do_floodfill:
            img_filtered = _perform_floodfill(img_filtered, tolerance=tolerance)

        # Add color channel
        if use_seg_imgs:
            copy_dat[i] =_normalize_img(np.expand_dims(img_filtered, axis=2))

        # Mask filtered (segmented) Image on Original image
        numpy_array[i][np.where(img_filtered == 0)] = 0.0

    # Do centre square cropping.
    if do_square_crop:
        numpy_array = _centre_crop_square(numpy_array, cx, cy, r)
    
    print("\nmax tree segmenter time: {:.01f}s, masked data shape={}\n".format(time.time() - t, numpy_array.shape), flush=True)
    if use_seg_imgs:
        return _centre_crop_square(copy_dat, cx, cy, r)
    return numpy_array


######################## end-functions ########################